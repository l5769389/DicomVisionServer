import asyncio
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from time import perf_counter

import socketio

from app.core.workspace import DEFAULT_WORKSPACE_ID, normalize_workspace_id
from app.core.logging import get_logger
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service
from app.services.volume_rendering.vtk_threading import should_run_3d_view_on_main_thread
from app.services.webrtc_3d_transport import webrtc_3d_transport_manager

MPR_PREVIEW_BATCH_MIN_INTERVAL_SECONDS = 0.0
logger = get_logger(__name__)


@dataclass
class RenderRequest:
    image_format: str = "webp"
    fast_preview: bool = False
    fast_preview_full_resolution: bool = False
    metadata_mode: str = "full"
    target_sids: tuple[str, ...] | None = None
    mpr_revision: int | None = None
    render_revision: int | None = None
    interaction_id: str | None = None


class ViewSocketHub:
    def __init__(self) -> None:
        self._server: socketio.AsyncServer | None = None
        self._view_sids: dict[str, set[str]] = defaultdict(set)
        self._sid_views: dict[str, set[str]] = defaultdict(set)
        self._sid_workspaces: dict[str, str] = {}
        self._render_locks: dict[str, asyncio.Lock] = {}
        self._pending_render_requests: dict[str, dict[str, RenderRequest]] = {}
        self._preview_worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._mpr_preview_worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._last_mpr_preview_batch_started_at: dict[str, float] = {}
        self._mpr_final_preemption_tokens: dict[str, int] = {}
        self._mpr_final_preemption_revisions: dict[str, int] = {}
        self._view_final_render_revisions: dict[str, int] = {}
        self._render_revisions: dict[str, int] = defaultdict(int)
        self._view_active_interaction_ids: dict[str, str] = {}
        self._delayed_final_render_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed_view_ids: set[str] = set()

    def attach_server(self, server: socketio.AsyncServer) -> None:
        self._server = server

    def bind_sid_workspace(self, sid: str, workspace_id: str | None) -> str:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        self._sid_workspaces[sid] = normalized_workspace_id
        return normalized_workspace_id

    def get_sid_workspace(self, sid: str) -> str:
        return self._sid_workspaces.get(sid, DEFAULT_WORKSPACE_ID)

    def bind_view(self, sid: str, view_id: str) -> None:
        if view_id in self._closed_view_ids:
            return
        self._view_sids[view_id].add(sid)
        self._sid_views[sid].add(view_id)

    def get_view_sids(self, view_id: str, target_sids: tuple[str, ...] | None = None) -> tuple[str, ...]:
        return self._resolve_target_sids(view_id, target_sids)

    def unbind_sid(self, sid: str) -> None:
        self._sid_workspaces.pop(sid, None)
        view_ids = self._sid_views.pop(sid, set())
        for view_id in view_ids:
            sids = self._view_sids.get(view_id)
            if sids is None:
                continue
            sids.discard(sid)
            if not sids:
                self._view_sids.pop(view_id, None)

    def unbind_view(self, view_id: str) -> None:
        webrtc_3d_transport_manager.schedule_close_view(view_id)
        sids = self._view_sids.pop(view_id, set())
        for sid in sids:
            views = self._sid_views.get(sid)
            if views is None:
                continue
            views.discard(view_id)
            if not views:
                self._sid_views.pop(sid, None)
        for queue_key in tuple(self._pending_render_requests.keys()):
            pending_requests = self._pending_render_requests[queue_key]
            pending_requests.pop(view_id, None)
            if not pending_requests:
                self._pending_render_requests.pop(queue_key, None)
        view_queue_key = f"view:{view_id}"
        self._render_locks.pop(view_queue_key, None)
        self._view_final_render_revisions.pop(view_id, None)
        self._view_active_interaction_ids.pop(view_id, None)
        preview_task = self._preview_worker_tasks.pop(view_queue_key, None)
        if preview_task is not None and not preview_task.done():
            preview_task.cancel()
        self._cancel_delayed_final_render(view_id)

    def close_view(self, view_id: str) -> None:
        if not view_id:
            return
        self._closed_view_ids.add(view_id)
        self.unbind_view(view_id)

    def is_view_closed(self, view_id: str) -> bool:
        return view_id in self._closed_view_ids

    def _get_render_lock(self, queue_key: str) -> asyncio.Lock:
        lock = self._render_locks.get(queue_key)
        if lock is None:
            lock = asyncio.Lock()
            self._render_locks[queue_key] = lock
        return lock

    @staticmethod
    def _resolve_render_queue_key(view_id: str) -> str:
        try:
            view = view_registry.get(view_id)
        except Exception:
            return f"view:{view_id}"
        if view.view_group is not None and str(view.view_group.group_type).lower() in {"mpr", "fusion"}:
            return f"mpr-group:{view.view_group.group_id}"
        return f"view:{view_id}"

    @staticmethod
    def _should_render_on_main_thread(view_id: str) -> bool:
        try:
            view = view_registry.get(view_id)
        except Exception:
            return False
        view_type = getattr(view, "view_type", None)
        return should_run_3d_view_on_main_thread(view_type) if view_type is not None else False

    @staticmethod
    def _is_mpr_group_queue(queue_key: str) -> bool:
        return queue_key.startswith("mpr-group:")

    @staticmethod
    def _is_preview_render_request(request: RenderRequest) -> bool:
        return request.image_format == "jpeg" or request.fast_preview

    @staticmethod
    def _is_final_render_request(request: RenderRequest) -> bool:
        return request.image_format in {"png", "webp"} and not request.fast_preview

    @classmethod
    def _is_preview_render_batch(cls, request_batch: dict[str, RenderRequest]) -> bool:
        return bool(request_batch) and all(cls._is_preview_render_request(request) for request in request_batch.values())

    @classmethod
    def _has_final_render_request(cls, request_batch: dict[str, RenderRequest] | None) -> bool:
        return bool(request_batch) and any(cls._is_final_render_request(request) for request in request_batch.values())

    @staticmethod
    def _merge_render_request(current: RenderRequest, incoming: RenderRequest) -> RenderRequest:
        if current.target_sids is None or incoming.target_sids is None:
            target_sids = None
        else:
            target_sids = tuple(dict.fromkeys((*current.target_sids, *incoming.target_sids)))

        current_is_final = ViewSocketHub._is_final_render_request(current)
        incoming_is_final = ViewSocketHub._is_final_render_request(incoming)
        incoming_is_newer = (
            current.mpr_revision is not None
            and incoming.mpr_revision is not None
            and int(incoming.mpr_revision) > int(current.mpr_revision)
        )
        if current_is_final and not incoming_is_final and not incoming_is_newer:
            chosen = current
        else:
            chosen = incoming

        return RenderRequest(
            image_format=chosen.image_format,
            fast_preview=chosen.fast_preview,
            fast_preview_full_resolution=chosen.fast_preview_full_resolution,
            metadata_mode=chosen.metadata_mode,
            target_sids=target_sids,
            mpr_revision=chosen.mpr_revision
            if chosen.mpr_revision is not None
            else ViewSocketHub._choose_render_mpr_revision(current.mpr_revision, incoming.mpr_revision),
            render_revision=chosen.render_revision,
            interaction_id=chosen.interaction_id,
        )

    def next_render_revision(self, view_id: str) -> int:
        self._render_revisions[view_id] += 1
        return self._render_revisions[view_id]

    def _cancel_delayed_final_render(self, view_id: str) -> None:
        task = self._delayed_final_render_tasks.pop(view_id, None)
        if task is not None and not task.done():
            task.cancel()

    def mark_view_interaction(self, view_id: str, interaction_id: str | None) -> None:
        if not interaction_id:
            return
        normalized_interaction_id = str(interaction_id)
        self._view_active_interaction_ids[view_id] = normalized_interaction_id
        self._cancel_delayed_final_render(view_id)
        queue_key = self._resolve_render_queue_key(view_id)
        pending_requests = self._pending_render_requests.get(queue_key)
        if not pending_requests:
            return
        pending_request = pending_requests.get(view_id)
        if pending_request is not None and pending_request.interaction_id != normalized_interaction_id:
            pending_requests.pop(view_id, None)
        if not pending_requests:
            self._pending_render_requests.pop(queue_key, None)

    def make_render_request(
        self,
        view_id: str,
        *,
        image_format: str = "webp",
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        target_sids: tuple[str, ...] | None = None,
        mpr_revision: int | None = None,
        render_revision: int | None = None,
        interaction_id: str | None = None,
    ) -> RenderRequest:
        return RenderRequest(
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            target_sids=target_sids,
            mpr_revision=mpr_revision,
            render_revision=render_revision if render_revision is not None else self.next_render_revision(view_id),
            interaction_id=interaction_id,
        )

    @staticmethod
    def _choose_render_mpr_revision(current: int | None, incoming: int | None) -> int | None:
        if current is None:
            return incoming
        if incoming is None:
            return current
        return max(int(current), int(incoming))

    def _queue_pending_render(self, queue_key: str, view_id: str, incoming_request: RenderRequest) -> None:
        if self.is_view_closed(view_id):
            return
        if (
            self._is_stale_preview_after_final(queue_key, view_id, incoming_request)
            or self._is_stale_interaction_request(view_id, incoming_request)
        ):
            return
        pending_requests = self._pending_render_requests.setdefault(queue_key, {})
        current_request = pending_requests.get(view_id)
        pending_requests[view_id] = (
            incoming_request
            if current_request is None
            else self._merge_render_request(current_request, incoming_request)
        )

    def _pop_pending_render_batch(self, queue_key: str) -> dict[str, RenderRequest]:
        return self._pending_render_requests.pop(queue_key, {})

    async def _coalesce_mpr_preview_batch(
        self,
        queue_key: str,
        request_batch: dict[str, RenderRequest],
    ) -> dict[str, RenderRequest]:
        if not self._is_mpr_group_queue(queue_key) or not self._is_preview_render_batch(request_batch):
            return request_batch

        request_batch = {
            view_id: request
            for view_id, request in request_batch.items()
            if not self._is_stale_preview_after_final(queue_key, view_id, request)
        }
        if not request_batch:
            return {}

        preemption_token = self._mpr_final_preemption_tokens.get(queue_key, 0)
        latest_pending_batch = self._pending_render_requests.get(queue_key)
        if self._has_final_render_request(latest_pending_batch):
            return self._pop_pending_render_batch(queue_key)

        last_started_at = self._last_mpr_preview_batch_started_at.get(queue_key)
        if last_started_at is not None:
            elapsed_seconds = perf_counter() - last_started_at
            delay_seconds = MPR_PREVIEW_BATCH_MIN_INTERVAL_SECONDS - elapsed_seconds
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        if self._mpr_final_preemption_tokens.get(queue_key, 0) != preemption_token:
            return {}

        latest_pending_batch = self._pending_render_requests.get(queue_key)
        if latest_pending_batch:
            request_batch = self._pop_pending_render_batch(queue_key)

        self._last_mpr_preview_batch_started_at[queue_key] = perf_counter()
        return request_batch

    def _pop_sibling_pending_render_batch(
        self,
        queue_key: str,
        current_batch: dict[str, RenderRequest],
    ) -> dict[str, RenderRequest]:
        pending_requests = self._pending_render_requests.get(queue_key)
        if not pending_requests:
            return {}

        sibling_batch: dict[str, RenderRequest] = {}
        for pending_view_id in tuple(pending_requests.keys()):
            if pending_view_id in current_batch:
                continue
            sibling_batch[pending_view_id] = pending_requests.pop(pending_view_id)

        if not pending_requests:
            self._pending_render_requests.pop(queue_key, None)
        return sibling_batch

    def _discard_pending_preview_requests(self, queue_key: str) -> None:
        pending_requests = self._pending_render_requests.get(queue_key)
        if not pending_requests:
            return

        for pending_view_id, pending_request in tuple(pending_requests.items()):
            if self._is_preview_render_request(pending_request):
                pending_requests.pop(pending_view_id, None)

        if not pending_requests:
            self._pending_render_requests.pop(queue_key, None)

    def _replace_pending_preview_batch(self, queue_key: str, request_batch: dict[str, RenderRequest]) -> bool:
        self._discard_pending_preview_requests(queue_key)
        if self._has_final_render_request(self._pending_render_requests.get(queue_key)):
            return False
        for view_id, request in tuple(request_batch.items()):
            if self._is_stale_preview_after_final(queue_key, view_id, request):
                request_batch.pop(view_id, None)
                continue
            self._queue_pending_render(queue_key, view_id, request)
        return bool(request_batch)

    def _cancel_mpr_preview_worker(self, queue_key: str) -> None:
        task = self._mpr_preview_worker_tasks.pop(queue_key, None)
        if task is not None and not task.done():
            task.cancel()

    def _cancel_preview_worker(self, queue_key: str) -> None:
        task = self._preview_worker_tasks.pop(queue_key, None)
        if task is not None and not task.done():
            task.cancel()

    def _ensure_mpr_preview_worker(self, queue_key: str) -> None:
        task = self._mpr_preview_worker_tasks.get(queue_key)
        if task is not None and not task.done():
            return
        self._mpr_preview_worker_tasks[queue_key] = asyncio.create_task(self._run_mpr_preview_worker(queue_key))

    def _ensure_preview_worker(self, queue_key: str) -> None:
        task = self._preview_worker_tasks.get(queue_key)
        if task is not None and not task.done():
            return
        self._preview_worker_tasks[queue_key] = asyncio.create_task(self._run_preview_worker(queue_key))

    def _mark_mpr_final_preemption(self, queue_key: str) -> None:
        if not self._is_mpr_group_queue(queue_key):
            return
        self._mpr_final_preemption_tokens[queue_key] = self._mpr_final_preemption_tokens.get(queue_key, 0) + 1

    def _remember_mpr_final_revision(self, queue_key: str, request: RenderRequest) -> None:
        if not self._is_mpr_group_queue(queue_key) or request.mpr_revision is None:
            return
        self._mpr_final_preemption_revisions[queue_key] = max(
            int(request.mpr_revision),
            self._mpr_final_preemption_revisions.get(queue_key, -1),
        )

    def _remember_view_final_revision(self, view_id: str, request: RenderRequest) -> None:
        if request.render_revision is None:
            return
        self._view_final_render_revisions[view_id] = max(
            int(request.render_revision),
            self._view_final_render_revisions.get(view_id, -1),
        )

    def _is_stale_mpr_preview_after_final(self, queue_key: str, request: RenderRequest) -> bool:
        if not self._is_mpr_group_queue(queue_key) or not self._is_preview_render_request(request):
            return False
        final_revision = self._mpr_final_preemption_revisions.get(queue_key)
        if final_revision is None or request.mpr_revision is None:
            return False
        request_revision = int(request.mpr_revision)
        final_revision_value = int(final_revision)
        return request_revision < final_revision_value

    def _is_stale_view_preview_after_final(self, view_id: str, request: RenderRequest) -> bool:
        if not self._is_preview_render_request(request) or request.render_revision is None:
            return False
        final_revision = self._view_final_render_revisions.get(view_id)
        if final_revision is None:
            return False
        return int(request.render_revision) < int(final_revision)

    def _is_stale_preview_after_final(self, queue_key: str, view_id: str, request: RenderRequest) -> bool:
        return (
            self._is_stale_mpr_preview_after_final(queue_key, request)
            or self._is_stale_view_preview_after_final(view_id, request)
        )

    def _is_stale_interaction_request(self, view_id: str, request: RenderRequest) -> bool:
        if request.interaction_id is None:
            return False
        active_interaction_id = self._view_active_interaction_ids.get(view_id)
        return active_interaction_id is not None and request.interaction_id != active_interaction_id

    def _resolve_target_sids(self, view_id: str, target_sids: tuple[str, ...] | None) -> tuple[str, ...]:
        if self.is_view_closed(view_id):
            return ()
        if target_sids is not None:
            return target_sids
        return tuple(self._view_sids.get(view_id, ()))

    def _should_suppress_preview_emit(self, view_id: str, request: RenderRequest, preemption_token: int) -> bool:
        if self._is_stale_interaction_request(view_id, request):
            return True
        if not self._is_preview_render_request(request):
            return False

        queue_key = self._resolve_render_queue_key(view_id)
        if self._is_stale_preview_after_final(queue_key, view_id, request):
            return True
        if not self._is_mpr_group_queue(queue_key):
            return False

        if self._has_final_render_request(self._pending_render_requests.get(queue_key)):
            return True

        return self._mpr_final_preemption_tokens.get(queue_key, 0) != preemption_token

    async def _sleep_until_next_mpr_preview_batch(self, queue_key: str) -> None:
        last_started_at = self._last_mpr_preview_batch_started_at.get(queue_key)
        if last_started_at is None:
            return
        elapsed_seconds = perf_counter() - last_started_at
        delay_seconds = MPR_PREVIEW_BATCH_MIN_INTERVAL_SECONDS - elapsed_seconds
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    def _filter_stale_preview_batch(
        self,
        queue_key: str,
        request_batch: dict[str, RenderRequest],
    ) -> dict[str, RenderRequest]:
        return {
            view_id: request
            for view_id, request in request_batch.items()
            if not self._is_stale_preview_after_final(queue_key, view_id, request)
            and not self._is_stale_interaction_request(view_id, request)
        }

    async def _run_mpr_preview_worker(self, queue_key: str) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                request_batch = self._pop_pending_render_batch(queue_key)
                request_batch = self._filter_stale_preview_batch(queue_key, request_batch)
                if not request_batch:
                    return

                preemption_token = self._mpr_final_preemption_tokens.get(queue_key, 0)
                await self._sleep_until_next_mpr_preview_batch(queue_key)
                if self._mpr_final_preemption_tokens.get(queue_key, 0) != preemption_token:
                    return

                latest_batch = self._pop_pending_render_batch(queue_key)
                if latest_batch:
                    request_batch = latest_batch
                request_batch = self._filter_stale_preview_batch(queue_key, request_batch)
                if not request_batch:
                    continue

                self._last_mpr_preview_batch_started_at[queue_key] = perf_counter()
                await asyncio.gather(
                    *(
                        self._emit_render_message_safely(view_id, request)
                        for view_id, request in request_batch.items()
                    )
                )
        except asyncio.CancelledError:
            return
        finally:
            if self._mpr_preview_worker_tasks.get(queue_key) is current_task:
                self._mpr_preview_worker_tasks.pop(queue_key, None)

    async def _run_preview_worker(self, queue_key: str) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                request_batch = self._pop_pending_render_batch(queue_key)
                if not request_batch:
                    return
                await asyncio.gather(
                    *(
                        self.emit_render_for_view(
                            view_id,
                            image_format=request.image_format,
                            fast_preview=request.fast_preview,
                            fast_preview_full_resolution=request.fast_preview_full_resolution,
                            metadata_mode=request.metadata_mode,
                            target_sids=request.target_sids,
                            mpr_revision=request.mpr_revision,
                            render_revision=request.render_revision,
                            interaction_id=request.interaction_id,
                        )
                        for view_id, request in request_batch.items()
                    )
                )
        except asyncio.CancelledError:
            return
        finally:
            if self._preview_worker_tasks.get(queue_key) is current_task:
                self._preview_worker_tasks.pop(queue_key, None)

    @staticmethod
    def _resolve_render_intent(request: RenderRequest) -> str:
        if request.metadata_mode in {"stack-pixel-preview", "mpr-pixel-preview"}:
            return "pixel-only"
        if request.metadata_mode in {
            "stack-geometry-preview",
            "stack-zoom-preview",
            "mpr-pan-zoom-preview",
            "mpr-zoom-preview",
            "mpr-crosshair-preview",
            "stack-preview-lite",
            "fusion-zoom-preview",
        }:
            return "geometry-preview"
        if request.metadata_mode in {"mpr-segmentation-preview", "fusion-registration-layer-preview"}:
            return "overlay-preview"
        if request.fast_preview:
            return "geometry-preview"
        return "full"

    @classmethod
    def build_image_update_payload(cls, result_meta, request: RenderRequest) -> dict[str, object]:
        payload = result_meta.model_dump(by_alias=True)
        payload["fastPreview"] = bool(request.fast_preview)
        payload["fastPreviewFullResolution"] = bool(request.fast_preview_full_resolution)
        payload["metadataMode"] = request.metadata_mode
        payload["renderIntent"] = cls._resolve_render_intent(request)
        if request.render_revision is not None:
            payload["renderRevision"] = int(request.render_revision)
        if request.interaction_id is not None:
            payload["interactionId"] = request.interaction_id

        if request.metadata_mode in {"stack-preview-lite", "stack-pixel-preview"}:
            payload.pop("measurements", None)
            payload.pop("annotations", None)
        elif request.metadata_mode == "mpr-pixel-preview":
            payload.pop("measurements", None)
            payload.pop("annotations", None)
            payload.pop("mprSegmentationOverlay", None)
            payload.pop("mpr_segmentation_overlay", None)
        elif request.metadata_mode in {"mpr-pan-zoom-preview", "mpr-zoom-preview"}:
            payload.pop("cornerInfo", None)
            payload.pop("orientation", None)
        elif request.metadata_mode == "mpr-crosshair-preview":
            payload.pop("cornerInfo", None)
            payload.pop("orientation", None)
            payload.pop("scaleBar", None)
            payload.pop("measurements", None)
            payload.pop("annotations", None)
            payload.pop("mprSegmentationOverlay", None)
            payload.pop("mpr_segmentation_overlay", None)
        elif request.metadata_mode == "fusion-registration-layer-preview":
            payload.pop("cornerInfo", None)
            payload.pop("orientation", None)
            payload.pop("scaleBar", None)
            payload.pop("measurements", None)
            payload.pop("annotations", None)
            payload.pop("fusionProjection", None)
        elif request.metadata_mode == "fusion-zoom-preview":
            payload.pop("cornerInfo", None)
            payload.pop("orientation", None)
        return payload

    @staticmethod
    def _build_image_update_payload(result_meta, request: RenderRequest) -> dict[str, object]:
        return ViewSocketHub.build_image_update_payload(result_meta, request)

    async def _emit_progress_message(self, view_id: str, sids: tuple[str, ...], payload: dict[str, object]) -> None:
        if self._server is None or not sids or self.is_view_closed(view_id):
            return

        message = {"viewId": view_id, **payload}
        for sid in sids:
            await self._server.emit("view_progress", message, to=sid)

    @staticmethod
    def _consume_progress_future(future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            pass

    async def _emit_render_error_message(self, view_id: str, request: RenderRequest, exc: Exception) -> None:
        if self._server is None or self.is_view_closed(view_id):
            return

        sids = self._resolve_target_sids(view_id, request.target_sids)
        if not sids:
            return

        error = {"message": getattr(exc, "detail", str(exc))}
        for sid in sids:
            await self._server.emit("image_error", error, to=sid)
            await self._server.emit("render_error", error, to=sid)

    async def _emit_render_message(self, view_id: str, request: RenderRequest) -> bool:
        if self._server is None or self.is_view_closed(view_id):
            return False

        sids = self._resolve_target_sids(view_id, request.target_sids)
        if not sids:
            return False

        queue_key = self._resolve_render_queue_key(view_id)
        preemption_token = self._mpr_final_preemption_tokens.get(queue_key, 0)
        if self._should_suppress_preview_emit(view_id, request, preemption_token):
            return False

        should_emit_progress = not request.fast_preview
        if should_emit_progress:
            await self._emit_progress_message(view_id, sids, {"phase": "queued", "progressPercent": 2})
        loop = asyncio.get_running_loop()

        def progress_callback(payload: dict[str, object]) -> None:
            if self._server is None or not should_emit_progress:
                return
            if self._should_suppress_preview_emit(view_id, request, preemption_token):
                return
            future = asyncio.run_coroutine_threadsafe(
                self._emit_progress_message(view_id, sids, payload),
                loop,
            )
            future.add_done_callback(self._consume_progress_future)

        render_started_at = perf_counter()
        webrtc_sids = webrtc_3d_transport_manager.get_active_sids(view_id, sids)
        webp_sids = tuple(sid for sid in sids if sid not in webrtc_sids)
        render_kwargs = {
            "image_format": request.image_format,
            "fast_preview": request.fast_preview,
            "fast_preview_full_resolution": request.fast_preview_full_resolution,
            "metadata_mode": request.metadata_mode,
            "progress_callback": progress_callback if should_emit_progress else None,
            "raw_3d_output": bool(webrtc_sids) and not webp_sids,
        }
        if self._should_render_on_main_thread(view_id):
            result = viewer_service.render_view_by_id(view_id, **render_kwargs)
        else:
            result = await asyncio.to_thread(viewer_service.render_view_by_id, view_id, **render_kwargs)
        render_ms = (perf_counter() - render_started_at) * 1000.0
        if self.is_view_closed(view_id) or self._should_suppress_preview_emit(view_id, request, preemption_token):
            return False
        payload = self._build_image_update_payload(result.meta, request)
        extra_image_bytes = getattr(result, "extra_image_bytes", None) or {}
        message = (payload, result.image_bytes, extra_image_bytes) if extra_image_bytes else (payload, result.image_bytes)
        emit_started_at = perf_counter()
        for sid in webp_sids:
            await self._server.emit("image_update", message, to=sid)
        webrtc_publish_ms = 0.0
        raw_image = getattr(result, "raw_image", None)
        if raw_image is not None:
            webrtc_payload = {**payload, "imageTransport": "webrtc"}
            for sid in webrtc_sids:
                publish_ms = webrtc_3d_transport_manager.publish(sid, view_id, raw_image)
                if publish_ms is None:
                    continue
                webrtc_publish_ms += publish_ms
                await self._server.emit("image_update_metadata", webrtc_payload, to=sid)
        emit_ms = (perf_counter() - emit_started_at) * 1000.0
        performance_timings = dict(getattr(result, "performance_timings", None) or {})
        if payload.get("render3dMode"):
            performance_timings["socket_send_ms"] = emit_ms
            log_method = logger.debug if request.fast_preview else logger.info
            log_method(
                (
                    "3d pipeline timing view_id=%s mode=%s fast_preview=%s sids=%s bytes=%s "
                    "vtk_render_ms=%.1f gpu_readback_ms=%.1f webp_encode_ms=%.1f "
                    "socket_send_ms=%.1f webrtc_publish_ms=%.1f gpu_ipc_ms=%.1f total_ms=%.1f"
                ),
                view_id,
                payload.get("render3dMode"),
                request.fast_preview,
                len(sids),
                len(result.image_bytes),
                float(performance_timings.get("vtk_render_ms", 0.0)),
                float(performance_timings.get("gpu_readback_ms", 0.0)),
                float(performance_timings.get("webp_encode_ms", 0.0)),
                emit_ms,
                webrtc_publish_ms,
                float(performance_timings.get("ipc_ms", 0.0)),
                (perf_counter() - render_started_at) * 1000.0,
            )
        if request.metadata_mode == "fusion-registration-layer-preview":
            logger.info(
                (
                    "fusion registration preview socket view_id=%s sids=%s render_ms=%.1f "
                    "emit_ms=%.1f total_ms=%.1f primary_bytes=%s extra_pet_bytes=%s"
                ),
                view_id,
                len(sids),
                render_ms,
                emit_ms,
                (perf_counter() - render_started_at) * 1000.0,
                len(result.image_bytes),
                len(extra_image_bytes.get("pet", b"")) if extra_image_bytes else 0,
            )
        if request.fast_preview and payload.get("render3dMode"):
            logger.debug(
                (
                    "3d preview socket timing view_id=%s mode=%s sids=%s "
                    "format=%s metadata_mode=%s render_ms=%.1f emit_ms=%.1f total_ms=%.1f bytes=%s"
                ),
                view_id,
                payload.get("render3dMode"),
                len(sids),
                request.image_format,
                request.metadata_mode,
                render_ms,
                emit_ms,
                (perf_counter() - render_started_at) * 1000.0,
                len(result.image_bytes),
            )
        if should_emit_progress:
            await self._emit_progress_message(view_id, sids, {"phase": "complete", "progressPercent": 100})
        return True

    async def _emit_render_message_safely(self, view_id: str, request: RenderRequest) -> bool:
        try:
            return await self._emit_render_message(view_id, request)
        except Exception as exc:
            await self._emit_render_error_message(view_id, request, exc)
            return False

    async def _drain_render_requests(self, queue_key: str, view_id: str, initial_request: RenderRequest) -> bool:
        emitted = False
        request_batch = {view_id: initial_request}
        while True:
            request_batch.update(self._pop_sibling_pending_render_batch(queue_key, request_batch))
            request_batch = await self._coalesce_mpr_preview_batch(queue_key, request_batch)
            if request_batch:
                results = await asyncio.gather(
                    *(
                        self._emit_render_message_safely(next_view_id, request)
                        for next_view_id, request in request_batch.items()
                    )
                )
                emitted = any(results) or emitted
            request_batch = self._pop_pending_render_batch(queue_key)
            if not request_batch:
                return emitted

    async def _emit_final_render_batch(
        self,
        queue_key: str,
        request_batch: dict[str, RenderRequest],
    ) -> bool:
        self._mark_mpr_final_preemption(queue_key)
        for request in request_batch.values():
            self._remember_mpr_final_revision(queue_key, request)
        self._discard_pending_preview_requests(queue_key)
        self._cancel_mpr_preview_worker(queue_key)
        results = await asyncio.gather(
            *(
                self._emit_render_message_safely(view_id, request)
                for view_id, request in request_batch.items()
            )
        )
        return any(results)

    async def schedule_render_batch(
        self,
        view_ids: tuple[str, ...],
        *,
        image_format: str = "webp",
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        target_sids: tuple[str, ...] | None = None,
        mpr_revision: int | None = None,
        interaction_id: str | None = None,
    ) -> bool:
        if self._server is None:
            return False

        unique_view_ids = tuple(dict.fromkeys(view_ids))
        unique_view_ids = tuple(view_id for view_id in unique_view_ids if not self.is_view_closed(view_id))
        if not unique_view_ids:
            return False

        requests_by_queue: dict[str, dict[str, RenderRequest]] = {}
        for view_id in unique_view_ids:
            queue_key = self._resolve_render_queue_key(view_id)
            requests_by_queue.setdefault(queue_key, {})[view_id] = RenderRequest(
                image_format=image_format,
                fast_preview=fast_preview,
                fast_preview_full_resolution=fast_preview_full_resolution,
                metadata_mode=metadata_mode,
                target_sids=target_sids,
                mpr_revision=mpr_revision,
                render_revision=self.next_render_revision(view_id),
                interaction_id=interaction_id,
            )

        emitted = False
        for queue_key, request_batch in requests_by_queue.items():
            is_mpr_group = self._is_mpr_group_queue(queue_key)
            is_preview_batch = self._is_preview_render_batch(request_batch)
            is_final_batch = all(self._is_final_render_request(request) for request in request_batch.values())

            if is_mpr_group and is_preview_batch:
                request_batch = self._filter_stale_preview_batch(queue_key, request_batch)
                if not request_batch:
                    continue
                if self._replace_pending_preview_batch(queue_key, request_batch):
                    self._ensure_mpr_preview_worker(queue_key)
                continue

            if is_preview_batch:
                request_batch = self._filter_stale_preview_batch(queue_key, request_batch)
                if request_batch and self._replace_pending_preview_batch(queue_key, request_batch):
                    self._ensure_preview_worker(queue_key)
                continue

            if is_mpr_group and is_final_batch:
                emitted = await self._emit_final_render_batch(queue_key, request_batch) or emitted
                continue

            results = await asyncio.gather(
                *(
                    self.emit_render_for_view(
                        view_id,
                        image_format=request.image_format,
                        fast_preview=request.fast_preview,
                        fast_preview_full_resolution=request.fast_preview_full_resolution,
                        metadata_mode=request.metadata_mode,
                        target_sids=request.target_sids,
                        mpr_revision=request.mpr_revision,
                        render_revision=request.render_revision,
                        interaction_id=request.interaction_id,
                    )
                    for view_id, request in request_batch.items()
                )
            )
            emitted = any(results) or emitted

        return emitted

    async def emit_render_for_view(
        self,
        view_id: str,
        *,
        image_format: str = "webp",
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        target_sids: tuple[str, ...] | None = None,
        mpr_revision: int | None = None,
        render_revision: int | None = None,
        interaction_id: str | None = None,
    ) -> bool:
        if self._server is None or self.is_view_closed(view_id):
            return False

        queue_key = self._resolve_render_queue_key(view_id)
        lock = self._get_render_lock(queue_key)
        incoming_request = RenderRequest(
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            target_sids=target_sids,
            mpr_revision=mpr_revision,
            render_revision=render_revision if render_revision is not None else self.next_render_revision(view_id),
            interaction_id=interaction_id,
        )
        if self._is_mpr_group_queue(queue_key) and self._is_final_render_request(incoming_request):
            self._mark_mpr_final_preemption(queue_key)
            self._remember_mpr_final_revision(queue_key, incoming_request)
            self._discard_pending_preview_requests(queue_key)
        elif self._is_final_render_request(incoming_request):
            self._remember_view_final_revision(view_id, incoming_request)
            self._discard_pending_preview_requests(queue_key)
            self._cancel_preview_worker(queue_key)
        elif self._is_stale_preview_after_final(queue_key, view_id, incoming_request):
            return False
        elif self._is_stale_interaction_request(view_id, incoming_request):
            return False

        if lock.locked():
            if self._is_mpr_group_queue(queue_key) and self._is_final_render_request(incoming_request):
                return await self._emit_render_message_safely(view_id, incoming_request)
            self._queue_pending_render(queue_key, view_id, incoming_request)
            return False

        async with lock:
            if self._is_mpr_group_queue(queue_key):
                await asyncio.sleep(0)
            return await self._drain_render_requests(queue_key, view_id, incoming_request)

    def schedule_delayed_final_render_for_view(
        self,
        view_id: str,
        *,
        delay_seconds: float,
        image_format: str = "webp",
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        target_sids: tuple[str, ...] | None = None,
        mpr_revision: int | None = None,
        interaction_id: str | None = None,
    ) -> asyncio.Task[None]:
        request = self.make_render_request(
            view_id,
            image_format=image_format,
            fast_preview=False,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            target_sids=target_sids,
            mpr_revision=mpr_revision,
            interaction_id=interaction_id,
        )
        self._cancel_delayed_final_render(view_id)

        async def run_delayed_render() -> None:
            try:
                await asyncio.sleep(max(0.0, float(delay_seconds)))
                if self.is_view_closed(view_id) or self._is_stale_interaction_request(view_id, request):
                    return
                await self.emit_render_for_view(
                    view_id,
                    image_format=request.image_format,
                    fast_preview=False,
                    fast_preview_full_resolution=request.fast_preview_full_resolution,
                    metadata_mode=request.metadata_mode,
                    target_sids=request.target_sids,
                    mpr_revision=request.mpr_revision,
                    render_revision=request.render_revision,
                    interaction_id=request.interaction_id,
                )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("delayed final render failed view_id=%s", view_id)
            finally:
                if self._delayed_final_render_tasks.get(view_id) is task:
                    self._delayed_final_render_tasks.pop(view_id, None)

        task = asyncio.create_task(run_delayed_render())
        self._delayed_final_render_tasks[view_id] = task
        return task

    async def emit_error_for_view(self, view_id: str, message: str) -> bool:
        if self._server is None:
            return False

        sids = tuple(self._view_sids.get(view_id, ()))
        if not sids:
            return False

        error = {"message": message}
        for sid in sids:
            await self._server.emit("image_error", error, to=sid)
            await self._server.emit("render_error", error, to=sid)
        return True


view_socket_hub = ViewSocketHub()
