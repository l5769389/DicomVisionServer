import asyncio
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass

import socketio

from app.core.workspace import DEFAULT_WORKSPACE_ID, normalize_workspace_id
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service


RENDER_FORMAT_QUALITY_RANK = {"jpeg": 0, "png": 1}


@dataclass
class RenderRequest:
    image_format: str = "png"
    fast_preview: bool = False
    target_sids: tuple[str, ...] | None = None


class ViewSocketHub:
    def __init__(self) -> None:
        self._server: socketio.AsyncServer | None = None
        self._view_sids: dict[str, set[str]] = defaultdict(set)
        self._sid_views: dict[str, set[str]] = defaultdict(set)
        self._sid_workspaces: dict[str, str] = {}
        self._render_locks: dict[str, asyncio.Lock] = {}
        self._pending_render_requests: dict[str, dict[str, RenderRequest]] = {}

    def attach_server(self, server: socketio.AsyncServer) -> None:
        self._server = server

    def bind_sid_workspace(self, sid: str, workspace_id: str | None) -> str:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        self._sid_workspaces[sid] = normalized_workspace_id
        return normalized_workspace_id

    def get_sid_workspace(self, sid: str) -> str:
        return self._sid_workspaces.get(sid, DEFAULT_WORKSPACE_ID)

    def bind_view(self, sid: str, view_id: str) -> None:
        self._view_sids[view_id].add(sid)
        self._sid_views[sid].add(view_id)

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
        if view.view_group is not None and str(view.view_group.group_type).lower() == "mpr":
            return f"mpr-group:{view.view_group.group_id}"
        return f"view:{view_id}"

    @staticmethod
    def _merge_render_request(current: RenderRequest, incoming: RenderRequest) -> RenderRequest:
        if current.target_sids is None or incoming.target_sids is None:
            target_sids = None
        else:
            target_sids = tuple(dict.fromkeys((*current.target_sids, *incoming.target_sids)))

        # Pending renders use "latest state wins", but a settled frame should not
        # be downgraded by an older drag-preview request still waiting in the queue.
        return RenderRequest(
            image_format=ViewSocketHub._choose_render_image_format(current.image_format, incoming.image_format),
            fast_preview=current.fast_preview and incoming.fast_preview,
            target_sids=target_sids,
        )

    @staticmethod
    def _choose_render_image_format(current: str, incoming: str) -> str:
        current_rank = RENDER_FORMAT_QUALITY_RANK.get(current, 0)
        incoming_rank = RENDER_FORMAT_QUALITY_RANK.get(incoming, 0)
        if incoming_rank > current_rank:
            return incoming
        if current_rank > incoming_rank:
            return current
        return incoming

    def _queue_pending_render(self, queue_key: str, view_id: str, incoming_request: RenderRequest) -> None:
        pending_requests = self._pending_render_requests.setdefault(queue_key, {})
        current_request = pending_requests.get(view_id)
        pending_requests[view_id] = (
            incoming_request
            if current_request is None
            else self._merge_render_request(current_request, incoming_request)
        )

    def _pop_pending_render_request(self, queue_key: str, view_id: str) -> RenderRequest | None:
        pending_requests = self._pending_render_requests.get(queue_key)
        if pending_requests is None:
            return None
        pending_request = pending_requests.pop(view_id, None)
        if not pending_requests:
            self._pending_render_requests.pop(queue_key, None)
        return pending_request

    def _resolve_target_sids(self, view_id: str, target_sids: tuple[str, ...] | None) -> tuple[str, ...]:
        if target_sids is not None:
            return target_sids
        return tuple(self._view_sids.get(view_id, ()))

    async def _emit_progress_message(self, view_id: str, sids: tuple[str, ...], payload: dict[str, object]) -> None:
        if self._server is None or not sids:
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
        if self._server is None:
            return

        sids = self._resolve_target_sids(view_id, request.target_sids)
        if not sids:
            return

        error = {"message": getattr(exc, "detail", str(exc))}
        for sid in sids:
            await self._server.emit("image_error", error, to=sid)
            await self._server.emit("render_error", error, to=sid)

    async def _emit_render_message(self, view_id: str, request: RenderRequest) -> bool:
        if self._server is None:
            return False

        sids = self._resolve_target_sids(view_id, request.target_sids)
        if not sids:
            return False

        await self._emit_progress_message(view_id, sids, {"phase": "queued", "progressPercent": 2})
        loop = asyncio.get_running_loop()

        def progress_callback(payload: dict[str, object]) -> None:
            if self._server is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._emit_progress_message(view_id, sids, payload),
                loop,
            )
            future.add_done_callback(self._consume_progress_future)

        result = await asyncio.to_thread(
            viewer_service.render_view_by_id,
            view_id,
            image_format=request.image_format,
            fast_preview=request.fast_preview,
            progress_callback=progress_callback,
        )
        message = (result.meta.model_dump(by_alias=True), result.image_bytes)
        for sid in sids:
            await self._server.emit("image_update", message, to=sid)
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
            merged_batch: dict[str, RenderRequest] = {}
            for next_view_id, request in tuple(request_batch.items()):
                pending_request = self._pop_pending_render_request(queue_key, next_view_id)
                if pending_request is not None:
                    request = self._merge_render_request(request, pending_request)
                merged_batch[next_view_id] = request
            if merged_batch:
                results = await asyncio.gather(
                    *(
                        self._emit_render_message_safely(next_view_id, request)
                        for next_view_id, request in merged_batch.items()
                    )
                )
                emitted = any(results) or emitted
            request_batch = self._pending_render_requests.pop(queue_key, {})
            if not request_batch:
                return emitted

    async def emit_render_for_view(
        self,
        view_id: str,
        *,
        image_format: str = "png",
        fast_preview: bool = False,
        target_sids: tuple[str, ...] | None = None,
    ) -> bool:
        if self._server is None:
            return False

        queue_key = self._resolve_render_queue_key(view_id)
        lock = self._get_render_lock(queue_key)
        incoming_request = RenderRequest(
            image_format=image_format,
            fast_preview=fast_preview,
            target_sids=target_sids,
        )

        if lock.locked():
            self._queue_pending_render(queue_key, view_id, incoming_request)
            return False

        async with lock:
            return await self._drain_render_requests(queue_key, view_id, incoming_request)

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
