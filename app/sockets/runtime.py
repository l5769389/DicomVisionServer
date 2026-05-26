import asyncio
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass

import socketio

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
        self._render_locks: dict[str, asyncio.Lock] = {}
        self._pending_render_requests: dict[str, RenderRequest] = {}

    def attach_server(self, server: socketio.AsyncServer) -> None:
        self._server = server

    def bind_view(self, sid: str, view_id: str) -> None:
        self._view_sids[view_id].add(sid)
        self._sid_views[sid].add(view_id)

    def unbind_sid(self, sid: str) -> None:
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
        self._pending_render_requests.pop(view_id, None)
        self._render_locks.pop(view_id, None)

    def _get_render_lock(self, view_id: str) -> asyncio.Lock:
        lock = self._render_locks.get(view_id)
        if lock is None:
            lock = asyncio.Lock()
            self._render_locks[view_id] = lock
        return lock

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

    def _queue_pending_render(self, view_id: str, incoming_request: RenderRequest) -> None:
        current_request = self._pending_render_requests.get(view_id)
        self._pending_render_requests[view_id] = (
            incoming_request
            if current_request is None
            else self._merge_render_request(current_request, incoming_request)
        )

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

    async def _drain_render_requests(self, view_id: str, initial_request: RenderRequest) -> bool:
        emitted = False
        request = initial_request
        while True:
            emitted = await self._emit_render_message(view_id, request) or emitted
            next_request = self._pending_render_requests.pop(view_id, None)
            if next_request is None:
                return emitted
            request = next_request

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

        lock = self._get_render_lock(view_id)
        incoming_request = RenderRequest(
            image_format=image_format,
            fast_preview=fast_preview,
            target_sids=target_sids,
        )

        if lock.locked():
            self._queue_pending_render(view_id, incoming_request)
            return False

        async with lock:
            return await self._drain_render_requests(view_id, incoming_request)

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
