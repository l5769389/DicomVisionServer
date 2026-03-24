from collections import defaultdict

import socketio

from app.services.viewer_service import viewer_service


class ViewSocketHub:
    def __init__(self) -> None:
        self._server: socketio.AsyncServer | None = None
        self._view_sids: dict[str, set[str]] = defaultdict(set)
        self._sid_views: dict[str, set[str]] = defaultdict(set)

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

    async def emit_render_for_view(self, view_id: str) -> bool:
        if self._server is None:
            return False

        sids = tuple(self._view_sids.get(view_id, ()))
        if not sids:
            return False

        result = viewer_service.render_view_by_id(view_id)
        message = (result.meta.model_dump(by_alias=True), result.image_bytes)
        for sid in sids:
            await self._server.emit("image_update", message, to=sid)
        return True

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
