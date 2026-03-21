import socketio

from app.schemas.view import ViewOperationRequest
from app.services.viewer_service import viewer_service


async def _handle_operation(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = ViewOperationRequest.model_validate(data)
        result = viewer_service.handle_view_operation(payload)
        message = result.model_dump(by_alias=True)
        await server.emit("image_update", message, to=sid)
    except Exception as exc:
        error = {"message": getattr(exc, "detail", str(exc))}
        await server.emit("image_error", error, to=sid)
        await server.emit("render_error", error, to=sid)


def register_socket_handlers(server: socketio.AsyncServer) -> None:
    @server.event
    async def connect(sid: str, environ: dict, auth: dict | None = None) -> None:
        await server.emit("connected", {"sid": sid}, to=sid)

    @server.event
    async def disconnect(sid: str) -> None:
        return None

    @server.on("view_operation")
    async def view_operation(sid: str, data: dict) -> None:
        await _handle_operation(server, sid, data)

    @server.on("image_operation")
    async def image_operation(sid: str, data: dict) -> None:
        await _handle_operation(server, sid, data)
