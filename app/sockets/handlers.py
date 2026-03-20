import socketio

from app.schemas.dicom import DicomRenderRequest
from app.services.dicom_service import dicom_service


def register_socket_handlers(server: socketio.AsyncServer) -> None:
    @server.event
    async def connect(sid: str, environ: dict, auth: dict | None = None) -> None:
        await server.emit("connected", {"sid": sid}, to=sid)

    @server.event
    async def disconnect(sid: str) -> None:
        return None

    @server.on("render_dicom")
    async def render_dicom(sid: str, data: dict) -> None:
        try:
            payload = DicomRenderRequest.model_validate(data)
            result = dicom_service.render_from_request(payload)
            await server.emit("render_result", result.model_dump(), to=sid)
        except Exception as exc:
            await server.emit(
                "render_error",
                {"message": getattr(exc, "detail", str(exc))},
                to=sid,
            )
