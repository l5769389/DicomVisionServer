import asyncio
import socketio

from app.core.logging import get_logger
from app.schemas.view import ViewHoverRequest, ViewOperationRequest, ViewSetSizeRequest
from app.sockets.runtime import view_socket_hub
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service
from app.utils.utils import timer

logger = get_logger(__name__)


async def _emit_render(server: socketio.AsyncServer, sid: str, view_id: str) -> None:
    view_socket_hub.bind_view(sid, view_id)
    await view_socket_hub.emit_render_for_view(view_id, target_sids=(sid,))
    logger.debug("socket image_update sid=%s view_id=%s", sid, view_id)


@timer
async def _handle_operation(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = ViewOperationRequest.model_validate(data)
        view_socket_hub.bind_view(sid, payload.view_id)
        result = await asyncio.to_thread(viewer_service.handle_view_operation, payload)
        if result.primary_result is not None:
            await server.emit(
                "image_update",
                (result.primary_result.meta.model_dump(by_alias=True), result.primary_result.image_bytes),
                to=sid,
            )
        if result.broadcast_view_ids:
            tasks = [
                view_socket_hub.emit_render_for_view(
                    view_id,
                    image_format=result.broadcast_image_format,
                    fast_preview=result.broadcast_fast_preview,
                )
                for view_id in result.broadcast_view_ids
            ]
            await asyncio.gather(*tasks)
        if result.deferred_view_ids:
            for view_id in result.deferred_view_ids:
                asyncio.create_task(
                    view_socket_hub.emit_render_for_view(
                        view_id,
                        image_format=result.deferred_image_format,
                        fast_preview=result.deferred_fast_preview,
                        target_sids=(sid,),
                    )
                )
        logger.info("socket view_operation sid=%s view_id=%s op_type=%s", sid, payload.view_id, payload.op_type)
    except Exception as exc:
        error = {"message": getattr(exc, "detail", str(exc))}
        logger.exception("socket view_operation failed sid=%s", sid)
        await server.emit("image_error", error, to=sid)
        await server.emit("render_error", error, to=sid)


async def _handle_hover(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = ViewHoverRequest.model_validate(data)
        view_socket_hub.bind_view(sid, payload.view_id)
        result = await asyncio.to_thread(viewer_service.handle_view_hover, payload)
        await server.emit("hover_info", result.model_dump(by_alias=True), to=sid)
    except Exception as exc:
        error = {"message": getattr(exc, "detail", str(exc))}
        logger.exception("socket view_hover failed sid=%s", sid)
        await server.emit("image_error", error, to=sid)


async def _handle_set_size(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = ViewSetSizeRequest.model_validate(data)
        view_socket_hub.bind_view(sid, payload.view_id)
        result = await asyncio.to_thread(viewer_service.set_view_size, payload)
        await server.emit("view_ack", result.model_dump(by_alias=True), to=sid)
        await _emit_render(server, sid, payload.view_id)
        logger.info("socket set_view_size sid=%s view_id=%s", sid, payload.view_id)
    except Exception as exc:
        error = {"message": getattr(exc, "detail", str(exc))}
        logger.exception("socket set_view_size failed sid=%s", sid)
        await server.emit("image_error", error, to=sid)
        await server.emit("render_error", error, to=sid)


def register_socket_handlers(server: socketio.AsyncServer) -> None:
    view_socket_hub.attach_server(server)

    @server.event
    async def connect(sid: str, environ: dict, auth: dict | None = None) -> None:
        logger.info("socket connected sid=%s", sid)
        await server.emit("connected", {"sid": sid}, to=sid)

    @server.event
    async def disconnect(sid: str) -> None:
        view_socket_hub.unbind_sid(sid)
        logger.info("socket disconnected sid=%s", sid)
        return None

    @server.on("bind_view")
    async def bind_view(sid: str, data: dict) -> None:
        view_id = str(data.get("viewId") or data.get("view_id") or "")
        if not view_id:
            await server.emit("image_error", {"message": "viewId is required"}, to=sid)
            return
        view_socket_hub.bind_view(sid, view_id)
        logger.info("socket bind_view sid=%s view_id=%s", sid, view_id)
        await server.emit("view_bound", {"viewId": view_id}, to=sid)
        try:
            view = view_registry.get(view_id)
            if view.width and view.height:
                await _emit_render(server, sid, view_id)
        except Exception as exc:
            logger.exception("socket bind_view initial render failed sid=%s view_id=%s", sid, view_id)
            await server.emit("render_error", {"message": getattr(exc, "detail", str(exc))}, to=sid)

    @server.on("set_view_size")
    async def set_view_size(sid: str, data: dict) -> None:
        await _handle_set_size(server, sid, data)

    @server.on("view_hover")
    async def view_hover(sid: str, data: dict) -> None:
        await _handle_hover(server, sid, data)

    @server.on("view_operation")
    async def view_operation(sid: str, data: dict) -> None:
        await _handle_operation(server, sid, data)

    @server.on("image_operation")
    async def image_operation(sid: str, data: dict) -> None:
        await _handle_operation(server, sid, data)
