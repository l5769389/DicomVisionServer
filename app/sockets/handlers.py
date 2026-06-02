import asyncio
import socketio

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
)
from app.core.logging import get_logger
from app.schemas.dicom import (
    FourDPlaybackFpsRequest,
    FourDPlaybackStartRequest,
    FourDPlaybackStopRequest,
)
from app.schemas.view import ViewHoverRequest, ViewOperationRequest, ViewSetSizeRequest
from app.sockets.four_d_playback import four_d_playback_hub
from app.sockets.runtime import view_socket_hub
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service
from app.services.workspace_activity import workspace_activity_service
from app.utils.utils import timer

logger = get_logger(__name__)


MPR_LOW_LATENCY_OPERATION_TYPES = {
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
}


def _build_error_payload(exc: Exception) -> dict[str, str]:
    return {"message": getattr(exc, "detail", str(exc))}


async def _emit_errors(
    server: socketio.AsyncServer,
    sid: str,
    *,
    events: tuple[str, ...],
    exc: Exception,
) -> None:
    error = _build_error_payload(exc)
    for event_name in events:
        await server.emit(event_name, error, to=sid)


async def _emit_render(server: socketio.AsyncServer, sid: str, view_id: str) -> None:
    workspace_id = view_socket_hub.get_sid_workspace(sid)
    view_registry.get(view_id, workspace_id=workspace_id)
    view_socket_hub.bind_view(sid, view_id)
    await view_socket_hub.emit_render_for_view(view_id, target_sids=(sid,))
    logger.debug("socket image_update sid=%s view_id=%s", sid, view_id)


def _schedule_render_for_view(
    server: socketio.AsyncServer,
    sid: str,
    view_id: str,
    *,
    image_format: str,
    fast_preview: bool,
    target_sids: tuple[str, ...] | None = None,
) -> asyncio.Task[None]:
    async def run_render() -> None:
        try:
            logger.debug(
                "socket background render scheduled sid=%s view_id=%s image_format=%s fast_preview=%s",
                sid,
                view_id,
                image_format,
                fast_preview,
            )
            await view_socket_hub.emit_render_for_view(
                view_id,
                image_format=image_format,
                fast_preview=fast_preview,
                target_sids=target_sids,
            )
            logger.debug(
                "socket background render completed sid=%s view_id=%s image_format=%s fast_preview=%s",
                sid,
                view_id,
                image_format,
                fast_preview,
            )
        except Exception as exc:
            logger.exception("socket background render failed sid=%s view_id=%s", sid, view_id)
            await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)

    return asyncio.create_task(run_render())


def _should_handle_view_operation_inline(view_type: str, payload: ViewOperationRequest) -> bool:
    if view_type not in {"MPR", "AX", "COR", "SAG"}:
        return False
    if payload.op_type not in MPR_LOW_LATENCY_OPERATION_TYPES:
        return False
    # These MPR drag operations only mutate cursor/window/transform state and
    # return broadcast render decisions. Keeping them out of the shared render
    # threadpool prevents background reslices from delaying the next pointer move.
    return payload.action_type in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}


async def _handle_view_operation_for_socket(payload: ViewOperationRequest, workspace_id: str, view_type: str):
    if _should_handle_view_operation_inline(view_type, payload):
        return viewer_service.handle_view_operation(payload, workspace_id)
    return await asyncio.to_thread(viewer_service.handle_view_operation, payload, workspace_id)


@timer
async def _handle_operation(server: socketio.AsyncServer, sid: str, data: dict) -> dict[str, object]:
    """Apply an interactive viewer operation and push any resulting frames.

    This is the realtime counterpart to the REST APIs: high-frequency operations
    such as scroll, window, pan, zoom, MPR crosshair, 3D rotation, and measurement
    edits enter here so the client can receive image_update events without polling.
    """
    try:
        payload = ViewOperationRequest.model_validate(data)
        workspace_id = view_socket_hub.get_sid_workspace(sid)
        view = view_registry.get(payload.view_id, workspace_id=workspace_id)
        view_socket_hub.bind_view(sid, payload.view_id)
        result = await _handle_view_operation_for_socket(payload, workspace_id, view.view_type)
        if result.draft_measurement is not None:
            await server.emit("measurement_draft", result.draft_measurement, to=sid)
        if result.primary_result is not None:
            await server.emit(
                "image_update",
                (result.primary_result.meta.model_dump(by_alias=True), result.primary_result.image_bytes),
                to=sid,
            )
        for view_id in result.broadcast_view_ids:
            _schedule_render_for_view(
                server,
                sid,
                view_id,
                image_format=result.broadcast_image_format,
                fast_preview=result.broadcast_fast_preview,
            )
        if result.deferred_view_ids:
            for view_id in result.deferred_view_ids:
                _schedule_render_for_view(
                    server,
                    sid,
                    view_id,
                    image_format=result.deferred_image_format,
                    fast_preview=result.deferred_fast_preview,
                    target_sids=(sid,),
                )
        logger.info("socket view_operation sid=%s view_id=%s op_type=%s", sid, payload.view_id, payload.op_type)
        return {"ok": True}
    except Exception as exc:
        logger.exception("socket view_operation failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)
        return {"ok": False, "message": _build_error_payload(exc)["message"]}


async def _handle_hover(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = ViewHoverRequest.model_validate(data)
        workspace_id = view_socket_hub.get_sid_workspace(sid)
        view_registry.get(payload.view_id, workspace_id=workspace_id)
        view_socket_hub.bind_view(sid, payload.view_id)
        result = await asyncio.to_thread(viewer_service.handle_view_hover, payload, workspace_id)
        await server.emit("hover_info", result.model_dump(by_alias=True), to=sid)
    except Exception as exc:
        logger.exception("socket view_hover failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error",), exc=exc)


async def _handle_set_size(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = ViewSetSizeRequest.model_validate(data)
        workspace_id = view_socket_hub.get_sid_workspace(sid)
        view_registry.get(payload.view_id, workspace_id=workspace_id)
        view_socket_hub.bind_view(sid, payload.view_id)
        result = await asyncio.to_thread(viewer_service.set_view_size, payload, workspace_id)
        await server.emit("view_ack", result.model_dump(by_alias=True), to=sid)
        await _emit_render(server, sid, payload.view_id)
        logger.info("socket set_view_size sid=%s view_id=%s", sid, payload.view_id)
    except Exception as exc:
        logger.exception("socket set_view_size failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)


async def _handle_four_d_playback_start(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = FourDPlaybackStartRequest.model_validate(data)
        await four_d_playback_hub.start(sid, payload)
        logger.info(
            "socket four_d_playback_start sid=%s tab_key=%s phase_index=%s fps=%s",
            sid,
            payload.tab_key,
            payload.phase_index,
            payload.fps,
        )
    except Exception as exc:
        logger.exception("socket four_d_playback_start failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)


async def _handle_four_d_playback_stop(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = FourDPlaybackStopRequest.model_validate(data)
        await four_d_playback_hub.stop(sid, payload)
        logger.info("socket four_d_playback_stop sid=%s tab_key=%s", sid, payload.tab_key)
    except Exception as exc:
        logger.exception("socket four_d_playback_stop failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)


async def _handle_four_d_playback_fps(server: socketio.AsyncServer, sid: str, data: dict) -> None:
    try:
        payload = FourDPlaybackFpsRequest.model_validate(data)
        await four_d_playback_hub.update_fps(sid, payload)
        logger.info("socket four_d_playback_fps sid=%s tab_key=%s fps=%s", sid, payload.tab_key, payload.fps)
    except Exception as exc:
        logger.exception("socket four_d_playback_fps failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)


def register_socket_handlers(server: socketio.AsyncServer) -> None:
    view_socket_hub.attach_server(server)
    four_d_playback_hub.attach_server(server)

    @server.event
    async def connect(sid: str, environ: dict, auth: dict | None = None) -> None:
        workspace_id = view_socket_hub.bind_sid_workspace(
            sid,
            str((auth or {}).get("workspaceId") or ""),
        )
        workspace_activity_service.touch(workspace_id)
        logger.info("socket connected sid=%s workspace_id=%s", sid, workspace_id)
        await server.emit("connected", {"sid": sid, "workspaceId": workspace_id}, to=sid)

    @server.event
    async def disconnect(sid: str) -> None:
        await four_d_playback_hub.unbind_sid(sid)
        view_socket_hub.unbind_sid(sid)
        logger.info("socket disconnected sid=%s", sid)
        return None

    @server.on("bind_view")
    async def bind_view(sid: str, data: dict) -> dict[str, object] | None:
        """Subscribe this Socket connection to a view's image_update events."""
        view_id = str(data.get("viewId") or data.get("view_id") or "")
        if not view_id:
            await server.emit("image_error", {"message": "viewId is required"}, to=sid)
            return {"ok": False, "message": "viewId is required"}
        should_render = bool(data.get("render", True))
        workspace_id = view_socket_hub.get_sid_workspace(sid)
        try:
            view = view_registry.get(view_id, workspace_id=workspace_id)
        except Exception as exc:
            logger.exception("socket bind_view failed sid=%s view_id=%s", sid, view_id)
            await _emit_errors(server, sid, events=("render_error",), exc=exc)
            return {"ok": False, "message": _build_error_payload(exc)["message"]}
        view_socket_hub.bind_view(sid, view_id)
        logger.info("socket bind_view sid=%s view_id=%s", sid, view_id)
        await server.emit("view_bound", {"viewId": view_id}, to=sid)
        if not should_render:
            return {"ok": True}
        try:
            if view.width and view.height:
                await _emit_render(server, sid, view_id)
            return {"ok": True}
        except Exception as exc:
            logger.exception("socket bind_view initial render failed sid=%s view_id=%s", sid, view_id)
            await _emit_errors(server, sid, events=("render_error",), exc=exc)
            return {"ok": False, "message": _build_error_payload(exc)["message"]}

    @server.on("set_view_size")
    async def set_view_size(sid: str, data: dict) -> None:
        await _handle_set_size(server, sid, data)

    @server.on("view_hover")
    async def view_hover(sid: str, data: dict) -> None:
        await _handle_hover(server, sid, data)

    @server.on("view_operation")
    async def view_operation(sid: str, data: dict) -> dict[str, object]:
        return await _handle_operation(server, sid, data)

    @server.on("image_operation")
    async def image_operation(sid: str, data: dict) -> dict[str, object]:
        return await _handle_operation(server, sid, data)

    @server.on("four_d_playback_start")
    async def four_d_playback_start(sid: str, data: dict) -> None:
        await _handle_four_d_playback_start(server, sid, data)

    @server.on("four_d_playback_stop")
    async def four_d_playback_stop(sid: str, data: dict) -> None:
        await _handle_four_d_playback_stop(server, sid, data)

    @server.on("four_d_playback_fps")
    async def four_d_playback_fps(sid: str, data: dict) -> None:
        await _handle_four_d_playback_fps(server, sid, data)
