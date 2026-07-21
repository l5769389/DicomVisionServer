import asyncio
from dataclasses import dataclass
import socketio

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_FUSION_CONFIG,
    VIEW_OP_TYPE_FUSION_REGISTRATION,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    VIEW_OP_TYPE_MPR_SEGMENTATION,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_PSEUDOCOLOR,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_SCROLL,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
)
from app.core.logging import get_logger
from app.schemas.dicom import (
    FourDPlaybackFpsRequest,
    FourDPlaybackStartRequest,
    FourDPlaybackStopRequest,
)
from app.schemas.view import ViewHoverRequest, ViewOperationRequest, ViewSetSizeRequest, normalize_image_format
from app.sockets.four_d_playback import four_d_playback_hub
from app.sockets.runtime import view_socket_hub
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service
from app.services.volume_rendering.vtk_threading import should_run_3d_view_on_main_thread
from app.services.webrtc_3d_transport import (
    get_webrtc_3d_client_config,
    webrtc_3d_transport_manager,
)
from app.services.workspace_activity import workspace_activity_service

logger = get_logger(__name__)


MPR_LOW_LATENCY_OPERATION_TYPES = {
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    VIEW_OP_TYPE_MPR_SEGMENTATION,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
}
MPR_CROSSHAIR_STATE_OPERATION_TYPES = {
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_MPR_OBLIQUE,
}
MPR_CROSSHAIR_PREVIEW_INTERVAL_SECONDS = 0.05
MPR_VIEW_TYPES = {"MPR", "AX", "COR", "SAG"}
FUSION_VIEW_TYPES = {
    "FusionCTAxial",
    "FusionPETAxial",
    "FusionOverlayAxial",
    "FusionPETCoronalMip",
}
FUSION_LOW_LATENCY_OPERATION_TYPES = {
    VIEW_OP_TYPE_FUSION_CONFIG,
    VIEW_OP_TYPE_FUSION_REGISTRATION,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_PSEUDOCOLOR,
    VIEW_OP_TYPE_SCROLL,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
}
ROTATE3D_FINAL_RENDER_TARGET_SPACING_SECONDS = 0.05
ROTATE3D_FINAL_RENDER_MIN_DELAY_SECONDS = 0.01
PAN_ZOOM_FINAL_RENDER_TARGET_SPACING_SECONDS = 0.035
PAN_ZOOM_FINAL_RENDER_MIN_DELAY_SECONDS = 0.005
MPR_END_REQUIRES_PENDING_MOVE_TYPES = {
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_ZOOM,
}


@dataclass
class _QueuedMprOperation:
    payload: ViewOperationRequest
    server: socketio.AsyncServer
    sid: str
    workspace_id: str


@dataclass
class _MprOperationQueueState:
    pending_start: _QueuedMprOperation | None = None
    pending_move: _QueuedMprOperation | None = None
    pending_end: _QueuedMprOperation | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class _MprCrosshairPreviewRequest:
    server: socketio.AsyncServer
    sid: str
    view_ids: tuple[str, ...]
    image_format: str
    fast_preview: bool
    fast_preview_full_resolution: bool
    metadata_mode: str
    mpr_revision: int | None
    generation: int = 0


@dataclass
class _MprCrosshairPreviewState:
    pending: _MprCrosshairPreviewRequest | None = None
    task: asyncio.Task[None] | None = None
    last_dispatch_at: float = 0.0
    generation: int = 0


_mpr_operation_queues: dict[str, _MprOperationQueueState] = {}
_mpr_crosshair_state_queues: dict[str, _MprOperationQueueState] = {}
_mpr_crosshair_preview_states: dict[str, _MprCrosshairPreviewState] = {}

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


async def _emit_render(server: socketio.AsyncServer, sid: str, view_id: str, *, image_format: str = "webp") -> None:
    workspace_id = view_socket_hub.get_sid_workspace(sid)
    view_registry.get(view_id, workspace_id=workspace_id)
    view_socket_hub.bind_view(sid, view_id)
    await view_socket_hub.emit_render_for_view(view_id, image_format=image_format, target_sids=(sid,))
    logger.debug("socket image_update sid=%s view_id=%s", sid, view_id)


def _schedule_render_for_view(
    server: socketio.AsyncServer,
    sid: str,
    view_id: str,
    *,
    image_format: str,
    fast_preview: bool,
    fast_preview_full_resolution: bool = False,
    metadata_mode: str = "full",
    target_sids: tuple[str, ...] | None = None,
    mpr_revision: int | None = None,
    interaction_id: str | None = None,
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
                fast_preview_full_resolution=fast_preview_full_resolution,
                metadata_mode=metadata_mode,
                target_sids=target_sids,
                mpr_revision=mpr_revision,
                interaction_id=interaction_id,
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


def _schedule_render_batch_for_views(
    server: socketio.AsyncServer,
    sid: str,
    view_ids: tuple[str, ...],
    *,
    image_format: str,
    fast_preview: bool,
    fast_preview_full_resolution: bool = False,
    metadata_mode: str = "full",
    target_sids: tuple[str, ...] | None = None,
    mpr_revision: int | None = None,
    interaction_id: str | None = None,
) -> asyncio.Task[None]:
    async def run_render_batch() -> None:
        try:
            logger.debug(
                "socket background render batch scheduled sid=%s view_ids=%s image_format=%s fast_preview=%s",
                sid,
                view_ids,
                image_format,
                fast_preview,
            )
            await view_socket_hub.schedule_render_batch(
                view_ids,
                image_format=image_format,
                fast_preview=fast_preview,
                fast_preview_full_resolution=fast_preview_full_resolution,
                metadata_mode=metadata_mode,
                target_sids=target_sids,
                mpr_revision=mpr_revision,
                interaction_id=interaction_id,
            )
            logger.debug(
                "socket background render batch completed sid=%s view_ids=%s image_format=%s fast_preview=%s",
                sid,
                view_ids,
                image_format,
                fast_preview,
            )
        except Exception as exc:
            logger.exception("socket background render batch failed sid=%s view_ids=%s", sid, view_ids)
            await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)

    return asyncio.create_task(run_render_batch())


def _should_queue_mpr_operation(view_type: str, payload: ViewOperationRequest) -> bool:
    if view_type in MPR_VIEW_TYPES:
        allowed_operation_types = MPR_LOW_LATENCY_OPERATION_TYPES
    elif view_type in FUSION_VIEW_TYPES:
        allowed_operation_types = FUSION_LOW_LATENCY_OPERATION_TYPES
    else:
        return False
    if payload.op_type not in allowed_operation_types:
        return False
    # High-frequency MPR drags are lossy: start/end are preserved, while move
    # events are coalesced to the latest payload by the group operation queue.
    return payload.action_type in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}


def _should_queue_mpr_crosshair_state_operation(view_type: str, payload: ViewOperationRequest) -> bool:
    return (
        view_type in MPR_VIEW_TYPES
        and payload.op_type in MPR_CROSSHAIR_STATE_OPERATION_TYPES
        and payload.action_type in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}
    )


async def _handle_view_operation_for_socket(payload: ViewOperationRequest, workspace_id: str, view_type: str):
    if should_run_3d_view_on_main_thread(view_type):
        return viewer_service.handle_view_operation(payload, workspace_id)
    return await asyncio.to_thread(viewer_service.handle_view_operation, payload, workspace_id)


def _resolve_mpr_operation_queue_key(view, workspace_id: str) -> str:
    view_group = getattr(view, "view_group", None)
    if view_group is not None:
        return f"mpr-op:{workspace_id}:{view_group.group_id}"
    return f"mpr-op:{workspace_id}:view:{view.view_id}"


def _pop_next_mpr_operation(state: _MprOperationQueueState) -> _QueuedMprOperation | None:
    if state.pending_start is not None:
        operation = state.pending_start
        state.pending_start = None
        return operation
    if (
        state.pending_end is not None
        and state.pending_move is not None
        and state.pending_move.payload.op_type in MPR_END_REQUIRES_PENDING_MOVE_TYPES
    ):
        operation = state.pending_move
        state.pending_move = None
        return operation
    if state.pending_end is not None:
        operation = state.pending_end
        state.pending_end = None
        return operation
    if state.pending_move is not None:
        operation = state.pending_move
        state.pending_move = None
        return operation
    return None


def _enqueue_mpr_operation(queue_key: str, operation: _QueuedMprOperation) -> None:
    state = _mpr_operation_queues.setdefault(queue_key, _MprOperationQueueState())
    action_type = operation.payload.action_type
    if action_type == DRAG_ACTION_START:
        state.pending_start = operation
        state.pending_move = None
        state.pending_end = None
    elif action_type == DRAG_ACTION_END:
        has_authoritative_end_state = (
            operation.payload.interaction_id is not None
            and operation.payload.canvas_width is not None
            and operation.payload.canvas_height is not None
        )
        if has_authoritative_end_state or operation.payload.op_type not in MPR_END_REQUIRES_PENDING_MOVE_TYPES:
            state.pending_move = None
        state.pending_end = operation
    elif action_type == DRAG_ACTION_MOVE:
        if state.pending_end is None:
            state.pending_move = operation
    else:
        state.pending_move = operation

    if state.task is None or state.task.done():
        state.task = asyncio.create_task(_run_mpr_operation_queue(queue_key, state))


def _enqueue_mpr_crosshair_state_operation(queue_key: str, operation: _QueuedMprOperation) -> None:
    state = _mpr_crosshair_state_queues.setdefault(queue_key, _MprOperationQueueState())
    action_type = operation.payload.action_type
    if action_type == DRAG_ACTION_START:
        state.pending_start = operation
        state.pending_move = None
        state.pending_end = None
    elif action_type == DRAG_ACTION_END:
        state.pending_move = None
        state.pending_end = operation
    elif action_type == DRAG_ACTION_MOVE:
        if state.pending_end is None:
            state.pending_move = operation
    else:
        state.pending_move = operation

    if state.task is None or state.task.done():
        state.task = asyncio.create_task(_run_mpr_crosshair_state_queue(queue_key, state))


async def _emit_mpr_state_updates(
    server: socketio.AsyncServer,
    sid: str,
    view_ids: tuple[str, ...],
    *,
    mpr_revision: int | None = None,
) -> None:
    if not view_ids:
        return
    workspace_id = view_socket_hub.get_sid_workspace(sid)
    state_payload_map = await asyncio.to_thread(
        viewer_service.build_mpr_state_update_payloads,
        view_ids,
        workspace_id=workspace_id,
        mpr_revision=mpr_revision,
    )
    for state_view_id in view_ids:
        state_payload = state_payload_map.get(state_view_id)
        if not state_payload:
            continue
        for target_sid in view_socket_hub.get_view_sids(state_view_id):
            await server.emit("mpr_state_update", state_payload, to=target_sid)


def _schedule_mpr_crosshair_preview(
    queue_key: str,
    request: _MprCrosshairPreviewRequest,
) -> None:
    if not request.view_ids:
        return
    state = _mpr_crosshair_preview_states.setdefault(queue_key, _MprCrosshairPreviewState())
    state.generation += 1
    request.generation = state.generation
    state.pending = request
    if state.task is None or state.task.done():
        state.task = asyncio.create_task(_run_mpr_crosshair_preview_queue(queue_key, state))


def _cancel_mpr_crosshair_preview(queue_key: str) -> None:
    state = _mpr_crosshair_preview_states.pop(queue_key, None)
    if state and state.task is not None and not state.task.done():
        state.task.cancel()


async def _run_mpr_crosshair_preview_queue(queue_key: str, state: _MprCrosshairPreviewState) -> None:
    current_task = asyncio.current_task()
    try:
        while True:
            request = state.pending
            if request is None:
                return
            state.pending = None
            loop = asyncio.get_running_loop()
            delay_seconds = MPR_CROSSHAIR_PREVIEW_INTERVAL_SECONDS - (loop.time() - state.last_dispatch_at)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            if request.generation != state.generation and state.pending is not None:
                continue
            try:
                await view_socket_hub.schedule_render_batch(
                    request.view_ids,
                    image_format=request.image_format,
                    fast_preview=request.fast_preview,
                    fast_preview_full_resolution=request.fast_preview_full_resolution,
                    metadata_mode=request.metadata_mode,
                    mpr_revision=request.mpr_revision,
                )
                state.last_dispatch_at = asyncio.get_running_loop().time()
            except Exception as exc:
                logger.exception("socket MPR crosshair preview failed sid=%s view_ids=%s", request.sid, request.view_ids)
                await _emit_errors(request.server, request.sid, events=("image_error", "render_error"), exc=exc)
            if state.pending is None:
                return
    except asyncio.CancelledError:
        raise
    finally:
        if state.task is current_task:
            if state.pending is None:
                _mpr_crosshair_preview_states.pop(queue_key, None)
            else:
                state.task = asyncio.create_task(_run_mpr_crosshair_preview_queue(queue_key, state))


async def _dispatch_operation_result(
    server: socketio.AsyncServer,
    sid: str,
    view,
    payload: ViewOperationRequest,
    result,
) -> dict[str, object]:
    if result.draft_measurement is not None:
        await server.emit("measurement_draft", result.draft_measurement, to=sid)
    if result.primary_result is not None:
        primary_request = view_socket_hub.make_render_request(
            result.primary_result.meta.view_id,
            image_format=result.primary_image_format,
            fast_preview=result.primary_fast_preview,
            fast_preview_full_resolution=result.primary_fast_preview_full_resolution,
            metadata_mode=result.primary_metadata_mode,
            mpr_revision=result.mpr_revision,
            interaction_id=payload.interaction_id,
        )
        primary_payload = view_socket_hub.build_image_update_payload(result.primary_result.meta, primary_request)
        await server.emit(
            "image_update",
            (primary_payload, result.primary_result.image_bytes),
            to=sid,
        )
    is_mpr_view = view.view_type in MPR_VIEW_TYPES
    if result.mpr_state_view_ids:
        await _emit_mpr_state_updates(server, sid, result.mpr_state_view_ids, mpr_revision=result.mpr_revision)
    if result.broadcast_view_ids:
        broadcast_fast_preview = result.broadcast_fast_preview
        broadcast_fast_preview_full_resolution = result.broadcast_fast_preview_full_resolution
        broadcast_metadata_mode = result.broadcast_metadata_mode
        if broadcast_fast_preview or result.broadcast_image_format == "jpeg":
            await view_socket_hub.schedule_render_batch(
                result.broadcast_view_ids,
                image_format=result.broadcast_image_format,
                fast_preview=broadcast_fast_preview,
                fast_preview_full_resolution=broadcast_fast_preview_full_resolution,
                metadata_mode=broadcast_metadata_mode,
                mpr_revision=result.mpr_revision,
                interaction_id=payload.interaction_id,
            )
        else:
            _schedule_render_batch_for_views(
                server,
                sid,
                result.broadcast_view_ids,
                image_format=result.broadcast_image_format,
                fast_preview=broadcast_fast_preview,
                fast_preview_full_resolution=broadcast_fast_preview_full_resolution,
                metadata_mode=broadcast_metadata_mode,
                mpr_revision=result.mpr_revision,
                interaction_id=payload.interaction_id,
            )
    if result.deferred_view_ids:
        if result.deferred_fast_preview or result.deferred_image_format == "jpeg":
            await view_socket_hub.schedule_render_batch(
                result.deferred_view_ids,
                image_format=result.deferred_image_format,
                fast_preview=result.deferred_fast_preview,
                fast_preview_full_resolution=result.deferred_fast_preview_full_resolution,
                metadata_mode=result.deferred_metadata_mode,
                target_sids=(sid,),
                mpr_revision=result.mpr_revision,
                interaction_id=payload.interaction_id,
            )
        elif (
            payload.op_type in {VIEW_OP_TYPE_PAN, VIEW_OP_TYPE_ZOOM}
            and payload.action_type == DRAG_ACTION_END
        ):
            for view_id in result.deferred_view_ids:
                view_socket_hub.schedule_delayed_final_render_for_view(
                    view_id,
                    delay_seconds=view_socket_hub.adaptive_final_render_delay(
                        view_id,
                        target_preview_spacing_seconds=PAN_ZOOM_FINAL_RENDER_TARGET_SPACING_SECONDS,
                        minimum_delay_seconds=PAN_ZOOM_FINAL_RENDER_MIN_DELAY_SECONDS,
                    ),
                    image_format=result.deferred_image_format,
                    fast_preview_full_resolution=result.deferred_fast_preview_full_resolution,
                    metadata_mode=result.deferred_metadata_mode,
                    target_sids=(sid,),
                    mpr_revision=result.mpr_revision,
                    interaction_id=payload.interaction_id,
                )
        elif is_mpr_view:
            _schedule_render_batch_for_views(
                server,
                sid,
                result.deferred_view_ids,
                image_format=result.deferred_image_format,
                fast_preview=result.deferred_fast_preview,
                fast_preview_full_resolution=result.deferred_fast_preview_full_resolution,
                metadata_mode=result.deferred_metadata_mode,
                target_sids=(sid,),
                mpr_revision=result.mpr_revision,
                interaction_id=payload.interaction_id,
            )
        else:
            for view_id in result.deferred_view_ids:
                if (
                    view.view_type == "3D"
                    and payload.op_type == VIEW_OP_TYPE_ROTATE_3D
                    and payload.action_type == DRAG_ACTION_END
                    and not result.deferred_fast_preview
                ):
                    view_socket_hub.schedule_delayed_final_render_for_view(
                        view_id,
                        delay_seconds=view_socket_hub.adaptive_final_render_delay(
                            view_id,
                            target_preview_spacing_seconds=ROTATE3D_FINAL_RENDER_TARGET_SPACING_SECONDS,
                            minimum_delay_seconds=ROTATE3D_FINAL_RENDER_MIN_DELAY_SECONDS,
                        ),
                        image_format=result.deferred_image_format,
                        fast_preview_full_resolution=result.deferred_fast_preview_full_resolution,
                        metadata_mode=result.deferred_metadata_mode,
                        target_sids=(sid,),
                        mpr_revision=result.mpr_revision,
                        interaction_id=payload.interaction_id,
                    )
                    continue
                _schedule_render_for_view(
                    server,
                    sid,
                    view_id,
                    image_format=result.deferred_image_format,
                    fast_preview=result.deferred_fast_preview,
                    fast_preview_full_resolution=result.deferred_fast_preview_full_resolution,
                    metadata_mode=result.deferred_metadata_mode,
                    target_sids=(sid,),
                    mpr_revision=result.mpr_revision,
                    interaction_id=payload.interaction_id,
                )
    log_method = logger.debug if payload.action_type == DRAG_ACTION_MOVE else logger.info
    log_method("socket view_operation sid=%s view_id=%s op_type=%s", sid, payload.view_id, payload.op_type)
    response: dict[str, object] = {"ok": True}
    if result.mpr_revision is not None:
        response["mprRevision"] = result.mpr_revision
    return response


async def _process_queued_mpr_operation(operation: _QueuedMprOperation) -> None:
    try:
        view = view_registry.get(operation.payload.view_id, workspace_id=operation.workspace_id)
        if view.view_type in FUSION_VIEW_TYPES:
            result = await asyncio.to_thread(viewer_service.handle_view_operation, operation.payload, operation.workspace_id)
        else:
            result = viewer_service.handle_view_operation(operation.payload, operation.workspace_id)
        await _dispatch_operation_result(operation.server, operation.sid, view, operation.payload, result)
    except Exception as exc:
        logger.exception("socket queued MPR operation failed sid=%s view_id=%s", operation.sid, operation.payload.view_id)
        await _emit_errors(operation.server, operation.sid, events=("image_error", "render_error"), exc=exc)


async def _run_mpr_operation_queue(queue_key: str, state: _MprOperationQueueState) -> None:
    current_task = asyncio.current_task()
    try:
        while True:
            operation = _pop_next_mpr_operation(state)
            if operation is None:
                return
            await _process_queued_mpr_operation(operation)
    finally:
        if state.task is current_task:
            if state.pending_start is None and state.pending_move is None and state.pending_end is None:
                _mpr_operation_queues.pop(queue_key, None)
            else:
                state.task = asyncio.create_task(_run_mpr_operation_queue(queue_key, state))


async def _process_queued_mpr_crosshair_state_operation(queue_key: str, operation: _QueuedMprOperation) -> None:
    try:
        view = view_registry.get(operation.payload.view_id, workspace_id=operation.workspace_id)
        result = viewer_service.handle_view_operation(operation.payload, operation.workspace_id)
        await _emit_mpr_state_updates(
            operation.server,
            operation.sid,
            result.mpr_state_view_ids,
            mpr_revision=result.mpr_revision,
        )
        if operation.payload.action_type == DRAG_ACTION_MOVE and result.broadcast_view_ids:
            _schedule_mpr_crosshair_preview(
                queue_key,
                _MprCrosshairPreviewRequest(
                    server=operation.server,
                    sid=operation.sid,
                    view_ids=result.broadcast_view_ids,
                    image_format=result.broadcast_image_format,
                    fast_preview=result.broadcast_fast_preview,
                    fast_preview_full_resolution=result.broadcast_fast_preview_full_resolution,
                    metadata_mode=result.broadcast_metadata_mode,
                    mpr_revision=result.mpr_revision,
                ),
            )
        if operation.payload.action_type == DRAG_ACTION_END:
            _cancel_mpr_crosshair_preview(queue_key)
            if result.broadcast_view_ids:
                _schedule_render_batch_for_views(
                    operation.server,
                    operation.sid,
                    result.broadcast_view_ids,
                    image_format=result.broadcast_image_format,
                    fast_preview=result.broadcast_fast_preview,
                    fast_preview_full_resolution=result.broadcast_fast_preview_full_resolution,
                    metadata_mode=result.broadcast_metadata_mode,
                    mpr_revision=result.mpr_revision,
                )
        logger.debug(
            "socket mpr_crosshair_state sid=%s view_id=%s action=%s",
            operation.sid,
            operation.payload.view_id,
            operation.payload.action_type,
        )
    except Exception as exc:
        logger.exception(
            "socket queued MPR crosshair state failed sid=%s view_id=%s",
            operation.sid,
            operation.payload.view_id,
        )
        await _emit_errors(operation.server, operation.sid, events=("image_error", "render_error"), exc=exc)


async def _run_mpr_crosshair_state_queue(queue_key: str, state: _MprOperationQueueState) -> None:
    current_task = asyncio.current_task()
    try:
        while True:
            operation = _pop_next_mpr_operation(state)
            if operation is None:
                return
            await _process_queued_mpr_crosshair_state_operation(queue_key, operation)
    finally:
        if state.task is current_task:
            if state.pending_start is None and state.pending_move is None and state.pending_end is None:
                _mpr_crosshair_state_queues.pop(queue_key, None)
            else:
                state.task = asyncio.create_task(_run_mpr_crosshair_state_queue(queue_key, state))


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
        if (
            payload.op_type in {VIEW_OP_TYPE_PAN, VIEW_OP_TYPE_ZOOM, VIEW_OP_TYPE_ROTATE_3D}
            and payload.action_type == DRAG_ACTION_START
        ):
            view_socket_hub.mark_view_interaction(payload.view_id, payload.interaction_id)
        if _should_queue_mpr_operation(view.view_type, payload):
            _enqueue_mpr_operation(
                _resolve_mpr_operation_queue_key(view, workspace_id),
                _QueuedMprOperation(
                    payload=payload,
                    server=server,
                    sid=sid,
                    workspace_id=workspace_id,
                ),
            )
            return {"ok": True}
        result = await _handle_view_operation_for_socket(payload, workspace_id, view.view_type)
        return await _dispatch_operation_result(server, sid, view, payload, result)
    except Exception as exc:
        logger.exception("socket view_operation failed sid=%s", sid)
        await _emit_errors(server, sid, events=("image_error", "render_error"), exc=exc)
        return {"ok": False, "message": _build_error_payload(exc)["message"]}


async def _handle_mpr_crosshair_state(server: socketio.AsyncServer, sid: str, data: dict) -> dict[str, object]:
    try:
        payload = ViewOperationRequest.model_validate(data)
        workspace_id = view_socket_hub.get_sid_workspace(sid)
        view = view_registry.get(payload.view_id, workspace_id=workspace_id)
        view_socket_hub.bind_view(sid, payload.view_id)
        if not _should_queue_mpr_crosshair_state_operation(view.view_type, payload):
            message = "mpr_crosshair_state requires an MPR crosshair or mprOblique start/move/end payload"
            await server.emit("render_error", {"message": message}, to=sid)
            return {"ok": False, "message": message}
        queue_key = _resolve_mpr_operation_queue_key(view, workspace_id)
        _enqueue_mpr_crosshair_state_operation(
            queue_key,
            _QueuedMprOperation(
                payload=payload,
                server=server,
                sid=sid,
                workspace_id=workspace_id,
            ),
        )
        return {"ok": True}
    except Exception as exc:
        logger.exception("socket mpr_crosshair_state failed sid=%s", sid)
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
        await _emit_render(server, sid, payload.view_id, image_format=payload.image_format)
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
        await webrtc_3d_transport_manager.close_sid(sid)
        view_socket_hub.unbind_sid(sid)
        logger.info("socket disconnected sid=%s", sid)
        return None

    @server.on("webrtc_3d_config")
    async def webrtc_3d_config(sid: str, _data: dict | None = None) -> dict[str, object]:
        return get_webrtc_3d_client_config()

    @server.on("webrtc_3d_offer")
    async def webrtc_3d_offer(sid: str, data: dict) -> dict[str, object]:
        view_id = str(data.get("viewId") or "")
        workspace_id = view_socket_hub.get_sid_workspace(sid)
        try:
            view = view_registry.get(view_id, workspace_id=workspace_id)
            if view.view_type != "3D":
                return {"ok": False, "message": "WebRTC transport is only available for 3D views"}
            view_socket_hub.bind_view(sid, view_id)
            return await webrtc_3d_transport_manager.create_answer(
                sid,
                view_id,
                str(data.get("sdp") or ""),
                str(data.get("type") or ""),
            )
        except Exception as exc:
            logger.exception("3d WebRTC offer failed sid=%s view_id=%s", sid, view_id)
            return {"ok": False, "message": _build_error_payload(exc)["message"]}

    @server.on("webrtc_3d_close")
    async def webrtc_3d_close(sid: str, data: dict) -> dict[str, object]:
        view_id = str(data.get("viewId") or "")
        if view_id:
            await webrtc_3d_transport_manager.close(sid, view_id)
        return {"ok": True, "viewId": view_id}

    @server.on("bind_view")
    async def bind_view(sid: str, data: dict) -> dict[str, object] | None:
        """Subscribe this Socket connection to a view's image_update events."""
        view_id = str(data.get("viewId") or data.get("view_id") or "")
        if not view_id:
            await server.emit("image_error", {"message": "viewId is required"}, to=sid)
            return {"ok": False, "message": "viewId is required"}
        should_render = bool(data.get("render", True))
        image_format = normalize_image_format(data.get("imageFormat") or data.get("image_format"))
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
                await _emit_render(server, sid, view_id, image_format=image_format)
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

    @server.on("mpr_crosshair_state")
    async def mpr_crosshair_state(sid: str, data: dict) -> dict[str, object]:
        return await _handle_mpr_crosshair_state(server, sid, data)

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
