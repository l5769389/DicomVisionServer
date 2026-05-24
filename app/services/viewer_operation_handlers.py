from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from app.core import (
    DRAG_ACTION_MOVE,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_PSEUDOCOLOR,
    VIEW_OP_TYPE_SCROLL,
    VIEW_OP_TYPE_TRANSFORM_2D,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_RESET,
    VIEW_OP_TYPE_VOLUME_PRESET,
    VIEW_OP_TYPE_VOLUME_CONFIG,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_MPR_STATE_SYNC,
    VIEW_OP_TYPE_MEASUREMENT,
)
from app.models.viewer import SeriesRecord, ViewRecord
from app.schemas.view import ImageFormat, ViewOperationRequest
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer
from app.services.view_registry import view_registry

if TYPE_CHECKING:
    from app.services.viewer_service import RenderedImageResult, ViewerService


RenderMode = Literal["none", "single", "broadcast"]

@dataclass(frozen=True)
class OperationRenderOutcome:
    primary_result: RenderedImageResult | None = None
    draft_measurement: dict[str, object] | None = None
    broadcast_view_ids: tuple[str, ...] = ()
    broadcast_image_format: ImageFormat = "png"
    broadcast_fast_preview: bool = False
    deferred_view_ids: tuple[str, ...] = ()
    deferred_image_format: ImageFormat = "png"
    deferred_fast_preview: bool = False


@dataclass(frozen=True)
class RenderDecision:
    mode: RenderMode
    image_format: ImageFormat = "png"
    fast_preview: bool = False
    draft_measurement: dict[str, object] | None = None


OperationHandler = Callable[
    ["ViewerService", ViewRecord, SeriesRecord, ViewOperationRequest, bool],
    RenderDecision,
]


def handle_view_operation(service: ViewerService, payload: ViewOperationRequest) -> OperationRenderOutcome:
    view = view_registry.get(payload.view_id)
    series = series_registry.get(view.series_id)
    is_mpr_view = service._is_mpr_view_type(view.view_type)

    active_viewport_changed = _sync_mpr_active_viewport(service, view) if is_mpr_view else False
    operation_handler = _get_operation_handler(payload)
    render_decision = operation_handler(service, view, series, payload, is_mpr_view)
    _apply_shared_view_mutations(view, payload)

    if is_mpr_view and active_viewport_changed:
        render_decision = _promote_render_decision_to_broadcast(render_decision)

    if render_decision.mode == "none":
        return OperationRenderOutcome(draft_measurement=render_decision.draft_measurement)

    _log_view_operation_state(service, view, payload)
    return _build_operation_render_outcome(service, view, render_decision)


def _render_none() -> RenderDecision:
    return RenderDecision(mode="none")


def _render_single(image_format: ImageFormat = "png", *, fast_preview: bool = False) -> RenderDecision:
    return RenderDecision(mode="single", image_format=image_format, fast_preview=fast_preview)


def _render_broadcast(image_format: ImageFormat = "png", *, fast_preview: bool = False) -> RenderDecision:
    return RenderDecision(mode="broadcast", image_format=image_format, fast_preview=fast_preview)


def _promote_render_decision_to_broadcast(render_decision: RenderDecision) -> RenderDecision:
    if render_decision.mode == "broadcast":
        return render_decision
    return RenderDecision(
        mode="broadcast",
        image_format=render_decision.image_format,
        fast_preview=render_decision.fast_preview,
    )


def _get_operation_handler(payload: ViewOperationRequest) -> OperationHandler:
    return OPERATION_HANDLERS.get(payload.op_type, _handle_generic_operation)


def _sync_mpr_active_viewport(service: ViewerService, view: ViewRecord) -> bool:
    target_viewport = service._resolve_mpr_viewport(view)
    active_viewport_changed = view.mpr_active_viewport != target_viewport
    view.mpr_active_viewport = target_viewport
    return active_viewport_changed


def _resolve_drag_single_render_decision(
    service: ViewerService,
    view: ViewRecord,
    payload: ViewOperationRequest,
    *,
    fast_preview_on_move: bool | None = None,
) -> RenderDecision:
    if payload.action_type == "start":
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_single(
            "jpeg",
            fast_preview=service._is_3d_view_type(view.view_type) if fast_preview_on_move is None else fast_preview_on_move,
        )
    return _render_single()


def _handle_scroll_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.delta is None:
        return _render_none()
    service._handle_scroll(view, series, int(payload.delta))
    return _render_broadcast() if is_mpr_view else _render_single()


def _handle_crosshair_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    should_broadcast_group = service._handle_mpr_crosshair(view, payload)
    if not should_broadcast_group:
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_broadcast("jpeg")
    return _render_broadcast()


def _handle_zoom_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.action_type is None:
        return _handle_generic_operation(service, view, series, payload, is_mpr_view)
    service._handle_drag_zoom(view, payload)
    return _resolve_drag_single_render_decision(service, view, payload)


def _handle_window_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.action_type is None and (payload.ww is not None or payload.wl is not None):
        if payload.ww is not None:
            view.window_width = float(payload.ww)
        if payload.wl is not None:
            view.window_center = float(payload.wl)
        view.is_initialized = True
        return _render_broadcast() if is_mpr_view else _render_single()

    if payload.action_type is None:
        return _handle_generic_operation(service, view, series, payload, is_mpr_view)
    service._handle_drag_window(view, payload)
    if is_mpr_view:
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_broadcast("jpeg", fast_preview=True)
        return _render_broadcast()
    return _resolve_drag_single_render_decision(service, view, payload)


def _handle_pan_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.action_type is None:
        return _handle_generic_operation(service, view, series, payload, is_mpr_view)
    service._handle_drag_pan(view, payload)
    return _resolve_drag_single_render_decision(service, view, payload)


def _handle_pseudocolor_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if payload.pseudocolor_preset is None:
        return _render_none()
    service._handle_pseudocolor(view, payload)
    if not view.width or not view.height:
        return _render_none()
    return _render_single()


def _handle_rotate_3d_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    if is_mpr_view:
        if not service._handle_mpr_model_rotate_3d(view, payload):
            return _render_none()
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_broadcast("jpeg", fast_preview=True)
        return _render_broadcast()
    service._handle_drag_rotate_3d(view, payload)
    return _resolve_drag_single_render_decision(service, view, payload, fast_preview_on_move=True)


def _handle_transform_2d_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del service, view, series, is_mpr_view
    if payload.rotation_degrees is None and payload.hor_flip is None and payload.ver_flip is None:
        return _render_none()
    return _render_single()


def _handle_reset_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    reset_target = str(payload.sub_op_type or "view").strip().lower()

    if reset_target == "measurements":
        if not service._clear_measurements(view):
            return _render_none()
        return _render_broadcast() if is_mpr_view else _render_single()

    if reset_target in {"mtf", "annotations"}:
        return _render_broadcast() if is_mpr_view else _render_single()

    if reset_target == "all":
        service._clear_measurements(view)
        service._reset_view(view)
        return _render_broadcast() if is_mpr_view else _render_single()

    service._reset_view(view)
    return _render_broadcast() if is_mpr_view else _render_single()


def _handle_volume_preset_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    service._handle_volume_preset(view, payload)
    return _render_single()


def _handle_volume_config_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    service._handle_volume_config(view, payload)
    return _render_single()


def _handle_measurement_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if payload.action_type == "delete":
        if service._delete_measurement(view, payload.measurement_id):
            return _render_single()
        return _render_none()
    if payload.action_type in {"start", "move"}:
        return RenderDecision(mode="none", draft_measurement=service._build_measurement_preview(view, payload))
    if payload.action_type == "end":
        if service._handle_measurement(view, payload):
            return _render_single()
        return _render_none()
    return _render_none()


def _handle_mpr_mip_config_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    if not is_mpr_view:
        return _render_none()
    if not service._handle_mpr_mip_config(view, payload):
        return _render_none()
    return _render_broadcast()


def _handle_mpr_oblique_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    if not is_mpr_view:
        return _render_none()
    if not service._handle_mpr_oblique(view, payload):
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_broadcast("jpeg", fast_preview=True)
    return _render_broadcast()


def _handle_mpr_state_sync_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    if not is_mpr_view or not payload.source_view_id:
        return _render_none()
    if not service._sync_mpr_state_from_source_view(view, payload.source_view_id):
        return _render_none()
    return _render_broadcast()


def _handle_generic_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if payload.zoom is not None and payload.zoom > 0:
        next_zoom = viewport_transformer.clamp_zoom(payload.zoom)
        if service._is_3d_view_type(view.view_type):
            next_zoom = service._clamp_3d_zoom(next_zoom)
        view.zoom = next_zoom
        view.is_initialized = True
        return _render_single()
    return _render_none()


OPERATION_HANDLERS: dict[str, OperationHandler] = {
    VIEW_OP_TYPE_SCROLL: _handle_scroll_operation,
    VIEW_OP_TYPE_CROSSHAIR: _handle_crosshair_operation,
    VIEW_OP_TYPE_ZOOM: _handle_zoom_operation,
    VIEW_OP_TYPE_WINDOW: _handle_window_operation,
    VIEW_OP_TYPE_PSEUDOCOLOR: _handle_pseudocolor_operation,
    VIEW_OP_TYPE_PAN: _handle_pan_operation,
    VIEW_OP_TYPE_TRANSFORM_2D: _handle_transform_2d_operation,
    VIEW_OP_TYPE_ROTATE_3D: _handle_rotate_3d_operation,
    VIEW_OP_TYPE_RESET: _handle_reset_operation,
    VIEW_OP_TYPE_VOLUME_PRESET: _handle_volume_preset_operation,
    VIEW_OP_TYPE_VOLUME_CONFIG: _handle_volume_config_operation,
    VIEW_OP_TYPE_MPR_MIP_CONFIG: _handle_mpr_mip_config_operation,
    VIEW_OP_TYPE_MPR_OBLIQUE: _handle_mpr_oblique_operation,
    VIEW_OP_TYPE_MPR_STATE_SYNC: _handle_mpr_state_sync_operation,
    VIEW_OP_TYPE_MEASUREMENT: _handle_measurement_operation,
}


def _apply_shared_view_mutations(view: ViewRecord, payload: ViewOperationRequest) -> None:
    handled_drag_ops = {VIEW_OP_TYPE_CROSSHAIR, VIEW_OP_TYPE_MPR_OBLIQUE, VIEW_OP_TYPE_ZOOM, VIEW_OP_TYPE_WINDOW, VIEW_OP_TYPE_PAN, VIEW_OP_TYPE_ROTATE_3D}
    if payload.x is not None and payload.op_type not in handled_drag_ops:
        view.offset_x += float(payload.x)
        view.is_initialized = True
    if payload.y is not None and payload.op_type not in handled_drag_ops:
        view.offset_y += float(payload.y)
        view.is_initialized = True
    if payload.hor_flip is not None:
        view.hor_flip = payload.hor_flip
        view.is_initialized = True
    if payload.ver_flip is not None:
        view.ver_flip = payload.ver_flip
        view.is_initialized = True
    if payload.rotation_degrees is not None:
        view.rotation_degrees = viewport_transformer.normalize_rotation_degrees(payload.rotation_degrees)
        view.is_initialized = True


def _log_view_operation_state(service: ViewerService, view: ViewRecord, payload: ViewOperationRequest) -> None:
    service._logger.info(
        "view operation view_id=%s series_id=%s group_id=%s view_type=%s op_type=%s action_type=%s sub_op_type=%s index=%s zoom=%.4f offset_x=%.2f offset_y=%.2f rotation=%s hor_flip=%s ver_flip=%s ww=%s wl=%s axial=%s coronal=%s sagittal=%s",
        view.view_id,
        view.series_id,
        view.view_group.group_id if view.view_group is not None else None,
        view.view_type,
        payload.op_type,
        payload.action_type,
        payload.sub_op_type,
        view.current_index,
        view.zoom,
        view.offset_x,
        view.offset_y,
        view.rotation_degrees,
        view.hor_flip,
        view.ver_flip,
        view.window_width,
        view.window_center,
        view.mpr_axial_index,
        view.mpr_coronal_index,
        view.mpr_sagittal_index,
    )


def _build_operation_render_outcome(
    service: ViewerService,
    view: ViewRecord,
    render_decision: RenderDecision,
) -> OperationRenderOutcome:
    if render_decision.mode == "none":
        return OperationRenderOutcome(draft_measurement=render_decision.draft_measurement)

    if render_decision.mode == "broadcast":
        return OperationRenderOutcome(
            broadcast_view_ids=tuple(group_view.view_id for group_view in service._get_mpr_group_views(view)),
            broadcast_image_format=render_decision.image_format,
            broadcast_fast_preview=render_decision.fast_preview,
        )

    if service._is_3d_view_type(view.view_type) and render_decision.fast_preview:
        # 3D drag interactions first schedule a quick preview frame and let the
        # socket runtime follow up with the heavier render path asynchronously.
        return OperationRenderOutcome(
            deferred_view_ids=(view.view_id,),
            deferred_image_format=render_decision.image_format,
            deferred_fast_preview=True,
        )

    return OperationRenderOutcome(
        draft_measurement=render_decision.draft_measurement,
        primary_result=service._render_by_view_type(
            view,
            image_format=render_decision.image_format,
            fast_preview=render_decision.fast_preview,
        )
    )





