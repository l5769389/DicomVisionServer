from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from app.core import (
    DRAG_ACTION_MOVE,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_SCROLL,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
    VIEW_OP_TYPE_ROTATE_3D,
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

    if payload.op_type == VIEW_OP_TYPE_CROSSHAIR and render_decision.mode == "none":
        return OperationRenderOutcome()

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


def _get_operation_handler(payload: ViewOperationRequest):
    if payload.op_type == VIEW_OP_TYPE_SCROLL:
        return _handle_scroll_operation
    if payload.op_type == VIEW_OP_TYPE_CROSSHAIR:
        return _handle_crosshair_operation
    if payload.op_type == VIEW_OP_TYPE_ZOOM:
        return _handle_zoom_operation
    if payload.op_type == VIEW_OP_TYPE_WINDOW:
        return _handle_window_operation
    if payload.op_type == VIEW_OP_TYPE_PAN:
        return _handle_pan_operation
    if payload.op_type == VIEW_OP_TYPE_ROTATE_3D:
        return _handle_rotate_3d_operation
    return _handle_generic_operation


def _sync_mpr_active_viewport(service: ViewerService, view: ViewRecord) -> bool:
    target_viewport = service._resolve_mpr_viewport(view)
    active_viewport_changed = view.mpr_active_viewport != target_viewport
    view.mpr_active_viewport = target_viewport
    if active_viewport_changed:
        view.is_initialized = True
    return active_viewport_changed


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
    if payload.action_type == "start":
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_single("jpeg", fast_preview=service._is_3d_view_type(view.view_type))
    return _render_single()


def _handle_window_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.action_type is None:
        return _handle_generic_operation(service, view, series, payload, is_mpr_view)
    service._handle_drag_window(view, payload)
    if is_mpr_view:
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_broadcast("jpeg", fast_preview=True)
        return _render_broadcast()
    if payload.action_type == "start":
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_single("jpeg", fast_preview=service._is_3d_view_type(view.view_type))
    return _render_single()


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
    if payload.action_type == "start":
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_single("jpeg", fast_preview=service._is_3d_view_type(view.view_type))
    return _render_single()


def _handle_rotate_3d_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    service._handle_drag_rotate_3d(view, payload)
    if payload.action_type == "start":
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_single("jpeg", fast_preview=True)
    return _render_single()


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


def _apply_shared_view_mutations(view: ViewRecord, payload: ViewOperationRequest) -> None:
    handled_drag_ops = {VIEW_OP_TYPE_CROSSHAIR, VIEW_OP_TYPE_ZOOM, VIEW_OP_TYPE_WINDOW, VIEW_OP_TYPE_PAN, VIEW_OP_TYPE_ROTATE_3D}
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


def _log_view_operation_state(service: ViewerService, view: ViewRecord, payload: ViewOperationRequest) -> None:
    service._logger.info(
        "view operation view_id=%s view_type=%s op_type=%s action_type=%s sub_op_type=%s index=%s zoom=%.4f offset_x=%.2f offset_y=%.2f ww=%s wl=%s axial=%s coronal=%s sagittal=%s",
        view.view_id,
        view.view_type,
        payload.op_type,
        payload.action_type,
        payload.sub_op_type,
        view.current_index,
        view.zoom,
        view.offset_x,
        view.offset_y,
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
        return OperationRenderOutcome()

    if render_decision.mode == "broadcast":
        return OperationRenderOutcome(
            broadcast_view_ids=tuple(group_view.view_id for group_view in service._get_mpr_group_views(view)),
            broadcast_image_format=render_decision.image_format,
            broadcast_fast_preview=render_decision.fast_preview,
        )

    if service._is_3d_view_type(view.view_type) and render_decision.fast_preview:
        return OperationRenderOutcome(
            deferred_view_ids=(view.view_id,),
            deferred_image_format=render_decision.image_format,
            deferred_fast_preview=True,
        )

    return OperationRenderOutcome(
        primary_result=service._render_by_view_type(
            view,
            image_format=render_decision.image_format,
            fast_preview=render_decision.fast_preview,
        )
    )
