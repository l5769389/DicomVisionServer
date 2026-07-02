from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    FUSION_PANE_OVERLAY_AXIAL,
    FUSION_PANE_PET_AXIAL,
    MPR_VIEWPORT_AXIAL,
    MPR_VIEWPORT_CORONAL,
    MPR_VIEWPORT_SAGITTAL,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_FUSION_CONFIG,
    VIEW_OP_TYPE_FUSION_REGISTRATION,
    VIEW_OP_TYPE_PET_CONFIG,
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
    VIEW_OP_TYPE_RENDER_3D_MODE,
    VIEW_OP_TYPE_SURFACE_CONFIG,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    VIEW_OP_TYPE_MPR_SEGMENTATION,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_MPR_CROSSHAIR_MODE,
    VIEW_OP_TYPE_MPR_STATE_SYNC,
    VIEW_OP_TYPE_MEASUREMENT,
    VIEW_OP_TYPE_ANNOTATION,
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
    primary_image_format: ImageFormat = "png"
    primary_fast_preview: bool = False
    primary_fast_preview_full_resolution: bool = False
    primary_metadata_mode: str = "full"
    draft_measurement: dict[str, object] | None = None
    mpr_revision: int | None = None
    broadcast_view_ids: tuple[str, ...] = ()
    broadcast_image_format: ImageFormat = "png"
    broadcast_fast_preview: bool = False
    broadcast_fast_preview_full_resolution: bool = False
    broadcast_metadata_mode: str = "full"
    deferred_view_ids: tuple[str, ...] = ()
    deferred_image_format: ImageFormat = "png"
    deferred_fast_preview: bool = False
    deferred_fast_preview_full_resolution: bool = False
    deferred_metadata_mode: str = "full"
    mpr_state_view_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderDecision:
    mode: RenderMode
    image_format: ImageFormat = "png"
    fast_preview: bool = False
    fast_preview_full_resolution: bool = False
    defer_single: bool = False
    metadata_mode: str = "full"
    draft_measurement: dict[str, object] | None = None
    broadcast_viewports: tuple[str, ...] | None = None
    emit_mpr_state_update: bool = False


OperationHandler = Callable[
    ["ViewerService", ViewRecord, SeriesRecord, ViewOperationRequest, bool],
    RenderDecision,
]


def handle_view_operation(
    service: ViewerService,
    payload: ViewOperationRequest,
    workspace_id: str | None = None,
) -> OperationRenderOutcome:
    view = (
        view_registry.get(payload.view_id)
        if workspace_id is None
        else view_registry.get(payload.view_id, workspace_id=workspace_id)
    )
    series = (
        series_registry.get(view.series_id)
        if workspace_id is None
        else series_registry.get(view.series_id, workspace_id=workspace_id)
    )
    if payload.op_type == VIEW_OP_TYPE_MPR_STATE_SYNC and payload.source_view_id:
        if workspace_id is None:
            view_registry.get(payload.source_view_id)
        else:
            view_registry.get(payload.source_view_id, workspace_id=workspace_id)
    is_mpr_view = service._is_mpr_view_type(view.view_type)
    is_fusion_view = service._is_fusion_view_type(view.view_type)

    active_viewport_changed = _sync_mpr_active_viewport(service, view) if is_mpr_view else False
    operation_handler = _get_operation_handler(payload)
    render_decision = operation_handler(service, view, series, payload, is_mpr_view)
    _apply_shared_view_mutations(service, view, payload)

    if is_mpr_view and active_viewport_changed and render_decision.mode != "none":
        render_decision = _promote_render_decision_to_broadcast(render_decision)

    if render_decision.mode == "none" and not render_decision.emit_mpr_state_update:
        return OperationRenderOutcome(draft_measurement=render_decision.draft_measurement)

    _log_view_operation_state(service, view, payload)
    mpr_revision = _resolve_operation_revision_for_render(service, view, payload, is_mpr_view, is_fusion_view)
    return _build_operation_render_outcome(
        service,
        view,
        render_decision,
        mpr_revision=mpr_revision,
        image_format=payload.image_format,
    )


def _render_none() -> RenderDecision:
    return RenderDecision(mode="none")


def _render_mpr_state_update(
    *,
    viewports: tuple[str, ...] | None = None,
) -> RenderDecision:
    return RenderDecision(
        mode="none",
        broadcast_viewports=viewports,
        emit_mpr_state_update=True,
    )


def _render_single(
    image_format: ImageFormat = "png",
    *,
    fast_preview: bool = False,
    fast_preview_full_resolution: bool = False,
    defer: bool = False,
    metadata_mode: str = "full",
) -> RenderDecision:
    return RenderDecision(
        mode="single",
        image_format=image_format,
        fast_preview=fast_preview,
        fast_preview_full_resolution=fast_preview_full_resolution,
        defer_single=defer,
        metadata_mode=metadata_mode,
    )


def _render_broadcast(
    image_format: ImageFormat = "png",
    *,
    fast_preview: bool = False,
    fast_preview_full_resolution: bool = False,
    metadata_mode: str = "full",
    viewports: tuple[str, ...] | None = None,
    emit_mpr_state_update: bool = False,
) -> RenderDecision:
    return RenderDecision(
        mode="broadcast",
        image_format=image_format,
        fast_preview=fast_preview,
        fast_preview_full_resolution=fast_preview_full_resolution,
        metadata_mode=metadata_mode,
        broadcast_viewports=viewports,
        emit_mpr_state_update=emit_mpr_state_update,
    )


def _render_full_resolution_preview_single(*, defer: bool = False, metadata_mode: str = "full") -> RenderDecision:
    return _render_single(
        "png",
        fast_preview=True,
        fast_preview_full_resolution=True,
        defer=defer,
        metadata_mode=metadata_mode,
    )


def _render_fast_preview_single(*, defer: bool = False, metadata_mode: str = "full") -> RenderDecision:
    return _render_single(
        "png",
        fast_preview=True,
        fast_preview_full_resolution=False,
        defer=defer,
        metadata_mode=metadata_mode,
    )


def _render_full_resolution_preview_broadcast(
    *,
    viewports: tuple[str, ...] | None = None,
    metadata_mode: str = "full",
) -> RenderDecision:
    return _render_broadcast(
        "png",
        fast_preview=True,
        fast_preview_full_resolution=True,
        metadata_mode=metadata_mode,
        viewports=viewports,
    )


def _render_mpr_crosshair_preview_broadcast(
    *,
    viewports: tuple[str, ...] | None = None,
) -> RenderDecision:
    return _render_broadcast(
        "png",
        fast_preview=True,
        fast_preview_full_resolution=True,
        metadata_mode="mpr-crosshair-preview",
        viewports=viewports,
        emit_mpr_state_update=True,
    )


def _promote_render_decision_to_broadcast(render_decision: RenderDecision) -> RenderDecision:
    if render_decision.mode == "broadcast":
        return render_decision
    return RenderDecision(
        mode="broadcast",
        image_format=render_decision.image_format,
        fast_preview=render_decision.fast_preview,
        fast_preview_full_resolution=render_decision.fast_preview_full_resolution,
        metadata_mode=render_decision.metadata_mode,
        broadcast_viewports=render_decision.broadcast_viewports,
        emit_mpr_state_update=render_decision.emit_mpr_state_update,
    )


MPR_GEOMETRY_REVISION_OPERATION_TYPES = {
    VIEW_OP_TYPE_SCROLL,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    VIEW_OP_TYPE_MPR_SEGMENTATION,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_RESET,
    VIEW_OP_TYPE_MPR_OBLIQUE,
    VIEW_OP_TYPE_MPR_CROSSHAIR_MODE,
    VIEW_OP_TYPE_MPR_STATE_SYNC,
}


def _resolve_operation_revision_for_render(
    service: ViewerService,
    view: ViewRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
    is_fusion_view: bool,
) -> int | None:
    if is_mpr_view:
        if _should_bump_mpr_geometry_revision(payload):
            return service._bump_mpr_revision(view.view_group)
        return service._get_mpr_revision(view.view_group)

    if is_fusion_view:
        return service._get_fusion_revision(view.view_group)

    return None


def _should_bump_mpr_geometry_revision(payload: ViewOperationRequest) -> bool:
    if payload.op_type not in MPR_GEOMETRY_REVISION_OPERATION_TYPES:
        return False

    if payload.op_type == VIEW_OP_TYPE_RESET:
        reset_target = str(payload.sub_op_type or "view").strip().lower()
        return reset_target not in {"measurements", "mtf", "annotations"}

    return True


MPR_VIEWPORT_ORDER = (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)


def _reference_mpr_viewports(service: ViewerService, view: ViewRecord) -> tuple[str, ...]:
    active_viewport = service._resolve_mpr_viewport(view)
    return tuple(viewport_key for viewport_key in MPR_VIEWPORT_ORDER if viewport_key != active_viewport)


def _viewports_with_active(
    service: ViewerService,
    view: ViewRecord,
    viewports: tuple[str, ...],
) -> tuple[str, ...]:
    active_viewport = service._resolve_mpr_viewport(view)
    return tuple(dict.fromkeys((active_viewport, *viewports)))


def _target_mpr_oblique_preview_viewports(
    service: ViewerService,
    view: ViewRecord,
    payload: ViewOperationRequest,
) -> tuple[str, ...]:
    if payload.line not in {"horizontal", "vertical"}:
        return _reference_mpr_viewports(service, view)
    if service._get_mpr_crosshair_mode(view.view_group) != "double-oblique":
        return _reference_mpr_viewports(service, view)
    active_viewport = service._resolve_mpr_viewport(view)
    return (service._resolve_mpr_oblique_target_viewport(active_viewport, payload.line),)


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
    move_image_format: ImageFormat = "jpeg",
    fast_preview_on_move: bool | None = None,
    fast_preview_full_resolution_on_move: bool = False,
    defer_on_move: bool = False,
    defer_on_end: bool = False,
    move_metadata_mode: str = "full",
) -> RenderDecision:
    if payload.action_type == "start":
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_single(
            move_image_format,
            fast_preview=service._is_3d_view_type(view.view_type) if fast_preview_on_move is None else fast_preview_on_move,
            fast_preview_full_resolution=fast_preview_full_resolution_on_move,
            defer=defer_on_move,
            metadata_mode=move_metadata_mode,
        )
    return _render_single(defer=defer_on_end)


def _handle_scroll_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.delta is None:
        return _render_none()
    if service._is_fusion_view_type(view.view_type):
        if not service._handle_fusion_scroll(view, payload):
            return _render_none()
        return _render_broadcast()
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
        return _render_mpr_crosshair_preview_broadcast(
            viewports=_reference_mpr_viewports(service, view),
        )
    if payload.action_type == "end":
        return _render_broadcast(viewports=_viewports_with_active(service, view, _reference_mpr_viewports(service, view)))
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
    if service._is_fusion_view_type(view.view_type):
        service._handle_fusion_drag_zoom(view, payload)
        if payload.action_type == DRAG_ACTION_START:
            return _render_none()
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_full_resolution_preview_broadcast(metadata_mode="fusion-zoom-preview")
        return _render_broadcast()
    service._handle_drag_zoom(view, payload)
    if is_mpr_view:
        if payload.action_type == DRAG_ACTION_START:
            return _render_none()
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_fast_preview_single(defer=True, metadata_mode="mpr-zoom-preview")
        return _render_single(defer=True)
    is_3d_view = service._is_3d_view_type(view.view_type)
    return _resolve_drag_single_render_decision(
        service,
        view,
        payload,
        move_image_format="png",
        fast_preview_on_move=True,
        fast_preview_full_resolution_on_move=not is_3d_view,
        defer_on_move=True,
        defer_on_end=True,
        move_metadata_mode="stack-zoom-preview",
    )


def _handle_window_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if service._is_fusion_view_type(view.view_type):
        if not service._handle_fusion_window(view, payload):
            return _render_none()
        if payload.action_type == DRAG_ACTION_START:
            return _render_none()
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_full_resolution_preview_broadcast(metadata_mode="mpr-pixel-preview")
        return _render_broadcast()

    if service._is_pet_view_type(view.view_type):
        if not service._handle_pet_window(view, payload):
            return _render_none()
        if payload.action_type == DRAG_ACTION_START:
            return _render_none()
        return _render_single(
            "png",
            fast_preview=payload.action_type == DRAG_ACTION_MOVE,
            defer=payload.action_type == DRAG_ACTION_MOVE,
            metadata_mode="stack-pixel-preview" if payload.action_type == DRAG_ACTION_MOVE else "full",
        )

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
            return _render_full_resolution_preview_broadcast(metadata_mode="mpr-pixel-preview")
        return _render_broadcast()
    return _resolve_drag_single_render_decision(
        service,
        view,
        payload,
        move_image_format="png",
        fast_preview_on_move=True,
        defer_on_move=True,
        defer_on_end=True,
        move_metadata_mode="stack-pixel-preview",
    )


def _handle_pan_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if payload.action_type is None:
        return _handle_generic_operation(service, view, series, payload, is_mpr_view)
    if service._is_fusion_view_type(view.view_type):
        service._handle_fusion_drag_pan(view, payload)
        if payload.action_type == DRAG_ACTION_START:
            return _render_none()
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_full_resolution_preview_broadcast()
        return _render_broadcast()
    service._handle_drag_pan(view, payload)
    if is_mpr_view:
        if payload.action_type == DRAG_ACTION_START:
            return _render_none()
        if payload.action_type == DRAG_ACTION_MOVE:
            return _render_fast_preview_single(defer=True, metadata_mode="mpr-pan-zoom-preview")
        return _render_single(defer=True)
    return _resolve_drag_single_render_decision(
        service,
        view,
        payload,
        move_image_format="png",
        fast_preview_on_move=True,
        defer_on_move=True,
        defer_on_end=True,
        move_metadata_mode="stack-geometry-preview",
    )


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
    if service._is_fusion_view_type(view.view_type):
        if not service._handle_fusion_pseudocolor(view, payload):
            return _render_none()
        return _render_broadcast()
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
            return _render_full_resolution_preview_broadcast(metadata_mode="mpr-pan-zoom-preview")
        return _render_broadcast()
    service._handle_drag_rotate_3d(view, payload)
    return _resolve_drag_single_render_decision(
        service,
        view,
        payload,
        fast_preview_on_move=True,
        move_metadata_mode="stack-geometry-preview",
    )


def _handle_transform_2d_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if (
        payload.x is None
        and payload.y is None
        and payload.zoom is None
        and payload.rotation_degrees is None
        and payload.hor_flip is None
        and payload.ver_flip is None
    ):
        return _render_none()
    if service._is_fusion_view_type(view.view_type):
        service._clear_fusion_registration_overlay_frame_locks(view.view_group)
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

    if reset_target == "annotations":
        if not service._clear_annotations(view):
            return _render_none()
        return _render_broadcast() if is_mpr_view else _render_single()

    if reset_target in {"mprcrosshair", "mpr-crosshair", "crosshair"}:
        if not service._reset_mpr_crosshair_state(view):
            return _render_none()
        return _render_broadcast()

    if reset_target in {"rotate3d", "rotate-3d", "3drotation"}:
        if not service._reset_rotate_3d_state(view):
            return _render_none()
        return _render_broadcast() if is_mpr_view else _render_single()

    if service._is_fusion_view_type(view.view_type):
        service._reset_view(view)
        return _render_broadcast()

    if reset_target in {"mtf"}:
        return _render_broadcast() if is_mpr_view else _render_single()

    if reset_target == "all":
        service._clear_measurements(view)
        service._clear_annotations(view)
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


def _handle_render_3d_mode_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    service._handle_render_3d_mode(view, payload)
    return _render_single()


def _handle_surface_config_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    service._handle_surface_config(view, payload)
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


def _handle_annotation_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    if payload.action_type == "delete":
        if service._delete_annotation(view, payload.annotation_id or payload.measurement_id):
            return _render_broadcast() if is_mpr_view else _render_single()
        return _render_none()
    if payload.action_type in {"start", "move"}:
        return _render_none()
    if payload.action_type == "end" or payload.action_type is None:
        if service._handle_annotation(view, payload):
            return _render_broadcast() if is_mpr_view else _render_single()
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
    if payload.action_type == DRAG_ACTION_START:
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_broadcast(
            "png",
            fast_preview=True,
            fast_preview_full_resolution=True,
            metadata_mode="mpr-pan-zoom-preview",
            emit_mpr_state_update=True,
        )
    return _render_broadcast()


def _handle_mpr_segmentation_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    if not is_mpr_view:
        return _render_none()
    refresh_stats = payload.action_type != DRAG_ACTION_MOVE
    if not service._handle_mpr_segmentation_config(view, payload, series=series, refresh_stats=refresh_stats):
        return _render_none()
    if payload.action_type == DRAG_ACTION_START:
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_full_resolution_preview_single(defer=True, metadata_mode="mpr-segmentation-preview")
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
        return _render_mpr_crosshair_preview_broadcast(
            viewports=_target_mpr_oblique_preview_viewports(service, view, payload),
        )
    if payload.action_type == "end":
        return _render_broadcast(
            viewports=_viewports_with_active(
                service,
                view,
                _target_mpr_oblique_preview_viewports(service, view, payload),
            )
        )
    return _render_broadcast()


def _handle_mpr_crosshair_mode_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series
    if not is_mpr_view:
        return _render_none()
    if not service._handle_mpr_crosshair_mode(view, payload):
        return _render_none()
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


def _handle_fusion_config_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if not service._is_fusion_view_type(view.view_type):
        return _render_none()
    if not service._handle_fusion_config(view, payload):
        return _render_none()
    if payload.action_type == DRAG_ACTION_MOVE:
        return _render_full_resolution_preview_broadcast()
    return _render_broadcast()


def _handle_fusion_registration_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if not service._is_fusion_view_type(view.view_type):
        return _render_none()
    if not service._handle_fusion_registration(view, payload):
        return _render_none()
    if payload.action_type == DRAG_ACTION_START:
        return _render_none()
    if payload.action_type in {DRAG_ACTION_MOVE, DRAG_ACTION_END}:
        return _render_broadcast(
            "png",
            fast_preview=True,
            metadata_mode="fusion-registration-layer-preview",
            viewports=(FUSION_PANE_OVERLAY_AXIAL, FUSION_PANE_PET_AXIAL),
        )
    return _render_broadcast()


def _handle_pet_config_operation(
    service: ViewerService,
    view: ViewRecord,
    series: SeriesRecord,
    payload: ViewOperationRequest,
    is_mpr_view: bool,
) -> RenderDecision:
    del series, is_mpr_view
    if not service._is_pet_view_type(view.view_type):
        return _render_none()
    if not service._handle_pet_config(view, payload):
        return _render_none()
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
    VIEW_OP_TYPE_RENDER_3D_MODE: _handle_render_3d_mode_operation,
    VIEW_OP_TYPE_SURFACE_CONFIG: _handle_surface_config_operation,
    VIEW_OP_TYPE_MPR_MIP_CONFIG: _handle_mpr_mip_config_operation,
    VIEW_OP_TYPE_MPR_SEGMENTATION: _handle_mpr_segmentation_operation,
    VIEW_OP_TYPE_MPR_OBLIQUE: _handle_mpr_oblique_operation,
    VIEW_OP_TYPE_MPR_CROSSHAIR_MODE: _handle_mpr_crosshair_mode_operation,
    VIEW_OP_TYPE_MPR_STATE_SYNC: _handle_mpr_state_sync_operation,
    VIEW_OP_TYPE_FUSION_CONFIG: _handle_fusion_config_operation,
    VIEW_OP_TYPE_FUSION_REGISTRATION: _handle_fusion_registration_operation,
    VIEW_OP_TYPE_PET_CONFIG: _handle_pet_config_operation,
    VIEW_OP_TYPE_MEASUREMENT: _handle_measurement_operation,
    VIEW_OP_TYPE_ANNOTATION: _handle_annotation_operation,
}


def _apply_shared_view_mutations(service: ViewerService, view: ViewRecord, payload: ViewOperationRequest) -> None:
    handled_drag_ops = {
        VIEW_OP_TYPE_CROSSHAIR,
        VIEW_OP_TYPE_FUSION_CONFIG,
        VIEW_OP_TYPE_FUSION_REGISTRATION,
        VIEW_OP_TYPE_MPR_OBLIQUE,
        VIEW_OP_TYPE_ZOOM,
        VIEW_OP_TYPE_WINDOW,
        VIEW_OP_TYPE_PAN,
        VIEW_OP_TYPE_ROTATE_3D,
    }
    is_transform_2d = payload.op_type == VIEW_OP_TYPE_TRANSFORM_2D
    mutated = False
    if payload.x is not None:
        if is_transform_2d:
            view.offset_x = float(payload.x)
            mutated = True
        elif payload.op_type not in handled_drag_ops:
            view.offset_x += float(payload.x)
            mutated = True
    if payload.y is not None:
        if is_transform_2d:
            view.offset_y = float(payload.y)
            mutated = True
        elif payload.op_type not in handled_drag_ops:
            view.offset_y += float(payload.y)
            mutated = True
    if payload.zoom is not None and is_transform_2d:
        next_zoom = viewport_transformer.clamp_zoom(float(payload.zoom))
        if service._is_3d_view_type(view.view_type):
            next_zoom = service._clamp_3d_zoom(next_zoom)
        view.zoom = next_zoom
        mutated = True
    if payload.hor_flip is not None:
        view.hor_flip = payload.hor_flip
        mutated = True
    if payload.ver_flip is not None:
        view.ver_flip = payload.ver_flip
        mutated = True
    if payload.rotation_degrees is not None:
        view.rotation_degrees = viewport_transformer.normalize_rotation_degrees(payload.rotation_degrees)
        mutated = True
    if mutated:
        view.is_initialized = True
        if is_transform_2d:
            service._reset_drag_state(view)


def _log_view_operation_state(service: ViewerService, view: ViewRecord, payload: ViewOperationRequest) -> None:
    log_method = (
        service._logger.debug
        if payload.action_type == DRAG_ACTION_MOVE
        and payload.op_type in {
            VIEW_OP_TYPE_CROSSHAIR,
            VIEW_OP_TYPE_MPR_OBLIQUE,
            VIEW_OP_TYPE_FUSION_CONFIG,
            VIEW_OP_TYPE_FUSION_REGISTRATION,
            VIEW_OP_TYPE_ZOOM,
            VIEW_OP_TYPE_WINDOW,
            VIEW_OP_TYPE_PAN,
            VIEW_OP_TYPE_ROTATE_3D,
        }
        else service._logger.info
    )
    log_method(
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
    *,
    mpr_revision: int | None = None,
    image_format: ImageFormat = "png",
) -> OperationRenderOutcome:
    if render_decision.mode == "none":
        mpr_state_view_ids: tuple[str, ...] = ()
        if render_decision.emit_mpr_state_update:
            mpr_state_view_ids = tuple(
                group_view.view_id
                for group_view in service._get_mpr_group_views(view)
                if group_view.width
                and group_view.height
                and (
                    render_decision.broadcast_viewports is None
                    or service._resolve_mpr_viewport(group_view) in render_decision.broadcast_viewports
                )
            )
        return OperationRenderOutcome(
            draft_measurement=render_decision.draft_measurement,
            mpr_revision=mpr_revision,
            mpr_state_view_ids=mpr_state_view_ids,
        )

    if render_decision.mode == "broadcast":
        if service._is_fusion_view_type(view.view_type):
            sized_group_view_ids = tuple(
                group_view.view_id
                for group_view in service._get_group_views(view)
                if group_view.width and group_view.height
                and (
                    render_decision.broadcast_viewports is None
                    or service._resolve_fusion_pane_role(group_view) in render_decision.broadcast_viewports
                )
            )
            if not sized_group_view_ids:
                return OperationRenderOutcome(draft_measurement=render_decision.draft_measurement)

            return OperationRenderOutcome(
                mpr_revision=mpr_revision,
                broadcast_view_ids=sized_group_view_ids,
                broadcast_image_format=image_format,
                broadcast_fast_preview=render_decision.fast_preview,
                broadcast_fast_preview_full_resolution=render_decision.fast_preview_full_resolution,
                broadcast_metadata_mode=render_decision.metadata_mode,
            )

        sized_group_view_ids = tuple(
            group_view.view_id
            for group_view in service._get_mpr_group_views(view)
            if group_view.width
            and group_view.height
            and (
                render_decision.broadcast_viewports is None
                or service._resolve_mpr_viewport(group_view) in render_decision.broadcast_viewports
            )
        )
        if not sized_group_view_ids:
            return OperationRenderOutcome(draft_measurement=render_decision.draft_measurement)

        return OperationRenderOutcome(
            mpr_revision=mpr_revision,
            broadcast_view_ids=sized_group_view_ids,
            broadcast_image_format=image_format,
            broadcast_fast_preview=render_decision.fast_preview,
            broadcast_fast_preview_full_resolution=render_decision.fast_preview_full_resolution,
            broadcast_metadata_mode=render_decision.metadata_mode,
            mpr_state_view_ids=sized_group_view_ids if render_decision.emit_mpr_state_update else (),
        )

    if service._is_3d_view_type(view.view_type) or render_decision.defer_single:
        # Expensive settled frames are routed through the socket render hub so
        # rapid previews and final frames can be coalesced without blocking the
        # interactive operation path.
        return OperationRenderOutcome(
            mpr_revision=mpr_revision,
            deferred_view_ids=(view.view_id,),
            deferred_image_format=image_format,
            deferred_fast_preview=render_decision.fast_preview,
            deferred_fast_preview_full_resolution=render_decision.fast_preview_full_resolution,
            deferred_metadata_mode=render_decision.metadata_mode,
        )

    return OperationRenderOutcome(
        draft_measurement=render_decision.draft_measurement,
        mpr_revision=mpr_revision,
        primary_image_format=image_format,
        primary_fast_preview=render_decision.fast_preview,
        primary_fast_preview_full_resolution=render_decision.fast_preview_full_resolution,
        primary_metadata_mode=render_decision.metadata_mode,
        primary_result=service._render_by_view_type(
            view,
            image_format=image_format,
            fast_preview=render_decision.fast_preview,
            fast_preview_full_resolution=render_decision.fast_preview_full_resolution,
            metadata_mode=render_decision.metadata_mode,
        )
    )
