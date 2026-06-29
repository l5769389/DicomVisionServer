from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL, ZOOM_MAX, ZOOM_MIN
from app.models.viewer import MprMipViewportState, MprRotationDragRecord, ViewGroupRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.mpr import build_geometry_from_patient_transform, ijk_to_world_point
from app.services.mpr_geometry import VolumePatientTransform
from app.services import viewer_service as viewer_service_module
from app.services.view_registry import view_registry
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import viewport_transformer


def _build_service_with_stubbed_series(monkeypatch):
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id, workspace_id=None: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series, *args, **kwargs: volume)
    return service, series, volume


def _build_service_with_left_handed_patient_geometry(monkeypatch):
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    transform = VolumePatientTransform(
        origin=np.zeros(3, dtype=np.float64),
        axis_vectors=(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        ),
        shape=volume.shape,
    )
    geometry = build_geometry_from_patient_transform(transform)
    monkeypatch.setattr(service, "_get_series_volume_geometry", lambda resolved_series, shape: geometry)
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda resolved_series: transform)
    return service, series, volume


def _build_axial_view(service: ViewerService, series, volume: np.ndarray) -> tuple[ViewGroupRecord, ViewRecord]:
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id=series.series_id)
    view = ViewRecord(view_id="v-ax", series_id=series.series_id, view_type="MPR", view_group=group)
    view.width = 240
    view.height = 240
    service._reset_mpr_group_geometry(group, volume.shape, series=series)
    return group, view


def test_mpr_render_by_id_uses_view_state_snapshot(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)
    view.is_initialized = True
    group.mpr_revision = 7
    captured_views: list[ViewRecord] = []

    def fake_render_by_view_type(render_view: ViewRecord, **kwargs):
        del kwargs
        captured_views.append(render_view)
        return SimpleNamespace(meta=None, image_bytes=b"")

    monkeypatch.setattr(service, "_render_by_view_type", fake_render_by_view_type)
    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[view.view_id] = view
        service.render_view_by_id(view.view_id)
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert captured_views
    snapshot = captured_views[0]
    assert snapshot is not view
    assert snapshot.view_group is not group
    assert snapshot.view_group is not None
    assert snapshot.view_group.mpr_revision == 7
    group.mpr_revision = 9
    assert snapshot.view_group.mpr_revision == 7


def test_mpr_state_update_payload_builds_crosshair_metadata_without_reslice(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    coronal_view.width = axial_view.width
    coronal_view.height = axial_view.height
    coronal_view.is_initialized = True
    group.mpr_revision = 11

    def fail_reslice(*args, **kwargs):
        del args, kwargs
        raise AssertionError("MPR state update must not reslice image pixels")

    monkeypatch.setattr(service, "_extract_mpr_plane", fail_reslice)
    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[coronal_view.view_id] = coronal_view
        payload = service.build_mpr_state_update_payload(coronal_view.view_id, mpr_revision=11)
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert payload is not None
    assert payload["viewId"] == coronal_view.view_id
    assert payload["mprRevision"] == 11
    slice_info = payload["slice_info"]
    assert isinstance(slice_info, dict)
    assert slice_info["total"] == 6
    assert 0 <= slice_info["current"] < slice_info["total"]
    assert payload["mprCrosshairMode"] == "orthogonal"
    assert "mprFrame" in payload
    assert "mprCursor" in payload
    assert "mprPlane" in payload
    assert "mpr_crosshair" in payload


def test_mpr_state_update_payloads_reuse_group_pose_context(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for view in (axial_view, coronal_view, sagittal_view):
        view.width = 240
        view.height = 240
        view.is_initialized = True
    group.mpr_revision = 13

    build_pose_context_calls = 0
    original_build_pose_context = service._build_mpr_pose_context

    def count_build_pose_context(*args, **kwargs):
        nonlocal build_pose_context_calls
        build_pose_context_calls += 1
        return original_build_pose_context(*args, **kwargs)

    monkeypatch.setattr(service, "_build_mpr_pose_context", count_build_pose_context)
    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )
        payloads = service.build_mpr_state_update_payloads(
            (coronal_view.view_id, sagittal_view.view_id),
            mpr_revision=13,
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert build_pose_context_calls == 1
    assert set(payloads) == {coronal_view.view_id, sagittal_view.view_id}
    assert payloads[coronal_view.view_id]["mprRevision"] == 13
    assert payloads[sagittal_view.view_id]["mprRevision"] == 13
    assert payloads[coronal_view.view_id]["mprPlane"] != payloads[sagittal_view.view_id]["mprPlane"]


def test_png_encoding_always_uses_low_compression_without_changing_format(monkeypatch) -> None:
    save_calls: list[dict[str, object]] = []

    def fake_save(self, output, **kwargs):
        del self
        save_calls.append(kwargs)
        output.write(b"png")

    monkeypatch.setattr(Image.Image, "save", fake_save)

    ViewerService._encode_image(Image.new("L", (1, 1)), "png", fast_preview=True)
    ViewerService._encode_image(Image.new("L", (1, 1)), "png", fast_preview=False)

    assert save_calls[0]["format"] == "PNG"
    assert save_calls[0]["compress_level"] == 1
    assert save_calls[1]["format"] == "PNG"
    assert save_calls[1]["compress_level"] == 1


def test_webp_encoding_uses_lossless_format(monkeypatch) -> None:
    save_calls: list[dict[str, object]] = []

    def fake_save(self, output, **kwargs):
        del self
        save_calls.append(kwargs)
        output.write(b"webp")

    monkeypatch.setattr(Image.Image, "save", fake_save)

    encoded = ViewerService._encode_image(Image.new("L", (1, 1)), "webp", fast_preview=True)

    assert encoded == b"webp"
    assert save_calls == [{"format": "WEBP", "lossless": True}]


def test_view_transform_payload_reports_clamped_zoom() -> None:
    view = ViewRecord(view_id="v", series_id="s", view_type="Stack")

    assert ZOOM_MIN == pytest.approx(0.1)
    assert ZOOM_MAX == pytest.approx(10.0)

    view.zoom = ZOOM_MAX * 4
    payload = ViewerService._build_view_transform_payload(view)
    assert payload.zoom == ZOOM_MAX

    view.zoom = ZOOM_MIN / 4
    payload = ViewerService._build_view_transform_payload(view)
    assert payload.zoom == ZOOM_MIN


def test_mpr_plane_cache_reuses_reslice_when_only_pan_zoom_changes(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, view = _build_axial_view(service, series, volume)
    reslice_calls = 0

    def fake_reslice_plane(*args, **kwargs):
        nonlocal reslice_calls
        del args, kwargs
        reslice_calls += 1
        return np.ones((6, 7), dtype=np.float32) * reslice_calls

    monkeypatch.setattr(viewer_service_module, "reslice_plane", fake_reslice_plane)

    first_plane, _, _ = service._extract_mpr_plane(view, volume)
    view.offset_x = 18
    view.offset_y = -7
    view.zoom = 1.2
    second_plane, _, _ = service._extract_mpr_plane(view, volume)

    assert reslice_calls == 1
    assert np.array_equal(first_plane, second_plane)


def _get_pose_context(service: ViewerService, view: ViewRecord, series, volume: np.ndarray):
    return service._build_mpr_pose_context(view, volume.shape, series=series)


def _get_plane(service: ViewerService, view: ViewRecord, series, volume: np.ndarray, viewport_key: str):
    return _get_pose_context(service, view, series, volume).poses[viewport_key]


def _screen_point_for_angle(
    service: ViewerService,
    view: ViewRecord,
    series,
    volume: np.ndarray,
    angle_rad: float,
    *,
    radius: float = 0.25,
) -> tuple[float, float]:
    pose_context = _get_pose_context(service, view, series, volume)
    active_viewport = service._resolve_mpr_viewport(view)
    active_plane = pose_context.poses[active_viewport]
    pixel_aspect_x, pixel_aspect_y = service._get_mpr_display_aspect_xy_from_pose(active_plane)
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=int(active_plane.output_shape[1]),
        image_height=int(active_plane.output_shape[0]),
        canvas_width=int(view.width or 0),
        canvas_height=int(view.height or 0),
        view=view,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )
    center_image_x, center_image_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )
    center_canvas = image_transform.matrix @ np.array([center_image_x, center_image_y, 1.0], dtype=np.float64)
    center_x = float(center_canvas[0]) / float(view.width or 1)
    center_y = float(center_canvas[1]) / float(view.height or 1)
    return (
        center_x + math.cos(angle_rad) * radius,
        center_y + math.sin(angle_rad) * radius,
    )


def _line_angles(service: ViewerService, view: ViewRecord, series, volume: np.ndarray) -> tuple[float, float]:
    pose_context = _get_pose_context(service, view, series, volume)
    return service._get_mpr_crosshair_line_angles_from_poses(
        pose_context.poses,
        service._resolve_mpr_viewport(view),
    )


def _visible_line_angles(service: ViewerService, view: ViewRecord, series, volume: np.ndarray) -> tuple[float, float]:
    pose_context = _get_pose_context(service, view, series, volume)
    return service._get_mpr_visible_crosshair_line_angles(
        view.view_group,
        pose_context.poses,
        service._resolve_mpr_viewport(view),
    )


def _crosshair_canvas_center(service: ViewerService, view: ViewRecord, series, volume: np.ndarray) -> tuple[float, float]:
    pose_context = _get_pose_context(service, view, series, volume)
    active_plane = pose_context.poses[service._resolve_mpr_viewport(view)]
    pixel_aspect_x, pixel_aspect_y = service._get_mpr_display_aspect_xy_from_pose(active_plane)
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=int(active_plane.output_shape[1]),
        image_height=int(active_plane.output_shape[0]),
        canvas_width=int(view.width or 0),
        canvas_height=int(view.height or 0),
        view=view,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )
    center_image_x, center_image_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )
    center_canvas = image_transform.matrix @ np.array([center_image_x, center_image_y, 1.0], dtype=np.float64)
    return (
        float(center_canvas[0]) / float(view.width or 1),
        float(center_canvas[1]) / float(view.height or 1),
    )


def _set_cursor_center(
    service: ViewerService,
    view: ViewRecord,
    series,
    volume: np.ndarray,
    center_ijk: tuple[float, float, float],
) -> None:
    pose_context = _get_pose_context(service, view, series, volume)
    next_cursor = replace(
        pose_context.cursor,
        center_world=ijk_to_world_point(pose_context.geometry, center_ijk),
    )
    service._sync_group_from_mpr_cursor(view.view_group, next_cursor, pose_context.geometry, volume.shape)


def _orientation_labels(
    service: ViewerService,
    view: ViewRecord,
    series,
    volume: np.ndarray,
) -> tuple[str | None, str | None, str | None, str | None]:
    pose_context = _get_pose_context(service, view, series, volume)
    viewport_key = service._resolve_mpr_viewport(view)
    plane_pose = pose_context.poses[viewport_key]
    overlay = service._build_mpr_orientation_overlay(
        view,
        viewport_key,
        service._plane_state_from_pose(plane_pose),
        plane_pose=plane_pose,
    )
    return overlay.top, overlay.right, overlay.bottom, overlay.left


def _undirected_angle_delta(first_angle: float, second_angle: float) -> float:
    delta = abs(float(second_angle) - float(first_angle)) % math.pi
    return min(delta, math.pi - delta)


def test_mpr_oblique_request_contract_no_longer_accepts_frontend_angle_delta() -> None:
    assert "delta_angle_rad" not in ViewOperationRequest.model_fields
    assert "angle_rad" not in ViewOperationRequest.model_fields


def test_mpr_crosshair_mode_defaults_to_orthogonal_and_keeps_lines_perpendicular(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)

    assert group.mpr_crosshair_mode == "orthogonal"
    start_horizontal_angle, _ = _visible_line_angles(service, view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, view, series, volume, start_horizontal_angle + 0.3)

    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=start_x,
            y=start_y,
        ),
    )
    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=move_x,
            y=move_y,
        ),
    )

    horizontal_angle, vertical_angle = _visible_line_angles(service, view, series, volume)
    assert _undirected_angle_delta(horizontal_angle, vertical_angle) == pytest.approx(math.pi / 2.0)


def test_mpr_double_oblique_drag_updates_only_dragged_target_plane(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)

    assert service._handle_mpr_crosshair_mode(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprCrosshairMode",
            mprCrosshairMode="double-oblique",
        ),
    )
    before_planes = {
        viewport_key: _get_plane(service, view, series, volume, viewport_key)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
    }

    start_horizontal_angle, _ = _visible_line_angles(service, view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, view, series, volume, start_horizontal_angle + 0.35)
    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=start_x,
            y=start_y,
        ),
    )
    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=move_x,
            y=move_y,
        ),
    )

    after_coronal = _get_plane(service, view, series, volume, MPR_VIEWPORT_CORONAL)
    after_sagittal = _get_plane(service, view, series, volume, MPR_VIEWPORT_SAGITTAL)
    assert not np.allclose(after_coronal.normal_world, before_planes[MPR_VIEWPORT_CORONAL].normal_world)
    assert np.allclose(after_sagittal.normal_world, before_planes[MPR_VIEWPORT_SAGITTAL].normal_world)
    horizontal_angle, vertical_angle = _visible_line_angles(service, view, series, volume)
    assert abs(_undirected_angle_delta(horizontal_angle, vertical_angle) - math.pi / 2.0) > 0.05
    assert group.mpr_crosshair_mode == "double-oblique"


def test_mpr_double_oblique_relock_reorthogonalizes_planes(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)

    service._handle_mpr_crosshair_mode(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprCrosshairMode",
            mprCrosshairMode="double-oblique",
        ),
    )
    start_horizontal_angle, _ = _visible_line_angles(service, view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, view, series, volume, start_horizontal_angle + 0.3)
    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=start_x,
            y=start_y,
        ),
    )
    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=move_x,
            y=move_y,
        ),
    )

    assert service._handle_mpr_crosshair_mode(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprCrosshairMode",
            mprCrosshairMode="orthogonal",
        ),
    )
    pose_context = _get_pose_context(service, view, series, volume)
    normals = [pose_context.poses[viewport_key].normal_world for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)]
    assert abs(float(np.dot(normals[0], normals[1]))) < 1e-6
    assert abs(float(np.dot(normals[0], normals[2]))) < 1e-6
    assert abs(float(np.dot(normals[1], normals[2]))) < 1e-6
    assert group.mpr_crosshair_mode == "orthogonal"
    assert group.mpr_independent_plane_normals == {}


def test_mpr_reset_restores_orthogonal_crosshair_mode(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)
    group.mpr_revision = 42

    service._handle_mpr_crosshair_mode(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprCrosshairMode",
            mprCrosshairMode="double-oblique",
        ),
    )
    assert group.mpr_crosshair_mode == "double-oblique"
    assert group.mpr_independent_plane_normals

    service._reset_mpr_group_geometry(group, volume.shape, series=series)
    assert group.mpr_crosshair_mode == "orthogonal"
    assert group.mpr_independent_plane_normals == {}
    assert group.mpr_revision == 42


def test_mpr_state_sync_copies_double_oblique_mode_and_independent_normals(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    source_group, source_view = _build_axial_view(service, series, volume)
    target_group = ViewGroupRecord(group_id="g-target", group_type="MPR", series_id=series.series_id)
    target_view = ViewRecord(view_id="v-target-ax", series_id=series.series_id, view_type="AX", view_group=target_group)
    target_view.width = 240
    target_view.height = 240
    service._reset_mpr_group_geometry(target_group, volume.shape, series=series)

    def get_view(view_id: str) -> ViewRecord:
        if view_id == source_view.view_id:
            return source_view
        if view_id == target_view.view_id:
            return target_view
        raise KeyError(view_id)

    monkeypatch.setattr(view_registry, "get", get_view)
    assert service._handle_mpr_crosshair_mode(
        source_view,
        ViewOperationRequest(
            viewId=source_view.view_id,
            opType="mprCrosshairMode",
            mprCrosshairMode="double-oblique",
        ),
    )
    source_group.mpr_independent_plane_normals[MPR_VIEWPORT_CORONAL] = (0.0, 0.70710678, 0.70710678)

    assert target_group.mpr_crosshair_mode == "orthogonal"
    assert service._sync_mpr_state_from_source_view(target_view, source_view.view_id)

    assert target_group.mpr_crosshair_mode == "double-oblique"
    assert target_group.mpr_independent_plane_normals == source_group.mpr_independent_plane_normals
    assert target_group.mpr_independent_plane_normals is not source_group.mpr_independent_plane_normals


@pytest.mark.parametrize(
    ("line", "dragged_target_viewport", "paired_target_viewport"),
    [
        ("horizontal", MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL),
        ("vertical", MPR_VIEWPORT_SAGITTAL, MPR_VIEWPORT_CORONAL),
    ],
)
def test_mpr_oblique_drag_uses_backend_pointer_position_to_rotate_paired_target_planes(
    monkeypatch,
    line: str,
    dragged_target_viewport: str,
    paired_target_viewport: str,
) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)

    before_planes = {
        viewport_key: _get_plane(service, view, series, volume, viewport_key)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
    }
    start_horizontal_angle, start_vertical_angle = _line_angles(service, view, series, volume)
    start_angle = start_horizontal_angle if line == "horizontal" else start_vertical_angle
    target_angle = start_angle + 0.35
    start_x, start_y = _screen_point_for_angle(service, view, series, volume, start_angle)
    move_x, move_y = _screen_point_for_angle(service, view, series, volume, target_angle)

    start_result = service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="start",
            line=line,
            x=start_x,
            y=start_y,
        ),
    )
    assert start_result is False
    assert group.rotation_drag is not None

    move_result = service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="move",
            line=line,
            x=move_x,
            y=move_y,
        ),
    )

    assert move_result is True
    after_planes = {
        viewport_key: _get_plane(service, view, series, volume, viewport_key)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
    }
    assert np.allclose(after_planes[MPR_VIEWPORT_AXIAL].normal_world, before_planes[MPR_VIEWPORT_AXIAL].normal_world)
    assert not np.allclose(after_planes[dragged_target_viewport].normal_world, before_planes[dragged_target_viewport].normal_world)
    assert not np.allclose(after_planes[paired_target_viewport].normal_world, before_planes[paired_target_viewport].normal_world)
    assert after_planes[dragged_target_viewport].is_oblique is True
    assert after_planes[paired_target_viewport].is_oblique is True
    after_horizontal_angle, after_vertical_angle = _line_angles(service, view, series, volume)
    assert math.isclose(
        _undirected_angle_delta(after_horizontal_angle, after_vertical_angle),
        math.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


@pytest.mark.parametrize("line", ["horizontal", "vertical"])
def test_mpr_oblique_drag_aligns_dragged_line_to_backend_pointer_angle(monkeypatch, line: str) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, view = _build_axial_view(service, series, volume)

    start_horizontal_angle, start_vertical_angle = _line_angles(service, view, series, volume)
    start_angle = start_horizontal_angle if line == "horizontal" else start_vertical_angle
    target_angle = start_angle + 0.42
    start_x, start_y = _screen_point_for_angle(service, view, series, volume, start_angle)
    move_x, move_y = _screen_point_for_angle(service, view, series, volume, target_angle)

    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="start",
            line=line,
            x=start_x,
            y=start_y,
        ),
    )
    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="move",
            line=line,
            x=move_x,
            y=move_y,
        ),
    )

    after_horizontal_angle, after_vertical_angle = _line_angles(service, view, series, volume)
    after_angle = after_horizontal_angle if line == "horizontal" else after_vertical_angle
    assert math.isfinite(after_angle)
    assert math.isclose(
        service._normalize_screen_half_turn_angle(after_angle),
        service._normalize_screen_half_turn_angle(target_angle),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    paired_angle = after_vertical_angle if line == "horizontal" else after_horizontal_angle
    assert math.isclose(
        _undirected_angle_delta(after_angle, paired_angle),
        math.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_mpr_oblique_second_rotation_keeps_active_plane_fixed_and_updates_other_views(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)

    first_start_angle, _ = _line_angles(service, axial_view, series, volume)
    first_start_x, first_start_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle)
    first_move_x, first_move_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle + 0.35)
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=first_start_x,
            y=first_start_y,
        ),
    )
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=first_move_x,
            y=first_move_y,
        ),
    )

    coronal_view = ViewRecord(
        view_id="v-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=axial_view.view_group,
    )
    coronal_view.width = axial_view.width
    coronal_view.height = axial_view.height
    before_second = {
        viewport_key: _get_plane(service, coronal_view, series, volume, viewport_key)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
    }

    _, second_start_angle = _line_angles(service, coronal_view, series, volume)
    second_start_x, second_start_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle)
    second_move_x, second_move_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle + 0.25)
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="vertical",
            x=second_start_x,
            y=second_start_y,
        ),
    )
    move_result = service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="vertical",
            x=second_move_x,
            y=second_move_y,
        ),
    )

    after_second = {
        viewport_key: _get_plane(service, coronal_view, series, volume, viewport_key)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
    }
    assert move_result is True
    assert np.allclose(after_second[MPR_VIEWPORT_CORONAL].normal_world, before_second[MPR_VIEWPORT_CORONAL].normal_world)
    assert not np.allclose(after_second[MPR_VIEWPORT_AXIAL].normal_world, before_second[MPR_VIEWPORT_AXIAL].normal_world)
    assert not np.allclose(after_second[MPR_VIEWPORT_SAGITTAL].normal_world, before_second[MPR_VIEWPORT_SAGITTAL].normal_world)
    after_horizontal_angle, after_vertical_angle = _line_angles(service, coronal_view, series, volume)
    assert math.isclose(
        _undirected_angle_delta(after_horizontal_angle, after_vertical_angle),
        math.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_mpr_oblique_second_view_rotation_preserves_first_view_crosshair_position(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)
    _set_cursor_center(service, axial_view, series, volume, (2.0, 1.0, 5.0))

    first_start_angle, _ = _line_angles(service, axial_view, series, volume)
    first_start_x, first_start_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle)
    first_move_x, first_move_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle + 0.35)
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=first_start_x,
            y=first_start_y,
        ),
    )
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=first_move_x,
            y=first_move_y,
        ),
    )
    axial_center_after_first = _crosshair_canvas_center(service, axial_view, series, volume)

    coronal_view = ViewRecord(
        view_id="v-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=axial_view.view_group,
    )
    coronal_view.width = axial_view.width
    coronal_view.height = axial_view.height
    _, second_start_angle = _line_angles(service, coronal_view, series, volume)
    second_start_x, second_start_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle)
    second_move_x, second_move_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle + 0.25)
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="vertical",
            x=second_start_x,
            y=second_start_y,
        ),
    )
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="vertical",
            x=second_move_x,
            y=second_move_y,
        ),
    )

    axial_center_after_second = _crosshair_canvas_center(service, axial_view, series, volume)
    assert axial_center_after_second[0] == pytest.approx(axial_center_after_first[0])
    assert axial_center_after_second[1] == pytest.approx(axial_center_after_first[1])


def test_mpr_oblique_target_view_crosshair_angles_stay_fixed_while_images_update(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(
        view_id="v-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=axial_view.view_group,
    )
    sagittal_view = ViewRecord(
        view_id="v-sag",
        series_id=series.series_id,
        view_type="SAG",
        view_group=axial_view.view_group,
    )
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    coronal_angles_before = _visible_line_angles(service, coronal_view, series, volume)
    sagittal_angles_before = _visible_line_angles(service, sagittal_view, series, volume)
    coronal_plane_before = _get_plane(service, coronal_view, series, volume, MPR_VIEWPORT_CORONAL)
    sagittal_plane_before = _get_plane(service, sagittal_view, series, volume, MPR_VIEWPORT_SAGITTAL)

    start_horizontal_angle, _ = _visible_line_angles(service, axial_view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle + 0.35)
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=start_x,
            y=start_y,
        ),
    )
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=move_x,
            y=move_y,
        ),
    )

    coronal_plane_after = _get_plane(service, coronal_view, series, volume, MPR_VIEWPORT_CORONAL)
    sagittal_plane_after = _get_plane(service, sagittal_view, series, volume, MPR_VIEWPORT_SAGITTAL)
    assert not np.allclose(coronal_plane_after.normal_world, coronal_plane_before.normal_world)
    assert not np.allclose(sagittal_plane_after.normal_world, sagittal_plane_before.normal_world)
    assert _visible_line_angles(service, coronal_view, series, volume) == pytest.approx(coronal_angles_before)
    assert _visible_line_angles(service, sagittal_view, series, volume) == pytest.approx(sagittal_angles_before)


def test_mpr_oblique_axial_rotation_preserves_left_handed_coronal_orientation(monkeypatch) -> None:
    service, series, volume = _build_service_with_left_handed_patient_geometry(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(
        view_id="v-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=axial_view.view_group,
    )
    coronal_view.width = axial_view.width
    coronal_view.height = axial_view.height

    assert _orientation_labels(service, coronal_view, series, volume) == ("S", "L", "I", "R")

    start_horizontal_angle, _ = _visible_line_angles(service, axial_view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle + 0.05)
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=start_x,
            y=start_y,
        ),
    )
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=move_x,
            y=move_y,
        ),
    )

    assert _orientation_labels(service, coronal_view, series, volume) == ("S", "L", "I", "R")


def test_mpr_oblique_second_view_rotation_preserves_left_handed_axial_orientation(monkeypatch) -> None:
    service, series, volume = _build_service_with_left_handed_patient_geometry(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(
        view_id="v-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=axial_view.view_group,
    )
    coronal_view.width = axial_view.width
    coronal_view.height = axial_view.height

    first_start_angle, _ = _visible_line_angles(service, axial_view, series, volume)
    first_start_x, first_start_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle)
    first_move_x, first_move_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle + 0.25)
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=first_start_x,
            y=first_start_y,
        ),
    )
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=first_move_x,
            y=first_move_y,
        ),
    )
    axial_labels_after_first = _orientation_labels(service, axial_view, series, volume)

    _, second_start_angle = _visible_line_angles(service, coronal_view, series, volume)
    second_start_x, second_start_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle)
    second_move_x, second_move_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle + 0.2)
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="vertical",
            x=second_start_x,
            y=second_start_y,
        ),
    )
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="vertical",
            x=second_move_x,
            y=second_move_y,
        ),
    )

    assert axial_labels_after_first == ("A", "L", "P", "R")
    assert _orientation_labels(service, axial_view, series, volume) == ("A", "L", "P", "R")


def test_mpr_oblique_second_view_rotation_preserves_first_view_crosshair_angle(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)

    first_start_angle, _ = _visible_line_angles(service, axial_view, series, volume)
    first_start_x, first_start_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle)
    first_move_x, first_move_y = _screen_point_for_angle(service, axial_view, series, volume, first_start_angle + 0.35)
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="horizontal",
            x=first_start_x,
            y=first_start_y,
        ),
    )
    service._handle_mpr_oblique(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="horizontal",
            x=first_move_x,
            y=first_move_y,
        ),
    )
    axial_angles_after_first = _visible_line_angles(service, axial_view, series, volume)

    coronal_view = ViewRecord(
        view_id="v-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=axial_view.view_group,
    )
    coronal_view.width = axial_view.width
    coronal_view.height = axial_view.height
    _, second_start_angle = _visible_line_angles(service, coronal_view, series, volume)
    second_start_x, second_start_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle)
    second_move_x, second_move_y = _screen_point_for_angle(service, coronal_view, series, volume, second_start_angle + 0.25)
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="start",
            line="vertical",
            x=second_start_x,
            y=second_start_y,
        ),
    )
    service._handle_mpr_oblique(
        coronal_view,
        ViewOperationRequest(
            viewId=coronal_view.view_id,
            opType="mprOblique",
            actionType="move",
            line="vertical",
            x=second_move_x,
            y=second_move_y,
        ),
    )

    assert _visible_line_angles(service, axial_view, series, volume) == pytest.approx(axial_angles_after_first)


def test_mpr_oblique_move_broadcasts_reference_full_resolution_preview(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    start_horizontal_angle, _ = _line_angles(service, axial_view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle + 0.25)

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="start",
                line="horizontal",
                x=start_x,
                y=start_y,
            )
        )
        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="move",
                line="horizontal",
                x=move_x,
                y=move_y,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert set(outcome.broadcast_view_ids) == {coronal_view.view_id, sagittal_view.view_id}
    assert set(outcome.mpr_state_view_ids) == {coronal_view.view_id, sagittal_view.view_id}
    assert outcome.broadcast_image_format == "png"
    assert outcome.broadcast_fast_preview is True
    assert outcome.broadcast_fast_preview_full_resolution is False
    assert outcome.broadcast_metadata_mode == "mpr-crosshair-preview"


def test_mpr_crosshair_move_uses_low_latency_preview_broadcast(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="crosshair",
                actionType="start",
                x=0.5,
                y=0.5,
            )
        )
        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="crosshair",
                actionType="move",
                x=0.55,
                y=0.55,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert set(outcome.broadcast_view_ids) == {coronal_view.view_id, sagittal_view.view_id}
    assert set(outcome.mpr_state_view_ids) == {coronal_view.view_id, sagittal_view.view_id}
    assert outcome.broadcast_image_format == "png"
    assert outcome.broadcast_fast_preview is True
    assert outcome.broadcast_fast_preview_full_resolution is False
    assert outcome.broadcast_metadata_mode == "mpr-crosshair-preview"


def test_mpr_window_keeps_geometry_revision_after_crosshair_move(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="crosshair",
                actionType="start",
                x=0.5,
                y=0.5,
            )
        )
        crosshair_outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="crosshair",
                actionType="move",
                x=0.55,
                y=0.55,
            )
        )
        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="window",
                actionType="start",
            )
        )
        window_outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="window",
                actionType="move",
                x=12,
                y=-8,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert crosshair_outcome.mpr_revision == 1
    assert crosshair_outcome.broadcast_metadata_mode == "mpr-crosshair-preview"
    assert window_outcome.mpr_revision == 1
    assert window_outcome.broadcast_metadata_mode == "mpr-pixel-preview"
    assert group.mpr_revision == 1


def test_stack_window_and_zoom_moves_use_png_render(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    view = ViewRecord(view_id="stack-view", series_id=series.series_id, view_type="Stack")
    view.window_width = 400.0
    view.window_center = 40.0
    view.zoom = 1.0

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[view.view_id] = view
        monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)

        service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="window", actionType="start")
        )
        window_move = service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="window", actionType="move", x=12, y=-8)
        )
        window_end = service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="window", actionType="end", x=12, y=-8)
        )

        service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="zoom", actionType="start", x=0, y=0)
        )
        zoom_move = service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="zoom", actionType="move", x=0, y=-10)
        )
        zoom_end = service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="zoom", actionType="end", x=0, y=-10)
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert window_move.primary_result is None
    assert window_move.deferred_view_ids == (view.view_id,)
    assert window_move.deferred_image_format == "png"
    assert window_move.deferred_fast_preview is True
    assert window_move.deferred_metadata_mode == "stack-pixel-preview"
    assert view.window_width == pytest.approx(412.0)
    assert view.window_center == pytest.approx(48.0)
    assert window_end.primary_result is None
    assert window_end.deferred_view_ids == (view.view_id,)
    assert window_end.deferred_image_format == "png"
    assert window_end.deferred_fast_preview is False
    assert zoom_move.primary_result is None
    assert zoom_move.deferred_view_ids == (view.view_id,)
    assert zoom_move.deferred_image_format == "png"
    assert zoom_move.deferred_fast_preview is True
    assert zoom_move.deferred_metadata_mode == "stack-geometry-preview"
    assert zoom_end.primary_result is None
    assert zoom_end.deferred_view_ids == (view.view_id,)
    assert zoom_end.deferred_image_format == "png"
    assert zoom_end.deferred_fast_preview is False


def test_window_drag_sensitivity_scales_with_current_width(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    view = ViewRecord(view_id="stack-view", series_id=series.series_id, view_type="Stack")
    view.window_width = 8.0
    view.window_center = 4.0

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[view.view_id] = view
        monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)

        service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="window", actionType="start")
        )
        service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="window", actionType="move", x=100, y=-50)
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert view.window_width == pytest.approx(10.0)
    assert view.window_center == pytest.approx(5.0)


def test_stack_pan_move_uses_png_render(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    view = ViewRecord(view_id="stack-view", series_id=series.series_id, view_type="Stack")

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[view.view_id] = view
        monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)

        service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="pan", actionType="start", x=0, y=0)
        )
        pan_move = service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="pan", actionType="move", x=12, y=-8)
        )
        pan_end = service.handle_view_operation(
            ViewOperationRequest(viewId=view.view_id, opType="pan", actionType="end", x=12, y=-8)
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert pan_move.primary_result is None
    assert pan_move.deferred_view_ids == (view.view_id,)
    assert pan_move.deferred_image_format == "png"
    assert pan_move.deferred_fast_preview is True
    assert pan_move.deferred_metadata_mode == "stack-geometry-preview"
    assert pan_end.primary_result is None
    assert pan_end.deferred_view_ids == (view.view_id,)
    assert pan_end.deferred_image_format == "png"
    assert pan_end.deferred_fast_preview is False


def test_mpr_pan_move_schedules_deferred_full_resolution_preview(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[axial_view.view_id] = axial_view

        service.handle_view_operation(
            ViewOperationRequest(viewId=axial_view.view_id, opType="pan", actionType="start", x=0, y=0)
        )
        move_outcome = service.handle_view_operation(
            ViewOperationRequest(viewId=axial_view.view_id, opType="pan", actionType="move", x=18, y=-7)
        )
        end_outcome = service.handle_view_operation(
            ViewOperationRequest(viewId=axial_view.view_id, opType="pan", actionType="end", x=0, y=0)
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert axial_view.offset_x == 18
    assert axial_view.offset_y == -7
    assert move_outcome.primary_result is None
    assert move_outcome.broadcast_view_ids == ()
    assert move_outcome.deferred_view_ids == (axial_view.view_id,)
    assert move_outcome.deferred_image_format == "png"
    assert move_outcome.deferred_fast_preview is True
    assert move_outcome.deferred_fast_preview_full_resolution is True
    assert move_outcome.deferred_metadata_mode == "mpr-pan-zoom-preview"
    assert end_outcome.primary_result is None
    assert end_outcome.deferred_view_ids == (axial_view.view_id,)
    assert end_outcome.deferred_image_format == "png"
    assert end_outcome.deferred_fast_preview is False


def test_mpr_zoom_move_schedules_deferred_full_resolution_preview(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)
    axial_view.zoom = 1.0

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[axial_view.view_id] = axial_view

        service.handle_view_operation(
            ViewOperationRequest(viewId=axial_view.view_id, opType="zoom", actionType="start", x=0, y=0)
        )
        move_outcome = service.handle_view_operation(
            ViewOperationRequest(viewId=axial_view.view_id, opType="zoom", actionType="move", x=0, y=-10)
        )
        end_outcome = service.handle_view_operation(
            ViewOperationRequest(viewId=axial_view.view_id, opType="zoom", actionType="end", x=0, y=0)
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert axial_view.zoom == pytest.approx(1.1)
    assert move_outcome.primary_result is None
    assert move_outcome.broadcast_view_ids == ()
    assert move_outcome.deferred_view_ids == (axial_view.view_id,)
    assert move_outcome.deferred_image_format == "png"
    assert move_outcome.deferred_fast_preview is True
    assert move_outcome.deferred_fast_preview_full_resolution is True
    assert end_outcome.primary_result is None
    assert end_outcome.deferred_view_ids == (axial_view.view_id,)
    assert end_outcome.deferred_image_format == "png"
    assert end_outcome.deferred_fast_preview is False


def test_mpr_mip_config_move_renders_full_resolution_preview_and_end_finalizes(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (axial_view, coronal_view, sagittal_view):
        candidate_view.width = 240
        candidate_view.height = 240

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update({
            axial_view.view_id: axial_view,
            coronal_view.view_id: coronal_view,
            sagittal_view.view_id: sagittal_view,
        })

        move_outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprMipConfig",
                actionType="move",
                mprMipConfig={
                    "enabled": True,
                    "algorithm": "maximum",
                    "viewports": {
                        "mpr-ax": {"thickness": 20},
                        "mpr-cor": {"thickness": 16},
                        "mpr-sag": {"thickness": 0},
                    },
                },
            )
        )
        end_outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprMipConfig",
                actionType="end",
                mprMipConfig={
                    "enabled": True,
                    "algorithm": "maximum",
                    "viewports": {
                        "mpr-ax": {"thickness": 22},
                        "mpr-cor": {"thickness": 16},
                        "mpr-sag": {"thickness": 0},
                    },
                },
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert group.mpr_mip.enabled is True
    assert group.mpr_mip.viewports["mpr-ax"].thickness == 22
    assert group.mpr_mip.viewports["mpr-sag"].thickness == 0
    assert move_outcome.broadcast_view_ids == ("v-ax", "v-cor", "v-sag")
    assert move_outcome.broadcast_image_format == "png"
    assert move_outcome.broadcast_fast_preview is True
    assert move_outcome.broadcast_fast_preview_full_resolution is True
    assert end_outcome.broadcast_view_ids == ("v-ax", "v-cor", "v-sag")
    assert end_outcome.broadcast_image_format == "png"
    assert end_outcome.broadcast_fast_preview is False
    assert move_outcome.mpr_revision == 1
    assert end_outcome.mpr_revision == 2


def test_mpr_crosshair_info_includes_mip_slab_offsets(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    group.mpr_mip.enabled = True
    group.mpr_mip.viewports["mpr-cor"] = MprMipViewportState(thickness=0)
    group.mpr_mip.viewports["mpr-sag"] = MprMipViewportState(thickness=2)

    plane = _get_plane(service, axial_view, series, volume, MPR_VIEWPORT_AXIAL)
    pixel_aspect_x, pixel_aspect_y = service._get_mpr_display_aspect_xy_from_pose(plane)
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=int(plane.output_shape[1]),
        image_height=int(plane.output_shape[0]),
        canvas_width=int(axial_view.width or 0),
        canvas_height=int(axial_view.height or 0),
        view=axial_view,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )

    overlay = service._build_mpr_crosshair_overlay(
        axial_view,
        volume.shape,
        plane.output_shape,
        image_transform,
    )
    crosshair_info = service._build_mpr_crosshair_info(overlay)

    assert crosshair_info is not None
    assert abs(crosshair_info.horizontal_slab_offset_x or 0.0) < 1e-6
    assert abs(crosshair_info.horizontal_slab_offset_y or 0.0) > 0.001
    assert abs(crosshair_info.vertical_slab_offset_x or 0.0) > 0.001
    assert abs(crosshair_info.vertical_slab_offset_y or 0.0) < 1e-6


def test_mpr_crosshair_mip_slab_offsets_project_in_double_oblique(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    group.mpr_crosshair_mode = "double-oblique"
    group.mpr_independent_plane_normals["mpr-cor"] = (0.6, 0.8, 0.0)
    group.mpr_mip.enabled = True
    group.mpr_mip.viewports["mpr-cor"] = MprMipViewportState(thickness=8)

    plane = _get_plane(service, axial_view, series, volume, MPR_VIEWPORT_AXIAL)
    pixel_aspect_x, pixel_aspect_y = service._get_mpr_display_aspect_xy_from_pose(plane)
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=int(plane.output_shape[1]),
        image_height=int(plane.output_shape[0]),
        canvas_width=int(axial_view.width or 0),
        canvas_height=int(axial_view.height or 0),
        view=axial_view,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )

    overlay = service._build_mpr_crosshair_overlay(
        axial_view,
        volume.shape,
        plane.output_shape,
        image_transform,
    )
    crosshair_info = service._build_mpr_crosshair_info(overlay)

    assert crosshair_info is not None
    assert abs(crosshair_info.horizontal_slab_offset_x or 0.0) < 1e-6
    # The target normal projection length is 0.8, so the active-plane offset is
    # 8 / 2 / 0.8 = 5mm before canvas normalization.
    assert abs(crosshair_info.horizontal_slab_offset_y or 0.0) == pytest.approx(5 / 240, rel=0.1)


def test_mpr_window_move_uses_full_resolution_preview_broadcast(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="window",
                actionType="start",
            )
        )
        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="window",
                actionType="move",
                x=12,
                y=-8,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert set(outcome.broadcast_view_ids) == {axial_view.view_id, coronal_view.view_id, sagittal_view.view_id}
    assert outcome.broadcast_image_format == "png"
    assert outcome.broadcast_fast_preview is True
    assert outcome.broadcast_fast_preview_full_resolution is True


def test_mpr_fast_preview_includes_backend_corner_state(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    _, axial_view = _build_axial_view(service, series, volume)
    axial_view.is_initialized = True
    axial_view.window_width = 512.0
    axial_view.window_center = 42.0
    axial_view.zoom = 1.5
    axial_view.offset_x = 18.0
    axial_view.offset_y = -7.0

    result = service._render_mpr_view(
        axial_view,
        image_format="jpeg",
        fast_preview=True,
        fast_preview_full_resolution=False,
    )

    assert result.meta.corner_info is not None
    assert result.meta.scale_bar is not None
    assert result.meta.scale_bar.label == "10 cm"
    assert result.meta.orientation is not None
    assert "W: 512 L: 42" in result.meta.corner_info.bottom_left
    assert "Zoom:1.5x" in result.meta.corner_info.bottom_right
    assert "X:18 Y:-7" in result.meta.corner_info.bottom_right


def test_mpr_crosshair_end_broadcasts_full_quality_to_all_mpr_views(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="crosshair",
                actionType="start",
                x=0.5,
                y=0.5,
            )
        )
        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="crosshair",
                actionType="end",
                x=0.55,
                y=0.55,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert set(outcome.broadcast_view_ids) == {axial_view.view_id, coronal_view.view_id, sagittal_view.view_id}
    assert outcome.broadcast_image_format == "png"
    assert outcome.broadcast_fast_preview is False


def test_mpr_oblique_end_broadcasts_full_quality_to_active_and_reference_views(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    start_horizontal_angle, _ = _visible_line_angles(service, axial_view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle)
    end_x, end_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle + 0.25)

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="start",
                line="horizontal",
                x=start_x,
                y=start_y,
            )
        )
        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="end",
                line="horizontal",
                x=end_x,
                y=end_y,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert set(outcome.broadcast_view_ids) == {axial_view.view_id, coronal_view.view_id, sagittal_view.view_id}
    assert outcome.broadcast_image_format == "png"
    assert outcome.broadcast_fast_preview is False


def test_mpr_double_oblique_move_broadcasts_target_and_end_syncs_active_view(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, axial_view = _build_axial_view(service, series, volume)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (coronal_view, sagittal_view):
        candidate_view.width = axial_view.width
        candidate_view.height = axial_view.height

    assert service._handle_mpr_crosshair_mode(
        axial_view,
        ViewOperationRequest(
            viewId=axial_view.view_id,
            opType="mprCrosshairMode",
            mprCrosshairMode="double-oblique",
        ),
    )
    start_horizontal_angle, _ = _visible_line_angles(service, axial_view, series, volume)
    start_x, start_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle)
    move_x, move_y = _screen_point_for_angle(service, axial_view, series, volume, start_horizontal_angle + 0.25)

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(
            {
                axial_view.view_id: axial_view,
                coronal_view.view_id: coronal_view,
                sagittal_view.view_id: sagittal_view,
            }
        )

        service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="start",
                line="horizontal",
                x=start_x,
                y=start_y,
            )
        )
        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="move",
                line="horizontal",
                x=move_x,
                y=move_y,
            )
        )
        end_outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprOblique",
                actionType="end",
                line="horizontal",
                x=move_x,
                y=move_y,
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert outcome.broadcast_view_ids == (coronal_view.view_id,)
    assert outcome.mpr_state_view_ids == (coronal_view.view_id,)
    assert outcome.broadcast_image_format == "png"
    assert outcome.broadcast_fast_preview is True
    assert outcome.broadcast_fast_preview_full_resolution is False
    assert outcome.broadcast_metadata_mode == "mpr-crosshair-preview"
    assert set(end_outcome.broadcast_view_ids) == {axial_view.view_id, coronal_view.view_id}
    assert end_outcome.broadcast_image_format == "png"
    assert end_outcome.broadcast_fast_preview is False


def test_mpr_oblique_end_applies_final_pointer_position_and_clears_drag(monkeypatch) -> None:
    service, series, volume = _build_service_with_stubbed_series(monkeypatch)
    group, view = _build_axial_view(service, series, volume)

    _, start_vertical_angle = _line_angles(service, view, series, volume)
    target_vertical_angle = start_vertical_angle + 0.2
    start_x, start_y = _screen_point_for_angle(service, view, series, volume, start_vertical_angle)
    end_x, end_y = _screen_point_for_angle(service, view, series, volume, target_vertical_angle)

    service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="start",
            line="vertical",
            x=start_x,
            y=start_y,
        ),
    )
    assert isinstance(group.rotation_drag, MprRotationDragRecord)

    end_result = service._handle_mpr_oblique(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="mprOblique",
            actionType="end",
            line="vertical",
            x=end_x,
            y=end_y,
        ),
    )

    assert end_result is True
    assert group.rotation_drag is None
    _, after_vertical_angle = _line_angles(service, view, series, volume)
    assert math.isclose(
        service._normalize_screen_half_turn_angle(after_vertical_angle),
        service._normalize_screen_half_turn_angle(target_vertical_angle),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
