from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.models.viewer import MprFrameState, ViewGroupRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.render_layers.render_context import CornerInfoOverlay, MprCrosshairOverlay
from app.services import viewer_service as viewer_service_module
from app.services.mpr import (
    axis_angle_rotation_matrix,
    build_geometry_from_patient_transform,
    build_identity_geometry,
    create_default_cursor,
    cursor_to_legacy_frame,
    derive_plane_pose,
    ijk_to_world_point,
    legacy_frame_to_cursor,
    reslice_plane,
    world_to_ijk_point,
)
from app.services.mpr_geometry import VolumePatientTransform
from app.services.viewer_service import ViewerService


def test_legacy_frame_round_trips_through_cursor_with_identity_geometry() -> None:
    geometry = build_identity_geometry((9, 11, 13))
    frame = MprFrameState(
        center=(4.0, 5.0, 6.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 0.0, 1.0),
        axis_col=(0.0, -1.0, 0.0),
    )

    cursor = legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)
    rebuilt = cursor_to_legacy_frame(cursor, geometry)

    assert np.allclose(rebuilt.center, frame.center, atol=1e-6)
    assert np.allclose(rebuilt.axis_slice, frame.axis_slice, atol=1e-6)
    assert np.allclose(rebuilt.axis_row, frame.axis_row, atol=1e-6)
    assert np.allclose(rebuilt.axis_col, frame.axis_col, atol=1e-6)


def test_cursor_to_legacy_frame_preserves_independent_orientation_columns() -> None:
    geometry = build_identity_geometry((9, 11, 13))
    cursor = replace(
        create_default_cursor(geometry),
        orientation_world=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 0.8660254037844386],
            ],
            dtype=np.float64,
        ),
    )

    frame = cursor_to_legacy_frame(cursor, geometry)

    assert np.allclose(frame.axis_slice, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(frame.axis_row, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(frame.axis_col, [0.0, 0.5, 0.8660254037844386], atol=1e-6)


def test_derive_plane_pose_matches_legacy_default_viewport_conventions() -> None:
    geometry = build_identity_geometry((5, 6, 7))
    frame = MprFrameState(
        center=(2.0, 3.0, 4.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )
    cursor = legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)

    axial = derive_plane_pose(cursor, "mpr-ax", geometry)
    coronal = derive_plane_pose(cursor, "mpr-cor", geometry)
    sagittal = derive_plane_pose(cursor, "mpr-sag", geometry)

    assert np.allclose(axial.row_world, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(axial.col_world, [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(axial.normal_world, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(coronal.row_world, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(coronal.col_world, [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(coronal.normal_world, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(sagittal.row_world, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(sagittal.col_world, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(sagittal.normal_world, [0.0, 0.0, 1.0], atol=1e-6)


def test_reslice_plane_matches_legacy_orthogonal_default_planes() -> None:
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    geometry = build_identity_geometry(volume.shape)
    frame = MprFrameState(
        center=(2.0, 3.0, 4.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )
    cursor = legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)

    axial = reslice_plane(volume, geometry, derive_plane_pose(cursor, "mpr-ax", geometry), mip=None)
    coronal = reslice_plane(volume, geometry, derive_plane_pose(cursor, "mpr-cor", geometry), mip=None)
    sagittal = reslice_plane(volume, geometry, derive_plane_pose(cursor, "mpr-sag", geometry), mip=None)

    assert np.allclose(axial, volume[2, :, :], atol=1e-6)
    assert np.allclose(coronal, np.flipud(volume[:, 3, :]), atol=1e-6)
    assert np.allclose(sagittal, np.flipud(volume[:, :, 4]), atol=1e-6)


def test_mpr_fast_preview_extracts_lower_resolution_plane(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.zeros((32, 320, 300), dtype=np.float32)
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id=series.series_id)
    view = ViewRecord(
        view_id="v",
        series_id=series.series_id,
        view_type="MPR",
        view_group=group,
        width=300,
        height=300,
        is_initialized=True,
    )
    service._reset_mpr_group_geometry(group, volume.shape, series=series)
    captured_shapes: list[tuple[int, int] | None] = []
    captured_interpolation_orders: list[int] = []

    def fake_extract_mpr_plane(
        view_arg,
        volume_arg,
        viewport_key=None,
        output_shape=None,
        interpolation_order=1,
    ):
        del view_arg
        captured_shapes.append(output_shape)
        captured_interpolation_orders.append(int(interpolation_order))
        target_viewport = viewport_key or MPR_VIEWPORT_AXIAL
        plane_shape = output_shape or service._get_mpr_plane_shape(volume_arg.shape, target_viewport)
        return np.zeros(plane_shape, dtype=np.float32), 1, 10

    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series, progress_callback=None: volume)
    monkeypatch.setattr(service, "_extract_mpr_plane", fake_extract_mpr_plane)
    corner_info_calls = 0

    def fake_build_slice_corner_info_overlay(*args, **kwargs):
        nonlocal corner_info_calls
        del args, kwargs
        corner_info_calls += 1
        return CornerInfoOverlay()

    monkeypatch.setattr(service, "_get_reference_instance_and_cache", lambda resolved_series: (None, None))
    monkeypatch.setattr(service, "_build_slice_corner_info_overlay", fake_build_slice_corner_info_overlay)
    monkeypatch.setattr(
        service,
        "_build_mpr_crosshair_overlay",
        lambda *args, **kwargs: MprCrosshairOverlay(
            width=300,
            height=300,
            image_left=0.0,
            image_top=0.0,
            image_width=300.0,
            image_height=300.0,
            horizontal_position=150.0,
            horizontal_color=(255, 255, 255, 255),
            vertical_position=150.0,
            vertical_color=(255, 255, 255, 255),
            center_x=150.0,
            center_y=150.0,
        ),
    )
    monkeypatch.setattr(service, "_render_fast_mpr_preview", lambda context: Image.new("L", (300, 300)))
    monkeypatch.setattr(service, "_encode_image", lambda image, image_format: b"image")

    service._render_mpr_view(view, image_format="jpeg", fast_preview=True)
    assert corner_info_calls == 0
    service._render_mpr_view(view, image_format="jpeg", fast_preview=True, fast_preview_full_resolution=True)
    assert corner_info_calls == 0
    service._render_mpr_view(view, image_format="png", fast_preview=False)
    assert corner_info_calls == 1

    assert captured_shapes[0] == service._get_mpr_fast_preview_plane_shape(
        volume.shape,
        MPR_VIEWPORT_AXIAL,
        viewport_size=(view.height, view.width),
    )
    assert captured_shapes[1] is None
    assert captured_shapes[2] is None
    assert captured_interpolation_orders == [0, 0, 1]


def test_mpr_fast_preview_shape_respects_viewport_size() -> None:
    service = ViewerService()

    assert service._get_mpr_fast_preview_plane_shape(
        (64, 1024, 1024),
        MPR_VIEWPORT_AXIAL,
        viewport_size=(360, 480),
    ) == (119, 158)

    assert service._get_mpr_fast_preview_plane_shape(
        (64, 1024, 1024),
        MPR_VIEWPORT_AXIAL,
        viewport_size=(120, 120),
    ) == (96, 96)


def test_derive_plane_pose_uses_stable_display_axes_after_large_axial_rotation() -> None:
    geometry = build_identity_geometry((5, 6, 7))
    frame = MprFrameState(
        center=(2.0, 3.0, 4.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, -0.1, 0.995),
        axis_col=(0.0, -0.995, -0.1),
    )
    cursor = legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)

    coronal = derive_plane_pose(cursor, "mpr-cor", geometry)
    sagittal = derive_plane_pose(cursor, "mpr-sag", geometry)

    assert np.allclose(coronal.row_world, [-1.0, 0.0, 0.0], atol=1e-6)
    assert float(np.dot(coronal.col_world, frame.axis_col)) > 0.999
    assert np.allclose(sagittal.row_world, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.isclose(float(np.dot(sagittal.row_world, sagittal.col_world)), 0.0, atol=1e-6)
    assert np.isclose(float(np.dot(sagittal.col_world, sagittal.normal_world)), 0.0, atol=1e-6)
    assert np.isclose(float(np.linalg.norm(sagittal.col_world)), 1.0, atol=1e-6)


def test_oblique_plane_pose_keeps_dragged_cursor_projection_near_target_image_point() -> None:
    service = ViewerService()
    geometry = build_identity_geometry((7, 8, 9))
    cursor = create_default_cursor(geometry)
    normal_world = np.asarray([2.0, -2.0, 1.0], dtype=np.float64)
    normal_world = normal_world / float(np.linalg.norm(normal_world))
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    initial_pose = derive_plane_pose(
        cursor,
        MPR_VIEWPORT_AXIAL,
        geometry,
        normal_world_override=normal_world,
        use_display_basis_for_cursor_offsets=True,
    )
    origin_x, origin_y = service._project_world_point_to_plane_image(initial_pose, cursor.center_world)
    group.crosshair_drag_origin_center = tuple(
        float(value) for value in world_to_ijk_point(geometry, cursor.center_world)
    )
    group.crosshair_drag_origin_image = (origin_x, origin_y)

    target_x = origin_x + 1.0
    target_y = origin_y + 1.0
    next_center_world = service._resolve_mpr_center_from_image_point(
        group,
        initial_pose,
        geometry,
        target_x,
        target_y,
    )
    next_pose = derive_plane_pose(
        replace(cursor, center_world=np.asarray(next_center_world, dtype=np.float64)),
        MPR_VIEWPORT_AXIAL,
        geometry,
        normal_world_override=normal_world,
        use_display_basis_for_cursor_offsets=True,
    )
    next_x, next_y = service._project_world_point_to_plane_image(next_pose, next_pose.cursor_center_world)

    assert next_x == pytest.approx(target_x, abs=0.12)
    assert next_y == pytest.approx(target_y, abs=0.12)


def test_mpr_display_aspect_uses_plane_pose_physical_spacing() -> None:
    transform = VolumePatientTransform(
        origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        axis_vectors=(
            np.asarray([3.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 2.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.5], dtype=np.float64),
        ),
        shape=(5, 6, 7),
    )
    geometry = build_geometry_from_patient_transform(transform)
    cursor = create_default_cursor(geometry)
    service = ViewerService()

    axial = derive_plane_pose(cursor, "mpr-ax", geometry)
    coronal = derive_plane_pose(cursor, "mpr-cor", geometry)
    sagittal = derive_plane_pose(cursor, "mpr-sag", geometry)

    assert np.allclose(service._get_mpr_display_aspect_xy_from_pose(axial), (0.5, 2.0), atol=1e-6)
    assert np.allclose(service._get_mpr_display_aspect_xy_from_pose(coronal), (0.5, 3.0), atol=1e-6)
    assert np.allclose(service._get_mpr_display_aspect_xy_from_pose(sagittal), (2.0, 3.0), atol=1e-6)


def test_viewer_service_uses_cursor_as_group_geometry_source() -> None:
    service = ViewerService()
    geometry = build_identity_geometry((5, 6, 7))
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group)
    first_frame = MprFrameState(
        center=(2.0, 3.0, 4.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )
    first_cursor = legacy_frame_to_cursor(first_frame, geometry, reference_center=first_frame.center)
    service._sync_group_from_mpr_cursor(group, first_cursor, geometry, geometry.shape_ijk)

    resolved_cursor = service._get_mpr_cursor_state(view, geometry, geometry.shape_ijk)
    assert group.mpr_cursor is not None
    assert np.allclose(resolved_cursor.center_world, [2.0, 3.0, 4.0], atol=1e-6)

    next_frame = MprFrameState(
        center=(1.0, 2.0, 3.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 0.0, 1.0),
        axis_col=(0.0, -1.0, 0.0),
    )
    next_cursor = legacy_frame_to_cursor(next_frame, geometry, reference_center=next_frame.center)
    group.mpr_cursor = service._serialize_mpr_cursor_record(next_cursor)
    group.axial_index = 4
    group.coronal_index = 4
    group.sagittal_index = 4

    rebuilt_cursor = service._get_mpr_cursor_state(view, geometry, geometry.shape_ijk)
    rebuilt_frame = cursor_to_legacy_frame(rebuilt_cursor, geometry)

    assert np.allclose(rebuilt_frame.center, next_frame.center, atol=1e-6)
    assert np.allclose(rebuilt_frame.axis_slice, next_frame.axis_slice, atol=1e-6)
    assert np.allclose(rebuilt_frame.axis_row, next_frame.axis_row, atol=1e-6)
    assert np.allclose(rebuilt_frame.axis_col, next_frame.axis_col, atol=1e-6)
    assert not np.allclose(rebuilt_cursor.center_world, ijk_to_world_point(geometry, (4.0, 4.0, 4.0)), atol=1e-6)


def test_mpr_model_rotation_changes_reslice_without_rotating_cursor() -> None:
    service = ViewerService()
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    geometry = build_identity_geometry(volume.shape)
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group)
    frame = MprFrameState(
        center=(2.0, 3.0, 4.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )
    cursor = legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)
    service._sync_group_from_mpr_cursor(group, cursor, geometry, volume.shape)

    base_plane, _, _ = service._extract_mpr_plane(view, volume, MPR_VIEWPORT_AXIAL)
    active_plane = service._build_mpr_pose_context(view, volume.shape).poses[MPR_VIEWPORT_AXIAL]
    service._set_mpr_model_rotation_matrix(
        group,
        axis_angle_rotation_matrix(np.asarray(active_plane.normal_world, dtype=np.float64), np.pi / 2.0),
    )
    rotated_plane, _, _ = service._extract_mpr_plane(view, volume, MPR_VIEWPORT_AXIAL)
    resolved_cursor = service._get_mpr_cursor_state(view, geometry, volume.shape)

    assert not np.allclose(rotated_plane, base_plane, atol=1e-6)
    assert np.allclose(resolved_cursor.orientation_world, cursor.orientation_world, atol=1e-6)


def test_mpr_model_rotation_uses_fixed_pivot_when_crosshair_center_moves() -> None:
    service = ViewerService()
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    geometry = build_identity_geometry(volume.shape)
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group)
    service._reset_mpr_group_geometry(group, volume.shape)
    pose_context = service._build_mpr_pose_context(view, volume.shape)
    cursor = pose_context.cursor
    active_plane = pose_context.poses[MPR_VIEWPORT_AXIAL]
    service._set_mpr_model_rotation_matrix(
        group,
        axis_angle_rotation_matrix(np.asarray(active_plane.normal_world, dtype=np.float64), np.pi / 2.0),
        pivot_world=active_plane.cursor_center_world,
    )
    rotated_before, _, _ = service._extract_mpr_plane(view, volume, MPR_VIEWPORT_AXIAL)
    origin_x, origin_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )
    next_center_world = service._resolve_mpr_center_from_image_point(
        group,
        active_plane,
        geometry,
        origin_x + 1.0,
        origin_y,
    )

    service._sync_group_from_mpr_cursor(
        group,
        replace(cursor, center_world=np.asarray(next_center_world, dtype=np.float64)),
        geometry,
        volume.shape,
    )
    rotated_after, _, _ = service._extract_mpr_plane(view, volume, MPR_VIEWPORT_AXIAL)

    assert not np.allclose(next_center_world, active_plane.cursor_center_world, atol=1e-6)
    assert np.allclose(rotated_after, rotated_before, atol=1e-6)


@pytest.mark.parametrize(
    ("view_type", "expected_viewport", "expected_center"),
    [
        ("AX", MPR_VIEWPORT_AXIAL, (3.0, 3.0, 3.0)),
        ("COR", MPR_VIEWPORT_CORONAL, (2.0, 4.0, 3.0)),
        ("SAG", MPR_VIEWPORT_SAGITTAL, (2.0, 3.0, 4.0)),
    ],
)
def test_mpr_scroll_moves_crosshair_center_slice(
    monkeypatch,
    view_type: str,
    expected_viewport: str,
    expected_center: tuple[float, float, float],
) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series: volume)

    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type=view_type, view_group=group)
    service._reset_mpr_group_geometry(group, volume.shape, series=series)

    service._handle_scroll(view, series, 1)
    pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    center_ijk = world_to_ijk_point(pose_context.geometry, pose_context.cursor.center_world)

    assert service._resolve_mpr_viewport(view) == expected_viewport
    assert np.allclose(center_ijk, expected_center, atol=1e-6)
    assert (group.axial_index, group.coronal_index, group.sagittal_index) == (
        round(expected_center[0]),
        round(expected_center[1]),
        round(expected_center[2]),
    )


def test_mpr_rotate3d_drag_updates_model_rotation_without_rotating_cursor(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series: volume)
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group, width=240, height=240)
    service._reset_mpr_group_geometry(group, volume.shape, series=series)
    before_cursor = service._get_mpr_cursor_state(view, build_identity_geometry(volume.shape), volume.shape)
    active_plane = service._build_mpr_pose_context(view, volume.shape, series=series).poses[MPR_VIEWPORT_AXIAL]

    assert not service._handle_mpr_model_rotate_3d(
        view,
        ViewOperationRequest(viewId="v", opType="rotate3d", actionType="start", x=0.5, y=0.25),
    )
    assert service._handle_mpr_model_rotate_3d(
        view,
        ViewOperationRequest(viewId="v", opType="rotate3d", actionType="move", x=0.75, y=0.5),
    )

    after_cursor = service._get_mpr_cursor_state(view, build_identity_geometry(volume.shape), volume.shape)
    rotation_matrix = service._get_mpr_model_rotation_matrix(group)
    top_direction = -np.asarray(active_plane.row_world, dtype=np.float64)
    right_direction = np.asarray(active_plane.col_world, dtype=np.float64)
    normal_direction = np.asarray(active_plane.normal_world, dtype=np.float64)
    assert float(np.dot(rotation_matrix @ top_direction, -right_direction)) > 0.99
    assert np.allclose(rotation_matrix @ normal_direction, normal_direction, atol=1e-6)
    assert np.allclose(after_cursor.orientation_world, before_cursor.orientation_world, atol=1e-6)


def test_mpr_crosshair_move_uses_screen_plane_direction_after_model_rotation() -> None:
    service = ViewerService()
    volume_shape = (5, 6, 7)
    geometry = build_identity_geometry(volume_shape)
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group)
    service._reset_mpr_group_geometry(group, volume_shape)
    active_plane = service._build_mpr_pose_context(view, volume_shape).poses[MPR_VIEWPORT_AXIAL]
    service._set_mpr_model_rotation_matrix(
        group,
        axis_angle_rotation_matrix(np.asarray(active_plane.normal_world, dtype=np.float64), np.pi / 2.0),
    )
    origin_x, origin_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )

    next_center_world = service._resolve_mpr_center_from_image_point(
        group,
        active_plane,
        geometry,
        origin_x + 1.0,
        origin_y,
    )

    assert np.allclose(
        next_center_world - active_plane.cursor_center_world,
        np.asarray(active_plane.col_world, dtype=np.float64) * active_plane.pixel_spacing_col_mm,
        atol=1e-6,
    )


def test_mpr_crosshair_move_uses_image_normalized_coordinates_when_canvas_has_letterbox(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series: volume)

    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group, width=320, height=240)
    service._reset_mpr_group_geometry(group, volume.shape, series=series)

    pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    active_plane = pose_context.poses[MPR_VIEWPORT_AXIAL]
    center_image_x, center_image_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )
    start_x = float(center_image_x) / float(active_plane.output_shape[1])
    start_y = float(center_image_y) / float(active_plane.output_shape[0])
    target_x = min(1.0, start_x + 0.15)
    target_y = min(1.0, start_y + 0.1)

    assert not service._handle_mpr_crosshair(
        view,
        ViewOperationRequest(viewId="v", opType="crosshair", actionType="start", x=start_x, y=start_y),
    )
    assert service._handle_mpr_crosshair(
        view,
        ViewOperationRequest(viewId="v", opType="crosshair", actionType="move", x=target_x, y=target_y),
    )

    next_pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    next_plane = next_pose_context.poses[MPR_VIEWPORT_AXIAL]
    next_center_image_x, next_center_image_y = service._project_world_point_to_plane_image(
        next_plane,
        next_pose_context.cursor.center_world,
    )

    assert next_center_image_x / float(next_plane.output_shape[1]) == pytest.approx(target_x, abs=1e-6)
    assert next_center_image_y / float(next_plane.output_shape[0]) == pytest.approx(target_y, abs=1e-6)


@pytest.mark.parametrize(
    ("view_type", "viewport_key"),
    [
        ("AX", MPR_VIEWPORT_AXIAL),
        ("COR", MPR_VIEWPORT_CORONAL),
        ("SAG", MPR_VIEWPORT_SAGITTAL),
    ],
)
def test_mpr_crosshair_move_projects_backend_cursor_for_each_active_viewport(
    monkeypatch,
    view_type: str,
    viewport_key: str,
) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.arange(9 * 10 * 11, dtype=np.float32).reshape((9, 10, 11))
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series: volume)

    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id=f"v-{view_type.lower()}", series_id="s", view_type=view_type, view_group=group, width=320, height=240)
    service._reset_mpr_group_geometry(group, volume.shape, series=series)

    pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    active_plane = pose_context.poses[viewport_key]
    center_image_x, center_image_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )
    start_x = float(center_image_x) / float(active_plane.output_shape[1])
    start_y = float(center_image_y) / float(active_plane.output_shape[0])
    target_x = min(0.9, max(0.1, start_x + 0.1))
    target_y = min(0.9, max(0.1, start_y + 0.08))

    assert not service._handle_mpr_crosshair(
        view,
        ViewOperationRequest(viewId=view.view_id, opType="crosshair", actionType="start", x=start_x, y=start_y),
    )
    assert service._handle_mpr_crosshair(
        view,
        ViewOperationRequest(viewId=view.view_id, opType="crosshair", actionType="move", x=target_x, y=target_y),
    )

    next_pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    next_plane = next_pose_context.poses[viewport_key]
    next_center_image_x, next_center_image_y = service._project_world_point_to_plane_image(
        next_plane,
        next_pose_context.cursor.center_world,
    )

    assert next_center_image_x / float(next_plane.output_shape[1]) == pytest.approx(target_x, abs=1e-6)
    assert next_center_image_y / float(next_plane.output_shape[0]) == pytest.approx(target_y, abs=1e-6)


def test_mpr_crosshair_end_applies_final_pointer_position(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda resolved_series: volume)

    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group, width=320, height=240)
    service._reset_mpr_group_geometry(group, volume.shape, series=series)
    pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    active_plane = pose_context.poses[MPR_VIEWPORT_AXIAL]
    center_image_x, center_image_y = service._project_world_point_to_plane_image(
        active_plane,
        active_plane.cursor_center_world,
    )
    start_x = float(center_image_x) / float(active_plane.output_shape[1])
    start_y = float(center_image_y) / float(active_plane.output_shape[0])
    target_x = min(1.0, start_x + 0.12)
    target_y = min(1.0, start_y + 0.08)

    assert not service._handle_mpr_crosshair(
        view,
        ViewOperationRequest(viewId="v", opType="crosshair", actionType="start", x=start_x, y=start_y),
    )
    assert service._handle_mpr_crosshair(
        view,
        ViewOperationRequest(viewId="v", opType="crosshair", actionType="end", x=target_x, y=target_y),
    )

    next_pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
    next_plane = next_pose_context.poses[MPR_VIEWPORT_AXIAL]
    next_center_image_x, next_center_image_y = service._project_world_point_to_plane_image(
        next_plane,
        next_pose_context.cursor.center_world,
    )

    assert not group.crosshair_drag_active
    assert group.crosshair_drag_origin_center is None
    assert group.crosshair_drag_origin_image is None
    assert next_center_image_x / float(next_plane.output_shape[1]) == pytest.approx(target_x, abs=1e-6)
    assert next_center_image_y / float(next_plane.output_shape[0]) == pytest.approx(target_y, abs=1e-6)


def test_build_mpr_crosshair_info_normalizes_against_rendered_canvas() -> None:
    info = ViewerService._build_mpr_crosshair_info(
        MprCrosshairOverlay(
            width=320,
            height=240,
            image_left=40.0,
            image_top=20.0,
            image_width=160.0,
            image_height=120.0,
            horizontal_position=100.0,
            horizontal_color=(0, 0, 0, 255),
            vertical_position=200.0,
            vertical_color=(0, 0, 0, 255),
            center_x=200.0,
            center_y=100.0,
        )
    )

    assert info is not None
    assert info.center_x == pytest.approx(200.0 / 320.0, abs=1e-6)
    assert info.center_y == pytest.approx(100.0 / 240.0, abs=1e-6)
    assert info.vertical_position == pytest.approx(200.0 / 320.0, abs=1e-6)
    assert info.horizontal_position == pytest.approx(100.0 / 240.0, abs=1e-6)
    assert info.hit_radius == pytest.approx(12.0 / 240.0, abs=1e-6)


def test_mpr_model_rotation_keeps_active_view_labels_and_updates_other_view_labels() -> None:
    service = ViewerService()
    volume_shape = (5, 6, 7)
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group)
    service._reset_mpr_group_geometry(group, volume_shape)
    pose_context = service._build_mpr_pose_context(view, volume_shape)
    axial_plane = pose_context.poses[MPR_VIEWPORT_AXIAL]
    coronal_plane = pose_context.poses[MPR_VIEWPORT_CORONAL]
    axial_before = service._build_mpr_orientation_overlay(
        view,
        MPR_VIEWPORT_AXIAL,
        service._plane_state_from_pose(axial_plane),
        plane_pose=axial_plane,
    )
    coronal_before = service._build_mpr_orientation_overlay(
        view,
        MPR_VIEWPORT_CORONAL,
        service._plane_state_from_pose(coronal_plane),
        plane_pose=coronal_plane,
    )
    service._set_mpr_model_rotation_matrix(
        group,
        axis_angle_rotation_matrix(np.asarray(axial_plane.normal_world, dtype=np.float64), np.pi / 2.0),
    )

    axial_after = service._build_mpr_orientation_overlay(
        view,
        MPR_VIEWPORT_AXIAL,
        service._plane_state_from_pose(axial_plane),
        plane_pose=axial_plane,
    )
    coronal_after = service._build_mpr_orientation_overlay(
        view,
        MPR_VIEWPORT_CORONAL,
        service._plane_state_from_pose(coronal_plane),
        plane_pose=coronal_plane,
    )

    assert (axial_after.top, axial_after.right, axial_after.bottom, axial_after.left) == (
        axial_before.top,
        axial_before.right,
        axial_before.bottom,
        axial_before.left,
    )
    assert (coronal_after.top, coronal_after.right, coronal_after.bottom, coronal_after.left) != (
        coronal_before.top,
        coronal_before.right,
        coronal_before.bottom,
        coronal_before.left,
    )
