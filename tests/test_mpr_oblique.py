import math
from types import SimpleNamespace

import numpy as np

from app.models.viewer import MprObliquePlaneState, MprRotationDragRecord, ViewGroupRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.mpr import build_identity_geometry, cursor_to_legacy_frame, ijk_to_world_point
from app.services import mpr_geometry
from app.services.mpr_geometry import VolumePatientTransform
from app.services.viewer_service import ViewerService
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer


def _run_with_stubbed_mpr_volume(service: ViewerService, callback):
    original_get_volume = service._get_series_volume
    original_series_get = series_registry.get
    try:
        series_registry.get = lambda _series_id: SimpleNamespace(series_id="s", instances=[])  # type: ignore[method-assign]
        service._get_series_volume = lambda _series: np.zeros((5, 6, 7), dtype=np.float32)  # type: ignore[method-assign]
        return callback()
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]


def _apply_oblique_drag(
    service: ViewerService,
    view: ViewRecord,
    *,
    line: str,
    angle_rad: float,
):
    def run_drag():
        active_viewport = service._resolve_mpr_viewport(view)
        series = series_registry.get(view.series_id)
        volume = service._get_series_volume(series)
        pose_context = service._build_mpr_pose_context(view, volume.shape, series=series)
        horizontal_angle, vertical_angle = service._get_mpr_crosshair_line_angles_from_poses(pose_context.poses, active_viewport)
        start_angle = horizontal_angle if line == "horizontal" else vertical_angle
        return (
            service._handle_mpr_oblique(
                view,
                ViewOperationRequest(viewId=view.view_id, opType="mprOblique", actionType="start", line=line, deltaAngleRad=0.0),
            ),
            service._handle_mpr_oblique(
                view,
                ViewOperationRequest(viewId=view.view_id, opType="mprOblique", actionType="move", line=line, deltaAngleRad=angle_rad - start_angle),
            ),
        )

    return _run_with_stubbed_mpr_volume(service, run_drag)


def _get_pose_line_angles(service: ViewerService, view: ViewRecord, viewport_key: str | None = None) -> tuple[float, float]:
    series = SimpleNamespace(series_id=view.series_id, instances=[])
    volume_shape = (5, 6, 7)
    pose_context = service._build_mpr_pose_context(view, volume_shape, series=series)
    return service._get_mpr_crosshair_line_angles_from_poses(
        pose_context.poses,
        viewport_key or service._resolve_mpr_viewport(view),
    )


def _normalize_line_delta(before_angle: float, after_angle: float) -> float:
    delta = float(after_angle) - float(before_angle)
    while delta > (math.pi / 2.0):
        delta -= math.pi
    while delta <= (-math.pi / 2.0):
        delta += math.pi
    return delta


def _get_group_frame(service: ViewerService, group: ViewGroupRecord, volume_shape: tuple[int, int, int] = (5, 6, 7)):
    view = ViewRecord(view_id="v-frame", series_id=group.series_id, view_type="MPR", view_group=group)
    geometry = build_identity_geometry(volume_shape)
    cursor = service._get_mpr_cursor_state(view, geometry, volume_shape)
    return cursor_to_legacy_frame(cursor, geometry)


def _get_group_plane(service: ViewerService, group: ViewGroupRecord, viewport_key: str, volume_shape: tuple[int, int, int] = (5, 6, 7)):
    view = ViewRecord(view_id=f"v-{viewport_key}", series_id=group.series_id, view_type="MPR", view_group=group)
    pose_context = service._build_mpr_pose_context(view, volume_shape, series=SimpleNamespace(series_id=group.series_id, instances=[]))
    return service._plane_state_from_pose(pose_context.poses[viewport_key])


def _build_group_orientation_overlay(service: ViewerService, view: ViewRecord, viewport_key: str):
    pose_context = service._build_mpr_pose_context(view, (5, 6, 7), series=SimpleNamespace(series_id=view.series_id, instances=[]))
    plane = service._plane_state_from_pose(pose_context.poses[viewport_key])
    return service._build_mpr_orientation_overlay(view, viewport_key, plane, plane_pose=pose_context.poses[viewport_key])


def _expected_orientation_text(service: ViewerService, vector: np.ndarray, *, is_oblique: bool) -> str | None:
    if is_oblique:
        return service._orientation_text_for_vector(vector, minimum_magnitude=0.2, max_components=2, axis_priority=(1, 0, 2))
    return service._dominant_orientation_text_for_vector(vector)


def _expected_overlay_labels_from_pose(service: ViewerService, pose) -> tuple[str | None, str | None, str | None, str | None]:
    row_patient = mpr_geometry.fallback_volume_direction_to_patient_vector(pose.row_world)
    col_patient = mpr_geometry.fallback_volume_direction_to_patient_vector(pose.col_world)
    return (
        _expected_orientation_text(service, -col_patient, is_oblique=pose.is_oblique),
        _expected_orientation_text(service, col_patient, is_oblique=pose.is_oblique),
        _expected_orientation_text(service, -row_patient, is_oblique=pose.is_oblique),
        _expected_orientation_text(service, row_patient, is_oblique=pose.is_oblique),
    )


def _set_group_center(service: ViewerService, group: ViewGroupRecord, center_ijk: tuple[float, float, float], volume_shape: tuple[int, int, int] = (5, 6, 7)) -> None:
    view = ViewRecord(view_id="v-center", series_id=group.series_id, view_type="MPR", view_group=group)
    geometry = build_identity_geometry(volume_shape)
    cursor = service._get_mpr_cursor_state(view, geometry, volume_shape)
    next_cursor = type(cursor)(
        center_world=ijk_to_world_point(geometry, center_ijk),
        reference_center_world=cursor.reference_center_world,
        orientation_world=cursor.orientation_world,
        linked_to_volume_rotation=cursor.linked_to_volume_rotation,
    )
    service._sync_group_from_mpr_cursor(group, next_cursor, geometry, volume_shape)


def test_mpr_oblique_drag_updates_target_plane_and_reslices() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    view.mpr_axial_index = 2
    view.mpr_coronal_index = 3
    view.mpr_sagittal_index = 4
    initial_frame = _get_group_frame(service, group)

    start = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprOblique",
        actionType="start",
        line="horizontal",
        deltaAngleRad=0.0,
    )
    move = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprOblique",
        actionType="move",
        line="horizontal",
        deltaAngleRad=0.35,
    )

    start_result, move_result = _run_with_stubbed_mpr_volume(service, lambda: (
        service._handle_mpr_oblique(view, start),
        service._handle_mpr_oblique(view, move),
    ))

    assert start_result is False
    assert move_result is True
    axial_horizontal, axial_vertical = _get_pose_line_angles(service, view, "mpr-ax")
    assert not math.isclose(axial_horizontal, 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose((axial_vertical - axial_horizontal) % np.pi, np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert group.mpr_cursor is not None
    next_frame = _get_group_frame(service, group)
    assert not np.allclose(np.asarray(next_frame.axis_row, dtype=np.float64), np.asarray(initial_frame.axis_row, dtype=np.float64))
    assert not np.allclose(np.asarray(next_frame.axis_col, dtype=np.float64), np.asarray(initial_frame.axis_col, dtype=np.float64))
    axial_plane = _get_group_plane(service, group, "mpr-ax")
    coronal_plane = _get_group_plane(service, group, "mpr-cor")
    sagittal_plane = _get_group_plane(service, group, "mpr-sag")
    assert axial_plane.is_oblique is False
    assert coronal_plane.is_oblique is True
    assert sagittal_plane.is_oblique is True
    axial_normal = np.asarray(axial_plane.normal, dtype=np.float64)
    coronal_normal = np.asarray(coronal_plane.normal, dtype=np.float64)
    sagittal_normal = np.asarray(sagittal_plane.normal, dtype=np.float64)
    assert math.isclose(float(np.dot(axial_normal, coronal_normal)), 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(np.dot(axial_normal, sagittal_normal)), 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(np.dot(coronal_normal, sagittal_normal)), 0.0, rel_tol=0.0, abs_tol=1e-6)

    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    plane, current, total = service._extract_mpr_plane(coronal_view, volume, "mpr-cor")

    assert plane.shape == (5, 7)
    assert current == 3
    assert total == 6
    assert np.isfinite(plane).all()
    for viewport_key in ("mpr-cor", "mpr-sag"):
        horizontal, vertical = _get_pose_line_angles(service, view, viewport_key)
        assert math.isclose((vertical - horizontal) % np.pi, np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_drag_keeps_active_view_plane_static() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))

    initial_plane, _, _ = service._extract_mpr_plane(axial_view, volume, "mpr-ax")

    _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)

    rotated_plane, _, _ = service._extract_mpr_plane(axial_view, volume, "mpr-ax")
    assert np.allclose(rotated_plane, initial_plane)


def test_mpr_oblique_drag_keeps_dragged_vertical_line_identity_in_active_view() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)

    horizontal, vertical = _get_pose_line_angles(service, axial_view, "mpr-ax")
    assert not math.isclose(vertical, np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose((vertical - horizontal) % np.pi, np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_drag_preserves_target_view_column_orientation() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)

    coronal_plane = _get_group_plane(service, group, "mpr-cor")
    sagittal_plane = _get_group_plane(service, group, "mpr-sag")
    coronal_normal = np.asarray(coronal_plane.normal, dtype=np.float64)
    sagittal_normal = np.asarray(sagittal_plane.normal, dtype=np.float64)
    coronal_col = np.asarray(coronal_plane.col, dtype=np.float64)
    sagittal_col = np.asarray(sagittal_plane.col, dtype=np.float64)

    assert abs(float(np.dot(coronal_col, sagittal_normal))) > 0.999
    assert abs(float(np.dot(sagittal_col, coronal_normal))) > 0.999


def test_mpr_oblique_drag_keeps_axial_rotation_target_views_upright() -> None:
    service = ViewerService()

    for line in ("horizontal", "vertical"):
        for angle_rad in np.linspace(0.0, np.pi / 2.0, 5):
            group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
            axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

            _apply_oblique_drag(service, axial_view, line=line, angle_rad=float(angle_rad))

            for viewport_key in ("mpr-cor", "mpr-sag"):
                plane = _get_group_plane(service, group, viewport_key)
                row = np.asarray(plane.row, dtype=np.float64)
                col = np.asarray(plane.col, dtype=np.float64)
                normal = np.asarray(plane.normal, dtype=np.float64)
                assert math.isclose(float(np.linalg.norm(row)), 1.0, rel_tol=0.0, abs_tol=1e-6)
                assert math.isclose(float(np.linalg.norm(col)), 1.0, rel_tol=0.0, abs_tol=1e-6)
                assert math.isclose(float(np.dot(row, col)), 0.0, rel_tol=0.0, abs_tol=1e-6)
                assert math.isclose(float(np.dot(row, normal)), 0.0, rel_tol=0.0, abs_tol=1e-6)
                assert math.isclose(float(np.dot(col, normal)), 0.0, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_drag_preserves_target_plane_direction_on_small_moves() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    previous_normal = np.asarray(_get_group_plane(service, group, "mpr-sag").normal, dtype=np.float64)

    _, current_vertical_angle = _get_pose_line_angles(service, axial_view, "mpr-ax")
    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=current_vertical_angle + 0.0005)
    next_normal = np.asarray(_get_group_plane(service, group, "mpr-sag").normal, dtype=np.float64)

    assert float(np.dot(previous_normal, next_normal)) > 0.999


def test_mpr_crosshair_move_uses_oblique_plane_basis_after_rotation() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (2.0, 3.0, 3.0))
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    previous_center = np.asarray(_get_group_frame(service, group).center, dtype=np.float64)
    coronal_plane = _get_group_plane(service, group, "mpr-cor")
    row_dir = np.asarray(coronal_plane.row, dtype=np.float64)
    col_dir = np.asarray(coronal_plane.col, dtype=np.float64)
    volume_shape = (5, 6, 7)
    plane_height, plane_width = service._get_mpr_plane_shape(volume_shape, "mpr-cor")
    start_image_x = float(plane_width) / 2.0
    start_image_y = float(plane_height) / 2.0
    image_x = float(plane_width) / 2.0 + 1.0
    image_y = float(plane_height) / 2.0 + 1.0
    expected_center = previous_center + row_dir + col_dir

    original_get_volume = service._get_series_volume
    original_series_get = series_registry.get
    original_get_mpr_aspect = service._get_mpr_display_aspect_xy
    image_points = iter(((start_image_x, start_image_y), (image_x, image_y)))
    original_canvas_to_image = service._canvas_to_image_coordinates
    try:
        series_registry.get = lambda _series_id: SimpleNamespace(series_id="s", instances=[])  # type: ignore[method-assign]
        service._get_series_volume = lambda _series: np.zeros(volume_shape, dtype=np.float32)  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = lambda _series, _viewport, _plane_state=None: (1.0, 1.0)  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = lambda _transform, _canvas_x, _canvas_y: next(image_points)  # type: ignore[method-assign]
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="start", x=0.5, y=0.5))
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.75))
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = original_canvas_to_image  # type: ignore[method-assign]

    assert np.allclose(np.asarray(_get_group_frame(service, group).center, dtype=np.float64), expected_center, atol=1e-6)


def test_mpr_crosshair_move_preserves_active_view_oblique_angles() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (2.0, 3.0, 3.0))
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)
    volume_shape = (5, 6, 7)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    expected_horizontal, expected_vertical = _get_pose_line_angles(service, axial_view, "mpr-ax")

    original_get_volume = service._get_series_volume
    original_series_get = series_registry.get
    original_get_mpr_aspect = service._get_mpr_display_aspect_xy
    image_points = iter(((3.5, 2.5), (4.5, 3.0)))
    original_canvas_to_image = service._canvas_to_image_coordinates
    try:
        series_registry.get = lambda _series_id: SimpleNamespace(series_id="s", instances=[])  # type: ignore[method-assign]
        service._get_series_volume = lambda _series: np.zeros(volume_shape, dtype=np.float32)  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = lambda _series, _viewport, _plane_state=None: (1.0, 1.0)  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = lambda _transform, _canvas_x, _canvas_y: next(image_points)  # type: ignore[method-assign]
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="start", x=0.5, y=0.5))
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.5))
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = original_canvas_to_image  # type: ignore[method-assign]

    horizontal, vertical = _get_pose_line_angles(service, axial_view, "mpr-ax")
    assert math.isclose(horizontal, expected_horizontal, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vertical, expected_vertical, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_crosshair_move_does_not_accumulate_absolute_offset_every_frame() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (2.0, 3.0, 3.0))
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)
    volume_shape = (5, 6, 7)

    original_get_volume = service._get_series_volume
    original_series_get = series_registry.get
    original_get_mpr_aspect = service._get_mpr_display_aspect_xy
    image_points = iter(((3.5, 2.5), (4.5, 3.0), (4.5, 3.0)))
    original_canvas_to_image = service._canvas_to_image_coordinates
    try:
        series_registry.get = lambda _series_id: SimpleNamespace(series_id="s", instances=[])  # type: ignore[method-assign]
        service._get_series_volume = lambda _series: np.zeros(volume_shape, dtype=np.float32)  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = lambda _series, _viewport, _plane_state=None: (1.0, 1.0)  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = lambda _transform, _canvas_x, _canvas_y: next(image_points)  # type: ignore[method-assign]
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="start", x=0.5, y=0.5))
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.5))
        center_after_first_move = np.asarray(_get_group_frame(service, group).center, dtype=np.float64)
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.5))
        center_after_second_move = np.asarray(_get_group_frame(service, group).center, dtype=np.float64)
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = original_canvas_to_image  # type: ignore[method-assign]

    assert np.allclose(center_after_second_move, center_after_first_move, atol=1e-6)


def test_mpr_crosshair_overlay_centers_oblique_target_view() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)
    volume_shape = (5, 6, 7)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    plane_shape = service._get_mpr_plane_shape(volume_shape, "mpr-cor")
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=plane_shape[1],
        image_height=plane_shape[0],
        canvas_width=coronal_view.width or plane_shape[1],
        canvas_height=coronal_view.height or plane_shape[0],
        view=coronal_view,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )

    overlay = service._build_mpr_crosshair_overlay(coronal_view, volume_shape, plane_shape, image_transform)
    info = service._build_mpr_crosshair_info(overlay)

    assert info is not None
    assert info.horizontal_position is None
    assert info.vertical_position is None
    assert math.isclose(info.center_x, 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(info.center_y, 0.5, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_reset_restores_group_to_initial_state() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group, width=320, height=240)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=320, height=240)
    sagittal_view = ViewRecord(view_id="v-sag", series_id="s", view_type="SAG", view_group=group, width=320, height=240)

    group.axial_index = 1
    group.coronal_index = 2
    group.sagittal_index = 3
    group.active_viewport = "mpr-sag"
    group.crosshair_drag_active = True
    group.crosshair_drag_origin_center = (1.0, 2.0, 3.0)
    group.crosshair_drag_origin_image = (10.0, 20.0)
    group.oblique_drag_active = True
    axial_view.rotation_degrees = 90
    axial_view.hor_flip = True
    coronal_view.ver_flip = True
    sagittal_view.pseudocolor_preset = "hot"

    original_get_volume = service._get_series_volume
    original_get_mpr_aspect = service._get_mpr_display_aspect_xy
    original_get_group_views = service._get_mpr_group_views
    original_series_get = series_registry.get
    try:
        series_registry.get = lambda _series_id: SimpleNamespace(series_id="s", instances=[])  # type: ignore[method-assign]
        service._get_mpr_group_views = lambda _view: [axial_view, coronal_view, sagittal_view]  # type: ignore[method-assign]
        service._get_series_volume = lambda _series: np.zeros((8, 10, 12), dtype=np.float32)  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = lambda _series, _viewport, _plane_state=None: (1.0, 1.0)  # type: ignore[method-assign]
        service._reset_mpr_view_group(axial_view)
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_mpr_group_views = original_get_group_views  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]

    assert group.axial_index == 4
    assert group.coronal_index == 5
    assert group.sagittal_index == 6
    assert group.mpr_cursor is not None
    assert group.active_viewport == "mpr-ax"
    frame = _get_group_frame(service, group, (8, 10, 12))
    assert frame.center == (4.0, 5.0, 6.0)
    assert frame.axis_slice == (1.0, 0.0, 0.0)
    assert frame.axis_row == (0.0, 1.0, 0.0)
    assert frame.axis_col == (0.0, 0.0, 1.0)
    assert group.crosshair_drag_active is False
    assert group.crosshair_drag_origin_center is None
    assert group.crosshair_drag_origin_image is None
    assert group.oblique_drag_active is False
    assert _get_group_plane(service, group, "mpr-cor", (8, 10, 12)).is_oblique is False
    assert _get_group_plane(service, group, "mpr-sag", (8, 10, 12)).is_oblique is False
    for item in (axial_view, coronal_view, sagittal_view):
        assert item.rotation_degrees == 0
        assert item.hor_flip is False
        assert item.ver_flip is False
        assert item.pseudocolor_preset == "bw"
        assert item.is_initialized is True


def test_mpr_oblique_editing_second_view_does_not_restore_first_view_angles_to_default() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)
    first_axial_horizontal, first_axial_vertical = _get_pose_line_angles(service, axial_view, "mpr-ax")

    _apply_oblique_drag(service, coronal_view, line="vertical", angle_rad=1.2)

    assert not math.isclose(first_axial_horizontal, 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        (first_axial_vertical - first_axial_horizontal) % np.pi,
        np.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert _get_group_plane(service, group, "mpr-ax").is_oblique is True
    assert _get_group_plane(service, group, "mpr-cor").is_oblique is True
    assert _get_group_plane(service, group, "mpr-sag").is_oblique is True


def test_mpr_orientation_overlay_updates_after_oblique_rotation() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    initial_overlay = _build_group_orientation_overlay(service, coronal_view, "mpr-cor")
    assert initial_overlay.top == "S"
    assert initial_overlay.bottom == "I"
    assert initial_overlay.left == "R"
    assert initial_overlay.right == "L"

    _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)

    rotated_overlay = _build_group_orientation_overlay(service, coronal_view, "mpr-cor")
    pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    expected_left, expected_right, expected_top, expected_bottom = _expected_overlay_labels_from_pose(
        service,
        pose_context.poses["mpr-cor"],
    )
    assert (rotated_overlay.left, rotated_overlay.right, rotated_overlay.top, rotated_overlay.bottom) == (
        expected_left,
        expected_right,
        expected_top,
        expected_bottom,
    )


def test_mpr_coronal_initial_orientation_is_slir_with_patient_transform() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    transform = VolumePatientTransform(
        origin=np.zeros(3, dtype=np.float64),
        axis_vectors=(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        ),
        shape=(5, 6, 7),
    )

    original_get_transform = service._get_series_patient_transform
    original_series_get = series_registry.get
    try:
        service._get_series_patient_transform = lambda _series: transform  # type: ignore[method-assign]
        series_registry.get = lambda _series_id: series  # type: ignore[method-assign]
        pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=series)
        plane = service._plane_state_from_pose(pose_context.poses["mpr-cor"])
        overlay = service._build_mpr_orientation_overlay(
            coronal_view,
            "mpr-cor",
            plane,
            plane_pose=pose_context.poses["mpr-cor"],
        )
    finally:
        service._get_series_patient_transform = original_get_transform
        series_registry.get = original_series_get  # type: ignore[method-assign]

    assert (overlay.top, overlay.right, overlay.bottom, overlay.left) == ("S", "L", "I", "R")


def test_mpr_orientation_overlay_flips_target_direction_for_axial_vertical_drag() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id="s", view_type="SAG", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)

    coronal_overlay = _build_group_orientation_overlay(service, coronal_view, "mpr-cor")
    sagittal_overlay = _build_group_orientation_overlay(service, sagittal_view, "mpr-sag")
    coronal_pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    sagittal_pose_context = service._build_mpr_pose_context(sagittal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    assert (coronal_overlay.left, coronal_overlay.right, coronal_overlay.top, coronal_overlay.bottom) == _expected_overlay_labels_from_pose(
        service,
        coronal_pose_context.poses["mpr-cor"],
    )
    assert (sagittal_overlay.left, sagittal_overlay.right, sagittal_overlay.top, sagittal_overlay.bottom) == _expected_overlay_labels_from_pose(
        service,
        sagittal_pose_context.poses["mpr-sag"],
    )

def test_mpr_orientation_overlay_normalizes_axial_vertical_rotation_to_undirected_line_orientation() -> None:
    service = ViewerService()

    for angle_rad, expected in (
        (np.deg2rad(292.5), ("R", "L")),
        (np.deg2rad(315.0), ("P", "A")),
        (np.deg2rad(22.5), ("A", "P")),
        (np.deg2rad(45.0), ("A", "P")),
        (np.deg2rad(112.5), ("R", "L")),
    ):
        group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
        axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
        coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
        _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=float(angle_rad))
        overlay = _build_group_orientation_overlay(service, coronal_view, "mpr-cor")
        pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
        assert (overlay.left, overlay.right, overlay.top, overlay.bottom) == _expected_overlay_labels_from_pose(
            service,
            pose_context.poses["mpr-cor"],
        )


def test_mpr_orientation_overlay_uses_single_axis_labels_on_small_oblique_angle() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=float(np.deg2rad(2.0)))

    overlay = _build_group_orientation_overlay(service, coronal_view, "mpr-cor")
    pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    expected_left, expected_right, _, _ = _expected_overlay_labels_from_pose(service, pose_context.poses["mpr-cor"])
    assert (overlay.left, overlay.right) == (expected_left, expected_right)
    assert overlay.left is not None and len(overlay.left) == 1
    assert overlay.right is not None and len(overlay.right) == 1


def test_mpr_orientation_overlay_uses_combined_axis_labels_on_oblique_views() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id="s", view_type="SAG", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)

    overlay = _build_group_orientation_overlay(service, sagittal_view, "mpr-sag")
    pose_context = service._build_mpr_pose_context(sagittal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    expected_left, expected_right, expected_top, expected_bottom = _expected_overlay_labels_from_pose(
        service,
        pose_context.poses["mpr-sag"],
    )
    assert (overlay.left, overlay.right, overlay.top, overlay.bottom) == (
        expected_left,
        expected_right,
        expected_top,
        expected_bottom,
    )
    assert overlay.left is not None and len(overlay.left) > 1
    assert overlay.right is not None and len(overlay.right) > 1


def test_mpr_orientation_overlay_uses_plane_geometry_without_source_flags() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id="s", view_type="SAG", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)
    group.oblique_source_viewport = None
    group.oblique_source_line = None

    coronal_overlay = _build_group_orientation_overlay(service, coronal_view, "mpr-cor")
    sagittal_overlay = _build_group_orientation_overlay(service, sagittal_view, "mpr-sag")

    coronal_pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    sagittal_pose_context = service._build_mpr_pose_context(sagittal_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
    assert (coronal_overlay.left, coronal_overlay.right, coronal_overlay.top, coronal_overlay.bottom) == _expected_overlay_labels_from_pose(
        service,
        coronal_pose_context.poses["mpr-cor"],
    )
    assert (sagittal_overlay.left, sagittal_overlay.right, sagittal_overlay.top, sagittal_overlay.bottom) == _expected_overlay_labels_from_pose(
        service,
        sagittal_pose_context.poses["mpr-sag"],
    )


def test_mpr_oblique_orientation_overlay_labels_match_plane_screen_axes() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)

    for viewport_key in ("mpr-ax", "mpr-cor", "mpr-sag"):
        view = ViewRecord(view_id=f"v-{viewport_key}", series_id="s", view_type="MPR", view_group=group)
        pose_context = service._build_mpr_pose_context(view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
        pose = pose_context.poses[viewport_key]
        overlay = _build_group_orientation_overlay(service, view, viewport_key)
        expected_left, expected_right, expected_top, expected_bottom = _expected_overlay_labels_from_pose(service, pose)
        assert overlay.top == expected_top
        assert overlay.right == expected_right
        assert overlay.bottom == expected_bottom
        assert overlay.left == expected_left


def test_mpr_frame_axes_sync_from_oblique_plane_normals() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)

    axial_normal = np.asarray(_get_group_plane(service, group, "mpr-ax").normal, dtype=np.float64)
    coronal_normal = np.asarray(_get_group_plane(service, group, "mpr-cor").normal, dtype=np.float64)
    sagittal_normal = np.asarray(_get_group_plane(service, group, "mpr-sag").normal, dtype=np.float64)
    frame = _get_group_frame(service, group)

    assert np.allclose(np.asarray(frame.axis_slice, dtype=np.float64), axial_normal, atol=1e-6)
    assert np.allclose(np.asarray(frame.axis_row, dtype=np.float64), coronal_normal, atol=1e-6)
    assert np.allclose(np.asarray(frame.axis_col, dtype=np.float64), sagittal_normal, atol=1e-6)


def test_mpr_oblique_plane_normals_are_derived_from_cursor_axes() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)

    frame = _get_group_frame(service, group)
    for viewport_key in ("mpr-ax", "mpr-cor", "mpr-sag"):
        plane = _get_group_plane(service, group, viewport_key)
        expected_normal = {
            "mpr-ax": frame.axis_slice,
            "mpr-cor": frame.axis_row,
            "mpr-sag": frame.axis_col,
        }[viewport_key]
        assert np.allclose(np.asarray(plane.normal, dtype=np.float64), np.asarray(expected_normal, dtype=np.float64), atol=1e-6)


def test_mpr_extract_and_crosshair_overlay_work_from_cursor_without_plane_cache() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)
    _set_group_center(service, group, (2.0, 3.0, 3.0))

    plane, current, total = service._extract_mpr_plane(coronal_view, volume, "mpr-cor")
    assert plane.shape == (5, 7)
    assert current == 3
    assert total == 6

    plane_shape = service._get_mpr_plane_shape(volume.shape, "mpr-cor")
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=plane_shape[1],
        image_height=plane_shape[0],
        canvas_width=coronal_view.width or plane_shape[1],
        canvas_height=coronal_view.height or plane_shape[0],
        view=coronal_view,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
    )
    overlay = service._build_mpr_crosshair_overlay(coronal_view, volume.shape, plane_shape, image_transform)
    info = service._build_mpr_crosshair_info(overlay)

    assert info is not None
    assert info.horizontal_position is None
    assert info.vertical_position is None
    assert np.isfinite(overlay.horizontal_angle_rad)
    assert np.isfinite(overlay.vertical_angle_rad)


def test_mpr_crosshair_line_angles_can_be_derived_without_angle_cache() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)
    expected_horizontal, expected_vertical = _get_pose_line_angles(service, axial_view, "mpr-cor")

    horizontal, vertical = _get_pose_line_angles(service, axial_view, "mpr-cor")
    assert math.isclose(horizontal, expected_horizontal, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(vertical, expected_vertical, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_drag_applies_cumulative_delta_from_drag_start() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    def run_drag(delta_angle_rad: float) -> tuple[float, float]:
        group.mpr_cursor = None
        group.rotation_drag = None
        group.oblique_drag_active = False
        pose_context = service._build_mpr_pose_context(axial_view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
        drag = service._serialize_mpr_cursor_record(pose_context.cursor)
        group.rotation_drag = MprRotationDragRecord(
            viewport="mpr-ax",
            line="vertical",
            start_cursor=drag,
        )
        service._apply_mpr_rotation_drag(group, group.rotation_drag, delta_angle_rad, pose_context.geometry, (5, 6, 7))
        return _get_pose_line_angles(service, axial_view, "mpr-ax")

    first_angles = run_drag(0.35)
    second_angles = run_drag(0.35)

    assert math.isclose(first_angles[0], second_angles[0], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(first_angles[1], second_angles[1], rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_second_view_rotation_does_not_flip_at_right_angle() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)

    def run_drag():
        series = series_registry.get(coronal_view.series_id)
        volume = service._get_series_volume(series)
        pose_context = service._build_mpr_pose_context(coronal_view, volume.shape, series=series)
        drag = MprRotationDragRecord(
            viewport="mpr-cor",
            line="vertical",
            start_cursor=service._serialize_mpr_cursor_record(pose_context.cursor),
        )

        service._apply_mpr_rotation_drag(group, drag, float(np.deg2rad(89.0)), pose_context.geometry, volume.shape)
        before = _get_group_plane(service, group, "mpr-cor", volume.shape)
        service._apply_mpr_rotation_drag(group, drag, float(np.deg2rad(91.0)), pose_context.geometry, volume.shape)
        after = _get_group_plane(service, group, "mpr-cor", volume.shape)
        return before, after

    before_plane, after_plane = _run_with_stubbed_mpr_volume(service, run_drag)

    assert float(np.dot(before_plane.row, after_plane.row)) > 0.99
    assert float(np.dot(before_plane.col, after_plane.col)) > 0.99


def test_mpr_oblique_axial_small_rotation_does_not_mirror_coronal_view() -> None:
    service = ViewerService()

    for line in ("horizontal", "vertical"):
        for delta_angle_rad in (-0.01, 0.01):
            group = ViewGroupRecord(group_id=f"g-{line}-{delta_angle_rad}", group_type="MPR", series_id="s")
            axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
            before_plane = _get_group_plane(service, group, "mpr-cor")

            def run_drag():
                series = series_registry.get(axial_view.series_id)
                volume = service._get_series_volume(series)
                pose_context = service._build_mpr_pose_context(axial_view, volume.shape, series=series)
                drag = MprRotationDragRecord(
                    viewport="mpr-ax",
                    line=line,
                    start_cursor=service._serialize_mpr_cursor_record(pose_context.cursor),
                )
                service._apply_mpr_rotation_drag(group, drag, delta_angle_rad, pose_context.geometry, volume.shape)

            _run_with_stubbed_mpr_volume(service, run_drag)

            after_plane = _get_group_plane(service, group, "mpr-cor")
            assert float(np.dot(before_plane.row, after_plane.row)) > 0.999
            assert float(np.dot(before_plane.col, after_plane.col)) > 0.999


def test_mpr_oblique_second_view_crosshair_follows_drag_after_first_view_rotation() -> None:
    service = ViewerService()

    for viewport_key, view_type in (("mpr-cor", "COR"), ("mpr-sag", "SAG")):
        group = ViewGroupRecord(group_id=f"g-{viewport_key}", group_type="MPR", series_id="s")
        axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
        second_view = ViewRecord(view_id=f"v-{viewport_key}", series_id="s", view_type=view_type, view_group=group)

        _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)
        before_horizontal, before_vertical = _get_pose_line_angles(service, second_view, viewport_key)

        def run_drag():
            series = series_registry.get(second_view.series_id)
            volume = service._get_series_volume(series)
            pose_context = service._build_mpr_pose_context(second_view, volume.shape, series=series)
            drag = MprRotationDragRecord(
                viewport=viewport_key,
                line="vertical",
                start_cursor=service._serialize_mpr_cursor_record(pose_context.cursor),
            )
            service._apply_mpr_rotation_drag(group, drag, 0.25, pose_context.geometry, volume.shape)

        _run_with_stubbed_mpr_volume(service, run_drag)

        after_horizontal, after_vertical = _get_pose_line_angles(service, second_view, viewport_key)
        assert math.isclose(_normalize_line_delta(before_vertical, after_vertical), 0.25, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(
            (after_vertical - after_horizontal) % np.pi,
            np.pi / 2.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )


def test_mpr_oblique_axial_then_second_view_rotation_keeps_direction_and_orientation_labels_stable() -> None:
    service = ViewerService()

    for viewport_key, view_type in (("mpr-cor", "COR"), ("mpr-sag", "SAG")):
        for delta_angle_rad in (-0.1, 0.1):
            group = ViewGroupRecord(group_id=f"g-{viewport_key}-{delta_angle_rad}", group_type="MPR", series_id="s")
            axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
            second_view = ViewRecord(view_id=f"v-{viewport_key}", series_id="s", view_type=view_type, view_group=group)

            _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)
            before_horizontal, before_vertical = _get_pose_line_angles(service, second_view, viewport_key)
            before_overlay = _build_group_orientation_overlay(service, second_view, viewport_key)

            def run_drag():
                series = series_registry.get(second_view.series_id)
                volume = service._get_series_volume(series)
                pose_context = service._build_mpr_pose_context(second_view, volume.shape, series=series)
                drag = MprRotationDragRecord(
                    viewport=viewport_key,
                    line="vertical",
                    start_cursor=service._serialize_mpr_cursor_record(pose_context.cursor),
                )
                service._apply_mpr_rotation_drag(group, drag, delta_angle_rad, pose_context.geometry, volume.shape)

            _run_with_stubbed_mpr_volume(service, run_drag)

            after_horizontal, after_vertical = _get_pose_line_angles(service, second_view, viewport_key)
            after_overlay = _build_group_orientation_overlay(service, second_view, viewport_key)
            assert math.isclose(_normalize_line_delta(before_vertical, after_vertical), delta_angle_rad, rel_tol=0.0, abs_tol=1e-6)
            assert math.isclose(
                (after_vertical - after_horizontal) % np.pi,
                np.pi / 2.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            assert after_overlay.top == before_overlay.top
            assert after_overlay.bottom == before_overlay.bottom


def test_mpr_oblique_drag_uses_consistent_screen_delta_sign_across_viewports() -> None:
    service = ViewerService()

    for viewport_key, view_type in (("mpr-ax", "MPR"), ("mpr-cor", "COR"), ("mpr-sag", "SAG")):
        for line in ("horizontal", "vertical"):
            group = ViewGroupRecord(group_id=f"g-{viewport_key}-{line}", group_type="MPR", series_id="s")
            view = ViewRecord(view_id=f"v-{viewport_key}-{line}", series_id="s", view_type=view_type, view_group=group)
            pose_context = service._build_mpr_pose_context(view, (5, 6, 7), series=SimpleNamespace(series_id="s", instances=[]))
            before_horizontal, before_vertical = service._get_mpr_crosshair_line_angles_from_poses(pose_context.poses, viewport_key)
            drag = MprRotationDragRecord(
                viewport=viewport_key,
                line=line,
                start_cursor=service._serialize_mpr_cursor_record(pose_context.cursor),
            )

            service._apply_mpr_rotation_drag(group, drag, 0.1, pose_context.geometry, (5, 6, 7))

            after_horizontal, after_vertical = _get_pose_line_angles(service, view, viewport_key)
            before_angle = before_horizontal if line == "horizontal" else before_vertical
            after_angle = after_horizontal if line == "horizontal" else after_vertical
            assert math.isclose(_normalize_line_delta(before_angle, after_angle), 0.1, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_line_direction_matches_client_screen_angle_convention() -> None:
    current_normal = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    current_row = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    current_col = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    angle_rad = 0.35

    line_dir = mpr_geometry.build_mpr_oblique_line_direction(current_row, current_col, angle_rad, line="horizontal")
    expected_line_dir = np.asarray([0.0, math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float64)
    target_normal = np.cross(line_dir, current_normal)
    target_plane = MprObliquePlaneState(
        row=(0.0, 0.0, 1.0),
        col=(0.0, 1.0, 0.0),
        normal=tuple(float(value) for value in target_normal),
        is_oblique=True,
    )

    resolved_angle = mpr_geometry.resolve_mpr_crosshair_line_angle(
        current_normal,
        current_row,
        current_col,
        target_plane,
        fallback=0.0,
    )

    assert np.allclose(line_dir, expected_line_dir, atol=1e-6)
    assert math.isclose(resolved_angle, angle_rad, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_spacing_uses_resolved_oblique_plane_basis() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    plane = MprObliquePlaneState(
        row=(-1.0, 0.0, 0.0),
        col=(0.0, -math.sqrt(0.5), math.sqrt(0.5)),
        normal=(0.0, math.sqrt(0.5), math.sqrt(0.5)),
        is_oblique=True,
    )
    transform = VolumePatientTransform(
        origin=np.zeros(3, dtype=np.float64),
        axis_vectors=(
            np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 3.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
        ),
        shape=(16, 16, 16),
    )

    original_get_transform = service._get_series_patient_transform
    try:
        service._get_series_patient_transform = lambda _series: transform  # type: ignore[method-assign]
        spacing_x, spacing_y = service._get_mpr_spacing_xy(series, "mpr-cor", plane)
    finally:
        service._get_series_patient_transform = original_get_transform  # type: ignore[method-assign]

    assert math.isclose(spacing_x, math.sqrt(17.0), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(spacing_y, 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_oblique_corner_info_marks_viewport_and_projects_relative_mm_location() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (2.0, 3.0, 5.0), (16, 16, 16))
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    plane = MprObliquePlaneState(
        row=(-1.0, 0.0, 0.0),
        col=(0.0, -math.sqrt(0.5), math.sqrt(0.5)),
        normal=(0.0, math.sqrt(0.5), math.sqrt(0.5)),
        is_oblique=True,
    )
    transform = VolumePatientTransform(
        origin=np.zeros(3, dtype=np.float64),
        axis_vectors=(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ),
        shape=(16, 16, 16),
    )

    original_get_transform = service._get_series_patient_transform
    try:
        service._get_series_patient_transform = lambda _series: transform  # type: ignore[method-assign]
        pose_context = service._build_mpr_pose_context(coronal_view, (16, 16, 16), series=series)
        corner_info = service._build_slice_corner_info_overlay(
            coronal_view,
            series,
            None,
            current_index=3,
            total_slices=8,
            viewport_label=service._build_mpr_viewport_label("mpr-cor", plane),
            plane_state=plane,
            plane_pose=pose_context.poses["mpr-cor"],
            cursor=pose_context.cursor,
        )
    finally:
        service._get_series_patient_transform = original_get_transform  # type: ignore[method-assign]

    assert corner_info.top_left[0] == "OBLIQUE CORONAL  P 3.00mm"


def test_mpr_oblique_corner_info_uses_single_axis_mm_label_without_patient_transform() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (20.0, 30.0, 10.0), (32, 32, 32))
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    plane = MprObliquePlaneState(
        row=(-1.0, 0.0, 0.0),
        col=(0.0, 0.0, 1.0),
        normal=(0.0, math.sqrt(0.5), -math.sqrt(0.5)),
        is_oblique=True,
    )

    original_get_transform = service._get_series_patient_transform
    try:
        service._get_series_patient_transform = lambda _series: None  # type: ignore[method-assign]
        pose_context = service._build_mpr_pose_context(coronal_view, (32, 32, 32), series=series)
        corner_info = service._build_slice_corner_info_overlay(
            coronal_view,
            series,
            None,
            current_index=3,
            total_slices=8,
            viewport_label=service._build_mpr_viewport_label("mpr-cor", plane),
            plane_state=plane,
            plane_pose=pose_context.poses["mpr-cor"],
            cursor=pose_context.cursor,
        )
    finally:
        service._get_series_patient_transform = original_get_transform  # type: ignore[method-assign]

    assert corner_info.top_left[0] == "OBLIQUE CORONAL"


def test_mpr_oblique_corner_info_is_stable_for_repeated_drag_delta() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])

    original_get_transform = service._get_series_patient_transform
    try:
        service._get_series_patient_transform = lambda _series: None  # type: ignore[method-assign]
        resolved_labels: dict[tuple[float, int], str] = {}
        for delta_degrees in (-45.0, 45.0):
            for attempt in (0, 1):
                group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
                _set_group_center(service, group, (2.0, 3.0, 3.0))
                axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
                coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

                _run_with_stubbed_mpr_volume(
                    service,
                    lambda: (
                        service._handle_mpr_oblique(
                            axial_view,
                            ViewOperationRequest(
                                viewId=axial_view.view_id,
                                opType="mprOblique",
                                actionType="start",
                                line="vertical",
                                deltaAngleRad=0.0,
                            ),
                        ),
                        service._handle_mpr_oblique(
                            axial_view,
                            ViewOperationRequest(
                                viewId=axial_view.view_id,
                                opType="mprOblique",
                                actionType="move",
                                line="vertical",
                                deltaAngleRad=float(np.deg2rad(delta_degrees)),
                            ),
                        ),
                    ),
                )
                pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=series)
                plane = service._plane_state_from_pose(pose_context.poses["mpr-cor"])
                corner_info = service._build_slice_corner_info_overlay(
                    coronal_view,
                    series,
                    None,
                    current_index=3,
                    total_slices=8,
                    viewport_label=service._build_mpr_viewport_label("mpr-cor", plane),
                    plane_state=plane,
                    plane_pose=pose_context.poses["mpr-cor"],
                    cursor=pose_context.cursor,
                )

                resolved_labels[(delta_degrees, attempt)] = corner_info.top_left[0]

        assert resolved_labels[(-45.0, 0)] == resolved_labels[(-45.0, 1)]
        assert resolved_labels[(45.0, 0)] == resolved_labels[(45.0, 1)]
    finally:
        service._get_series_patient_transform = original_get_transform  # type: ignore[method-assign]


def test_mpr_oblique_corner_info_uses_initial_center_as_distance_origin_after_center_move() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (2.0, 4.0, 3.0))
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    original_get_transform = service._get_series_patient_transform
    try:
        service._get_series_patient_transform = lambda _series: None  # type: ignore[method-assign]
        _apply_oblique_drag(
            service,
            axial_view,
            line="vertical",
            angle_rad=float(np.deg2rad(-45.0)),
        )
        pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=series)
        plane = service._plane_state_from_pose(pose_context.poses["mpr-cor"])
        corner_info = service._build_slice_corner_info_overlay(
            coronal_view,
            series,
            None,
            current_index=4,
            total_slices=8,
            viewport_label=service._build_mpr_viewport_label("mpr-cor", plane),
            plane_state=plane,
            plane_pose=pose_context.poses["mpr-cor"],
            cursor=pose_context.cursor,
        )
    finally:
        service._get_series_patient_transform = original_get_transform  # type: ignore[method-assign]

    assert corner_info.top_left[0] != "OBLIQUE CORONAL  P 0mm"
    assert corner_info.top_left[0].startswith("OBLIQUE CORONAL  ")
    assert corner_info.top_left[0].endswith("0.71mm")


def test_mpr_oblique_corner_info_keeps_anterior_label_after_axial_rotation_and_center_move() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    _set_group_center(service, group, (2.0, 3.0, 3.0))
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    original_get_transform = service._get_series_patient_transform
    try:
        service._get_series_patient_transform = lambda _series: None  # type: ignore[method-assign]
        _apply_oblique_drag(
            service,
            axial_view,
            line="vertical",
            angle_rad=float(np.deg2rad(-30.0)),
        )
        _set_group_center(service, group, (2.0, 2.0, 3.0))
        pose_context = service._build_mpr_pose_context(coronal_view, (5, 6, 7), series=series)
        plane = service._plane_state_from_pose(pose_context.poses["mpr-cor"])
        corner_info = service._build_slice_corner_info_overlay(
            coronal_view,
            series,
            None,
            current_index=2,
            total_slices=8,
            viewport_label=service._build_mpr_viewport_label("mpr-cor", plane),
            plane_state=plane,
            plane_pose=pose_context.poses["mpr-cor"],
            cursor=pose_context.cursor,
        )
    finally:
        service._get_series_patient_transform = original_get_transform  # type: ignore[method-assign]

    assert corner_info.top_left[0] == "OBLIQUE CORONAL  L 0.5mm"


def test_mpr_axial_standard_location_uses_inferior_for_positive_patient_z() -> None:
    service = ViewerService()

    assert service._format_standard_physical_location("axial", np.asarray([0.0, 0.0, 1476.88])) == "I 1476.88mm"
