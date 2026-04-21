import math
from types import SimpleNamespace

import numpy as np

from app.models.viewer import ViewGroupRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
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
    return _run_with_stubbed_mpr_volume(service, lambda: (
        service._handle_mpr_oblique(
            view,
            ViewOperationRequest(viewId=view.view_id, opType="mprOblique", actionType="start", line=line, angleRad=angle_rad),
        ),
        service._handle_mpr_oblique(
            view,
            ViewOperationRequest(viewId=view.view_id, opType="mprOblique", actionType="move", line=line, angleRad=angle_rad),
        ),
    ))


def test_mpr_oblique_drag_updates_target_plane_and_reslices() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    view.mpr_axial_index = 2
    view.mpr_coronal_index = 3
    view.mpr_sagittal_index = 4
    initial_frame = group.mpr_frame

    start = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprOblique",
        actionType="start",
        line="horizontal",
        angleRad=0.0,
    )
    move = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprOblique",
        actionType="move",
        line="horizontal",
        angleRad=0.35,
    )

    start_result, move_result = _run_with_stubbed_mpr_volume(service, lambda: (
        service._handle_mpr_oblique(view, start),
        service._handle_mpr_oblique(view, move),
    ))

    assert start_result is False
    assert move_result is True
    assert group.oblique_line_angles["mpr-ax"]["horizontal"] == 0.35
    assert math.isclose(group.oblique_line_angles["mpr-ax"]["vertical"], 0.35 + np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert group.oblique_planes["mpr-ax"].is_oblique is False
    assert group.oblique_planes["mpr-cor"].is_oblique is True
    assert group.oblique_planes["mpr-sag"].is_oblique is True
    assert group.mpr_frame == initial_frame
    axial_normal = np.asarray(group.oblique_planes["mpr-ax"].normal, dtype=np.float64)
    coronal_normal = np.asarray(group.oblique_planes["mpr-cor"].normal, dtype=np.float64)
    sagittal_normal = np.asarray(group.oblique_planes["mpr-sag"].normal, dtype=np.float64)
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
    assert math.isclose(
        (group.oblique_line_angles["mpr-cor"]["vertical"] - group.oblique_line_angles["mpr-cor"]["horizontal"]) % np.pi,
        np.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        (group.oblique_line_angles["mpr-sag"]["vertical"] - group.oblique_line_angles["mpr-sag"]["horizontal"]) % np.pi,
        np.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


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

    assert math.isclose(group.oblique_line_angles["mpr-ax"]["vertical"], 1.2, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        group.oblique_line_angles["mpr-ax"]["horizontal"],
        (1.2 - np.pi / 2.0) % np.pi,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_mpr_oblique_drag_preserves_target_view_column_orientation() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)

    coronal_normal = np.asarray(group.oblique_planes["mpr-cor"].normal, dtype=np.float64)
    sagittal_normal = np.asarray(group.oblique_planes["mpr-sag"].normal, dtype=np.float64)
    coronal_col = np.asarray(group.oblique_planes["mpr-cor"].col, dtype=np.float64)
    sagittal_col = np.asarray(group.oblique_planes["mpr-sag"].col, dtype=np.float64)

    assert float(np.dot(coronal_col, sagittal_normal)) > 0.999
    assert float(np.dot(sagittal_col, coronal_normal)) > 0.999


def test_mpr_oblique_drag_preserves_target_plane_direction_on_small_moves() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    previous_normal = np.asarray(group.oblique_planes["mpr-sag"].normal, dtype=np.float64)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2005)
    next_normal = np.asarray(group.oblique_planes["mpr-sag"].normal, dtype=np.float64)

    assert float(np.dot(previous_normal, next_normal)) > 0.999


def test_mpr_crosshair_move_uses_oblique_plane_basis_after_rotation() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    group.mpr_frame.center = (2.0, 3.0, 3.0)
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    previous_center = np.asarray(group.mpr_frame.center, dtype=np.float64)
    coronal_plane = group.oblique_planes["mpr-cor"]
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
        service._get_mpr_display_aspect_xy = lambda _series, _viewport: (1.0, 1.0)  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = lambda _transform, _canvas_x, _canvas_y: next(image_points)  # type: ignore[method-assign]
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="start", x=0.5, y=0.5))
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.75))
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = original_canvas_to_image  # type: ignore[method-assign]

    assert np.allclose(np.asarray(group.mpr_frame.center, dtype=np.float64), expected_center, atol=1e-6)


def test_mpr_crosshair_move_preserves_active_view_oblique_angles() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    group.mpr_frame.center = (2.0, 3.0, 3.0)
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group, width=700, height=500)
    volume_shape = (5, 6, 7)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)
    expected_horizontal = group.oblique_line_angles["mpr-ax"]["horizontal"]
    expected_vertical = group.oblique_line_angles["mpr-ax"]["vertical"]

    original_get_volume = service._get_series_volume
    original_series_get = series_registry.get
    original_get_mpr_aspect = service._get_mpr_display_aspect_xy
    image_points = iter(((3.5, 2.5), (4.5, 3.0)))
    original_canvas_to_image = service._canvas_to_image_coordinates
    try:
        series_registry.get = lambda _series_id: SimpleNamespace(series_id="s", instances=[])  # type: ignore[method-assign]
        service._get_series_volume = lambda _series: np.zeros(volume_shape, dtype=np.float32)  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = lambda _series, _viewport: (1.0, 1.0)  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = lambda _transform, _canvas_x, _canvas_y: next(image_points)  # type: ignore[method-assign]
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="start", x=0.5, y=0.5))
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.5))
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = original_canvas_to_image  # type: ignore[method-assign]

    assert math.isclose(group.oblique_line_angles["mpr-ax"]["horizontal"], expected_horizontal, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(group.oblique_line_angles["mpr-ax"]["vertical"], expected_vertical, rel_tol=0.0, abs_tol=1e-6)


def test_mpr_crosshair_move_does_not_accumulate_absolute_offset_every_frame() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    group.mpr_frame.center = (2.0, 3.0, 3.0)
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
        service._get_mpr_display_aspect_xy = lambda _series, _viewport: (1.0, 1.0)  # type: ignore[method-assign]
        service._canvas_to_image_coordinates = lambda _transform, _canvas_x, _canvas_y: next(image_points)  # type: ignore[method-assign]
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="start", x=0.5, y=0.5))
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.5))
        center_after_first_move = np.asarray(group.mpr_frame.center, dtype=np.float64)
        service._handle_mpr_crosshair(coronal_view, ViewOperationRequest(viewId=coronal_view.view_id, opType="crosshair", actionType="move", x=0.75, y=0.5))
        center_after_second_move = np.asarray(group.mpr_frame.center, dtype=np.float64)
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
    group.crosshair_drag_active = True
    group.oblique_drag_active = True
    group.oblique_planes["mpr-cor"].is_oblique = True
    group.oblique_planes["mpr-sag"].is_oblique = True
    group.oblique_line_angles["mpr-ax"]["horizontal"] = 0.4
    group.oblique_line_angles["mpr-ax"]["vertical"] = 1.9
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
        service._get_mpr_display_aspect_xy = lambda _series, _viewport: (1.0, 1.0)  # type: ignore[method-assign]
        service._reset_mpr_view_group(axial_view)
    finally:
        series_registry.get = original_series_get  # type: ignore[method-assign]
        service._get_mpr_group_views = original_get_group_views  # type: ignore[method-assign]
        service._get_series_volume = original_get_volume  # type: ignore[method-assign]
        service._get_mpr_display_aspect_xy = original_get_mpr_aspect  # type: ignore[method-assign]

    assert group.axial_index == 4
    assert group.coronal_index == 5
    assert group.sagittal_index == 6
    assert group.mpr_frame.center == (4.0, 5.0, 6.0)
    assert group.mpr_frame.axis_slice == (1.0, 0.0, 0.0)
    assert group.mpr_frame.axis_row == (0.0, 1.0, 0.0)
    assert group.mpr_frame.axis_col == (0.0, 0.0, 1.0)
    assert group.crosshair_drag_active is False
    assert group.oblique_drag_active is False
    assert group.oblique_planes["mpr-cor"].is_oblique is False
    assert group.oblique_planes["mpr-sag"].is_oblique is False
    assert math.isclose(group.oblique_line_angles["mpr-ax"]["horizontal"], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(group.oblique_line_angles["mpr-ax"]["vertical"], np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)
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
    first_axial_horizontal = group.oblique_line_angles["mpr-ax"]["horizontal"]
    first_axial_vertical = group.oblique_line_angles["mpr-ax"]["vertical"]

    _apply_oblique_drag(service, coronal_view, line="vertical", angle_rad=1.2)

    assert not math.isclose(first_axial_horizontal, 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        (first_axial_vertical - first_axial_horizontal) % np.pi,
        np.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert group.oblique_planes["mpr-ax"].is_oblique is True
    assert group.oblique_planes["mpr-cor"].is_oblique is True
    assert group.oblique_planes["mpr-sag"].is_oblique is True


def test_mpr_orientation_overlay_updates_after_oblique_rotation() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)

    initial_overlay = service._build_mpr_orientation_overlay(coronal_view, "mpr-cor")
    assert initial_overlay.top == "S"
    assert initial_overlay.bottom == "I"
    assert initial_overlay.left == "R"
    assert initial_overlay.right == "L"

    _apply_oblique_drag(service, axial_view, line="horizontal", angle_rad=0.35)

    rotated_overlay = service._build_mpr_orientation_overlay(coronal_view, "mpr-cor")
    assert rotated_overlay.left == "R"
    assert rotated_overlay.right == "L"
    assert rotated_overlay.top != initial_overlay.top
    assert rotated_overlay.bottom != initial_overlay.bottom
    assert rotated_overlay.top == "SA"
    assert rotated_overlay.bottom == "IP"


def test_mpr_orientation_overlay_flips_target_direction_for_axial_vertical_drag() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    axial_view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id="s", view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id="s", view_type="SAG", view_group=group)

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.2)

    coronal_overlay = service._build_mpr_orientation_overlay(coronal_view, "mpr-cor")
    sagittal_overlay = service._build_mpr_orientation_overlay(sagittal_view, "mpr-sag")
    assert coronal_overlay.top == "S"
    assert coronal_overlay.right == "LP"
    assert coronal_overlay.bottom == "I"
    assert coronal_overlay.left == "RA"
    assert sagittal_overlay.top == "S"
    assert sagittal_overlay.right == "PR"
    assert sagittal_overlay.bottom == "I"
    assert sagittal_overlay.left == "AL"

    _apply_oblique_drag(service, axial_view, line="vertical", angle_rad=1.8)

    coronal_overlay = service._build_mpr_orientation_overlay(coronal_view, "mpr-cor")
    sagittal_overlay = service._build_mpr_orientation_overlay(sagittal_view, "mpr-sag")
    assert coronal_overlay.top == "S"
    assert coronal_overlay.right == "LA"
    assert coronal_overlay.bottom == "I"
    assert coronal_overlay.left == "RP"
    assert sagittal_overlay.top == "S"
    assert sagittal_overlay.right == "PL"
    assert sagittal_overlay.bottom == "I"
    assert sagittal_overlay.left == "AR"
