import math
from types import SimpleNamespace

import numpy as np

from app.models.viewer import ViewGroupRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.viewer_service import ViewerService
from app.services.series_registry import series_registry


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


def test_mpr_oblique_drag_updates_target_plane_and_reslices() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    view.mpr_axial_index = 2
    view.mpr_coronal_index = 3
    view.mpr_sagittal_index = 4

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
    assert group.oblique_planes["mpr-cor"].is_oblique is True
    assert group.oblique_planes["mpr-sag"].is_oblique is True

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

    _run_with_stubbed_mpr_volume(service, lambda: (
        service._handle_mpr_oblique(
            axial_view,
            ViewOperationRequest(viewId=axial_view.view_id, opType="mprOblique", actionType="start", line="horizontal", angleRad=0.35),
        ),
        service._handle_mpr_oblique(
            axial_view,
            ViewOperationRequest(viewId=axial_view.view_id, opType="mprOblique", actionType="move", line="horizontal", angleRad=0.35),
        ),
    ))
    first_axial_horizontal = group.oblique_line_angles["mpr-ax"]["horizontal"]
    first_axial_vertical = group.oblique_line_angles["mpr-ax"]["vertical"]

    _run_with_stubbed_mpr_volume(service, lambda: (
        service._handle_mpr_oblique(
            coronal_view,
            ViewOperationRequest(viewId=coronal_view.view_id, opType="mprOblique", actionType="start", line="vertical", angleRad=1.2),
        ),
        service._handle_mpr_oblique(
            coronal_view,
            ViewOperationRequest(viewId=coronal_view.view_id, opType="mprOblique", actionType="move", line="vertical", angleRad=1.2),
        ),
    ))

    assert not math.isclose(first_axial_horizontal, 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert not math.isclose(group.oblique_line_angles["mpr-ax"]["horizontal"], 0.0, rel_tol=0.0, abs_tol=1e-6)
    assert not math.isclose(group.oblique_line_angles["mpr-ax"]["vertical"], np.pi / 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        (group.oblique_line_angles["mpr-ax"]["vertical"] - group.oblique_line_angles["mpr-ax"]["horizontal"]) % np.pi,
        np.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        (first_axial_vertical - first_axial_horizontal) % np.pi,
        np.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert not np.allclose(group.mpr_frame.axis_slice, np.array([1.0, 0.0, 0.0], dtype=np.float64), atol=1e-6)
    assert not np.allclose(group.mpr_frame.axis_row, np.array([0.0, 1.0, 0.0], dtype=np.float64), atol=1e-6)


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

    _run_with_stubbed_mpr_volume(service, lambda: (
        service._handle_mpr_oblique(
            axial_view,
            ViewOperationRequest(viewId=axial_view.view_id, opType="mprOblique", actionType="start", line="horizontal", angleRad=0.35),
        ),
        service._handle_mpr_oblique(
            axial_view,
            ViewOperationRequest(viewId=axial_view.view_id, opType="mprOblique", actionType="move", line="horizontal", angleRad=0.35),
        ),
    ))

    rotated_overlay = service._build_mpr_orientation_overlay(coronal_view, "mpr-cor")
    assert rotated_overlay.left == "R"
    assert rotated_overlay.right == "L"
    assert rotated_overlay.top != initial_overlay.top
    assert rotated_overlay.bottom != initial_overlay.bottom
    assert rotated_overlay.top == "SP"
    assert rotated_overlay.bottom == "IA"
