from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL, VIEW_OP_TYPE_ROTATE_3D
from app.models.measurement import MeasurementPoint
from app.models.viewer import MprMipViewportState, ViewGroupRecord, ViewRecord
from app.schemas.dicom import LoadFolderRequest
from app.schemas.view import ViewCreateRequest, ViewOperationRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service
from app.services.mpr import (
    axis_angle_rotation_matrix,
    plane_image_point_to_world,
    world_point_to_plane_image,
    world_to_ijk_point,
)
from app.services.volume_rendering.camera_math import quaternion_to_rotation_matrix
from tests.support.dicom_phantoms import build_asymmetric_landmark_volume, write_ct_series


def _clear_viewer_state() -> None:
    view_registry._view_by_id.clear()
    for group in view_group_registry.list_all():
        view_group_registry.delete(group.group_id)
    viewer_service._series_volume_cache.clear()
    viewer_service._series_patient_transform_cache.clear()
    viewer_service._series_volume_geometry_cache.clear()
    viewer_service._mpr_plane_cache.clear()
    series_registry.clear()
    dicom_cache.clear()


@pytest.fixture(autouse=True)
def clear_viewer_state():
    _clear_viewer_state()
    try:
        yield
    finally:
        _clear_viewer_state()


def _load_synthetic_series(
    root: Path,
    volume: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    orientation: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    origin_patient_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
):
    write_ct_series(
        root,
        volume,
        spacing_zyx_mm=spacing,
        orientation=orientation,
        origin_patient_mm=origin_patient_mm,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        file_order=tuple(reversed(range(volume.shape[0]))),
    )
    loaded = series_registry.load_folder(LoadFolderRequest(folderPath=str(root)))
    return series_registry.get(loaded.series_list[0].series_id)


def _build_mpr_views_for_series(series, volume: np.ndarray) -> dict[str, ViewRecord]:
    group = ViewGroupRecord(group_id="physical-orientation", group_type="MPR", series_id=series.series_id)
    views = {
        viewport: ViewRecord(
            view_id=f"physical-orientation-{viewport.lower()}",
            series_id=series.series_id,
            view_type=view_type,
            view_group=group,
            width=360,
            height=280,
        )
        for viewport, view_type in (
            (MPR_VIEWPORT_AXIAL, "AX"),
            (MPR_VIEWPORT_CORONAL, "COR"),
            (MPR_VIEWPORT_SAGITTAL, "SAG"),
        )
    }
    viewer_service._reset_mpr_group_geometry(group, volume.shape, series=series)
    return views


def _mpr_orientation_labels(view: ViewRecord, series, volume: np.ndarray) -> tuple[str | None, ...]:
    viewport = viewer_service._resolve_mpr_viewport(view)
    pose = viewer_service._build_mpr_pose_context(view, volume.shape, series=series).poses[viewport]
    overlay = viewer_service._build_mpr_orientation_overlay(
        view,
        viewport,
        viewer_service._plane_state_from_pose(pose),
        plane_pose=pose,
    )
    return overlay.top, overlay.right, overlay.bottom, overlay.left


def test_rotated_acquisition_mpr_four_edge_labels_follow_screen_rotation(tmp_path: Path) -> None:
    angle = math.radians(30.0)
    orientation = (
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    )
    volume = np.zeros((5, 7, 9), dtype=np.int16)
    series = _load_synthetic_series(
        tmp_path,
        volume,
        spacing=(3.0, 2.0, 0.5),
        orientation=orientation,
        origin_patient_mm=(12.0, -25.0, 40.0),
    )
    built_volume = viewer_service._build_series_volume(series)
    views = _build_mpr_views_for_series(series, built_volume)
    expected_by_viewport = {
        MPR_VIEWPORT_AXIAL: {
            0: ("A", "L", "P", "R"),
            90: ("L", "P", "R", "A"),
            180: ("P", "R", "A", "L"),
            270: ("R", "A", "L", "P"),
        },
        MPR_VIEWPORT_CORONAL: {
            0: ("S", "L", "I", "R"),
            90: ("L", "I", "R", "S"),
            180: ("I", "R", "S", "L"),
            270: ("R", "S", "L", "I"),
        },
        MPR_VIEWPORT_SAGITTAL: {
            0: ("S", "P", "I", "A"),
            90: ("P", "I", "A", "S"),
            180: ("I", "A", "S", "P"),
            270: ("A", "S", "P", "I"),
        },
    }

    for viewport, view in views.items():
        for rotation_degrees, expected_labels in expected_by_viewport[viewport].items():
            view.rotation_degrees = rotation_degrees
            assert _mpr_orientation_labels(view, series, built_volume) == expected_labels


def test_rotated_acquisition_model_rotation_updates_all_mpr_edge_labels_exactly(tmp_path: Path) -> None:
    angle = math.radians(30.0)
    orientation = (
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    )
    volume = np.zeros((5, 7, 9), dtype=np.int16)
    series = _load_synthetic_series(
        tmp_path,
        volume,
        spacing=(3.0, 2.0, 0.5),
        orientation=orientation,
        origin_patient_mm=(12.0, -25.0, 40.0),
    )
    built_volume = viewer_service._build_series_volume(series)
    views = _build_mpr_views_for_series(series, built_volume)
    axial_pose = viewer_service._build_mpr_pose_context(
        views[MPR_VIEWPORT_AXIAL],
        built_volume.shape,
        series=series,
    ).poses[MPR_VIEWPORT_AXIAL]
    group = views[MPR_VIEWPORT_AXIAL].view_group
    assert group is not None
    viewer_service._set_mpr_model_rotation_matrix(
        group,
        axis_angle_rotation_matrix(np.asarray(axial_pose.normal_world), np.pi / 2.0),
        pivot_world=axial_pose.cursor_center_world,
    )

    assert _mpr_orientation_labels(views[MPR_VIEWPORT_AXIAL], series, built_volume) == ("A", "L", "P", "R")
    assert _mpr_orientation_labels(views[MPR_VIEWPORT_CORONAL], series, built_volume) == ("S", "AL", "I", "PR")
    assert _mpr_orientation_labels(views[MPR_VIEWPORT_SAGITTAL], series, built_volume) == ("S", "LP", "I", "RA")


def test_rotated_acquisition_double_oblique_crosshair_plane_uses_compound_patient_directions(
    tmp_path: Path,
) -> None:
    angle = math.radians(30.0)
    orientation = (
        math.cos(angle),
        math.sin(angle),
        0.0,
        -math.sin(angle),
        math.cos(angle),
        0.0,
    )
    volume = np.zeros((7, 9, 11), dtype=np.int16)
    series = _load_synthetic_series(
        tmp_path,
        volume,
        spacing=(2.5, 1.0, 0.5),
        orientation=orientation,
        origin_patient_mm=(12.0, -25.0, 40.0),
    )
    built_volume = viewer_service._build_series_volume(series)
    views = _build_mpr_views_for_series(series, built_volume)
    axial_view = views[MPR_VIEWPORT_AXIAL]
    initial_pose = viewer_service._build_mpr_pose_context(
        axial_view,
        built_volume.shape,
        series=series,
    ).poses[MPR_VIEWPORT_AXIAL]
    oblique_normal = np.asarray(initial_pose.normal_world) + np.asarray(initial_pose.col_world)
    oblique_normal /= np.linalg.norm(oblique_normal)
    group = axial_view.view_group
    assert group is not None
    group.mpr_crosshair_mode = "double-oblique"
    group.mpr_independent_plane_normals[MPR_VIEWPORT_AXIAL] = tuple(float(value) for value in oblique_normal)

    oblique_pose = viewer_service._build_mpr_pose_context(
        axial_view,
        built_volume.shape,
        series=series,
    ).poses[MPR_VIEWPORT_AXIAL]
    np.testing.assert_allclose(oblique_pose.normal_world, oblique_normal, atol=1e-6)
    assert _mpr_orientation_labels(axial_view, series, built_volume) == ("AL", "IL", "PR", "SR")


def test_rotated_acquisition_preserves_patient_geometry_and_physical_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    angle = math.radians(30.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    row_direction = np.asarray([cosine, sine, 0.0], dtype=np.float64)
    column_direction = np.asarray([-sine, cosine, 0.0], dtype=np.float64)
    slice_direction = np.cross(row_direction, column_direction)
    orientation = tuple(float(value) for value in np.concatenate((row_direction, column_direction)))
    origin = (12.0, -25.0, 40.0)
    spacing = (3.0, 2.0, 0.5)
    volume = np.zeros((5, 7, 9), dtype=np.int16)
    series = _load_synthetic_series(
        tmp_path,
        volume,
        spacing=spacing,
        orientation=orientation,
        origin_patient_mm=origin,
    )
    built_volume = viewer_service._build_series_volume(series)
    geometry = viewer_service._get_series_volume_geometry(series, built_volume.shape)

    np.testing.assert_allclose(geometry.ijk_to_world[:3, 0], slice_direction * spacing[0], atol=1e-6)
    np.testing.assert_allclose(geometry.ijk_to_world[:3, 1], column_direction * spacing[1], atol=1e-6)
    np.testing.assert_allclose(geometry.ijk_to_world[:3, 2], row_direction * spacing[2], atol=1e-6)
    np.testing.assert_allclose(geometry.ijk_to_world[:3, 3], origin, atol=1e-6)

    created = view_registry.create(
        ViewCreateRequest(seriesId=series.series_id, viewType="AX", viewGroupKey="rotated-acquisition-truth")
    )
    view = view_registry.get(created.view_id)
    assert view.view_group is not None
    viewer_service._reset_mpr_group_geometry(view.view_group, built_volume.shape, series=series)
    pose = viewer_service._build_mpr_pose_context(view, built_volume.shape, series=series).poses[MPR_VIEWPORT_AXIAL]
    center_x, center_y = world_point_to_plane_image(pose, pose.cursor_center_world)
    points = (
        MeasurementPoint(x=center_x, y=center_y),
        MeasurementPoint(x=center_x + 4.0, y=center_y),
    )
    monkeypatch.setattr(viewer_service, "_resolve_measurement_image_points", lambda active_view, payload: points)
    assert viewer_service._handle_measurement(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="measurement",
            subOpType="line",
            actionType="end",
            measurementId="rotated-acquisition-line",
            points=[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
        ),
    )

    [measurement] = view.measurements
    measured_world_delta = np.asarray(measurement.world_points[1]) - np.asarray(measurement.world_points[0])
    np.testing.assert_allclose(measured_world_delta, row_direction * 4.0 * spacing[2], atol=1e-6)
    assert measurement.metrics.length == pytest.approx(2.0)


def test_real_dicom_roi_reprojects_and_recomputes_statistics_on_oblique_plane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zz, yy, xx = np.indices((7, 9, 11), dtype=np.int16)
    volume = 100 * zz + 10 * yy + xx
    series = _load_synthetic_series(tmp_path, volume, spacing=(1.0, 1.0, 1.0))
    built_volume = viewer_service._build_series_volume(series)
    geometry = viewer_service._get_series_volume_geometry(series, built_volume.shape)
    created = view_registry.create(
        ViewCreateRequest(seriesId=series.series_id, viewType="AX", viewGroupKey="oblique-roi-truth")
    )
    view = view_registry.get(created.view_id)
    assert view.view_group is not None
    viewer_service._reset_mpr_group_geometry(view.view_group, built_volume.shape, series=series)
    initial_pose = viewer_service._build_mpr_pose_context(view, built_volume.shape, series=series).poses[MPR_VIEWPORT_AXIAL]
    center_x, center_y = world_point_to_plane_image(initial_pose, initial_pose.cursor_center_world)
    points = (
        MeasurementPoint(x=center_x - 2.0, y=center_y - 1.0),
        MeasurementPoint(x=center_x + 2.0, y=center_y + 1.0),
    )
    monkeypatch.setattr(viewer_service, "_resolve_measurement_image_points", lambda active_view, payload: points)
    assert viewer_service._handle_measurement(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="measurement",
            subOpType="rect",
            actionType="end",
            measurementId="oblique-roi",
            points=[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
        ),
    )
    [before] = viewer_service._build_visible_measurements(view)

    oblique_normal = np.asarray(initial_pose.normal_world) + np.asarray(initial_pose.col_world)
    oblique_normal /= np.linalg.norm(oblique_normal)
    view.view_group.mpr_crosshair_mode = "double-oblique"
    view.view_group.mpr_independent_plane_normals[MPR_VIEWPORT_AXIAL] = tuple(float(value) for value in oblique_normal)
    [after] = viewer_service._build_visible_measurements(view)
    oblique_pose = viewer_service._build_mpr_pose_context(view, built_volume.shape, series=series).poses[MPR_VIEWPORT_AXIAL]

    left = min(round(point.x) for point in after.points)
    right = max(round(point.x) for point in after.points)
    top = min(round(point.y) for point in after.points)
    bottom = max(round(point.y) for point in after.points)
    expected_values = []
    for image_y in range(top, bottom + 1):
        for image_x in range(left, right + 1):
            world = plane_image_point_to_world(oblique_pose, (image_x, image_y))
            voxel_zyx = world_to_ijk_point(geometry, world)
            expected_values.append(100.0 * voxel_zyx[0] + 10.0 * voxel_zyx[1] + voxel_zyx[2])
    expected = np.asarray(expected_values, dtype=np.float64)

    assert after.points != before.points
    assert after.metrics.minimum != pytest.approx(before.metrics.minimum)
    assert after.metrics.maximum != pytest.approx(before.metrics.maximum)
    assert after.metrics.mean == pytest.approx(float(np.mean(expected)), abs=1e-2)
    assert after.metrics.standard_deviation == pytest.approx(float(np.std(expected)), abs=1e-2)
    assert after.metrics.minimum == pytest.approx(float(np.min(expected)), abs=1e-2)
    assert after.metrics.maximum == pytest.approx(float(np.max(expected)), abs=1e-2)


def test_real_dicom_rescale_transform_drives_mpr_pixels_and_roi_hu_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zz, yy, xx = np.indices((5, 7, 9), dtype=np.int16)
    stored_truth = 20 * zz + 3 * yy + xx
    expected_hu = stored_truth.astype(np.float32) * 2.0 - 1024.0
    series = _load_synthetic_series(
        tmp_path,
        expected_hu,
        spacing=(2.0, 1.0, 0.5),
        rescale_slope=2.0,
        rescale_intercept=-1024.0,
    )
    built_volume = viewer_service._build_series_volume(series)
    np.testing.assert_array_equal(built_volume, expected_hu)

    created = view_registry.create(
        ViewCreateRequest(seriesId=series.series_id, viewType="AX", viewGroupKey="rescale-roi-truth")
    )
    view = view_registry.get(created.view_id)
    assert view.view_group is not None
    viewer_service._reset_mpr_group_geometry(view.view_group, built_volume.shape, series=series)
    plane, _current, _total = viewer_service._extract_mpr_plane(
        view,
        built_volume,
        MPR_VIEWPORT_AXIAL,
        interpolation_order=0,
    )
    np.testing.assert_array_equal(plane, expected_hu[expected_hu.shape[0] // 2])

    points = (MeasurementPoint(x=1.0, y=2.0), MeasurementPoint(x=5.0, y=4.0))
    monkeypatch.setattr(viewer_service, "_resolve_measurement_image_points", lambda active_view, payload: points)
    assert viewer_service._handle_measurement(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="measurement",
            subOpType="rect",
            actionType="end",
            measurementId="rescaled-hu-roi",
            points=[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
        ),
    )
    [measurement] = viewer_service._build_visible_measurements(view)
    expected_roi = expected_hu[expected_hu.shape[0] // 2, 2:5, 1:6]

    assert measurement.metrics.mean == pytest.approx(float(np.mean(expected_roi)))
    assert measurement.metrics.standard_deviation == pytest.approx(float(np.std(expected_roi)))
    assert measurement.metrics.minimum == pytest.approx(float(np.min(expected_roi)))
    assert measurement.metrics.maximum == pytest.approx(float(np.max(expected_roi)))


def test_real_dicom_mpr_measurements_use_each_plane_physical_spacing(tmp_path: Path, monkeypatch) -> None:
    volume = np.zeros((5, 6, 8), dtype=np.int16)
    series = _load_synthetic_series(tmp_path, volume, spacing=(3.0, 2.0, 0.5))
    built_volume = viewer_service._build_series_volume(series)
    np.testing.assert_array_equal(built_volume, volume)

    expected_spacing = {
        "AX": (MPR_VIEWPORT_AXIAL, 0.5, 2.0),
        "COR": (MPR_VIEWPORT_CORONAL, 0.5, 3.0),
        "SAG": (MPR_VIEWPORT_SAGITTAL, 2.0, 3.0),
    }
    for view_type, (viewport, spacing_x, spacing_y) in expected_spacing.items():
        created = view_registry.create(
            ViewCreateRequest(seriesId=series.series_id, viewType=view_type, viewGroupKey="measurement-truth")
        )
        view = view_registry.get(created.view_id)
        view.width = 320
        view.height = 240
        pose = viewer_service._build_mpr_pose_context(view, volume.shape, series=series).poses[viewport]
        center_x = (pose.output_shape[1] - 1.0) / 2.0
        center_y = (pose.output_shape[0] - 1.0) / 2.0
        image_points = (
            MeasurementPoint(x=center_x, y=center_y),
            MeasurementPoint(x=center_x + 4.0, y=center_y + 2.0),
        )
        monkeypatch.setattr(
            viewer_service,
            "_resolve_measurement_image_points",
            lambda active_view, payload, points=image_points: points,
        )

        changed = viewer_service._handle_measurement(
            view,
            ViewOperationRequest(
                viewId=view.view_id,
                opType="measurement",
                subOpType="line",
                actionType="end",
                measurementId=f"line-{view_type}",
                points=[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
            ),
        )

        assert changed is True
        [measurement] = view.measurements
        assert len(measurement.world_points) == 2
        assert measurement.metrics.length == pytest.approx(math.hypot(4.0 * spacing_x, 2.0 * spacing_y))


def test_real_dicom_mpr_measurement_reprojects_after_model_rotation(tmp_path: Path, monkeypatch) -> None:
    volume = np.zeros((5, 6, 8), dtype=np.int16)
    series = _load_synthetic_series(tmp_path, volume, spacing=(3.0, 2.0, 0.5))
    created = view_registry.create(
        ViewCreateRequest(seriesId=series.series_id, viewType="AX", viewGroupKey="measurement-rotation-truth")
    )
    view = view_registry.get(created.view_id)
    assert view.view_group is not None
    viewer_service._reset_mpr_group_geometry(view.view_group, volume.shape, series=series)
    pose = viewer_service._build_mpr_pose_context(view, volume.shape, series=series).poses[MPR_VIEWPORT_AXIAL]
    center_x, center_y = world_point_to_plane_image(pose, pose.cursor_center_world)
    image_points = (
        MeasurementPoint(x=center_x, y=center_y),
        MeasurementPoint(x=center_x + 2.0, y=center_y),
    )
    monkeypatch.setattr(
        viewer_service,
        "_resolve_measurement_image_points",
        lambda active_view, payload: image_points,
    )
    assert viewer_service._handle_measurement(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="measurement",
            subOpType="line",
            actionType="end",
            measurementId="rotating-line",
            points=[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
        ),
    )
    before_by_id = {
        measurement.measurement_id: measurement
        for measurement in viewer_service._build_visible_measurements(view)
    }
    viewer_service._set_mpr_model_rotation_matrix(
        view.view_group,
        axis_angle_rotation_matrix(np.asarray(pose.normal_world), np.pi / 2.0),
        pivot_world=pose.cursor_center_world,
    )

    after_by_id = {
        measurement.measurement_id: measurement
        for measurement in viewer_service._build_visible_measurements(view)
    }
    before = before_by_id["rotating-line"]
    after = after_by_id["rotating-line"]

    assert before.points[1].x == pytest.approx(center_x + 2.0)
    assert before.points[1].y == pytest.approx(center_y)
    assert after.points[1].x == pytest.approx(center_x)
    assert after.points[1].y == pytest.approx(center_y + 0.5)
    assert before.metrics.length == pytest.approx(1.0)
    assert after.metrics.length == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("algorithm", "expected_hu"),
    [
        ("maximum", 300.0),
        ("minimum", 100.0),
        ("average", 200.0),
        ("sum", 600.0),
    ],
)
def test_real_dicom_mpr_mip_uses_physical_slab_thickness(
    tmp_path: Path,
    algorithm: str,
    expected_hu: float,
) -> None:
    volume = np.stack(
        [np.full((6, 8), value, dtype=np.int16) for value in (0, 100, 200, 300, 400)],
        axis=0,
    )
    series = _load_synthetic_series(tmp_path, volume, spacing=(3.0, 2.0, 0.5))
    built_volume = viewer_service._build_series_volume(series)
    created = view_registry.create(
        ViewCreateRequest(seriesId=series.series_id, viewType="AX", viewGroupKey="mip-truth")
    )
    view = view_registry.get(created.view_id)
    assert view.view_group is not None
    viewer_service._reset_mpr_group_geometry(view.view_group, built_volume.shape, series=series)
    view.view_group.mpr_mip.enabled = True
    view.view_group.mpr_mip.algorithm = algorithm
    view.view_group.mpr_mip.viewports[MPR_VIEWPORT_AXIAL] = MprMipViewportState(thickness=9)

    plane, _current, _total = viewer_service._extract_mpr_plane(view, built_volume, MPR_VIEWPORT_AXIAL)

    np.testing.assert_allclose(plane, expected_hu, atol=1e-6)


def test_real_dicom_asymmetric_landmark_follows_3d_anatomical_left_orientation(tmp_path: Path) -> None:
    volume = build_asymmetric_landmark_volume()
    spacing = (2.5, 1.0, 0.5)
    series = _load_synthetic_series(tmp_path, volume, spacing=spacing)
    built_volume = viewer_service._build_series_volume(series)
    np.testing.assert_array_equal(built_volume, volume)

    created = view_registry.create(ViewCreateRequest(seriesId=series.series_id, viewType="3D"))
    view = view_registry.get(created.view_id)
    view.width = 640
    view.height = 480
    assert viewer_service._handle_anatomical_orientation(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            subOpType="orientation:L",
        ),
    )
    request = viewer_service._build_volume_render_request(
        view,
        volume=built_volume,
        spacing_xyz=viewer_service._get_3d_spacing_xyz(series),
        fast_preview=False,
    )

    marker_index = np.argwhere(built_volume == 1200)[0]
    center_zyx = (np.asarray(built_volume.shape, dtype=np.float64) - 1.0) / 2.0
    relative_xyz_mm = np.asarray(
        [
            (marker_index[2] - center_zyx[2]) * spacing[2],
            (marker_index[1] - center_zyx[1]) * spacing[1],
            (marker_index[0] - center_zyx[0]) * spacing[0],
        ]
    )
    displayed = quaternion_to_rotation_matrix(request.rotation_quaternion) @ relative_xyz_mm

    assert request.spacing_xyz == pytest.approx((0.5, 1.0, 2.5))
    assert displayed[1] < 0.0
    assert displayed[0] == pytest.approx(0.0, abs=1e-6)
    assert displayed[2] == pytest.approx(0.0, abs=1e-6)
