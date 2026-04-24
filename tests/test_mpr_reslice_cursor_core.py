import numpy as np

from app.models.viewer import MprFrameState, ViewGroupRecord, ViewRecord
from app.services.mpr import (
    build_identity_geometry,
    cursor_to_legacy_frame,
    derive_plane_pose,
    ijk_to_world_point,
    legacy_frame_to_cursor,
    reslice_plane,
)
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
