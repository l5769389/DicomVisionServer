from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.models.viewer import MprFrameState

from .geometry import VolumeGeometry, ijk_to_world_point, world_to_ijk_point


@dataclass(frozen=True)
class MprCursorState:
    center_world: np.ndarray
    reference_center_world: np.ndarray
    orientation_world: np.ndarray
    linked_to_volume_rotation: bool = False


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    next_vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(next_vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        next_vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(next_vector))
    return next_vector / max(norm, 1e-6)


def orthonormalize_matrix(matrix: np.ndarray) -> np.ndarray:
    candidate = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(candidate, full_matrices=False)
    orthonormal = u @ vh
    if float(np.linalg.det(orthonormal)) < 0.0:
        u[:, -1] *= -1.0
        orthonormal = u @ vh
    return orthonormal


def _geometry_axis_orientation(geometry: VolumeGeometry) -> np.ndarray:
    affine = geometry.ijk_to_world[:3, :3]
    columns = [
        _normalize_vector(affine[:, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float64)),
        _normalize_vector(affine[:, 1], np.asarray([0.0, 1.0, 0.0], dtype=np.float64)),
        _normalize_vector(affine[:, 2], np.asarray([0.0, 0.0, 1.0], dtype=np.float64)),
    ]
    return orthonormalize_matrix(np.column_stack(columns))


def create_default_cursor(geometry: VolumeGeometry) -> MprCursorState:
    center_ijk = np.asarray([(size - 1) / 2.0 for size in geometry.shape_ijk], dtype=np.float64)
    center_world = ijk_to_world_point(geometry, center_ijk)
    orientation_world = _geometry_axis_orientation(geometry)
    return MprCursorState(
        center_world=center_world,
        reference_center_world=center_world.copy(),
        orientation_world=orientation_world,
        linked_to_volume_rotation=False,
    )


def legacy_frame_to_cursor(
    frame: MprFrameState,
    geometry: VolumeGeometry,
    *,
    reference_center: tuple[float, float, float] | np.ndarray | None = None,
    linked_to_volume_rotation: bool = False,
) -> MprCursorState:
    center_world = ijk_to_world_point(geometry, frame.center)
    reference_source = frame.center if reference_center is None else reference_center
    reference_center_world = ijk_to_world_point(geometry, reference_source)
    affine = geometry.ijk_to_world[:3, :3]
    orientation_world = np.column_stack([
        _normalize_vector(affine @ np.asarray(frame.axis_slice, dtype=np.float64), affine[:, 0]),
        _normalize_vector(affine @ np.asarray(frame.axis_row, dtype=np.float64), affine[:, 1]),
        _normalize_vector(affine @ np.asarray(frame.axis_col, dtype=np.float64), affine[:, 2]),
    ])
    return MprCursorState(
        center_world=center_world,
        reference_center_world=reference_center_world,
        orientation_world=orthonormalize_matrix(orientation_world),
        linked_to_volume_rotation=linked_to_volume_rotation,
    )


def cursor_to_legacy_frame(cursor: MprCursorState, geometry: VolumeGeometry) -> MprFrameState:
    center_ijk = world_to_ijk_point(geometry, cursor.center_world)
    inverse_affine = geometry.world_to_ijk[:3, :3]
    orientation = orthonormalize_matrix(cursor.orientation_world)
    axis_slice = _normalize_vector(inverse_affine @ orientation[:, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    axis_row = _normalize_vector(inverse_affine @ orientation[:, 1], np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    axis_col = _normalize_vector(inverse_affine @ orientation[:, 2], np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    return MprFrameState(
        center=tuple(float(value) for value in center_ijk),
        axis_slice=tuple(float(value) for value in axis_slice),
        axis_row=tuple(float(value) for value in axis_row),
        axis_col=tuple(float(value) for value in axis_col),
    )


def clamp_world_to_geometry(point_world: np.ndarray | tuple[float, float, float], geometry: VolumeGeometry) -> np.ndarray:
    ijk = world_to_ijk_point(geometry, point_world)
    clamped_ijk = np.array(
        [
            max(0.0, min(float(ijk[0]), geometry.shape_ijk[0] - 1)),
            max(0.0, min(float(ijk[1]), geometry.shape_ijk[1] - 1)),
            max(0.0, min(float(ijk[2]), geometry.shape_ijk[2] - 1)),
        ],
        dtype=np.float64,
    )
    return ijk_to_world_point(geometry, clamped_ijk)


def translate_cursor(cursor: MprCursorState, delta_world: np.ndarray, geometry: VolumeGeometry) -> MprCursorState:
    next_center = clamp_world_to_geometry(np.asarray(cursor.center_world, dtype=np.float64) + np.asarray(delta_world, dtype=np.float64), geometry)
    return replace(cursor, center_world=next_center)


def rotate_cursor(cursor: MprCursorState, rotation_world: np.ndarray) -> MprCursorState:
    next_orientation = orthonormalize_matrix(np.asarray(rotation_world, dtype=np.float64) @ np.asarray(cursor.orientation_world, dtype=np.float64))
    return replace(cursor, orientation_world=next_orientation)


def axis_angle_rotation_matrix(axis_world: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize_vector(
        np.asarray(axis_world, dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    )
    x, y, z = axis
    cos_angle = float(np.cos(angle_rad))
    sin_angle = float(np.sin(angle_rad))
    one_minus_cos = 1.0 - cos_angle
    return np.asarray(
        [
            [
                cos_angle + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_angle,
                x * z * one_minus_cos + y * sin_angle,
            ],
            [
                y * x * one_minus_cos + z * sin_angle,
                cos_angle + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_angle,
            ],
            [
                z * x * one_minus_cos - y * sin_angle,
                z * y * one_minus_cos + x * sin_angle,
                cos_angle + z * z * one_minus_cos,
            ],
        ],
        dtype=np.float64,
    )


def rotate_cursor_from_drag(
    start_cursor: MprCursorState,
    axis_world: np.ndarray,
    start_angle_rad: float,
    current_angle_rad: float,
) -> MprCursorState:
    rotation_world = axis_angle_rotation_matrix(axis_world, float(current_angle_rad) - float(start_angle_rad))
    return rotate_cursor(start_cursor, rotation_world)
