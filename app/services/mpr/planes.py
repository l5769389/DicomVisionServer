from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL

from .cursor import MprCursorState, create_default_cursor, orthonormalize_matrix
from .geometry import VolumeGeometry, ijk_to_world_point, spacing_along_world_direction, world_to_ijk_point


@dataclass(frozen=True)
class PlaneConvention:
    row_axis_index: int
    col_axis_index: int
    normal_axis_index: int
    row_sign: float = 1.0
    col_sign: float = 1.0
    normal_sign: float = 1.0


DEFAULT_MPR_CONVENTION: dict[str, PlaneConvention] = {
    MPR_VIEWPORT_AXIAL: PlaneConvention(row_axis_index=1, col_axis_index=2, normal_axis_index=0),
    MPR_VIEWPORT_CORONAL: PlaneConvention(row_axis_index=0, col_axis_index=2, normal_axis_index=1, row_sign=-1.0),
    MPR_VIEWPORT_SAGITTAL: PlaneConvention(row_axis_index=0, col_axis_index=1, normal_axis_index=2, row_sign=-1.0),
}


@dataclass(frozen=True)
class OutputShapePolicy:
    viewport_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanePose:
    viewport: str
    center_world: np.ndarray
    cursor_center_world: np.ndarray
    row_world: np.ndarray
    col_world: np.ndarray
    normal_world: np.ndarray
    pixel_spacing_row_mm: float
    pixel_spacing_col_mm: float
    output_shape: tuple[int, int]
    is_oblique: bool


def _normalize_world_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    next_vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(next_vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        next_vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(next_vector))
    return next_vector / max(norm, 1e-6)


def _project_vector_to_plane(direction: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = np.asarray(direction, dtype=np.float64) - float(np.dot(direction, normal)) * normal
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return projected / norm


def _resolve_output_shape(geometry: VolumeGeometry, viewport: str, policy: OutputShapePolicy) -> tuple[int, int]:
    if viewport in policy.viewport_shapes:
        return tuple(int(value) for value in policy.viewport_shapes[viewport])

    convention = DEFAULT_MPR_CONVENTION.get(viewport, DEFAULT_MPR_CONVENTION[MPR_VIEWPORT_AXIAL])
    height = int(geometry.shape_ijk[convention.row_axis_index])
    width = int(geometry.shape_ijk[convention.col_axis_index])
    return (height, width)


def derive_plane_pose(
    cursor: MprCursorState,
    viewport: str,
    geometry: VolumeGeometry,
    output_shape_policy: OutputShapePolicy | None = None,
) -> PlanePose:
    policy = output_shape_policy or OutputShapePolicy()
    convention = DEFAULT_MPR_CONVENTION.get(viewport, DEFAULT_MPR_CONVENTION[MPR_VIEWPORT_AXIAL])
    orientation = orthonormalize_matrix(np.asarray(cursor.orientation_world, dtype=np.float64))
    normal_world = _normalize_world_vector(
        orientation[:, convention.normal_axis_index] * convention.normal_sign,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    )
    default_orientation = create_default_cursor(geometry).orientation_world
    default_row_world = _normalize_world_vector(
        default_orientation[:, convention.row_axis_index] * convention.row_sign,
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )
    default_col_world = _normalize_world_vector(
        default_orientation[:, convention.col_axis_index] * convention.col_sign,
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    default_normal_world = _normalize_world_vector(
        default_orientation[:, convention.normal_axis_index] * convention.normal_sign,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    )
    projected_row = _project_vector_to_plane(default_row_world, normal_world)
    projected_col = _project_vector_to_plane(default_col_world, normal_world)
    if projected_row is not None and projected_col is not None:
        row_world = projected_row
        orthogonal_col = projected_col - float(np.dot(projected_col, row_world)) * row_world
        col_world = _normalize_world_vector(orthogonal_col, default_col_world)
    else:
        fallback_col = default_col_world - float(np.dot(default_col_world, normal_world)) * normal_world
        if float(np.linalg.norm(fallback_col)) <= 1e-6:
            fallback_col = default_row_world - float(np.dot(default_row_world, normal_world)) * normal_world
        col_world = _normalize_world_vector(fallback_col, default_col_world)
        row_world = _normalize_world_vector(np.cross(col_world, normal_world), default_row_world)
    is_oblique = float(np.linalg.norm(normal_world - default_normal_world)) > 1e-6
    output_shape = _resolve_output_shape(geometry, viewport, policy)
    cursor_center_world = np.asarray(cursor.center_world, dtype=np.float64)
    if is_oblique:
        center_world = cursor_center_world
    else:
        center_ijk = world_to_ijk_point(geometry, cursor_center_world)
        plane_center_ijk = np.array(center_ijk, dtype=np.float64)
        plane_center_ijk[convention.row_axis_index] = (geometry.shape_ijk[convention.row_axis_index] - 1) / 2.0
        plane_center_ijk[convention.col_axis_index] = (geometry.shape_ijk[convention.col_axis_index] - 1) / 2.0
        center_world = ijk_to_world_point(geometry, plane_center_ijk)
    return PlanePose(
        viewport=viewport,
        center_world=center_world,
        cursor_center_world=cursor_center_world,
        row_world=row_world,
        col_world=col_world,
        normal_world=normal_world,
        pixel_spacing_row_mm=spacing_along_world_direction(geometry, row_world),
        pixel_spacing_col_mm=spacing_along_world_direction(geometry, col_world),
        output_shape=output_shape,
        is_oblique=is_oblique,
    )
