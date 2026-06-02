from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL

from .cursor import MprCursorState, create_default_cursor
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


def _resolve_display_basis_from_normal(
    normal_world: np.ndarray,
    default_row_world: np.ndarray,
    default_col_world: np.ndarray,
    default_normal_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    default_handedness = 1.0
    default_basis_direction = float(np.dot(np.cross(default_normal_world, default_row_world), default_col_world))
    if np.isfinite(default_basis_direction) and default_basis_direction < 0.0:
        default_handedness = -1.0

    projected_row = _project_vector_to_plane(default_row_world, normal_world)
    if projected_row is not None:
        row_world = _normalize_world_vector(projected_row, default_row_world)
        col_world = _normalize_world_vector(default_handedness * np.cross(normal_world, row_world), default_col_world)
        return row_world, col_world

    projected_col = _project_vector_to_plane(default_col_world, normal_world)
    if projected_col is not None:
        col_world = _normalize_world_vector(projected_col, default_col_world)
        row_world = _normalize_world_vector(default_handedness * np.cross(col_world, normal_world), default_row_world)
        return row_world, col_world

    return default_row_world, default_col_world


def _resolve_output_shape(geometry: VolumeGeometry, viewport: str, policy: OutputShapePolicy) -> tuple[int, int]:
    if viewport in policy.viewport_shapes:
        return tuple(int(value) for value in policy.viewport_shapes[viewport])

    convention = DEFAULT_MPR_CONVENTION.get(viewport, DEFAULT_MPR_CONVENTION[MPR_VIEWPORT_AXIAL])
    height = int(geometry.shape_ijk[convention.row_axis_index])
    width = int(geometry.shape_ijk[convention.col_axis_index])
    return (height, width)


def _resolve_default_plane_center_world(
    cursor_center_world: np.ndarray,
    geometry: VolumeGeometry,
    convention: PlaneConvention,
) -> np.ndarray:
    center_ijk = world_to_ijk_point(geometry, cursor_center_world)
    plane_center_ijk = np.array(center_ijk, dtype=np.float64)
    plane_center_ijk[convention.row_axis_index] = (geometry.shape_ijk[convention.row_axis_index] - 1) / 2.0
    plane_center_ijk[convention.col_axis_index] = (geometry.shape_ijk[convention.col_axis_index] - 1) / 2.0
    return ijk_to_world_point(geometry, plane_center_ijk)


def _resolve_cursor_image_offsets(
    cursor_center_world: np.ndarray,
    default_center_world: np.ndarray,
    default_row_world: np.ndarray,
    default_col_world: np.ndarray,
    geometry: VolumeGeometry,
) -> tuple[float, float]:
    delta_world = np.asarray(cursor_center_world, dtype=np.float64) - np.asarray(default_center_world, dtype=np.float64)
    row_offset_px = float(np.dot(delta_world, default_row_world)) / max(
        spacing_along_world_direction(geometry, default_row_world),
        1e-6,
    )
    col_offset_px = float(np.dot(delta_world, default_col_world)) / max(
        spacing_along_world_direction(geometry, default_col_world),
        1e-6,
    )
    return row_offset_px, col_offset_px


def derive_plane_pose(
    cursor: MprCursorState,
    viewport: str,
    geometry: VolumeGeometry,
    output_shape_policy: OutputShapePolicy | None = None,
    normal_world_override: np.ndarray | tuple[float, float, float] | None = None,
    use_display_basis_for_cursor_offsets: bool = False,
) -> PlanePose:
    policy = output_shape_policy or OutputShapePolicy()
    convention = DEFAULT_MPR_CONVENTION.get(viewport, DEFAULT_MPR_CONVENTION[MPR_VIEWPORT_AXIAL])
    orientation = np.asarray(cursor.orientation_world, dtype=np.float64)
    cursor_normal_world = orientation[:, convention.normal_axis_index] * convention.normal_sign
    normal_source = cursor_normal_world if normal_world_override is None else np.asarray(normal_world_override, dtype=np.float64)
    normal_world = _normalize_world_vector(normal_source, cursor_normal_world)
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
    is_oblique = float(np.linalg.norm(normal_world - default_normal_world)) > 1e-6
    if is_oblique:
        row_world, col_world = _resolve_display_basis_from_normal(
            normal_world,
            default_row_world,
            default_col_world,
            default_normal_world,
        )
    else:
        row_world = default_row_world
        col_world = default_col_world
    cursor_center_world = np.asarray(cursor.center_world, dtype=np.float64)
    output_shape = _resolve_output_shape(geometry, viewport, policy)
    default_center_world = _resolve_default_plane_center_world(cursor_center_world, geometry, convention)
    row_offset_px, col_offset_px = _resolve_cursor_image_offsets(
        cursor_center_world,
        default_center_world,
        row_world if use_display_basis_for_cursor_offsets else default_row_world,
        col_world if use_display_basis_for_cursor_offsets else default_col_world,
        geometry,
    )
    row_spacing_mm = spacing_along_world_direction(geometry, row_world)
    col_spacing_mm = spacing_along_world_direction(geometry, col_world)
    center_world = (
        cursor_center_world
        - row_world * row_offset_px * row_spacing_mm
        - col_world * col_offset_px * col_spacing_mm
    )
    return PlanePose(
        viewport=viewport,
        center_world=center_world,
        cursor_center_world=cursor_center_world,
        row_world=row_world,
        col_world=col_world,
        normal_world=normal_world,
        pixel_spacing_row_mm=row_spacing_mm,
        pixel_spacing_col_mm=col_spacing_mm,
        output_shape=output_shape,
        is_oblique=is_oblique,
    )
