from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.mpr_geometry import VolumePatientTransform


@dataclass(frozen=True)
class VolumeGeometry:
    shape_ijk: tuple[int, int, int]
    ijk_to_world: np.ndarray
    world_to_ijk: np.ndarray
    spacing_hint_mm: tuple[float, float, float]


def _affine_from_origin_and_axes(origin: np.ndarray, axis_vectors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = np.asarray(axis_vectors[0], dtype=np.float64)
    affine[:3, 1] = np.asarray(axis_vectors[1], dtype=np.float64)
    affine[:3, 2] = np.asarray(axis_vectors[2], dtype=np.float64)
    affine[:3, 3] = np.asarray(origin, dtype=np.float64)
    return affine


def build_geometry_from_patient_transform(transform: VolumePatientTransform) -> VolumeGeometry:
    ijk_to_world = _affine_from_origin_and_axes(transform.origin, transform.axis_vectors)
    world_to_ijk = np.linalg.inv(ijk_to_world)
    spacing_hint_mm = tuple(
        max(float(np.linalg.norm(np.asarray(transform.axis_vectors[index], dtype=np.float64))), 1e-6)
        for index in range(3)
    )
    return VolumeGeometry(
        shape_ijk=tuple(int(value) for value in transform.shape),
        ijk_to_world=ijk_to_world,
        world_to_ijk=world_to_ijk,
        spacing_hint_mm=spacing_hint_mm,
    )


def build_identity_geometry(shape_ijk: tuple[int, int, int]) -> VolumeGeometry:
    shape = tuple(int(value) for value in shape_ijk)
    ijk_to_world = np.eye(4, dtype=np.float64)
    world_to_ijk = np.eye(4, dtype=np.float64)
    return VolumeGeometry(
        shape_ijk=shape,
        ijk_to_world=ijk_to_world,
        world_to_ijk=world_to_ijk,
        spacing_hint_mm=(1.0, 1.0, 1.0),
    )


def ijk_to_world_point(geometry: VolumeGeometry, point_ijk: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    point = np.asarray(point_ijk, dtype=np.float64)
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point
    return (geometry.ijk_to_world @ homogeneous)[:3]


def world_to_ijk_point(geometry: VolumeGeometry, point_world: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    point = np.asarray(point_world, dtype=np.float64)
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point
    return (geometry.world_to_ijk @ homogeneous)[:3]


def spacing_along_world_direction(geometry: VolumeGeometry, direction_world: np.ndarray | tuple[float, float, float]) -> float:
    direction = np.asarray(direction_world, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-6:
        return min(geometry.spacing_hint_mm)
    unit_direction = direction / norm
    ijk_step = geometry.world_to_ijk[:3, :3] @ unit_direction
    ijk_step_norm = float(np.linalg.norm(ijk_step))
    if not np.isfinite(ijk_step_norm) or ijk_step_norm <= 1e-6:
        return min(geometry.spacing_hint_mm)
    return max(1.0 / ijk_step_norm, 1e-6)
