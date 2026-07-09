from __future__ import annotations

from itertools import product
from math import atan, isfinite, radians, sin, tan
from typing import Iterable

import numpy as np


CAMERA_FIT_PADDING = 1.12
BASE_CAMERA_FORWARD = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
BASE_CAMERA_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def normalize_bounds(bounds: Iterable[float] | None) -> tuple[float, float, float, float, float, float] | None:
    if bounds is None:
        return None
    values = tuple(float(value) for value in bounds)
    if len(values) != 6 or not all(isfinite(value) for value in values):
        return None
    if values[0] > values[1] or values[2] > values[3] or values[4] > values[5]:
        return None
    if max(values[1] - values[0], values[3] - values[2], values[5] - values[4]) <= 1e-6:
        return None
    return values


def bounds_center(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.asarray(
        [
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        ],
        dtype=np.float64,
    )


def bounds_radius(bounds: tuple[float, float, float, float, float, float]) -> float:
    center = bounds_center(bounds)
    radius = 0.0
    for corner in product((bounds[0], bounds[1]), (bounds[2], bounds[3]), (bounds[4], bounds[5])):
        radius = max(radius, float(np.linalg.norm(np.asarray(corner, dtype=np.float64) - center)))
    return max(radius, 1e-3)


def _field_of_view_half_angles(view_angle_degrees: float, aspect_ratio: float) -> tuple[float, float]:
    aspect = max(float(aspect_ratio), 1e-3)
    half_fov_y = max(radians(max(float(view_angle_degrees), 1.0)) * 0.5, radians(0.5))
    half_fov_x = max(atan(tan(half_fov_y) * aspect), radians(0.5))
    return half_fov_x, half_fov_y


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _camera_basis(rotation_matrix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = np.asarray(rotation_matrix, dtype=np.float64) if rotation_matrix is not None else np.eye(3, dtype=np.float64)
    forward = _normalize_vector(rotation @ BASE_CAMERA_FORWARD)
    up = _normalize_vector(rotation @ BASE_CAMERA_UP)
    right = _normalize_vector(np.cross(forward, up))
    up = _normalize_vector(np.cross(right, forward))
    return right, up, forward


def _projected_half_extents(
    bounds: tuple[float, float, float, float, float, float],
    *,
    rotation_matrix: np.ndarray | None = None,
) -> tuple[float, float, float]:
    center = bounds_center(bounds)
    right, up, forward = _camera_basis(rotation_matrix)
    half_width = 0.0
    half_height = 0.0
    half_depth = 0.0
    for corner in product((bounds[0], bounds[1]), (bounds[2], bounds[3]), (bounds[4], bounds[5])):
        relative = np.asarray(corner, dtype=np.float64) - center
        half_width = max(half_width, abs(float(np.dot(relative, right))))
        half_height = max(half_height, abs(float(np.dot(relative, up))))
        half_depth = max(half_depth, abs(float(np.dot(relative, forward))))
    return max(half_width, 1e-3), max(half_height, 1e-3), max(half_depth, 1e-3)


def fit_distance_for_bounds(
    bounds: tuple[float, float, float, float, float, float],
    *,
    view_angle_degrees: float,
    aspect_ratio: float,
    rotation_matrix: np.ndarray | None = None,
    padding: float = CAMERA_FIT_PADDING,
) -> float:
    half_fov_x, half_fov_y = _field_of_view_half_angles(view_angle_degrees, aspect_ratio)
    half_width, half_height, half_depth = _projected_half_extents(bounds, rotation_matrix=rotation_matrix)
    safe_padding = max(float(padding), 1.0)
    distance_for_width = half_width * safe_padding / max(tan(half_fov_x), 1e-3)
    distance_for_height = half_height * safe_padding / max(tan(half_fov_y), 1e-3)
    return max(distance_for_width, distance_for_height) + half_depth * safe_padding


def fit_stable_distance_for_bounds(
    bounds: tuple[float, float, float, float, float, float],
    *,
    view_angle_degrees: float,
    aspect_ratio: float,
    padding: float = CAMERA_FIT_PADDING,
) -> float:
    radius = bounds_radius(bounds) * max(float(padding), 1.0)
    half_fov_x, half_fov_y = _field_of_view_half_angles(view_angle_degrees, aspect_ratio)
    limiting_half_angle = max(min(half_fov_x, half_fov_y), radians(0.5))
    return radius / max(sin(limiting_half_angle), 1e-3)


def fit_parallel_scale_for_bounds(
    bounds: tuple[float, float, float, float, float, float],
    *,
    aspect_ratio: float,
    rotation_matrix: np.ndarray | None = None,
    padding: float = CAMERA_FIT_PADDING,
) -> float:
    aspect = max(float(aspect_ratio), 1e-3)
    half_width, half_height, _half_depth = _projected_half_extents(bounds, rotation_matrix=rotation_matrix)
    return max(half_height, half_width / aspect) * max(float(padding), 1.0)


def fit_stable_parallel_scale_for_bounds(
    bounds: tuple[float, float, float, float, float, float],
    *,
    aspect_ratio: float,
    padding: float = CAMERA_FIT_PADDING,
) -> float:
    aspect = max(float(aspect_ratio), 1e-3)
    radius = bounds_radius(bounds)
    return max(radius, radius / aspect) * max(float(padding), 1.0)
