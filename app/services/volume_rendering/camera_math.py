from __future__ import annotations

from math import cos, radians, sin

import numpy as np


TRACKBALL_MOTION_FACTOR = 36.0
TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH = -20.0
TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT = -20.0


def normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    vector = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    vector /= norm
    return tuple(float(value) for value in vector)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return normalize_quaternion((float(x), float(y), float(z), float(w)))


def quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def axis_angle_rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    normalized_axis = normalize_vector(np.asarray(axis, dtype=np.float64))
    x, y, z = normalized_axis
    angle = radians(float(angle_degrees))
    c = cos(angle)
    s = sin(angle)
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def apply_trackball_delta_to_quaternion(
    quaternion: tuple[float, float, float, float],
    *,
    delta_x_pixels: float,
    delta_y_pixels: float,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float]:
    """Apply screen-space "grab the model" drag semantics.

    The quaternion represents model rotation. Rendering applies the inverse to
    the camera, so dragging right/up makes visible anatomy move right/up instead
    of orbiting the camera in the opposite direction.
    """

    width = max(float(canvas_width), 1.0)
    height = max(float(canvas_height), 1.0)
    delta_azimuth = (
        float(delta_x_pixels)
        * TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH
        / width
        * TRACKBALL_MOTION_FACTOR
    )
    delta_elevation = (
        float(delta_y_pixels)
        * TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT
        / height
        * TRACKBALL_MOTION_FACTOR
    )
    if abs(delta_azimuth) < 1e-12 and abs(delta_elevation) < 1e-12:
        return normalize_quaternion(quaternion)

    current_matrix = quaternion_to_rotation_matrix(quaternion)
    base_right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    base_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    current_up = normalize_vector(current_matrix @ base_up)
    yaw_matrix = axis_angle_rotation_matrix(current_up, delta_azimuth)
    yawed_matrix = yaw_matrix @ current_matrix

    yawed_right = normalize_vector(yawed_matrix @ base_right)
    pitch_matrix = axis_angle_rotation_matrix(yawed_right, delta_elevation)
    next_matrix = pitch_matrix @ yawed_matrix
    return rotation_matrix_to_quaternion(next_matrix)
