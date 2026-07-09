from __future__ import annotations

from math import acos, cos, radians, sin, sqrt

import numpy as np


VTK_TRACKBALL_MOTION_FACTOR = 10.0
VTK_TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH = -20.0
VTK_TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT = -20.0
DIRECT_MODEL_TRACKBALL_MOTION_FACTOR = 10.0
DIRECT_MODEL_TRACKBALL_DEGREES_PER_VIEW_WIDTH = 20.0
DIRECT_MODEL_TRACKBALL_DEGREES_PER_VIEW_HEIGHT = 20.0
DIRECT_MODEL_TRACKBALL_RADIUS_VIEW_FRACTION = 0.5

TRACKBALL_MOTION_FACTOR = VTK_TRACKBALL_MOTION_FACTOR
TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH = VTK_TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH
TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT = VTK_TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT


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


def apply_vtk_trackball_camera_delta_to_quaternion(
    quaternion: tuple[float, float, float, float],
    *,
    delta_x_pixels: float,
    delta_y_pixels: float,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float]:
    """Apply vtkInteractorStyleTrackballCamera rotate semantics.

    The stored quaternion represents model rotation, while rendering applies
    its inverse to the camera. This helper therefore updates the camera basis
    first, then stores the inverse as the next model quaternion.
    """

    width = max(float(canvas_width), 1.0)
    height = max(float(canvas_height), 1.0)
    delta_azimuth = (
        float(delta_x_pixels)
        * VTK_TRACKBALL_AZIMUTH_DEGREES_PER_VIEW_WIDTH
        / width
        * VTK_TRACKBALL_MOTION_FACTOR
    )
    delta_elevation = (
        float(delta_y_pixels)
        * VTK_TRACKBALL_ELEVATION_DEGREES_PER_VIEW_HEIGHT
        / height
        * VTK_TRACKBALL_MOTION_FACTOR
    )
    if abs(delta_azimuth) < 1e-12 and abs(delta_elevation) < 1e-12:
        return normalize_quaternion(quaternion)

    model_rotation = quaternion_to_rotation_matrix(quaternion)
    camera_rotation = model_rotation.T
    base_right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    base_forward = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    base_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    current_forward = normalize_vector(camera_rotation @ base_forward)
    current_up = normalize_vector(camera_rotation @ base_up)

    azimuth_matrix = axis_angle_rotation_matrix(current_up, delta_azimuth)
    current_forward = normalize_vector(azimuth_matrix @ current_forward)
    current_up = normalize_vector(azimuth_matrix @ current_up)

    elevation_axis = normalize_vector(np.cross(current_up, current_forward))
    elevation_matrix = axis_angle_rotation_matrix(elevation_axis, delta_elevation)
    current_forward = normalize_vector(elevation_matrix @ current_forward)
    current_up = normalize_vector(elevation_matrix @ current_up)

    current_right = normalize_vector(np.cross(current_forward, current_up))
    current_up = normalize_vector(np.cross(current_right, current_forward))

    base_basis = np.column_stack((base_right, base_forward, base_up))
    current_basis = np.column_stack((current_right, current_forward, current_up))
    next_camera_rotation = current_basis @ base_basis.T
    next_model_rotation = next_camera_rotation.T
    return rotation_matrix_to_quaternion(next_model_rotation)


def apply_direct_model_trackball_delta_to_quaternion(
    quaternion: tuple[float, float, float, float],
    *,
    delta_x_pixels: float,
    delta_y_pixels: float,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float]:
    """Rotate the stored model quaternion with fixed screen-axis controls.

    This is intentionally not an arcball axis derived from the current model.
    The initial camera looks at the volume from the negative-Y side, so the
    visible "front" face is the model's -Y side. Positive drag signs therefore
    intentionally differ from VTK camera orbit signs: the visible face follows
    the pointer instead of the camera orbiting around the model.

    A vertical drag should always mean the same screen-up pitch command, even
    after the model has been turned upside down or rolled around.
    """

    width = max(float(canvas_width), 1.0)
    height = max(float(canvas_height), 1.0)
    yaw_degrees = (
        float(delta_x_pixels)
        * DIRECT_MODEL_TRACKBALL_DEGREES_PER_VIEW_WIDTH
        / width
        * DIRECT_MODEL_TRACKBALL_MOTION_FACTOR
    )
    pitch_degrees = (
        float(delta_y_pixels)
        * DIRECT_MODEL_TRACKBALL_DEGREES_PER_VIEW_HEIGHT
        / height
        * DIRECT_MODEL_TRACKBALL_MOTION_FACTOR
    )
    if abs(yaw_degrees) < 1e-12 and abs(pitch_degrees) < 1e-12:
        return normalize_quaternion(quaternion)

    model_rotation = quaternion_to_rotation_matrix(quaternion)
    screen_right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    screen_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    yaw_matrix = axis_angle_rotation_matrix(screen_up, yaw_degrees)
    pitch_matrix = axis_angle_rotation_matrix(screen_right, pitch_degrees)
    next_model_rotation = pitch_matrix @ yaw_matrix @ model_rotation
    return rotation_matrix_to_quaternion(next_model_rotation)


def resolve_direct_model_trackball_control_point(
    *,
    canvas_x: float,
    canvas_y: float,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float]:
    """Project a canvas pointer to the front hemisphere of a virtual trackball.

    Screen axes are represented in render/world coordinates: +X is screen
    right, +Z is screen up, and -Y faces the user. The point returned here is
    the model control point that should remain under the pointer while the
    drag is active.
    """

    width = max(float(canvas_width), 1.0)
    height = max(float(canvas_height), 1.0)
    radius = max(min(width, height) * DIRECT_MODEL_TRACKBALL_RADIUS_VIEW_FRACTION, 1.0)
    x = (float(canvas_x) - width * 0.5) / radius
    z = (height * 0.5 - float(canvas_y)) / radius
    distance_squared = x * x + z * z
    if distance_squared <= 1.0:
        y = -sqrt(max(0.0, 1.0 - distance_squared))
        return tuple(float(value) for value in normalize_vector(np.asarray([x, y, z], dtype=np.float64)))

    distance = sqrt(distance_squared)
    return (float(x / distance), 0.0, float(z / distance))


def apply_direct_model_trackball_control_points_to_quaternion(
    quaternion: tuple[float, float, float, float],
    *,
    origin_control_point: tuple[float, float, float],
    current_control_point: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Rotate the model so the drag-start control point follows the pointer."""

    origin_vector = normalize_vector(np.asarray(origin_control_point, dtype=np.float64))
    current_vector = normalize_vector(np.asarray(current_control_point, dtype=np.float64))
    dot = float(np.clip(np.dot(origin_vector, current_vector), -1.0, 1.0))
    if dot >= 1.0 - 1e-12:
        return normalize_quaternion(quaternion)

    axis = np.cross(origin_vector, current_vector)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        fallback_axis = np.cross(origin_vector, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(fallback_axis)) <= 1e-12:
            fallback_axis = np.cross(origin_vector, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        axis = fallback_axis
    angle_degrees = float(np.degrees(acos(dot)))
    model_rotation = quaternion_to_rotation_matrix(quaternion)
    control_rotation = axis_angle_rotation_matrix(axis, angle_degrees)
    return rotation_matrix_to_quaternion(control_rotation @ model_rotation)


def apply_trackball_delta_to_quaternion(
    quaternion: tuple[float, float, float, float],
    *,
    delta_x_pixels: float,
    delta_y_pixels: float,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float]:
    return apply_direct_model_trackball_delta_to_quaternion(
        quaternion,
        delta_x_pixels=delta_x_pixels,
        delta_y_pixels=delta_y_pixels,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
