from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.models.viewer import MprFrameState, MprObliquePlaneState, create_default_mpr_oblique_planes


@dataclass(frozen=True)
class VolumePatientTransform:
    origin: np.ndarray
    axis_vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int, int]

    def point_to_patient(self, index: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        volume_index = np.asarray(index, dtype=np.float64)
        return (
            np.asarray(self.origin, dtype=np.float64)
            + np.asarray(self.axis_vectors[0], dtype=np.float64) * float(volume_index[0])
            + np.asarray(self.axis_vectors[1], dtype=np.float64) * float(volume_index[1])
            + np.asarray(self.axis_vectors[2], dtype=np.float64) * float(volume_index[2])
        )

    def clamped_point_to_patient(self, index: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        volume_index = np.asarray(index, dtype=np.float64)
        clamped = np.array(
            [
                max(0.0, min(float(volume_index[0]), float(self.shape[0] - 1))),
                max(0.0, min(float(volume_index[1]), float(self.shape[1] - 1))),
                max(0.0, min(float(volume_index[2]), float(self.shape[2] - 1))),
            ],
            dtype=np.float64,
        )
        return self.point_to_patient(clamped)

    def direction_to_patient(self, direction: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        volume_direction = normalize_oblique_vector(direction, fallback=(1.0, 0.0, 0.0))
        patient_vector = self.direction_step_to_patient(volume_direction)
        return normalize_patient_vector(patient_vector, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))

    def direction_step_to_patient(self, direction: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        volume_direction = np.asarray(direction, dtype=np.float64)
        return (
            np.asarray(self.axis_vectors[0], dtype=np.float64) * float(volume_direction[0])
            + np.asarray(self.axis_vectors[1], dtype=np.float64) * float(volume_direction[1])
            + np.asarray(self.axis_vectors[2], dtype=np.float64) * float(volume_direction[2])
        )

    def spacing_for_direction(self, direction: np.ndarray | tuple[float, float, float]) -> float:
        step_vector = self.direction_step_to_patient(normalize_oblique_vector(direction, fallback=(1.0, 0.0, 0.0)))
        return max(float(np.linalg.norm(step_vector)), 1e-3)

    def spacing_xyz(self) -> tuple[float, float, float]:
        return tuple(max(float(np.linalg.norm(self.axis_vectors[index])), 1e-3) for index in (2, 1, 0))


def normalize_oblique_vector(
    value: tuple[float, float, float] | np.ndarray,
    *,
    fallback: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


def default_mpr_frame_state(volume_shape: tuple[int, int, int]) -> MprFrameState:
    depth, height, width = volume_shape
    return MprFrameState(
        center=(float(depth // 2), float(height // 2), float(width // 2)),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )


def default_mpr_oblique_plane(viewport_key: str) -> MprObliquePlaneState:
    default_planes = create_default_mpr_oblique_planes()
    return default_planes.get(viewport_key) or default_planes[MPR_VIEWPORT_AXIAL]


def build_mpr_oblique_planes_from_frame(frame: MprFrameState) -> dict[str, MprObliquePlaneState]:
    slice_axis = normalize_oblique_vector(frame.axis_slice, fallback=(1.0, 0.0, 0.0))
    row_axis = normalize_oblique_vector(frame.axis_row, fallback=(0.0, 1.0, 0.0))
    col_axis = normalize_oblique_vector(frame.axis_col, fallback=(0.0, 0.0, 1.0))
    default_planes = create_default_mpr_oblique_planes()

    def build(row_dir: np.ndarray, col_dir: np.ndarray, normal_dir: np.ndarray, viewport_key: str) -> MprObliquePlaneState:
        default_plane = default_planes[viewport_key]
        default_normal = normalize_oblique_vector(default_plane.normal, fallback=(1.0, 0.0, 0.0))
        is_oblique = float(np.linalg.norm(normal_dir - default_normal)) > 1e-6
        return MprObliquePlaneState(
            row=tuple(float(value) for value in row_dir),
            col=tuple(float(value) for value in col_dir),
            normal=tuple(float(value) for value in normal_dir),
            is_oblique=is_oblique,
        )

    return {
        MPR_VIEWPORT_AXIAL: build(row_axis, col_axis, slice_axis, MPR_VIEWPORT_AXIAL),
        MPR_VIEWPORT_CORONAL: build(-slice_axis, col_axis, row_axis, MPR_VIEWPORT_CORONAL),
        MPR_VIEWPORT_SAGITTAL: build(-slice_axis, row_axis, col_axis, MPR_VIEWPORT_SAGITTAL),
    }


def resolve_mpr_plane_basis(
    frame: MprFrameState,
    viewport_key: str,
    *,
    cached_plane: MprObliquePlaneState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_plane = build_mpr_oblique_planes_from_frame(frame).get(viewport_key) or default_mpr_oblique_plane(viewport_key)
    default_plane = default_mpr_oblique_plane(viewport_key)
    plane = cached_plane or frame_plane
    normal_dir = normalize_oblique_vector(frame_plane.normal, fallback=tuple(default_plane.normal))
    row_dir = normalize_oblique_vector(plane.row, fallback=tuple(frame_plane.row))
    col_dir = normalize_oblique_vector(plane.col, fallback=tuple(frame_plane.col))
    projected_row = project_vector_to_plane(row_dir, normal_dir)
    projected_col = project_vector_to_plane(col_dir, normal_dir)
    if projected_row is not None and projected_col is not None:
        row_dir = projected_row
        col_dir = normalize_oblique_vector(
            projected_col - float(np.dot(projected_col, row_dir)) * row_dir,
            fallback=tuple(frame_plane.col),
        )
    else:
        row_dir, col_dir = get_mpr_display_basis(viewport_key, normal_dir)
    return row_dir, col_dir, normal_dir


def resolve_mpr_plane_state(
    frame: MprFrameState,
    viewport_key: str,
    *,
    cached_plane: MprObliquePlaneState | None = None,
) -> MprObliquePlaneState:
    row_dir, col_dir, normal_dir = resolve_mpr_plane_basis(frame, viewport_key, cached_plane=cached_plane)
    default_plane = default_mpr_oblique_plane(viewport_key)
    default_normal = normalize_oblique_vector(default_plane.normal, fallback=(0.0, 1.0, 0.0))
    is_oblique = float(np.linalg.norm(normal_dir - default_normal)) > 1e-6
    return MprObliquePlaneState(
        row=tuple(float(value) for value in row_dir),
        col=tuple(float(value) for value in col_dir),
        normal=tuple(float(value) for value in normal_dir),
        is_oblique=is_oblique,
    )


def build_mpr_plane_state_from_group_normals(
    viewport_key: str,
    axial_normal: np.ndarray,
    coronal_normal: np.ndarray,
    sagittal_normal: np.ndarray,
    *,
    reference_plane: MprObliquePlaneState | None = None,
) -> MprObliquePlaneState:
    axial_normal = normalize_oblique_vector(axial_normal, fallback=(1.0, 0.0, 0.0))
    coronal_normal = normalize_oblique_vector(coronal_normal, fallback=(0.0, 1.0, 0.0))
    sagittal_normal = normalize_oblique_vector(sagittal_normal, fallback=(0.0, 0.0, 1.0))

    if viewport_key == MPR_VIEWPORT_CORONAL:
        row_dir = -axial_normal
        col_dir = sagittal_normal
        normal_dir = coronal_normal
    elif viewport_key == MPR_VIEWPORT_SAGITTAL:
        row_dir = -axial_normal
        col_dir = coronal_normal
        normal_dir = sagittal_normal
    else:
        row_dir = coronal_normal
        col_dir = sagittal_normal
        normal_dir = axial_normal

    reference = reference_plane or default_mpr_oblique_plane(viewport_key)
    reference_row = normalize_oblique_vector(reference.row, fallback=tuple(row_dir))
    reference_col = normalize_oblique_vector(reference.col, fallback=tuple(col_dir))
    if float(np.dot(row_dir, reference_row)) < 0.0:
        row_dir = -row_dir
    if float(np.dot(col_dir, reference_col)) < 0.0:
        col_dir = -col_dir

    default_plane = default_mpr_oblique_plane(viewport_key)
    default_normal = normalize_oblique_vector(default_plane.normal, fallback=(0.0, 1.0, 0.0))
    is_oblique = float(np.linalg.norm(normal_dir - default_normal)) > 1e-6
    return MprObliquePlaneState(
        row=tuple(float(value) for value in row_dir),
        col=tuple(float(value) for value in col_dir),
        normal=tuple(float(value) for value in normal_dir),
        is_oblique=is_oblique,
    )


def get_mpr_display_basis(viewport_key: str, normal_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    default_plane = default_mpr_oblique_plane(viewport_key)
    canonical_row = normalize_oblique_vector(default_plane.row, fallback=(1.0, 0.0, 0.0))
    canonical_col = normalize_oblique_vector(default_plane.col, fallback=(0.0, 0.0, 1.0))
    normalized_normal = normalize_oblique_vector(normal_dir, fallback=tuple(default_plane.normal))
    projected_col = canonical_col - float(np.dot(canonical_col, normalized_normal)) * normalized_normal
    if float(np.linalg.norm(projected_col)) <= 1e-8:
        projected_col = canonical_row - float(np.dot(canonical_row, normalized_normal)) * normalized_normal
    col_dir = normalize_oblique_vector(projected_col, fallback=tuple(canonical_col))
    row_dir = normalize_oblique_vector(np.cross(col_dir, normalized_normal), fallback=tuple(canonical_row))
    projected_row = canonical_row - float(np.dot(canonical_row, normalized_normal)) * normalized_normal
    projected_row = normalize_oblique_vector(projected_row, fallback=tuple(canonical_row))
    if float(np.dot(row_dir, projected_row)) < 0.0:
        row_dir = -row_dir
        col_dir = -col_dir
    return row_dir, col_dir


def normalize_screen_half_turn_angle(angle_rad: float) -> float:
    return float(angle_rad % np.pi)


def direction_from_screen_angle(active_row: np.ndarray, active_col: np.ndarray, angle_rad: float) -> np.ndarray:
    return normalize_oblique_vector(
        np.cos(angle_rad) * active_col + np.sin(angle_rad) * active_row,
        fallback=tuple(active_col),
    )


def build_mpr_oblique_line_direction(active_row: np.ndarray, active_col: np.ndarray, angle_rad: float, *, line: str) -> np.ndarray:
    del line
    normalized_angle = normalize_screen_half_turn_angle(angle_rad)
    return direction_from_screen_angle(active_row, active_col, normalized_angle)


def resolve_mpr_crosshair_line_angle(
    current_normal: np.ndarray,
    current_row: np.ndarray,
    current_col: np.ndarray,
    target_plane: MprObliquePlaneState,
    *,
    fallback: float,
) -> float:
    target_normal = normalize_oblique_vector(target_plane.normal, fallback=tuple(current_col))
    line_dir = normalize_oblique_vector(np.cross(current_normal, target_normal), fallback=tuple(current_col))
    col_component = float(np.dot(line_dir, current_col))
    row_component = float(np.dot(line_dir, current_row))
    if not np.isfinite(col_component) or not np.isfinite(row_component):
        return fallback
    magnitude = float(np.hypot(col_component, row_component))
    if magnitude <= 1e-8:
        return fallback
    return normalize_screen_half_turn_angle(np.arctan2(row_component, col_component))


def project_vector_to_plane(direction: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = np.asarray(direction, dtype=np.float64) - float(np.dot(direction, normal)) * normal
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return projected / norm


def project_patient_direction_to_plane(direction: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    return project_vector_to_plane(direction, normal)


def normalize_patient_vector(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    next_vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(next_vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        next_vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(next_vector))
    return next_vector / max(norm, 1e-6)


def fallback_volume_direction_to_patient_vector(direction: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    volume_direction = normalize_oblique_vector(direction, fallback=(1.0, 0.0, 0.0))
    fallback_patient = np.asarray(
        [float(volume_direction[2]), float(volume_direction[1]), float(volume_direction[0])],
        dtype=np.float64,
    )
    return normalize_patient_vector(fallback_patient, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def volume_direction_to_patient_vector(
    direction: np.ndarray | tuple[float, float, float],
    transform: VolumePatientTransform | None,
) -> np.ndarray:
    if transform is not None:
        return transform.direction_to_patient(direction)
    return fallback_volume_direction_to_patient_vector(direction)


def resolve_mpr_orientation_screen_axes(
    normal_volume: np.ndarray | tuple[float, float, float],
    transform: VolumePatientTransform | None,
) -> tuple[np.ndarray, np.ndarray]:
    normal_patient = volume_direction_to_patient_vector(normal_volume, transform)
    patient_inferior = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    patient_posterior = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    y_vector = project_patient_direction_to_plane(patient_inferior, normal_patient)
    if y_vector is None:
        y_vector = project_patient_direction_to_plane(patient_posterior, normal_patient)
    if y_vector is None:
        y_vector = patient_posterior

    x_vector = normalize_patient_vector(np.cross(y_vector, normal_patient), fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    return x_vector, y_vector
