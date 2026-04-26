from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.models.viewer import MprFrameState, MprObliquePlaneState


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
    default_planes = {
        MPR_VIEWPORT_AXIAL: MprObliquePlaneState(
            row=(0.0, 1.0, 0.0),
            col=(0.0, 0.0, 1.0),
            normal=(1.0, 0.0, 0.0),
        ),
        MPR_VIEWPORT_CORONAL: MprObliquePlaneState(
            row=(-1.0, 0.0, 0.0),
            col=(0.0, 0.0, 1.0),
            normal=(0.0, 1.0, 0.0),
        ),
        MPR_VIEWPORT_SAGITTAL: MprObliquePlaneState(
            row=(-1.0, 0.0, 0.0),
            col=(0.0, 1.0, 0.0),
            normal=(0.0, 0.0, 1.0),
        ),
    }
    return default_planes.get(viewport_key) or default_planes[MPR_VIEWPORT_AXIAL]


def get_mpr_display_basis(viewport_key: str, normal_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    default_plane = default_mpr_oblique_plane(viewport_key)
    canonical_row = normalize_oblique_vector(default_plane.row, fallback=(1.0, 0.0, 0.0))
    canonical_col = normalize_oblique_vector(default_plane.col, fallback=(0.0, 0.0, 1.0))
    normalized_normal = normalize_oblique_vector(normal_dir, fallback=tuple(default_plane.normal))
    projected_row = canonical_row - float(np.dot(canonical_row, normalized_normal)) * normalized_normal
    if float(np.linalg.norm(projected_row)) > 1e-8:
        row_dir = normalize_oblique_vector(projected_row, fallback=tuple(canonical_row))
        col_dir = normalize_oblique_vector(np.cross(normalized_normal, row_dir), fallback=tuple(canonical_col))
        return row_dir, col_dir
    projected_col = canonical_col - float(np.dot(canonical_col, normalized_normal)) * normalized_normal
    col_dir = normalize_oblique_vector(projected_col, fallback=tuple(canonical_col))
    row_dir = normalize_oblique_vector(np.cross(col_dir, normalized_normal), fallback=tuple(canonical_row))
    return row_dir, col_dir


def normalize_screen_half_turn_angle(angle_rad: float) -> float:
    return float(angle_rad % np.pi)


def direction_from_screen_angle(active_row: np.ndarray, active_col: np.ndarray, angle_rad: float) -> np.ndarray:
    return normalize_oblique_vector(
        np.cos(angle_rad) * active_col + np.sin(angle_rad) * active_row,
        fallback=tuple(active_col),
    )


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
