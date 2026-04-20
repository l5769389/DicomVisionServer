from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StandardizedAxisMapping:
    row_direction: np.ndarray
    column_direction: np.ndarray
    slice_direction: np.ndarray
    patient_axes: tuple[int, int, int]
    canonical_signs: tuple[int, int, int]
    transpose_order: tuple[int, int, int]


def get_dataset_orientation(dataset: Any) -> np.ndarray | None:
    value = getattr(dataset, "ImageOrientationPatient", None)
    if value is None or len(value) < 6:
        return None
    try:
        orientation = np.asarray([float(item) for item in value[:6]], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return orientation if np.all(np.isfinite(orientation)) else None


def get_dataset_position(dataset: Any) -> np.ndarray | None:
    value = getattr(dataset, "ImagePositionPatient", None)
    if value is None or len(value) < 3:
        return None
    try:
        position = np.asarray([float(item) for item in value[:3]], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return position if np.all(np.isfinite(position)) else None


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def get_standardized_axis_mapping(
    orientation: np.ndarray,
    *,
    logger: Any | None = None,
) -> StandardizedAxisMapping | None:
    row_direction = normalize_vector(orientation[:3])
    column_direction = normalize_vector(orientation[3:6])
    if row_direction is None or column_direction is None:
        return None

    slice_direction = normalize_vector(np.cross(row_direction, column_direction))
    if slice_direction is None:
        return None

    raw_axis_vectors = (slice_direction, column_direction, row_direction)
    patient_axes: list[int] = []
    axis_signs: list[int] = []

    for vector in raw_axis_vectors:
        patient_axis = int(np.argmax(np.abs(vector)))
        if patient_axis in patient_axes:
            if logger is not None:
                logger.warning("falling back to non-standardized volume because orientation axes are not orthogonal enough")
            return None
        patient_axes.append(patient_axis)
        axis_signs.append(1 if vector[patient_axis] >= 0 else -1)

    return StandardizedAxisMapping(
        row_direction=row_direction,
        column_direction=column_direction,
        slice_direction=slice_direction,
        patient_axes=(patient_axes[0], patient_axes[1], patient_axes[2]),
        canonical_signs=(
            axis_signs[patient_axes.index(2)],
            axis_signs[patient_axes.index(1)],
            axis_signs[patient_axes.index(0)],
        ),
        transpose_order=(
            patient_axes.index(2),
            patient_axes.index(1),
            patient_axes.index(0),
        ),
    )


def build_standardized_volume(
    slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    *,
    logger: Any,
) -> np.ndarray:
    orientation = next((item[1] for item in slice_entries if item[1] is not None), None)
    if orientation is None:
        return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

    axis_mapping = get_standardized_axis_mapping(orientation, logger=logger)
    if axis_mapping is None:
        return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

    positions = [item[2] for item in slice_entries]
    if any(position is None for position in positions):
        ordered_entries = slice_entries
    else:
        ordered_entries = sorted(
            slice_entries,
            key=lambda item: float(np.dot(item[2], axis_mapping.slice_direction)) if item[2] is not None else 0.0,
        )

    raw_volume = np.stack([item[0] for item in ordered_entries], axis=0).astype(np.float32)
    volume = np.transpose(raw_volume, axes=axis_mapping.transpose_order)
    for axis, sign in enumerate(axis_mapping.canonical_signs):
        if sign < 0:
            volume = np.flip(volume, axis=axis)

    logger.info(
        "standardized MPR volume shape=%s raw_axes=%s canonical_signs=%s row_dir=%s col_dir=%s slice_dir=%s",
        volume.shape,
        axis_mapping.patient_axes,
        axis_mapping.canonical_signs,
        np.round(axis_mapping.row_direction, 4).tolist(),
        np.round(axis_mapping.column_direction, 4).tolist(),
        np.round(axis_mapping.slice_direction, 4).tolist(),
    )
    return volume.astype(np.float32, copy=False)
