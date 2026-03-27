from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException

from app.core import MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.models.viewer import SeriesRecord, ViewRecord
from app.services.dicom_cache import dicom_cache


def extract_mpr_plane(
    view: ViewRecord,
    volume: np.ndarray,
    target_viewport: str,
) -> tuple[np.ndarray, int, int]:
    depth, height, width = volume.shape
    if target_viewport == MPR_VIEWPORT_CORONAL:
        index = max(0, min(view.mpr_coronal_index, height - 1))
        plane = np.flipud(volume[:, index, :])
        return plane.astype(np.float32), index, height
    if target_viewport == MPR_VIEWPORT_SAGITTAL:
        index = max(0, min(view.mpr_sagittal_index, width - 1))
        plane = np.flipud(volume[:, :, index])
        return plane.astype(np.float32), index, width
    index = max(0, min(view.mpr_axial_index, depth - 1))
    view.current_index = index
    plane = volume[index, :, :]
    return plane.astype(np.float32), index, depth


def get_series_volume(
    series: SeriesRecord,
    volume_cache: dict[str, np.ndarray],
    *,
    logger: Any,
) -> np.ndarray:
    cached_volume = volume_cache.get(series.series_id)
    if cached_volume is not None:
        return cached_volume

    slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
    for instance in series.instances:
        if not instance.sop_instance_uid:
            continue
        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        dataset = cached.dataset
        orientation = get_dataset_orientation(dataset)
        position = get_dataset_position(dataset)
        slice_entries.append((cached.source_pixels, orientation, position))

    if not slice_entries:
        raise HTTPException(status_code=400, detail="Series does not contain readable pixel data")

    first_shape = slice_entries[0][0].shape
    if any(item[0].shape != first_shape for item in slice_entries):
        raise HTTPException(status_code=400, detail="MPR requires a series with consistent slice dimensions")

    volume = build_standardized_volume(slice_entries, logger=logger)
    volume_cache[series.series_id] = volume
    return volume


def get_dataset_orientation(dataset) -> np.ndarray | None:
    value = getattr(dataset, "ImageOrientationPatient", None)
    if value is None or len(value) < 6:
        return None
    try:
        orientation = np.asarray([float(item) for item in value[:6]], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return orientation if np.all(np.isfinite(orientation)) else None


def get_dataset_position(dataset) -> np.ndarray | None:
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


def build_standardized_volume(
    slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    *,
    logger: Any,
) -> np.ndarray:
    orientation = next((item[1] for item in slice_entries if item[1] is not None), None)
    if orientation is None:
        return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

    row_direction = normalize_vector(orientation[:3])
    column_direction = normalize_vector(orientation[3:6])
    if row_direction is None or column_direction is None:
        return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

    slice_direction = normalize_vector(np.cross(row_direction, column_direction))
    if slice_direction is None:
        return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

    positions = [item[2] for item in slice_entries]
    if any(position is None for position in positions):
        ordered_entries = slice_entries
    else:
        ordered_entries = sorted(
            slice_entries,
            key=lambda item: float(np.dot(item[2], slice_direction)) if item[2] is not None else 0.0,
        )

    raw_volume = np.stack([item[0] for item in ordered_entries], axis=0).astype(np.float32)
    raw_axis_vectors = (slice_direction, column_direction, row_direction)
    patient_axes: list[int] = []
    axis_signs: list[int] = []

    for vector in raw_axis_vectors:
        patient_axis = int(np.argmax(np.abs(vector)))
        if patient_axis in patient_axes:
            logger.warning("falling back to non-standardized volume because orientation axes are not orthogonal enough")
            return raw_volume
        patient_axes.append(patient_axis)
        axis_signs.append(1 if vector[patient_axis] >= 0 else -1)

    transpose_order = [patient_axes.index(2), patient_axes.index(1), patient_axes.index(0)]
    canonical_signs = [
        axis_signs[patient_axes.index(2)],
        axis_signs[patient_axes.index(1)],
        axis_signs[patient_axes.index(0)],
    ]
    volume = np.transpose(raw_volume, axes=transpose_order)
    for axis, sign in enumerate(canonical_signs):
        if sign < 0:
            volume = np.flip(volume, axis=axis)

    logger.info(
        "standardized MPR volume shape=%s raw_axes=%s canonical_signs=%s row_dir=%s col_dir=%s slice_dir=%s",
        volume.shape,
        patient_axes,
        canonical_signs,
        np.round(row_direction, 4).tolist(),
        np.round(column_direction, 4).tolist(),
        np.round(slice_direction, 4).tolist(),
    )
    return volume.astype(np.float32, copy=False)
