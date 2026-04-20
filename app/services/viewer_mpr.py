from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException

from app.core import MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.models.viewer import SeriesRecord, ViewRecord
from app.services.dicom_cache import dicom_cache
from app.services.dicom_geometry import build_standardized_volume, get_dataset_orientation, get_dataset_position


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


