from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .geometry import VolumeGeometry, spacing_along_world_direction
from .planes import PlanePose


DEFAULT_MIP_ALGORITHM = "maximum"


@dataclass(frozen=True)
class MipConfig:
    enabled: bool = False
    algorithm: str = DEFAULT_MIP_ALGORITHM
    thickness: int = 1
    max_samples: int | None = None


def _get_ndimage():
    from scipy import ndimage

    return ndimage


@lru_cache(maxsize=16)
def _plane_offset_grids(
    height: int,
    width: int,
    row_spacing_mm: float,
    col_spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_offsets_mm = (np.arange(height, dtype=np.float64) - (float(height) - 1.0) / 2.0) * float(row_spacing_mm)
    col_offsets_mm = (np.arange(width, dtype=np.float64) - (float(width) - 1.0) / 2.0) * float(col_spacing_mm)
    col_grid_mm, row_grid_mm = np.meshgrid(col_offsets_mm, row_offsets_mm)
    return col_grid_mm, row_grid_mm


def _slab_offsets_mm(geometry: VolumeGeometry, plane: PlanePose, mip: MipConfig | None) -> np.ndarray:
    if mip is None or not mip.enabled:
        return np.asarray([0.0], dtype=np.float64)

    step_mm = max(1e-6, float(spacing_along_world_direction(geometry, plane.normal_world)))
    thickness_mm = float(mip.thickness)
    if not np.isfinite(thickness_mm) or thickness_mm <= 0.0:
        return np.asarray([0.0], dtype=np.float64)
    sample_count = max(1, int(np.ceil(thickness_mm / step_mm)))
    # Offsets are centered on the displayed plane so enabling MIP does not shift
    # the current crosshair slice. Even sample counts must use half-step offsets
    # instead of one extra sample on either side.
    full_offsets = (np.arange(sample_count, dtype=np.float64) - (float(sample_count) - 1.0) / 2.0) * step_mm
    if mip.max_samples is None:
        return full_offsets.astype(np.float64)

    max_samples = max(1, int(mip.max_samples))
    if full_offsets.size <= max_samples:
        return full_offsets.astype(np.float64)
    if max_samples == 1:
        return np.asarray([0.0], dtype=np.float64)
    return np.linspace(float(full_offsets[0]), float(full_offsets[-1]), num=max_samples, dtype=np.float64)


def reslice_plane(
    volume: np.ndarray,
    geometry: VolumeGeometry,
    plane: PlanePose,
    mip: MipConfig | None,
    interpolation_order: int = 1,
) -> np.ndarray:
    height, width = plane.output_shape
    col_grid_mm, row_grid_mm = _plane_offset_grids(
        int(height),
        int(width),
        float(plane.pixel_spacing_row_mm),
        float(plane.pixel_spacing_col_mm),
    )
    slab_offsets_mm = _slab_offsets_mm(geometry, plane, mip)

    world_to_ijk = geometry.world_to_ijk[:3, :3]
    world_origin = geometry.world_to_ijk[:3, 3]
    center_world = np.asarray(plane.center_world, dtype=np.float64)
    row_world = np.asarray(plane.row_world, dtype=np.float64)
    col_world = np.asarray(plane.col_world, dtype=np.float64)
    normal_world = np.asarray(plane.normal_world, dtype=np.float64)
    center_ijk = world_to_ijk @ center_world + world_origin
    row_ijk_per_mm = world_to_ijk @ row_world
    col_ijk_per_mm = world_to_ijk @ col_world
    normal_ijk_per_mm = world_to_ijk @ normal_world
    ndimage = _get_ndimage()

    def sample_at_offset(slab_offset_mm: float) -> np.ndarray:
        coords = (
            center_ijk[:, None, None]
            + col_ijk_per_mm[:, None, None] * col_grid_mm[None, :, :]
            + row_ijk_per_mm[:, None, None] * row_grid_mm[None, :, :]
            + normal_ijk_per_mm[:, None, None] * float(slab_offset_mm)
        )
        return ndimage.map_coordinates(
            volume,
            coords,
            order=max(0, min(int(interpolation_order), 1)),
            mode="nearest",
        ).astype(np.float32, copy=False)

    if slab_offsets_mm.size == 1:
        return sample_at_offset(float(slab_offsets_mm[0]))

    algorithm = str(mip.algorithm or DEFAULT_MIP_ALGORITHM)
    accumulator = sample_at_offset(float(slab_offsets_mm[0])).astype(np.float32, copy=True)
    for slab_offset_mm in slab_offsets_mm[1:]:
        sample = sample_at_offset(float(slab_offset_mm))
        if algorithm == "minimum":
            np.minimum(accumulator, sample, out=accumulator)
        elif algorithm == "average" or algorithm == "sum":
            accumulator += sample
        else:
            np.maximum(accumulator, sample, out=accumulator)

    if algorithm == "average":
        accumulator /= float(slab_offsets_mm.size)
    return accumulator.astype(np.float32, copy=False)
