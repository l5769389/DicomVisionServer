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


SLAB_REDUCERS = {
    "minimum": np.min,
    "average": np.mean,
    "sum": np.sum,
    DEFAULT_MIP_ALGORITHM: np.max,
}


def _get_ndimage():
    from scipy import ndimage

    return ndimage


def _reduce_slab(slab: np.ndarray, algorithm: str) -> np.ndarray:
    reducer = SLAB_REDUCERS.get(algorithm, SLAB_REDUCERS[DEFAULT_MIP_ALGORITHM])
    return reducer(slab, axis=0)


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

    thickness = max(1, int(mip.thickness))
    half_before = (thickness - 1) // 2
    step_mm = spacing_along_world_direction(geometry, plane.normal_world)
    # Offsets are centered on the displayed plane so enabling MIP does not shift
    # the current crosshair slice.
    return (np.arange(-half_before, thickness - half_before, dtype=np.float64) * step_mm).astype(np.float64)


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

    sampled_planes: list[np.ndarray] = []
    for slab_offset_mm in slab_offsets_mm:
        sampled_planes.append(sample_at_offset(float(slab_offset_mm)))

    slab = np.stack(sampled_planes, axis=0)
    return _reduce_slab(slab, str(mip.algorithm or DEFAULT_MIP_ALGORITHM)).astype(np.float32, copy=False)
