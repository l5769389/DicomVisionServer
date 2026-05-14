from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

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


def _reduce_slab(slab: np.ndarray, algorithm: str) -> np.ndarray:
    reducer = SLAB_REDUCERS.get(algorithm, SLAB_REDUCERS[DEFAULT_MIP_ALGORITHM])
    return reducer(slab, axis=0)


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
) -> np.ndarray:
    height, width = plane.output_shape
    row_offsets_mm = (np.arange(height, dtype=np.float64) - (float(height) - 1.0) / 2.0) * float(plane.pixel_spacing_row_mm)
    col_offsets_mm = (np.arange(width, dtype=np.float64) - (float(width) - 1.0) / 2.0) * float(plane.pixel_spacing_col_mm)
    col_grid_mm, row_grid_mm = np.meshgrid(col_offsets_mm, row_offsets_mm)
    slab_offsets_mm = _slab_offsets_mm(geometry, plane, mip)

    sampled_planes: list[np.ndarray] = []
    world_to_ijk = geometry.world_to_ijk[:3, :3]
    world_origin = geometry.world_to_ijk[:3, 3]
    center_world = np.asarray(plane.center_world, dtype=np.float64)
    row_world = np.asarray(plane.row_world, dtype=np.float64)
    col_world = np.asarray(plane.col_world, dtype=np.float64)
    normal_world = np.asarray(plane.normal_world, dtype=np.float64)

    for slab_offset_mm in slab_offsets_mm:
        world_points = (
            center_world[:, None, None]
            + col_world[:, None, None] * col_grid_mm[None, :, :]
            + row_world[:, None, None] * row_grid_mm[None, :, :]
            + normal_world[:, None, None] * float(slab_offset_mm)
        )
        coords = np.tensordot(world_to_ijk, world_points, axes=([1], [0])) + world_origin[:, None, None]
        sampled = ndimage.map_coordinates(volume, coords, order=1, mode="nearest")
        sampled_planes.append(sampled.astype(np.float32, copy=False))

    slab = np.stack(sampled_planes, axis=0)
    if mip is None or not mip.enabled:
        return slab[0].astype(np.float32, copy=False)

    return _reduce_slab(slab, str(mip.algorithm or DEFAULT_MIP_ALGORITHM)).astype(np.float32, copy=False)
