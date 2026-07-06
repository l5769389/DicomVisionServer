from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VolumeRenderRequest:
    view_id: str
    volume: np.ndarray
    spacing_xyz: tuple[float, float, float]
    canvas_width: int
    canvas_height: int
    window_width: float
    window_center: float
    zoom: float
    offset_x: float
    offset_y: float
    rotation_quaternion: tuple[float, float, float, float]
    volume_preset: str = "bone"
    volume_config: dict[str, Any] | None = None
    fast_preview: bool = False
    volume_token: str | None = None


@dataclass(frozen=True)
class SurfaceRenderRequest:
    view_id: str
    volume: np.ndarray
    spacing_xyz: tuple[float, float, float]
    canvas_width: int
    canvas_height: int
    zoom: float
    offset_x: float
    offset_y: float
    rotation_quaternion: tuple[float, float, float, float]
    surface_config: dict[str, Any] | None = None
    fast_preview: bool = False
