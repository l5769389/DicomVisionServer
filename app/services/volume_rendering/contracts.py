from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class VtkRenderTimings:
    vtk_render_ms: float = 0.0
    gpu_readback_ms: float = 0.0
    session_ms: float = 0.0
    configure_ms: float = 0.0
    ipc_ms: float = 0.0
    source_dtype: str = ""
    vtk_dtype: str = ""

    def as_dict(self) -> dict[str, float | str]:
        return {
            "vtk_render_ms": self.vtk_render_ms,
            "gpu_readback_ms": self.gpu_readback_ms,
            "session_ms": self.session_ms,
            "configure_ms": self.configure_ms,
            "ipc_ms": self.ipc_ms,
            "source_dtype": self.source_dtype,
            "vtk_dtype": self.vtk_dtype,
        }


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
    volume_token: str | None = None
    progress_callback: Callable[[dict[str, object]], None] | None = None
