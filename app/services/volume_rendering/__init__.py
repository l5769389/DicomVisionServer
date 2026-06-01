from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from app.services.volume_rendering.contracts import SurfaceRenderRequest, VolumeRenderRequest

if TYPE_CHECKING:
    from app.services.volume_rendering.vtk_surface_renderer import VtkSurfaceRenderer
    from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer


__all__ = [
    "SurfaceRenderRequest",
    "VtkSurfaceRenderer",
    "vtk_surface_renderer",
    "VolumeRenderRequest",
    "VtkVolumeRenderer",
    "vtk_volume_renderer",
]


def __getattr__(name: str) -> Any:
    if name in {"VtkVolumeRenderer", "vtk_volume_renderer"}:
        volume_module = import_module("app.services.volume_rendering.vtk_volume_renderer")
        return getattr(volume_module, name)
    if name in {"VtkSurfaceRenderer", "vtk_surface_renderer"}:
        surface_module = import_module("app.services.volume_rendering.vtk_surface_renderer")
        return getattr(surface_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
