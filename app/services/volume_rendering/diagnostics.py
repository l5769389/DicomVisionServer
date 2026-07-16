from __future__ import annotations

import platform
import re
from typing import Any

import numpy as np
from vtkmodules.vtkCommonCore import vtkVersion
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.util.numpy_support import numpy_to_vtk


def collect_vtk_render_diagnostics() -> dict[str, Any]:
    """Create a tiny offscreen context and report the renderer VTK actually uses."""

    volume = np.zeros((8, 8, 8), dtype=np.int16)
    volume[2:6, 2:6, 2:6] = 200
    image_data = vtkImageData()
    image_data.SetDimensions(8, 8, 8)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    scalars = numpy_to_vtk(volume.ravel(order="C"), deep=True)
    image_data.GetPointData().SetScalars(scalars)

    mapper = vtkSmartVolumeMapper()
    mapper.SetInputData(image_data)
    if hasattr(mapper, "SetRequestedRenderModeToGPU"):
        mapper.SetRequestedRenderModeToGPU()

    color = vtkColorTransferFunction()
    color.AddRGBPoint(-1000.0, 0.0, 0.0, 0.0)
    color.AddRGBPoint(200.0, 1.0, 1.0, 1.0)
    opacity = vtkPiecewiseFunction()
    opacity.AddPoint(-1000.0, 0.0)
    opacity.AddPoint(200.0, 0.8)
    prop = vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)

    actor = vtkVolume()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    renderer = vtkRenderer()
    renderer.AddVolume(actor)
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(32, 32)
    window.AddRenderer(renderer)

    capabilities = ""
    error: str | None = None
    try:
        renderer.ResetCamera()
        window.Render()
        reporter = getattr(window, "ReportCapabilities", None)
        if callable(reporter):
            capabilities = str(reporter() or "")
    except Exception as exc:  # pragma: no cover - depends on the host OpenGL stack
        error = f"{type(exc).__name__}: {exc}"
    finally:
        window.Finalize()

    renderer_name = _extract_capability(capabilities, "OpenGL renderer string")
    vendor = _extract_capability(capabilities, "OpenGL vendor string")
    version = _extract_capability(capabilities, "OpenGL version string")
    mapper_mode = _safe_int_call(mapper, "GetLastUsedRenderMode")
    software = "llvmpipe" in renderer_name.lower() or "software" in renderer_name.lower()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vtk": vtkVersion.GetVTKVersion(),
        "opengl_vendor": vendor or "unknown",
        "opengl_renderer": renderer_name or "unknown",
        "opengl_version": version or "unknown",
        "mapper_mode": mapper_mode,
        "software_renderer": software,
        "error": error,
        "capabilities": capabilities,
    }


def _extract_capability(capabilities: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", capabilities, flags=re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _safe_int_call(target: object, method_name: str) -> int | None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return None
    try:
        return int(method())
    except Exception:
        return None
