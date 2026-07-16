from __future__ import annotations

import sys

from app.core.config import get_settings


THREE_D_VIEW_TYPE = "3D"


def vtk_requires_main_thread() -> bool:
    return sys.platform == "darwin"


def should_bypass_vtk_worker_thread() -> bool:
    return vtk_requires_main_thread()


def should_run_3d_view_on_main_thread(view_type: str | None) -> bool:
    return (
        vtk_requires_main_thread()
        and not get_settings().vtk_render_process_enabled
        and view_type == THREE_D_VIEW_TYPE
    )
