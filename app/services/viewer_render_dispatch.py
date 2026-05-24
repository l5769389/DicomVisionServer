from typing import TYPE_CHECKING, Callable

from app.models.viewer import ViewRecord
from app.schemas.view import ImageFormat

if TYPE_CHECKING:
    from app.services.viewer_service import RenderedImageResult, ViewerService

ViewRenderProgressCallback = Callable[[dict[str, object]], None]


def render_by_view_type(
    service: "ViewerService",
    view: ViewRecord,
    image_format: ImageFormat = "png",
    *,
    fast_preview: bool = False,
    progress_callback: ViewRenderProgressCallback | None = None,
) -> "RenderedImageResult":
    """Route a view record to the renderer that owns its view type."""

    if service._is_mpr_view_type(view.view_type):
        return service._render_mpr_view(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            progress_callback=progress_callback,
        )
    if service._is_3d_view_type(view.view_type):
        return service._render_3d_view(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            progress_callback=progress_callback,
        )
    return service._render_view(view, image_format=image_format, fast_preview=fast_preview)
