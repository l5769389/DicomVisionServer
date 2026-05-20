from typing import TYPE_CHECKING

from app.models.viewer import ViewRecord
from app.schemas.view import ImageFormat

if TYPE_CHECKING:
    from app.services.viewer_service import RenderedImageResult, ViewerService


def render_by_view_type(
    service: "ViewerService",
    view: ViewRecord,
    image_format: ImageFormat = "png",
    *,
    fast_preview: bool = False,
) -> "RenderedImageResult":
    """Route a view record to the renderer that owns its view type."""

    if service._is_mpr_view_type(view.view_type):
        return service._render_mpr_view(view, image_format=image_format, fast_preview=fast_preview)
    if service._is_3d_view_type(view.view_type):
        return service._render_3d_view(view, image_format=image_format, fast_preview=fast_preview)
    return service._render_view(view, image_format=image_format, fast_preview=fast_preview)
