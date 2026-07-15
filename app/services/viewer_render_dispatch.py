from typing import TYPE_CHECKING, Callable

from app.models.viewer import ViewRecord
from app.schemas.view import ImageFormat

if TYPE_CHECKING:
    from app.services.viewer_service import RenderedImageResult, ViewerService

ViewRenderProgressCallback = Callable[[dict[str, object]], None]
FUSION_VIEW_TYPES = {
    "FusionCTAxial",
    "FusionPETAxial",
    "FusionOverlayAxial",
    "FusionPETCoronalMip",
}
PET_VIEW_TYPES = {"PET"}


def render_by_view_type(
    service: "ViewerService",
    view: ViewRecord,
    image_format: ImageFormat = "webp",
    *,
    fast_preview: bool = False,
    fast_preview_full_resolution: bool = False,
    metadata_mode: str = "full",
    progress_callback: ViewRenderProgressCallback | None = None,
) -> "RenderedImageResult":
    """Route a view record to the renderer that owns its view type."""

    if service._is_mpr_view_type(view.view_type):
        return service._render_mpr_view(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
        )
    if service._is_3d_view_type(view.view_type):
        return service._render_3d_view(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            progress_callback=progress_callback,
        )
    if view.view_type in FUSION_VIEW_TYPES:
        return service._render_fusion_view(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
        )
    if view.view_type in PET_VIEW_TYPES:
        return service._render_pet_view(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
        )
    return service._render_view(
        view,
        image_format=image_format,
        fast_preview=fast_preview,
        metadata_mode=metadata_mode,
    )
