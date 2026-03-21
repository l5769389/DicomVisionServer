import base64
import io

from fastapi import HTTPException
from PIL import Image

from app.models.viewer import ViewRecord
from app.schemas.view import SliceInfo, ViewImageResponse, ViewOperationRequest, ViewSetSizeRequest, WindowInfo
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer
from app.services.view_registry import view_registry


class ViewerService:
    def set_view_size(self, payload: ViewSetSizeRequest) -> ViewImageResponse:
        if payload.op_type != "setSize":
            raise HTTPException(status_code=400, detail="opType must be setSize")

        view = view_registry.get(payload.view_id)
        view.width = payload.size.width
        view.height = payload.size.height
        return self._render_view(view)

    def render_view_by_id(self, view_id: str) -> ViewImageResponse:
        view = view_registry.get(view_id)
        return self._render_view(view)

    def handle_view_operation(self, payload: ViewOperationRequest) -> ViewImageResponse:
        view = view_registry.get(payload.view_id)
        series = series_registry.get(view.series_id)

        if payload.scroll is not None:
            next_index = view.current_index + int(payload.scroll)
            view.current_index = max(0, min(next_index, len(series.instances) - 1))

        if payload.zoom is not None and payload.zoom > 0:
            view.zoom = float(payload.zoom)

        if payload.x is not None:
            view.offset_x = float(payload.x)
        if payload.y is not None:
            view.offset_y = float(payload.y)
        if payload.hor_flip is not None:
            view.hor_flip = payload.hor_flip
        if payload.ver_flip is not None:
            view.ver_flip = payload.ver_flip

        return self._render_view(view)

    def _render_view(self, view: ViewRecord) -> ViewImageResponse:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        cached = dicom_cache.get(instance.path)
        image = Image.fromarray(cached.image_array, mode="L")
        image = viewport_transformer.build_canvas(image, view.width, view.height, view)

        return ViewImageResponse(
            slice_info=SliceInfo(current=view.current_index, total=len(series.instances)),
            window_info=WindowInfo(ww=cached.window_width, wl=cached.window_center),
            image=self._image_to_base64(image),
            viewId=view.view_id,
        )

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")


viewer_service = ViewerService()
