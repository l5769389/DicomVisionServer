import io
from dataclasses import dataclass, replace

import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_SET_SIZE,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
    WINDOW_DRAG_SENSITIVITY,
    WINDOW_WIDTH_MIN,
    ZOOM_DRAG_FACTOR_MIN,
    ZOOM_DRAG_SENSITIVITY,
)
from app.core.logging import get_logger
from app.models.viewer import ViewRecord
from app.schemas.view import (
    ImageFormat,
    OperationAcceptedResponse,
    SliceInfo,
    ViewImageResponse,
    ViewOperationRequest,
    ViewSetSizeRequest,
    WindowInfo,
)
from app.services.dicom_cache import CachedDicom, dicom_cache
from app.services.layered_renderer import RenderContext, layered_renderer
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer
from app.services.view_registry import view_registry


logger = get_logger(__name__)


@dataclass(frozen=True)
class RenderedImageResult:
    meta: ViewImageResponse
    image_bytes: bytes


@dataclass(frozen=True)
class RenderPlan:
    render_view: ViewRecord
    render_ratio: float


class ViewerService:
    def set_view_size(self, payload: ViewSetSizeRequest) -> OperationAcceptedResponse:
        if payload.op_type != VIEW_OP_TYPE_SET_SIZE:
            raise HTTPException(status_code=400, detail="opType must be setSize")

        view = view_registry.get(payload.view_id)
        view.width = payload.size.width
        view.height = payload.size.height
        logger.info("set view size view_id=%s width=%s height=%s", view.view_id, view.width, view.height)

        if not view.is_initialized:
            self._initialize_viewport(view)
            view.is_initialized = True

        return OperationAcceptedResponse(message="View size updated", viewId=view.view_id)

    def render_view_by_id(self, view_id: str) -> RenderedImageResult:
        view = view_registry.get(view_id)
        return self._render_view(view, image_format="png")

    def handle_view_operation(self, payload: ViewOperationRequest) -> RenderedImageResult | None:
        view = view_registry.get(payload.view_id)
        series = series_registry.get(view.series_id)

        if payload.scroll is not None:
            next_index = view.current_index + int(payload.scroll)
            view.current_index = max(0, min(next_index, len(series.instances) - 1))

        if payload.op_type == VIEW_OP_TYPE_ZOOM and payload.action_type is not None:
            self._handle_drag_zoom(view, payload)
        elif payload.op_type == VIEW_OP_TYPE_WINDOW and payload.action_type is not None:
            self._handle_drag_window(view, payload)
        elif payload.op_type == VIEW_OP_TYPE_PAN and payload.action_type is not None:
            self._handle_drag_pan(view, payload)
        elif payload.zoom is not None and payload.zoom > 0:
            view.zoom = viewport_transformer.clamp_zoom(payload.zoom)
            view.is_initialized = True

        if payload.x is not None and payload.op_type not in {VIEW_OP_TYPE_ZOOM, VIEW_OP_TYPE_WINDOW, VIEW_OP_TYPE_PAN}:
            view.offset_x += float(payload.x)
            view.is_initialized = True
        if payload.y is not None and payload.op_type not in {VIEW_OP_TYPE_ZOOM, VIEW_OP_TYPE_WINDOW, VIEW_OP_TYPE_PAN}:
            view.offset_y += float(payload.y)
            view.is_initialized = True
        if payload.hor_flip is not None:
            view.hor_flip = payload.hor_flip
            view.is_initialized = True
        if payload.ver_flip is not None:
            view.ver_flip = payload.ver_flip
            view.is_initialized = True

        logger.info(
            "view operation view_id=%s op_type=%s action_type=%s index=%s zoom=%.4f offset_x=%.2f offset_y=%.2f ww=%s wl=%s",
            view.view_id,
            payload.op_type,
            payload.action_type,
            view.current_index,
            view.zoom,
            view.offset_x,
            view.offset_y,
            view.window_width,
            view.window_center,
        )

        if payload.op_type in {VIEW_OP_TYPE_WINDOW, VIEW_OP_TYPE_ZOOM, VIEW_OP_TYPE_PAN} and payload.action_type == DRAG_ACTION_START:
            return None

        if payload.op_type == VIEW_OP_TYPE_WINDOW and payload.action_type == DRAG_ACTION_MOVE:
            return self._render_view(view, image_format="jpeg", fast_window=True)

        return self._render_view(view, image_format="png")

    def _initialize_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        image_height, image_width = cached.source_pixels.shape[:2]
        view.zoom = viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.window_width = cached.window_width or self._derive_default_window_width(cached)
        view.window_center = cached.window_center or self._derive_default_window_center(cached)
        self._reset_drag_state(view)
        logger.info(
            "viewport initialized view_id=%s image_width=%s image_height=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            image_width,
            image_height,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    def _render_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_window: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        render_plan = self._build_render_plan(view, cached)
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=cached.source_pixels.shape[1],
            image_height=cached.source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )

        if fast_window:
            image = self._render_fast_window_image(render_plan, cached, image_transform)
        else:
            image = layered_renderer.render(
                RenderContext(
                    view=render_plan.render_view,
                    instance=instance,
                    cached=cached,
                    image_transform=image_transform,
                )
            )

        logger.debug(
            "render completed view_id=%s index=%s viewport=%sx%s render=%sx%s ratio=%.4f zoom=%.4f ww=%s wl=%s image_format=%s fast_window=%s",
            view.view_id,
            view.current_index,
            view.width,
            view.height,
            render_plan.render_view.width,
            render_plan.render_view.height,
            render_plan.render_ratio,
            view.zoom,
            view.window_width,
            view.window_center,
            image_format,
            fast_window,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=len(series.instances)),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    @staticmethod
    def _render_fast_window_image(
        render_plan: RenderPlan,
        cached: CachedDicom,
        image_transform,
    ) -> Image.Image:
        base_pixels = ViewerService._window_pixels(
            cached,
            render_plan.render_view.window_width,
            render_plan.render_view.window_center,
        )
        transformed = viewport_transformer.apply_affine_array(
            base_pixels,
            render_plan.render_view.width or 0,
            render_plan.render_view.height or 0,
            image_transform,
            order=1,
            cval=0.0,
        )
        return Image.fromarray(transformed, mode="L")

    def _build_render_plan(self, view: ViewRecord, cached: CachedDicom) -> RenderPlan:
        render_ratio = self._resolve_render_ratio(view, cached)
        if render_ratio >= 0.999:
            return RenderPlan(render_view=view, render_ratio=1.0)

        render_width = max(1, int(round((view.width or 1) * render_ratio)))
        render_height = max(1, int(round((view.height or 1) * render_ratio)))
        render_view = replace(
            view,
            width=render_width,
            height=render_height,
            zoom=view.zoom * render_ratio,
            offset_x=view.offset_x * render_ratio,
            offset_y=view.offset_y * render_ratio,
        )
        return RenderPlan(render_view=render_view, render_ratio=render_ratio)

    @staticmethod
    def _resolve_render_ratio(view: ViewRecord, cached: CachedDicom) -> float:
        if not view.width or not view.height:
            return 1.0

        image_height, image_width = cached.source_pixels.shape[:2]
        if view.width <= image_width or view.height <= image_height:
            return 1.0

        contain_zoom = viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        if view.zoom > contain_zoom:
            return 1.0

        width_ratio = image_width / view.width
        height_ratio = image_height / view.height
        return max(width_ratio, height_ratio)

    def _handle_drag_pan(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_offset_x = view.offset_x
            view.drag_origin_offset_y = view.offset_y
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_offset_x = view.drag_origin_offset_x if view.drag_origin_offset_x is not None else view.offset_x
            base_offset_y = view.drag_origin_offset_y if view.drag_origin_offset_y is not None else view.offset_y
            view.offset_x = float(base_offset_x) + float(payload.x or 0.0)
            view.offset_y = float(base_offset_y) + float(payload.y or 0.0)
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_offset_x = None
            view.drag_origin_offset_y = None

    def _handle_drag_zoom(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_zoom = view.zoom
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_zoom = view.drag_origin_zoom if view.drag_origin_zoom is not None else view.zoom
            delta_y = float(payload.y or 0.0)
            zoom_factor = 1.0 - delta_y * ZOOM_DRAG_SENSITIVITY
            zoom_factor = max(ZOOM_DRAG_FACTOR_MIN, zoom_factor)
            view.zoom = viewport_transformer.clamp_zoom(float(base_zoom) * zoom_factor)
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_zoom = None

    def _handle_drag_window(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_window_width = view.window_width
            view.drag_origin_window_center = view.window_center
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_ww = view.drag_origin_window_width if view.drag_origin_window_width is not None else view.window_width
            base_wl = view.drag_origin_window_center if view.drag_origin_window_center is not None else view.window_center
            base_ww = max(WINDOW_WIDTH_MIN, float(base_ww or WINDOW_WIDTH_MIN))
            base_wl = float(base_wl or 0.0)
            delta_x = float(payload.x or 0.0)
            delta_y = float(payload.y or 0.0)
            view.window_width = max(WINDOW_WIDTH_MIN, base_ww + delta_x * WINDOW_DRAG_SENSITIVITY)
            view.window_center = base_wl - delta_y * WINDOW_DRAG_SENSITIVITY
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_window_width = None
            view.drag_origin_window_center = None

    @staticmethod
    def _window_pixels(cached: CachedDicom, window_width: float | None, window_center: float | None) -> np.ndarray:
        pixels = cached.source_pixels
        ww = window_width or cached.window_width
        wl = window_center or cached.window_center

        if ww is not None and ww > 0 and wl is not None:
            lower = wl - ww / 2.0
            upper = wl + ww / 2.0
        else:
            lower = cached.pixel_min
            upper = cached.pixel_max

        scale = upper - lower
        if scale <= 0:
            return np.zeros(pixels.shape, dtype=np.uint8)

        normalized = (pixels - lower) * (255.0 / scale)
        clipped = np.clip(normalized, 0.0, 255.0)
        return clipped.astype(np.uint8)

    @staticmethod
    def _derive_default_window_width(cached: CachedDicom) -> float:
        return max(WINDOW_WIDTH_MIN, cached.pixel_max - cached.pixel_min)

    @staticmethod
    def _derive_default_window_center(cached: CachedDicom) -> float:
        return (cached.pixel_max + cached.pixel_min) / 2.0

    @staticmethod
    def _reset_drag_state(view: ViewRecord) -> None:
        view.drag_origin_zoom = None
        view.drag_origin_offset_x = None
        view.drag_origin_offset_y = None
        view.drag_origin_window_width = None
        view.drag_origin_window_center = None

    @staticmethod
    def _encode_image(image: Image.Image, image_format: ImageFormat) -> bytes:
        output = io.BytesIO()
        if image_format == "jpeg":
            image.convert("RGB").save(output, format="JPEG", quality=20)
        else:
            image.save(output, format="PNG")
        return output.getvalue()


viewer_service = ViewerService()
