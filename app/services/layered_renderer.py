from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from app.models.viewer import InstanceRecord, ViewRecord
from app.services.dicom_cache import CachedDicom
from app.services.viewport_transformer import AffineTransform, viewport_transformer

LayerSpace = Literal["image", "screen"]


@dataclass(frozen=True)
class RenderContext:
    view: ViewRecord
    instance: InstanceRecord
    cached: CachedDicom
    image_transform: AffineTransform


class RenderLayer(Protocol):
    name: str
    space: LayerSpace
    resample: Image.Resampling

    def render(self, context: RenderContext) -> Image.Image | None: ...


class BaseImageLayer:
    name = "base_image"

    def render_pixels(self, context: RenderContext) -> np.ndarray:
        pixels = context.cached.source_pixels
        ww = context.view.window_width or context.cached.window_width
        wl = context.view.window_center or context.cached.window_center

        if ww is not None and ww > 0 and wl is not None:
            lower = wl - ww / 2.0
            upper = wl + ww / 2.0
        else:
            lower = context.cached.pixel_min
            upper = context.cached.pixel_max

        clipped = np.clip(pixels, lower, upper)
        scale = upper - lower
        if scale <= 0:
            return np.zeros_like(clipped, dtype=np.uint8)

        normalized = (clipped - lower) / scale
        return (normalized * 255.0).astype(np.uint8)


class CornerInfoLayer:
    name = "corner_info"
    space: LayerSpace = "screen"
    resample = Image.Resampling.NEAREST

    def render(self, context: RenderContext) -> Image.Image | None:
        return None


class OrientationLayer:
    name = "orientation"
    space: LayerSpace = "image"
    resample = Image.Resampling.BILINEAR

    def render(self, context: RenderContext) -> Image.Image | None:
        return None


class MeasurementLayer:
    name = "measurement"
    space: LayerSpace = "image"
    resample = Image.Resampling.BILINEAR

    def render(self, context: RenderContext) -> Image.Image | None:
        return None


class LayeredRenderer:
    def __init__(self) -> None:
        self._base_layer = BaseImageLayer()
        self._overlay_layers: tuple[RenderLayer, ...] = (
            CornerInfoLayer(),
            OrientationLayer(),
            MeasurementLayer(),
        )

    def render(self, context: RenderContext) -> Image.Image:
        canvas_width = context.view.width or 0
        canvas_height = context.view.height or 0

        base_pixels = self._base_layer.render_pixels(context)
        transformed_base = viewport_transformer.apply_affine_array(
            base_pixels,
            canvas_width,
            canvas_height,
            context.image_transform,
            order=1,
            cval=0.0,
        )

        overlay_images: list[Image.Image] = []
        for layer in self._overlay_layers:
            layer_image = layer.render(context)
            if layer_image is None:
                continue
            if layer.space == "image":
                layer_image = viewport_transformer.apply_affine(
                    layer_image,
                    canvas_width,
                    canvas_height,
                    context.image_transform,
                    resample=layer.resample,
                )
            overlay_images.append(layer_image)

        if not overlay_images:
            return Image.fromarray(transformed_base, mode="L")

        canvas = Image.fromarray(transformed_base, mode="L").convert("RGBA")
        for overlay_image in overlay_images:
            if overlay_image.mode != "RGBA":
                overlay_image = overlay_image.convert("RGBA")
            canvas.alpha_composite(overlay_image)
        return canvas


layered_renderer = LayeredRenderer()
