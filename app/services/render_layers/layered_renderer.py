from PIL import Image

from app.services.render_layers.base_image_layer import BaseImageLayer
from app.services.render_layers.corner_info_layer import CornerInfoLayer
from app.services.render_layers.measurement_layer import MeasurementLayer
from app.services.render_layers.render_context import RenderContext, RenderLayer
from app.services.viewport_transformer import viewport_transformer
from app.utils.utils import timer


class LayeredRenderer:
    def __init__(self) -> None:
        self._base_layer = BaseImageLayer()
        self._overlay_layers: tuple[RenderLayer, ...] = (
            CornerInfoLayer(),
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

        if not self._has_overlay_content(context):
            return Image.fromarray(transformed_base, mode="L")

        canvas = Image.fromarray(transformed_base, mode="L").convert("RGBA")
        return self.composite_overlays(canvas, context)

    def composite_overlays(self, canvas: Image.Image, context: RenderContext) -> Image.Image:
        canvas_width = context.view.width or 0
        canvas_height = context.view.height or 0

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
            if layer_image.mode != "RGBA":
                layer_image = layer_image.convert("RGBA")
            canvas.alpha_composite(layer_image)
        return canvas

    @staticmethod
    def _has_overlay_content(context: RenderContext) -> bool:
        return any(
            (
                context.corner_info is not None,
            )
        )


layered_renderer = LayeredRenderer()
