from PIL import Image

from app.services.render_layers.render_context import LayerSpace, RenderContext


class MeasurementLayer:
    name = "measurement"
    space: LayerSpace = "image"
    resample = Image.Resampling.BILINEAR

    def render(self, context: RenderContext) -> Image.Image | None:
        return None
