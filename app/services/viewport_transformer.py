from PIL import Image, ImageOps

from app.models.viewer import ViewRecord


class ViewportTransformer:
    def build_canvas(self, image: Image.Image, width: int, height: int, view: ViewRecord) -> Image.Image:
        transformed = image

        if view.hor_flip:
            transformed = ImageOps.mirror(transformed)
        if view.ver_flip:
            transformed = ImageOps.flip(transformed)

        zoom = max(view.zoom, 1.0)
        resized_width = max(1, int(transformed.width * zoom))
        resized_height = max(1, int(transformed.height * zoom))
        transformed = transformed.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        canvas = Image.new("L", (width, height), color=0)
        left = int((width - transformed.width) / 2 + view.offset_x)
        top = int((height - transformed.height) / 2 + view.offset_y)
        canvas.paste(transformed, (left, top))
        return canvas


viewport_transformer = ViewportTransformer()
