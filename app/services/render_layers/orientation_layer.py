from PIL import Image, ImageDraw, ImageFont

from app.services.render_layers.render_context import LayerSpace, RenderContext


class OrientationLayer:
    name = "orientation"
    space: LayerSpace = "screen"
    resample = Image.Resampling.NEAREST

    def render(self, context: RenderContext) -> Image.Image | None:
        overlay = context.orientation
        width = context.view.width or 0
        height = context.view.height or 0
        if overlay is None or width <= 0 or height <= 0:
            return None

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font_size = max(18, int(min(width, height) * 0.04))
        font = _load_font(font_size)
        edge_margin = max(14, int(font_size * 0.8))

        if overlay.top:
            _draw_centered_label(draw, (width // 2, edge_margin), overlay.top, font, "ma")
        if overlay.bottom:
            _draw_centered_label(draw, (width // 2, height - edge_margin), overlay.bottom, font, "md")
        if overlay.left:
            _draw_centered_label(draw, (edge_margin, height // 2), overlay.left, font, "lm")
        if overlay.right:
            _draw_centered_label(draw, (width - edge_margin, height // 2), overlay.right, font, "rm")

        return image


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    anchor: str,
) -> None:
    shadow = (0, 0, 0, 220)
    foreground = (255, 164, 164, 244)
    x, y = position
    for offset_x, offset_y in ((1, 1), (1, 0), (0, 1)):
        draw.text((x + offset_x, y + offset_y), text, fill=shadow, font=font, anchor=anchor)
    draw.text((x, y), text, fill=foreground, font=font, anchor=anchor)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("arialbd.ttf", "msyhbd.ttc", "arial.ttf", "msyh.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
