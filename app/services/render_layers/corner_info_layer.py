from PIL import Image, ImageDraw, ImageFont

from app.services.render_layers.render_context import LayerSpace, RenderContext


class CornerInfoLayer:
    name = "corner_info"
    space: LayerSpace = "screen"
    resample = Image.Resampling.NEAREST

    def render(self, context: RenderContext) -> Image.Image | None:
        overlay = context.corner_info
        width = context.view.width or 0
        height = context.view.height or 0
        if overlay is None or width <= 0 or height <= 0:
            return None

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font_size = max(12, int(min(width, height) * 0.024))
        font = _load_font(font_size)
        margin = max(12, int(font_size * 0.75))
        line_gap = max(2, font_size // 5)

        self._draw_block(draw, font, overlay.top_left, margin, margin, "left", "top", width, height, line_gap)
        self._draw_block(draw, font, overlay.top_right, width - margin, margin, "right", "top", width, height, line_gap)
        self._draw_block(
            draw,
            font,
            overlay.bottom_left,
            margin,
            height - margin,
            "left",
            "bottom",
            width,
            height,
            line_gap,
        )
        self._draw_block(
            draw,
            font,
            overlay.bottom_right,
            width - margin,
            height - margin,
            "right",
            "bottom",
            width,
            height,
            line_gap,
        )
        return image

    @staticmethod
    def _draw_block(
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        lines: tuple[str, ...],
        anchor_x: int,
        anchor_y: int,
        align_x: str,
        align_y: str,
        width: int,
        height: int,
        line_gap: int,
    ) -> None:
        normalized_lines = tuple(line for line in lines if line)
        if not normalized_lines:
            return

        line_heights = []
        line_widths = []
        for line in normalized_lines:
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            line_widths.append(right - left)
            line_heights.append(bottom - top)

        total_height = sum(line_heights) + line_gap * max(0, len(normalized_lines) - 1)
        start_y = anchor_y if align_y == "top" else anchor_y - total_height
        current_y = max(0, min(height - total_height, start_y))

        for index, line in enumerate(normalized_lines):
            line_width = line_widths[index]
            if align_x == "right":
                x = max(0, min(width - line_width, anchor_x - line_width))
            else:
                x = max(0, min(width - line_width, anchor_x))
            _draw_shadowed_text(draw, (x, current_y), line, font)
            current_y += line_heights[index] + line_gap


def _draw_shadowed_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    x, y = position
    shadow = (0, 0, 0, 220)
    foreground = (255, 164, 164, 244)
    for offset_x, offset_y in ((1, 1), (1, 0), (0, 1)):
        draw.text((x + offset_x, y + offset_y), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=foreground, font=font)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("msyh.ttc", "simhei.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
