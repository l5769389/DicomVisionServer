from PIL import Image, ImageDraw

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.services.render_layers.render_context import ColorRGBA, LayerSpace, RenderContext


class MprCrosshairLayer:
    name = "mpr_crosshair"
    space: LayerSpace = "screen"
    resample = Image.Resampling.NEAREST

    def render(self, context: RenderContext) -> Image.Image | None:
        overlay = context.mpr_crosshair
        if overlay is None or context.mpr_viewport not in {MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL}:
            return None

        image = Image.new("RGBA", (overlay.width, overlay.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        min_size = min(overlay.width, overlay.height)
        stroke_width = 2 if min_size >= 700 else 1
        outline_width = stroke_width + 1
        outline_alpha = 168 if overlay.is_active else 132
        outline_color: ColorRGBA = (0, 0, 0, outline_alpha)

        center_x: int | None = None
        center_y: int | None = None
        gap_radius = 6 if overlay.is_active else 5

        if overlay.center_x is not None and overlay.center_y is not None:
            center_x = max(0, min(overlay.width - 1, int(round(overlay.center_x))))
            center_y = max(0, min(overlay.height - 1, int(round(overlay.center_y))))

        if overlay.horizontal_position is not None and overlay.horizontal_color is not None:
            y = max(0, min(overlay.height - 1, int(round(overlay.horizontal_position))))
            if center_x is not None and center_y is not None and abs(y - center_y) <= gap_radius:
                left_end = max(-1, center_x - gap_radius)
                right_start = min(overlay.width, center_x + gap_radius + 1)
                if left_end >= 0:
                    draw.line([(0, y), (left_end, y)], fill=outline_color, width=outline_width)
                    draw.line([(0, y), (left_end, y)], fill=overlay.horizontal_color, width=stroke_width)
                if right_start <= overlay.width - 1:
                    draw.line([(right_start, y), (overlay.width - 1, y)], fill=outline_color, width=outline_width)
                    draw.line([(right_start, y), (overlay.width - 1, y)], fill=overlay.horizontal_color, width=stroke_width)
            else:
                draw.line([(0, y), (overlay.width - 1, y)], fill=outline_color, width=outline_width)
                draw.line([(0, y), (overlay.width - 1, y)], fill=overlay.horizontal_color, width=stroke_width)

        if overlay.vertical_position is not None and overlay.vertical_color is not None:
            x = max(0, min(overlay.width - 1, int(round(overlay.vertical_position))))
            if center_x is not None and center_y is not None and abs(x - center_x) <= gap_radius:
                top_end = max(-1, center_y - gap_radius)
                bottom_start = min(overlay.height, center_y + gap_radius + 1)
                if top_end >= 0:
                    draw.line([(x, 0), (x, top_end)], fill=outline_color, width=outline_width)
                    draw.line([(x, 0), (x, top_end)], fill=overlay.vertical_color, width=stroke_width)
                if bottom_start <= overlay.height - 1:
                    draw.line([(x, bottom_start), (x, overlay.height - 1)], fill=outline_color, width=outline_width)
                    draw.line([(x, bottom_start), (x, overlay.height - 1)], fill=overlay.vertical_color, width=stroke_width)
            else:
                draw.line([(x, 0), (x, overlay.height - 1)], fill=outline_color, width=outline_width)
                draw.line([(x, 0), (x, overlay.height - 1)], fill=overlay.vertical_color, width=stroke_width)

        return image
