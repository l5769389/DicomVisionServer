import numpy as np

from app.services.pseudocolor import DEFAULT_PSEUDOCOLOR_PRESET, apply_pseudocolor
from app.services.render_layers.render_context import RenderContext


class BaseImageLayer:
    name = "base_image"

    def render_pixels(self, context: RenderContext) -> np.ndarray:
        pixels = context.source_pixels
        ww = context.view.window_width or (context.cached.window_width if context.cached is not None else None)
        wl = context.view.window_center or (context.cached.window_center if context.cached is not None else None)

        if ww is not None and ww > 0 and wl is not None:
            lower = wl - ww / 2.0
            upper = wl + ww / 2.0
        else:
            lower = context.pixel_min
            upper = context.pixel_max

        clipped = np.clip(pixels, lower, upper)
        scale = upper - lower
        if scale <= 0:
            return np.zeros_like(clipped, dtype=np.uint8)

        normalized = (clipped - lower) / scale
        grayscale = (normalized * 255.0).astype(np.uint8)
        if context.view.pseudocolor_preset == DEFAULT_PSEUDOCOLOR_PRESET:
            return grayscale
        return apply_pseudocolor(grayscale, context.view.pseudocolor_preset)
