from __future__ import annotations

from typing import Any

import numpy as np


def map_normalized_canvas_to_image_row_col(
    normalized_x: float,
    normalized_y: float,
    *,
    image_width: int,
    image_height: int,
    canvas_width: int,
    canvas_height: int,
    image_transform: Any,
) -> tuple[int, int]:
    """Map a normalized canvas coordinate back into 1-based image row/col space.

    The frontend reports pointer positions normalized to the canvas size. Rendering
    may apply pan/zoom/fit transforms, so hover lookup must invert the image-to-canvas
    transform before converting to source pixel coordinates.
    """

    if image_width <= 0 or image_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
        return (0, 0)

    x = max(0.0, min(1.0, float(normalized_x)))
    y = max(0.0, min(1.0, float(normalized_y)))
    max_canvas_x = max(float(canvas_width) - 1e-6, 0.0)
    max_canvas_y = max(float(canvas_height) - 1e-6, 0.0)
    canvas_x = min(max(x * float(canvas_width), 0.0), max_canvas_x)
    canvas_y = min(max(y * float(canvas_height), 0.0), max_canvas_y)

    affine_matrix, offset = image_transform.inverse_components()
    source_point = affine_matrix @ np.asarray([canvas_x, canvas_y], dtype=np.float64) + offset
    source_x = float(source_point[0])
    source_y = float(source_point[1])

    if source_x < 0.0 or source_x >= float(image_width) or source_y < 0.0 or source_y >= float(image_height):
        return (0, 0)

    row = int(np.floor(source_y)) + 1
    col = int(np.floor(source_x)) + 1
    return (row, col)
