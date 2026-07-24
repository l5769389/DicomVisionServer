from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.measurement import MeasurementPoint, MeasurementRecord
from app.services.measurement_geometry import build_smooth_path_points
from app.services.render_layers.render_context import LayerSpace, RenderContext


MEASUREMENT_STROKE = (85, 231, 255, 255)
MEASUREMENT_STROKE_OUTLINE = (3, 15, 24, 220)
MEASUREMENT_LABEL_BG = (7, 16, 28, 232)
MEASUREMENT_LABEL_BORDER = (108, 201, 255, 188)
MEASUREMENT_TEXT = (235, 245, 255, 255)
MEASUREMENT_HANDLE_FILL = (255, 255, 255, 255)


class MeasurementLayer:
    name = "measurement"
    space: LayerSpace = "screen"
    resample = Image.Resampling.BILINEAR
    _SUPERSAMPLE_SCALE = 2

    def render(self, context: RenderContext) -> Image.Image | None:
        width = context.view.width or 0
        height = context.view.height or 0
        if width <= 0 or height <= 0 or not context.measurements:
            return None

        scale = self._SUPERSAMPLE_SCALE
        image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for measurement in context.measurements:
            screen_points = tuple(self._transform_point(point, context, scale=scale) for point in measurement.points)
            if not screen_points:
                continue
            self._draw_measurement(draw, font, measurement, screen_points, width * scale, height * scale, scale=scale)
        return image.resize((width, height), resample=Image.Resampling.LANCZOS)

    @staticmethod
    def _transform_point(point: MeasurementPoint, context: RenderContext, *, scale: int) -> tuple[float, float]:
        vector = np.asarray([point.x, point.y, 1.0], dtype=np.float64)
        mapped = context.image_transform.matrix @ vector
        return (float(mapped[0]) * scale, float(mapped[1]) * scale)

    def _draw_measurement(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        measurement: MeasurementRecord,
        points: tuple[tuple[float, float], ...],
        width: int,
        height: int,
        *,
        scale: int,
    ) -> None:
        if measurement.tool_type in {"line", "alignment-horizontal", "alignment-vertical"} and len(points) >= 2:
            self._draw_line(draw, points[:2], scale=scale)
        elif measurement.tool_type == "rect" and len(points) >= 2:
            self._draw_rect(draw, points[:2], scale=scale)
        elif measurement.tool_type == "ellipse" and len(points) >= 2:
            self._draw_ellipse(draw, points[:2], scale=scale)
        elif measurement.tool_type == "angle" and len(points) >= 3:
            self._draw_angle(draw, points[:3], scale=scale)
        elif measurement.tool_type == "curve" and len(points) >= 2:
            self._draw_curve(draw, points, scale=scale)
        elif measurement.tool_type == "freeform" and len(points) >= 3:
            self._draw_freeform(draw, points, scale=scale)
        else:
            return

        self._draw_handles(draw, points, scale=scale)
        self._draw_label(draw, font, measurement, points, width, height, scale=scale)

    def _draw_line(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        self._draw_polyline(draw, points, scale=scale)

    def _draw_rect(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        left = min(points[0][0], points[1][0])
        right = max(points[0][0], points[1][0])
        top = min(points[0][1], points[1][1])
        bottom = max(points[0][1], points[1][1])
        draw.rectangle((left, top, right, bottom), outline=MEASUREMENT_STROKE_OUTLINE, width=max(2, scale * 2))
        draw.rectangle((left, top, right, bottom), outline=MEASUREMENT_STROKE, width=max(1, scale))

    def _draw_ellipse(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        left = min(points[0][0], points[1][0])
        right = max(points[0][0], points[1][0])
        top = min(points[0][1], points[1][1])
        bottom = max(points[0][1], points[1][1])
        draw.ellipse((left, top, right, bottom), outline=MEASUREMENT_STROKE_OUTLINE, width=max(2, scale * 2))
        draw.ellipse((left, top, right, bottom), outline=MEASUREMENT_STROKE, width=max(1, scale))

    def _draw_angle(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        self._draw_polyline(draw, points[:2], scale=scale)
        self._draw_polyline(draw, points[1:3], scale=scale)

    def _draw_curve(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        self._draw_polyline(draw, build_smooth_path_points(points), scale=scale)

    def _draw_freeform(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        self._draw_polyline(draw, build_smooth_path_points(points, close_path=True), scale=scale)

    def _draw_polyline(self, draw: ImageDraw.ImageDraw, points: Iterable[tuple[float, float]], *, scale: int) -> None:
        point_list = list(points)
        draw.line(point_list, fill=MEASUREMENT_STROKE_OUTLINE, width=max(2, scale * 2), joint="curve")
        draw.line(point_list, fill=MEASUREMENT_STROKE, width=max(1, scale), joint="curve")

    def _draw_handles(self, draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...], *, scale: int) -> None:
        radius = 3 * scale
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=MEASUREMENT_HANDLE_FILL, outline=MEASUREMENT_STROKE_OUTLINE)

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        measurement: MeasurementRecord,
        points: tuple[tuple[float, float], ...],
        width: int,
        height: int,
        *,
        scale: int,
    ) -> None:
        lines = measurement.label_lines
        if not lines:
            return

        label_x, label_y = self._resolve_label_position(measurement, points)
        padding_x = 8 * scale
        padding_y = 6 * scale
        line_gap = 3 * scale
        line_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_width = max((bbox[2] - bbox[0]) for bbox in line_sizes)
        text_height = sum((bbox[3] - bbox[1]) for bbox in line_sizes) + max(0, len(lines) - 1) * line_gap
        margin = 6 * scale
        left = max(margin, min(width - text_width - padding_x * 2 - margin, int(round(label_x))))
        top = max(margin, min(height - text_height - padding_y * 2 - margin, int(round(label_y))))
        right = left + text_width + padding_x * 2
        bottom = top + text_height + padding_y * 2

        draw.rounded_rectangle((left, top, right, bottom), radius=7 * scale, fill=MEASUREMENT_LABEL_BG, outline=MEASUREMENT_LABEL_BORDER, width=max(1, scale))

        cursor_y = top + padding_y
        for index, line in enumerate(lines):
            bbox = line_sizes[index]
            draw.text((left + padding_x, cursor_y), line, fill=MEASUREMENT_TEXT, font=font)
            cursor_y += (bbox[3] - bbox[1]) + line_gap

    @staticmethod
    def _resolve_label_position(
        measurement: MeasurementRecord,
        points: tuple[tuple[float, float], ...],
    ) -> tuple[float, float]:
        if measurement.tool_type in {"line", "curve", "alignment-horizontal", "alignment-vertical"} and len(points) >= 2:
            anchor = points[-1]
            return (anchor[0] + 20.0, anchor[1] - 56.0)

        if measurement.tool_type in {"rect", "ellipse", "freeform"} and len(points) >= 2:
            top = min(point[1] for point in points)
            right = max(point[0] for point in points)
            return (right + 20.0, top - 48.0)

        if measurement.tool_type == "angle" and len(points) >= 3:
            vertex = points[1]
            return (vertex[0] + 20.0, vertex[1] - 56.0)

        return (points[0][0] + 20.0, points[0][1] - 56.0)
