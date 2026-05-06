from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.models.measurement import (
    MeasurementAreaUnit,
    MeasurementMetrics,
    MeasurementPoint,
    MeasurementToolType,
    MeasurementUnit,
)


@dataclass(frozen=True)
class _PixelStats:
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None


def clamp_point_to_image(point: MeasurementPoint, image_width: int, image_height: int) -> MeasurementPoint:
    max_x = max(float(image_width - 1), 0.0)
    max_y = max(float(image_height - 1), 0.0)
    return MeasurementPoint(
        x=max(0.0, min(max_x, float(point.x))),
        y=max(0.0, min(max_y, float(point.y))),
    )


def build_measurement_metrics(
    tool_type: MeasurementToolType,
    points: tuple[MeasurementPoint, ...],
    source_pixels: np.ndarray,
    spacing_xy: tuple[float, float] | None,
) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    if tool_type == "line":
        return _build_line_metrics(points[:2], spacing_xy)
    if tool_type == "rect":
        return _build_rect_metrics(points[:2], source_pixels, spacing_xy)
    if tool_type == "ellipse":
        return _build_ellipse_metrics(points[:2], source_pixels, spacing_xy)
    if tool_type == "angle":
        return _build_angle_metrics(points[:3])
    if tool_type == "curve":
        return _build_curve_metrics(points, spacing_xy)
    if tool_type == "freeform":
        return _build_freeform_metrics(points, source_pixels, spacing_xy)
    raise ValueError(f"Unsupported measurement tool type: {tool_type}")


def _build_line_metrics(
    points: tuple[MeasurementPoint, ...],
    spacing_xy: tuple[float, float] | None,
) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    start, end = points
    dx = float(end.x - start.x)
    dy = float(end.y - start.y)
    if spacing_xy is not None:
        length = math.hypot(dx * spacing_xy[0], dy * spacing_xy[1])
        metrics = MeasurementMetrics(unit="mm", area_unit="mm2", length=length)
        return (metrics, (f"{length:.1f} mm",))
    length = math.hypot(dx, dy)
    metrics = MeasurementMetrics(unit="px", area_unit="px2", length=length)
    return (metrics, (f"{length:.1f} px",))


def _build_rect_metrics(
    points: tuple[MeasurementPoint, ...],
    source_pixels: np.ndarray,
    spacing_xy: tuple[float, float] | None,
) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    left, top, right, bottom = _resolve_bounds(points)
    clipped_left, clipped_top, clipped_right, clipped_bottom = _clip_bounds_to_image(left, top, right, bottom, source_pixels)
    roi = source_pixels[clipped_top : clipped_bottom + 1, clipped_left : clipped_right + 1]
    stats = _sample_pixel_stats(roi)
    pixel_width = max(0, right - left)
    pixel_height = max(0, bottom - top)
    if spacing_xy is not None:
        width = pixel_width * spacing_xy[0]
        height = pixel_height * spacing_xy[1]
        area = width * height
        metrics = MeasurementMetrics(
            unit="mm",
            area_unit="mm2",
            width=width,
            height=height,
            area=area,
            mean=stats.mean,
            standard_deviation=stats.standard_deviation,
            minimum=stats.minimum,
            maximum=stats.maximum,
        )
        return (
            metrics,
            _build_roi_label_lines(
                width=width,
                height=height,
                area=area,
                length_unit="mm",
                area_unit="mm2",
                mean=stats.mean,
                minimum=stats.minimum,
                maximum=stats.maximum,
                standard_deviation=stats.standard_deviation,
            ),
        )

    area = float(pixel_width * pixel_height)
    metrics = MeasurementMetrics(
        unit="px",
        area_unit="px2",
        width=float(pixel_width),
        height=float(pixel_height),
        area=area,
        mean=stats.mean,
        standard_deviation=stats.standard_deviation,
        minimum=stats.minimum,
        maximum=stats.maximum,
    )
    return (
        metrics,
        _build_roi_label_lines(
            width=float(pixel_width),
            height=float(pixel_height),
            area=area,
            length_unit="px",
            area_unit="px2",
            mean=stats.mean,
            minimum=stats.minimum,
            maximum=stats.maximum,
            standard_deviation=stats.standard_deviation,
        ),
    )


def _build_ellipse_metrics(
    points: tuple[MeasurementPoint, ...],
    source_pixels: np.ndarray,
    spacing_xy: tuple[float, float] | None,
) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    left, top, right, bottom = _resolve_bounds(points)
    clipped_left, clipped_top, clipped_right, clipped_bottom = _clip_bounds_to_image(left, top, right, bottom, source_pixels)
    roi = source_pixels[clipped_top : clipped_bottom + 1, clipped_left : clipped_right + 1]
    if roi.size:
        yy, xx = np.indices(roi.shape, dtype=np.float64)
        radius_x = max((right - left) / 2.0, 1e-6)
        radius_y = max((bottom - top) / 2.0, 1e-6)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        source_x = xx + float(clipped_left)
        source_y = yy + float(clipped_top)
        mask = ((source_x - center_x) / radius_x) ** 2 + ((source_y - center_y) / radius_y) ** 2 <= 1.0
        masked = roi[mask]
    else:
        masked = np.asarray([], dtype=np.float32)

    stats = _sample_pixel_stats(masked)
    pixel_width = max(0.0, float(right - left))
    pixel_height = max(0.0, float(bottom - top))
    if spacing_xy is not None:
        width = pixel_width * spacing_xy[0]
        height = pixel_height * spacing_xy[1]
        area = math.pi * (width / 2.0) * (height / 2.0)
        metrics = MeasurementMetrics(
            unit="mm",
            area_unit="mm2",
            width=width,
            height=height,
            area=area,
            mean=stats.mean,
            standard_deviation=stats.standard_deviation,
            minimum=stats.minimum,
            maximum=stats.maximum,
        )
        return (
            metrics,
            _build_roi_label_lines(
                width=width,
                height=height,
                area=area,
                length_unit="mm",
                area_unit="mm2",
                mean=stats.mean,
                minimum=stats.minimum,
                maximum=stats.maximum,
                standard_deviation=stats.standard_deviation,
            ),
        )

    area = math.pi * (pixel_width / 2.0) * (pixel_height / 2.0)
    metrics = MeasurementMetrics(
        unit="px",
        area_unit="px2",
        width=pixel_width,
        height=pixel_height,
        area=area,
        mean=stats.mean,
        standard_deviation=stats.standard_deviation,
        minimum=stats.minimum,
        maximum=stats.maximum,
    )
    return (
        metrics,
        _build_roi_label_lines(
            width=pixel_width,
            height=pixel_height,
            area=area,
            length_unit="px",
            area_unit="px2",
            mean=stats.mean,
            minimum=stats.minimum,
            maximum=stats.maximum,
            standard_deviation=stats.standard_deviation,
        ),
    )


def _build_angle_metrics(points: tuple[MeasurementPoint, ...]) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    start, vertex, end = points
    vector_a = np.asarray([start.x - vertex.x, start.y - vertex.y], dtype=np.float64)
    vector_b = np.asarray([end.x - vertex.x, end.y - vertex.y], dtype=np.float64)
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    if length_a <= 1e-6 or length_b <= 1e-6:
        angle = 0.0
    else:
        cosine = float(np.dot(vector_a, vector_b) / (length_a * length_b))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    metrics = MeasurementMetrics(unit="px", area_unit="px2", angle_degrees=angle)
    return (metrics, (f"{angle:.1f}\u00b0",))


def _build_curve_metrics(
    points: tuple[MeasurementPoint, ...],
    spacing_xy: tuple[float, float] | None,
) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    length = 0.0
    for index in range(1, len(points)):
        dx = float(points[index].x - points[index - 1].x)
        dy = float(points[index].y - points[index - 1].y)
        if spacing_xy is not None:
            length += math.hypot(dx * spacing_xy[0], dy * spacing_xy[1])
        else:
            length += math.hypot(dx, dy)

    if spacing_xy is not None:
        metrics = MeasurementMetrics(unit="mm", area_unit="mm2", length=length)
        return (metrics, (f"{length:.1f} mm",))
    metrics = MeasurementMetrics(unit="px", area_unit="px2", length=length)
    return (metrics, (f"{length:.1f} px",))


def _build_freeform_metrics(
    points: tuple[MeasurementPoint, ...],
    source_pixels: np.ndarray,
    spacing_xy: tuple[float, float] | None,
) -> tuple[MeasurementMetrics, tuple[str, ...]]:
    left, top, right, bottom = _resolve_bounds_for_points(points)
    clipped_left, clipped_top, clipped_right, clipped_bottom = _clip_bounds_to_image(left, top, right, bottom, source_pixels)
    roi = source_pixels[clipped_top : clipped_bottom + 1, clipped_left : clipped_right + 1]
    mask = _build_polygon_mask(points, left=clipped_left, top=clipped_top, shape=roi.shape) if roi.size else np.asarray([], dtype=bool)
    masked = roi[mask] if roi.size and mask.size else np.asarray([], dtype=np.float32)

    stats = _sample_pixel_stats(masked)
    pixel_width = max(0.0, float(right - left))
    pixel_height = max(0.0, float(bottom - top))
    pixel_area = float(np.count_nonzero(mask)) if mask.size else 0.0

    if spacing_xy is not None:
        width = pixel_width * spacing_xy[0]
        height = pixel_height * spacing_xy[1]
        area = pixel_area * spacing_xy[0] * spacing_xy[1]
        metrics = MeasurementMetrics(
            unit="mm",
            area_unit="mm2",
            width=width,
            height=height,
            area=area,
            mean=stats.mean,
            standard_deviation=stats.standard_deviation,
            minimum=stats.minimum,
            maximum=stats.maximum,
        )
        return (
            metrics,
            _build_roi_label_lines(
                width=width,
                height=height,
                area=area,
                length_unit="mm",
                area_unit="mm2",
                mean=stats.mean,
                minimum=stats.minimum,
                maximum=stats.maximum,
                standard_deviation=stats.standard_deviation,
            ),
        )

    metrics = MeasurementMetrics(
        unit="px",
        area_unit="px2",
        width=pixel_width,
        height=pixel_height,
        area=pixel_area,
        mean=stats.mean,
        standard_deviation=stats.standard_deviation,
        minimum=stats.minimum,
        maximum=stats.maximum,
    )
    return (
        metrics,
        _build_roi_label_lines(
            width=pixel_width,
            height=pixel_height,
            area=pixel_area,
            length_unit="px",
            area_unit="px2",
            mean=stats.mean,
            minimum=stats.minimum,
            maximum=stats.maximum,
            standard_deviation=stats.standard_deviation,
        ),
    )


def _resolve_bounds(points: tuple[MeasurementPoint, ...]) -> tuple[int, int, int, int]:
    x0 = int(round(points[0].x))
    y0 = int(round(points[0].y))
    x1 = int(round(points[1].x))
    y1 = int(round(points[1].y))
    left = min(x0, x1)
    right = max(x0, x1)
    top = min(y0, y1)
    bottom = max(y0, y1)
    return (left, top, right, bottom)


def _resolve_bounds_for_points(points: tuple[MeasurementPoint, ...]) -> tuple[int, int, int, int]:
    xs = [int(round(point.x)) for point in points]
    ys = [int(round(point.y)) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _clip_bounds_to_image(
    left: int,
    top: int,
    right: int,
    bottom: int,
    source_pixels: np.ndarray,
) -> tuple[int, int, int, int]:
    height, width = source_pixels.shape[:2]
    if height <= 0 or width <= 0 or right < 0 or bottom < 0 or left >= width or top >= height:
        return (0, 0, -1, -1)
    return (
        max(0, left),
        max(0, top),
        min(width - 1, right),
        min(height - 1, bottom),
    )


def _sample_pixel_stats(values: np.ndarray) -> _PixelStats:
    if not values.size:
        return _PixelStats(mean=None, standard_deviation=None, minimum=None, maximum=None)

    return _PixelStats(
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )


def _build_polygon_mask(
    points: tuple[MeasurementPoint, ...],
    *,
    left: int,
    top: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    height, width = shape[:2]
    if height <= 0 or width <= 0:
        return np.asarray([], dtype=bool)

    yy, xx = np.indices((height, width), dtype=np.float64)
    x = xx + float(left)
    y = yy + float(top)
    inside = np.zeros((height, width), dtype=bool)
    previous = points[-1]
    for current in points:
        y_crosses = (current.y > y) != (previous.y > y)
        x_intersection = (previous.x - current.x) * (y - current.y) / ((previous.y - current.y) or 1e-9) + current.x
        inside ^= y_crosses & (x < x_intersection)
        previous = current
    return inside


def _build_roi_label_lines(
    *,
    width: float,
    height: float,
    area: float,
    length_unit: str,
    area_unit: str,
    mean: float | None,
    minimum: float | None,
    maximum: float | None,
    standard_deviation: float | None,
) -> tuple[str, ...]:
    return (
        f"Size {width:.1f} * {height:.1f} {length_unit}",
        f"Area {area:.1f} {area_unit}",
        _format_stat_label("Mean", mean),
        _format_stat_label("Min", minimum),
        _format_stat_label("Max", maximum),
        _format_stat_label("SD", standard_deviation),
    )


def _format_stat_label(name: str, value: float | None) -> str:
    return f"{name} {value:.1f}" if value is not None else f"{name} -"
