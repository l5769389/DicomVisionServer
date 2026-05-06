from __future__ import annotations

from typing import TypeAlias


Point2D: TypeAlias = tuple[float, float]

DEFAULT_SMOOTH_SAMPLES_PER_SEGMENT = 12


def build_smooth_path_points(
    points: tuple[Point2D, ...],
    *,
    close_path: bool = False,
    samples_per_segment: int = DEFAULT_SMOOTH_SAMPLES_PER_SEGMENT,
) -> tuple[Point2D, ...]:
    if len(points) < 2:
        return tuple(points)

    sample_count = max(1, int(samples_per_segment))
    sampled: list[Point2D] = [points[0]]
    segment_count = len(points) if close_path and len(points) > 2 else len(points) - 1

    for index in range(segment_count):
        start = points[index]
        end = points[(index + 1) % len(points)]
        previous = (
            points[(index - 1 + len(points)) % len(points)]
            if close_path and len(points) > 2
            else points[max(0, index - 1)]
        )
        next_point = (
            points[(index + 2) % len(points)]
            if close_path and len(points) > 2
            else points[min(len(points) - 1, index + 2)]
        )
        control_point1 = (
            start[0] + (end[0] - previous[0]) / 6.0,
            start[1] + (end[1] - previous[1]) / 6.0,
        )
        control_point2 = (
            end[0] - (next_point[0] - start[0]) / 6.0,
            end[1] - (next_point[1] - start[1]) / 6.0,
        )

        for step in range(1, sample_count + 1):
            sampled.append(_cubic_bezier_point(start, control_point1, control_point2, end, step / sample_count))

    return tuple(sampled)


def _cubic_bezier_point(
    start: Point2D,
    control_point1: Point2D,
    control_point2: Point2D,
    end: Point2D,
    t: float,
) -> Point2D:
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    return (
        mt2 * mt * start[0] + 3.0 * mt2 * t * control_point1[0] + 3.0 * mt * t2 * control_point2[0] + t2 * t * end[0],
        mt2 * mt * start[1] + 3.0 * mt2 * t * control_point1[1] + 3.0 * mt * t2 * control_point2[1] + t2 * t * end[1],
    )
