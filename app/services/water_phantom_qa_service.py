from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from fastapi import HTTPException
from scipy import ndimage

from app.models.viewer import ViewRecord
from app.schemas.view import (
    QaWaterAccuracyMetricsPayload,
    QaWaterMetricsPayload,
    QaWaterNoiseMetricsPayload,
    QaWaterRoiPayload,
    QaWaterRoiStatsPayload,
    QaWaterUniformityMetricsPayload,
    ViewQaWaterAnalyzeRequest,
    ViewQaWaterAnalyzeResponse,
)
from app.services.view_registry import view_registry


class WaterPhantomQaContext(Protocol):
    def _resolve_measurement_source_context(self, view: ViewRecord) -> tuple[np.ndarray, tuple[float, float] | None, object]:
        ...

    def _build_hover_mapping_context(self, view: ViewRecord) -> tuple[int, int, Any, int, int]:
        ...


RoiSource = tuple[str, str, str, float, float, float]


class WaterPhantomQaService:
    def __init__(self, context: WaterPhantomQaContext) -> None:
        self._context = context

    def analyze(self, payload: ViewQaWaterAnalyzeRequest) -> ViewQaWaterAnalyzeResponse:
        view = view_registry.get(payload.view_id)
        if view.view_type not in {"Stack", "MPR", "AX", "COR", "SAG"}:
            raise HTTPException(status_code=400, detail="Water phantom QA is only available for 2D views")

        source_pixels, spacing_xy, _ = self._context._resolve_measurement_source_context(view)
        image_height = int(source_pixels.shape[0])
        image_width = int(source_pixels.shape[1])
        image_width_for_transform, image_height_for_transform, image_transform, canvas_width, canvas_height = self._context._build_hover_mapping_context(view)
        if (
            image_width <= 0
            or image_height <= 0
            or image_width_for_transform <= 0
            or image_height_for_transform <= 0
            or canvas_width <= 0
            or canvas_height <= 0
        ):
            return ViewQaWaterAnalyzeResponse(
                viewId=view.view_id,
                viewportKey=payload.viewport_key,
                rois=[],
                status="error",
                message="Current view is not ready for water phantom QA.",
            )

        detected = self._detect_water_phantom_geometry(source_pixels)
        if detected is None:
            return ViewQaWaterAnalyzeResponse(
                viewId=view.view_id,
                viewportKey=payload.viewport_key,
                rois=[],
                status="error",
                message="No water phantom contour was detected in the current image.",
            )

        center_x, center_y, phantom_radius = detected
        water_roi_radius = max(6.0, phantom_radius * 0.12)
        air_roi_radius = water_roi_radius
        peripheral_distance = phantom_radius * 0.55
        water_positions = (
            ("center", "Center", center_x, center_y),
            ("top", "Top", center_x, center_y - peripheral_distance),
            ("right", "Right", center_x + peripheral_distance, center_y),
            ("bottom", "Bottom", center_x, center_y + peripheral_distance),
            ("left", "Left", center_x - peripheral_distance, center_y),
        )

        air_x, air_y = self._resolve_air_roi_center(
            center_x,
            center_y,
            phantom_radius,
            air_roi_radius,
            image_width,
            image_height,
        )

        roi_sources: list[RoiSource] = [
            (roi_id, label, "water", x, y, water_roi_radius)
            for roi_id, label, x, y in water_positions
        ]
        roi_sources.append(("air", "Air", "air", air_x, air_y, air_roi_radius))
        rois = [
            self._build_roi_payload(
                roi_id,
                label,
                kind,
                x,
                y,
                radius,
                image_transform,
                canvas_width,
                canvas_height,
            )
            for roi_id, label, kind, x, y, radius in roi_sources
        ]
        metrics = self._build_metrics(
            source_pixels,
            roi_sources,
            spacing_xy=spacing_xy,
            enabled_metrics={str(metric).strip().lower() for metric in payload.metrics},
        )

        return ViewQaWaterAnalyzeResponse(
            viewId=view.view_id,
            viewportKey=payload.viewport_key,
            rois=rois,
            metrics=metrics,
            status="ready",
            message=None,
        )

    @staticmethod
    def _resolve_air_roi_center(
        center_x: float,
        center_y: float,
        phantom_radius: float,
        air_radius: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        distance = phantom_radius + air_radius * 2.9
        diagonal_distance = distance / float(np.sqrt(2.0))
        candidates = (
            (center_x + diagonal_distance, center_y - diagonal_distance),
            (center_x + diagonal_distance, center_y + diagonal_distance),
            (center_x + distance, center_y),
            (center_x - distance, center_y),
            (center_x, center_y + distance),
            (center_x, center_y - distance),
        )
        min_clearance = phantom_radius + air_radius * 1.6
        for x, y in candidates:
            if (
                air_radius <= x <= float(image_width) - air_radius
                and air_radius <= y <= float(image_height) - air_radius
                and float(np.hypot(x - center_x, y - center_y)) >= min_clearance
            ):
                return x, y

        clamped_candidates = (
            (float(image_width) - air_radius, air_radius),
            (float(image_width) - air_radius, float(image_height) - air_radius),
            (float(image_width) - air_radius, center_y),
            (air_radius, center_y),
            (center_x, float(image_height) - air_radius),
            (center_x, air_radius),
        )
        fallback_points = tuple(
            (
                max(air_radius, min(float(image_width) - air_radius, x)),
                max(air_radius, min(float(image_height) - air_radius, y)),
            )
            for x, y in clamped_candidates
        )
        for point in fallback_points:
            if float(np.hypot(point[0] - center_x, point[1] - center_y)) >= min_clearance:
                return point
        return max(fallback_points, key=lambda point: float(np.hypot(point[0] - center_x, point[1] - center_y)))

    @staticmethod
    def _sample_circular_roi_stats(
        source_pixels: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> tuple[float, float, int]:
        height, width = source_pixels.shape[:2]
        min_x = max(0, int(np.floor(center_x - radius)))
        max_x = min(width - 1, int(np.ceil(center_x + radius)))
        min_y = max(0, int(np.floor(center_y - radius)))
        max_y = min(height - 1, int(np.ceil(center_y + radius)))
        if max_x < min_x or max_y < min_y:
            return 0.0, 0.0, 0

        y_grid, x_grid = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]
        mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
        values = np.asarray(source_pixels[min_y : max_y + 1, min_x : max_x + 1], dtype=np.float64)[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0, 0.0, 0

        return float(np.mean(values)), float(np.std(values)), int(values.size)

    def _build_metrics(
        self,
        source_pixels: np.ndarray,
        roi_sources: list[RoiSource],
        *,
        spacing_xy: tuple[float, float] | None,
        enabled_metrics: set[str],
    ) -> QaWaterMetricsPayload:
        stats_by_id: dict[str, tuple[float, float, int]] = {
            roi_id: self._sample_circular_roi_stats(source_pixels, center_x, center_y, radius)
            for roi_id, _, _, center_x, center_y, radius in roi_sources
        }
        center_mean, center_std_dev, _ = stats_by_id.get("center", (0.0, 0.0, 0))
        peripheral_means = [
            stats_by_id[roi_id][0]
            for roi_id in ("top", "right", "bottom", "left")
            if roi_id in stats_by_id and stats_by_id[roi_id][2] > 0
        ]
        metrics = QaWaterMetricsPayload()
        water_roi_stats = [
            self._build_roi_stats_payload(
                roi_id,
                label,
                kind,
                radius,
                stats_by_id[roi_id],
                center_mean=center_mean,
                spacing_xy=spacing_xy,
            )
            for roi_id, label, kind, _, _, radius in roi_sources
            if kind == "water" and roi_id in stats_by_id and stats_by_id[roi_id][2] > 0
        ]

        if "accuracy" in enabled_metrics:
            metrics.accuracy = QaWaterAccuracyMetricsPayload(
                centerMean=round(center_mean, 2),
                deviationHu=round(center_mean, 2),
                targetHu=0.0,
                unit="HU",
            )
        if "uniformity" in enabled_metrics:
            max_deviation = max((abs(mean - center_mean) for mean in peripheral_means), default=0.0)
            metrics.uniformity = QaWaterUniformityMetricsPayload(
                centerMean=round(center_mean, 2),
                maxDeviation=round(max_deviation, 2),
                peripheralMeans=[round(mean, 2) for mean in peripheral_means],
                roiStats=water_roi_stats,
                unit="HU",
            )
        if "noise" in enabled_metrics:
            metrics.noise = QaWaterNoiseMetricsPayload(
                stdDev=round(center_std_dev, 2),
                unit="HU",
            )

        return metrics

    @staticmethod
    def _build_roi_stats_payload(
        roi_id: str,
        label: str,
        kind: str,
        radius: float,
        stats: tuple[float, float, int],
        *,
        center_mean: float,
        spacing_xy: tuple[float, float] | None,
    ) -> QaWaterRoiStatsPayload:
        mean, std_dev, sample_count = stats
        pixel_width = float(radius * 2.0)
        pixel_height = float(radius * 2.0)
        if spacing_xy is not None:
            width = pixel_width * spacing_xy[0]
            height = pixel_height * spacing_xy[1]
            area = float(np.pi * (width / 2.0) * (height / 2.0))
            size_unit = "mm"
            area_unit = "mm2"
        else:
            width = pixel_width
            height = pixel_height
            area = float(np.pi * radius * radius)
            size_unit = "px"
            area_unit = "px2"

        return QaWaterRoiStatsPayload(
            id=roi_id,
            label=label,
            kind=kind,
            area=round(area, 2),
            width=round(width, 2),
            height=round(height, 2),
            mean=round(mean, 2),
            stdDev=round(std_dev, 2),
            sampleCount=sample_count,
            deviationFromCenter=round(mean - center_mean, 2) if roi_id != "center" else 0.0,
            sizeUnit=size_unit,
            areaUnit=area_unit,
            unit="HU",
        )

    @staticmethod
    def _compute_otsu_threshold(values: np.ndarray) -> int:
        histogram = np.bincount(values.ravel().astype(np.uint8), minlength=256)
        total = int(values.size)
        weighted_sum = float(np.dot(np.arange(256), histogram))
        background_weight = 0.0
        background_sum = 0.0
        max_variance = 0.0
        threshold = 0

        for index, count in enumerate(histogram):
            background_weight += float(count)
            if background_weight <= 0:
                continue
            foreground_weight = float(total) - background_weight
            if foreground_weight <= 0:
                break
            background_sum += float(index * count)
            background_mean = background_sum / background_weight
            foreground_mean = (weighted_sum - background_sum) / foreground_weight
            variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
            if variance > max_variance:
                max_variance = variance
                threshold = index

        return threshold

    @staticmethod
    def _find_largest_mask_component(mask: np.ndarray) -> tuple[int, int, int, int, int, float, float] | None:
        structure = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=bool)
        labels, label_count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
        if label_count <= 0:
            return None

        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0
        label_index = int(np.argmax(component_sizes))
        area = int(component_sizes[label_index])
        if area <= 0:
            return None

        component_slices = ndimage.find_objects(labels)
        component_slice = component_slices[label_index - 1] if label_index - 1 < len(component_slices) else None
        if component_slice is None:
            return None

        y_slice, x_slice = component_slice
        min_y = int(y_slice.start)
        max_y = int(y_slice.stop - 1)
        min_x = int(x_slice.start)
        max_x = int(x_slice.stop - 1)
        local_mask = labels[component_slice] == label_index
        local_y, local_x = np.nonzero(local_mask)
        sum_x = float(np.sum(local_x + min_x))
        sum_y = float(np.sum(local_y + min_y))
        return (area, min_x, max_x, min_y, max_y, sum_x, sum_y)

    def _detect_water_phantom_geometry(self, source_pixels: np.ndarray) -> tuple[float, float, float] | None:
        pixels = np.asarray(source_pixels, dtype=np.float64)
        finite_mask = np.isfinite(pixels)
        if not np.any(finite_mask):
            return None

        finite_values = pixels[finite_mask]
        pixel_min = float(np.min(finite_values))
        pixel_max = float(np.max(finite_values))
        if pixel_max <= pixel_min:
            return None

        normalized = np.zeros(pixels.shape, dtype=np.uint8)
        normalized[finite_mask] = np.clip((pixels[finite_mask] - pixel_min) * 255.0 / (pixel_max - pixel_min), 0, 255).astype(np.uint8)
        threshold = max(8, self._compute_otsu_threshold(normalized))
        candidates = (
            self._find_largest_mask_component(normalized > threshold),
            self._find_largest_mask_component((normalized <= threshold) & finite_mask),
        )
        image_area = int(normalized.shape[0] * normalized.shape[1])
        component = next(
            (
                candidate
                for candidate in sorted((item for item in candidates if item is not None), key=lambda item: item[0], reverse=True)
                if image_area * 0.02 <= candidate[0] <= image_area * 0.9
            ),
            None,
        )
        if component is None:
            return None

        area, min_x, max_x, min_y, max_y, sum_x, sum_y = component
        center_x = sum_x / float(area)
        center_y = sum_y / float(area)
        bounds_radius = min(max_x - min_x, max_y - min_y) / 2.0
        area_radius = float(np.sqrt(float(area) / np.pi))
        min_dimension = float(min(normalized.shape[1], normalized.shape[0]))
        phantom_radius = max(min_dimension * 0.12, min(min(bounds_radius, area_radius) * 0.92, min_dimension * 0.46))
        return center_x, center_y, phantom_radius

    @staticmethod
    def _build_roi_payload(
        roi_id: str,
        label: str,
        kind: str,
        center_x: float,
        center_y: float,
        radius: float,
        image_transform: Any,
        canvas_width: int,
        canvas_height: int,
    ) -> QaWaterRoiPayload:
        matrix = image_transform.matrix
        canvas_center = matrix @ np.asarray([center_x, center_y, 1.0], dtype=np.float64)
        canvas_edge_x = matrix @ np.asarray([center_x + radius, center_y, 1.0], dtype=np.float64)
        canvas_edge_y = matrix @ np.asarray([center_x, center_y + radius, 1.0], dtype=np.float64)
        width = max(float(canvas_width), 1.0)
        height = max(float(canvas_height), 1.0)
        screen_radius = (
            float(np.hypot(canvas_edge_x[0] - canvas_center[0], canvas_edge_x[1] - canvas_center[1]))
            + float(np.hypot(canvas_edge_y[0] - canvas_center[0], canvas_edge_y[1] - canvas_center[1]))
        ) / 2.0

        return QaWaterRoiPayload(
            id=roi_id,
            label=label,
            kind=kind,
            center={
                "x": max(0.0, min(1.0, float(canvas_center[0]) / width)),
                "y": max(0.0, min(1.0, float(canvas_center[1]) / height)),
            },
            radius=max(0.0, min(1.0, screen_radius / max(min(width, height), 1.0))),
        )
