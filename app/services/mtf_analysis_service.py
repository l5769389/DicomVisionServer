from __future__ import annotations

from typing import Protocol

import numpy as np
from fastapi import HTTPException

from app.models.measurement import MeasurementPoint
from app.models.viewer import ViewRecord
from app.schemas.view import (
    MtfCurvePointPayload,
    MtfMetricsPayload,
    ViewMtfAnalyzeRequest,
    ViewMtfAnalyzeResponse,
)
from app.services.view_registry import view_registry


class MtfAnalysisContext(Protocol):
    def _resolve_normalized_point_to_image_point(self, view: ViewRecord, normalized_x: float, normalized_y: float) -> MeasurementPoint:
        ...

    def _resolve_measurement_source_context(self, view: ViewRecord) -> tuple[np.ndarray, tuple[float, float] | None, object]:
        ...


class MtfAnalysisService:
    def __init__(self, context: MtfAnalysisContext) -> None:
        self._context = context

    def analyze(self, payload: ViewMtfAnalyzeRequest) -> ViewMtfAnalyzeResponse:
        view = view_registry.get(payload.view_id)
        if view.view_type not in {"Stack", "MPR", "AX", "COR", "SAG"}:
            raise HTTPException(status_code=400, detail="MTF analysis is only available for 2D views")
        if len(payload.points) < 2:
            raise HTTPException(status_code=400, detail="MTF analysis requires two ROI points")

        image_points = tuple(
            self._context._resolve_normalized_point_to_image_point(view, point.x, point.y)
            for point in payload.points[:2]
        )
        source_pixels, spacing_xy, _ = self._context._resolve_measurement_source_context(view)
        image_height = int(source_pixels.shape[0])
        image_width = int(source_pixels.shape[1])
        left = max(0, min(int(round(image_points[0].x)), int(round(image_points[1].x))))
        right = min(image_width - 1, max(int(round(image_points[0].x)), int(round(image_points[1].x))))
        top = max(0, min(int(round(image_points[0].y)), int(round(image_points[1].y))))
        bottom = min(image_height - 1, max(int(round(image_points[0].y)), int(round(image_points[1].y))))
        if right <= left or bottom <= top:
            raise HTTPException(status_code=400, detail="MTF ROI is too small")

        roi = np.asarray(source_pixels[top : bottom + 1, left : right + 1], dtype=np.float64)
        if roi.size == 0:
            raise HTTPException(status_code=400, detail="MTF ROI is empty")

        sample_count = int(roi.size)
        try:
            from app.services.mtf import MtfAnalyzer

            analysis = MtfAnalyzer.analyze_roi(roi, spacing_xy=spacing_xy)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        unit = "lp/mm" if spacing_xy is not None else "lp/pixel"
        curve = [
            MtfCurvePointPayload(frequency=round(float(freq), 6), value=round(float(value), 6))
            for freq, value in zip(analysis.frequencies, analysis.values)
        ]

        return ViewMtfAnalyzeResponse(
            viewId=view.view_id,
            viewportKey=payload.viewport_key,
            points=payload.points[:2],
            metrics=MtfMetricsPayload(
                mtf50=round(float(analysis.mtf50), 4),
                mtf10=round(float(analysis.mtf10), 4),
                fwhmW=round(float(analysis.fwhm_w), 4),
                fwhmH=round(float(analysis.fwhm_h), 4),
                peakValue=round(float(analysis.peak_value), 4),
                sampleCount=sample_count,
                unit=unit,
            ),
            curve=curve,
            isPlaceholder=False,
        )
