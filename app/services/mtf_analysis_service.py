from __future__ import annotations

from copy import deepcopy
import math
from typing import NoReturn, Protocol

import numpy as np
from fastapi import HTTPException

from app.models.measurement import MeasurementPoint
from app.models.viewer import ViewRecord
from app.schemas.view import (
    MtfCurvePointPayload,
    MtfMetricsPayload,
    MtfQualityWarningPayload,
    ViewMtfAnalyzeRequest,
    ViewMtfAnalyzeResponse,
)
from app.services.series_registry import series_registry
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
        if view.view_type not in {"Stack", "PET"}:
            self._raise_analysis_error(
                code="mtf-view-not-supported",
                message="MTF point-source analysis is only available for original 2D Stack or PET views.",
                suggestion="Open the original 2D series and select the point-source ROI there.",
            )
        if len(payload.points) < 2:
            self._raise_analysis_error(
                code="mtf-roi-points-missing",
                message="MTF analysis requires two ROI corner points.",
                suggestion="Draw a rectangular ROI around the complete point source.",
            )

        view_snapshot = deepcopy(view)
        if payload.source_slice_index is not None:
            series = series_registry.get(
                view_snapshot.series_id,
                workspace_id=view_snapshot.workspace_id,
            )
            if payload.source_slice_index >= len(series.instances):
                self._raise_analysis_error(
                    code="mtf-source-slice-unavailable",
                    message="The source slice selected for MTF analysis is no longer available.",
                    suggestion="Return to the source slice and select the point-source ROI again.",
                )
            view_snapshot.current_index = payload.source_slice_index

        try:
            image_points = tuple(
                self._context._resolve_normalized_point_to_image_point(view_snapshot, point.x, point.y)
                for point in payload.points[:2]
            )
            source_pixels, spacing_xy, _ = self._context._resolve_measurement_source_context(view_snapshot)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "The source image could not be read."
            self._raise_analysis_error(
                code="mtf-source-unavailable",
                message=detail,
                suggestion="Return to the source slice, wait for it to finish loading, and try again.",
            )

        source_pixels = np.asarray(source_pixels)
        if source_pixels.ndim != 2:
            self._raise_analysis_error(
                code="mtf-source-unavailable",
                message="MTF analysis requires a two-dimensional source image.",
                suggestion="Use an original single-frame 2D Stack or PET view.",
            )
        image_height = int(source_pixels.shape[0])
        image_width = int(source_pixels.shape[1])
        if image_height < 9 or image_width < 9:
            self._raise_analysis_error(
                code="mtf-source-too-small",
                message="The source image is smaller than the minimum 9 x 9 pixels required for MTF analysis.",
                suggestion="Use a source image with a larger pixel matrix.",
            )

        left = max(0, min(math.floor(image_points[0].x), math.floor(image_points[1].x)))
        right = min(image_width - 1, max(math.ceil(image_points[0].x), math.ceil(image_points[1].x)))
        top = max(0, min(math.floor(image_points[0].y), math.floor(image_points[1].y)))
        bottom = min(image_height - 1, max(math.ceil(image_points[0].y), math.ceil(image_points[1].y)))
        left, right, expanded_w = self._ensure_minimum_interval(left, right, image_width)
        top, bottom, expanded_h = self._ensure_minimum_interval(top, bottom, image_height)

        roi = np.asarray(source_pixels[top : bottom + 1, left : right + 1], dtype=np.float64)
        if roi.size == 0:
            self._raise_analysis_error(
                code="mtf-roi-empty",
                message="The selected MTF ROI does not overlap the source image.",
                suggestion="Select the point source again inside the image bounds.",
            )

        sample_count = int(roi.size)
        try:
            from app.services.mtf import MtfAnalyzer

            analysis = MtfAnalyzer.analyze_roi(roi, spacing_xy=spacing_xy)
        except ValueError as exc:
            self._raise_value_error(exc)

        unit = "lp/mm" if spacing_xy is not None else "lp/pixel"

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(float(value), 4)

        curve = [
            MtfCurvePointPayload(frequency=round(float(freq), 6), value=round(float(value), 6))
            for freq, value in zip(analysis.frequencies, analysis.values)
        ]

        quality_warnings = [
            MtfQualityWarningPayload(code=warning.code, message=warning.message)
            for warning in analysis.quality_warnings
        ]
        if expanded_w or expanded_h:
            quality_warnings.insert(
                0,
                MtfQualityWarningPayload(
                    code="roi-auto-expanded",
                    message="The analysis ROI was expanded internally to the minimum 9 x 9 pixel size.",
                ),
            )

        return ViewMtfAnalyzeResponse(
            viewId=view_snapshot.view_id,
            viewportKey=payload.viewport_key,
            points=payload.points[:2],
            metrics=MtfMetricsPayload(
                mtf50=rounded(analysis.mtf50),
                mtf10=rounded(analysis.mtf10),
                mtf50W=rounded(analysis.mtf50_w),
                mtf10W=rounded(analysis.mtf10_w),
                mtf50H=rounded(analysis.mtf50_h),
                mtf10H=rounded(analysis.mtf10_h),
                nyquistW=rounded(analysis.nyquist_w),
                nyquistH=rounded(analysis.nyquist_h),
                radialNyquist=rounded(analysis.radial_nyquist),
                fwhmW=rounded(analysis.fwhm_w),
                fwhmH=rounded(analysis.fwhm_h),
                peakValue=round(float(analysis.peak_value), 4),
                sampleCount=sample_count,
                unit=unit,
                sourceSizeCorrected=False,
            ),
            curve=curve,
            qualityWarnings=quality_warnings,
            isPlaceholder=False,
        )

    @staticmethod
    def _ensure_minimum_interval(start: int, end: int, limit: int, target: int = 9) -> tuple[int, int, bool]:
        if end - start + 1 >= target:
            return start, end, False
        center = (start + end) / 2.0
        expanded_start = int(round(center - (target - 1) / 2.0))
        expanded_start = max(0, min(expanded_start, limit - target))
        return expanded_start, expanded_start + target - 1, True

    @staticmethod
    def _raise_analysis_error(*, code: str, message: str, suggestion: str) -> NoReturn:
        raise HTTPException(
            status_code=422,
            detail={
                "code": code,
                "message": message,
                "suggestion": suggestion,
            },
        )

    @classmethod
    def _raise_value_error(cls, error: ValueError) -> NoReturn:
        message = str(error)
        if "non-finite" in message:
            cls._raise_analysis_error(
                code="mtf-non-finite-roi",
                message=message,
                suggestion="Choose a different ROI on a valid source slice.",
            )
        if "SNR is below" in message:
            cls._raise_analysis_error(
                code="mtf-snr-too-low",
                message=message,
                suggestion="Use a cleaner slice or draw a larger ROI that includes more background around the point source.",
            )
        if "detectable point-source" in message or "stable point-source" in message:
            cls._raise_analysis_error(
                code="mtf-no-detectable-source",
                message=message,
                suggestion="Place the ROI around one isolated bright or dark point source with visible surrounding background.",
            )
        cls._raise_analysis_error(
            code="mtf-analysis-failed",
            message=message,
            suggestion="Select a larger ROI containing one isolated point source and retry the analysis.",
        )
