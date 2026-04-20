import io
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont
from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    MPR_VIEWPORT_AXIAL,
    MPR_VIEWPORT_CORONAL,
    MPR_VIEWPORT_SAGITTAL,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_SET_SIZE,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_VOLUME_CONFIG,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    WINDOW_DRAG_SENSITIVITY,
    WINDOW_WIDTH_MIN,
    ZOOM_DRAG_FACTOR_MIN,
    ZOOM_DRAG_SENSITIVITY,
    ZOOM_DRAG_SENSITIVITY_3D,
    ZOOM_MAX_3D,
    ZOOM_MIN_3D,
    VIEW_OP_TYPE_SCROLL,
)
from app.core.logging import get_logger
from app.models.measurement import MeasurementPoint, MeasurementRecord, MeasurementSliceContext
from app.models.viewer import InstanceRecord, MprMipState, MprMipViewportState, SeriesRecord, ViewRecord
from app.schemas.dicom import CornerInfoPayload, CornerInfoRequest, CornerInfoResponse
from app.schemas.view import (
    ImageFormat,
    MtfCurvePointPayload,
    MtfMetricsPayload,
    MprCrosshairInfo,
    MprMipConfig,
    MprMipViewportConfig,
    MeasurementOverlayPayload,
    OperationAcceptedResponse,
    OrientationInfo,
    QaWaterAccuracyMetricsPayload,
    QaWaterMetricsPayload,
    QaWaterNoiseMetricsPayload,
    QaWaterRoiPayload,
    QaWaterRoiStatsPayload,
    QaWaterUniformityMetricsPayload,
    ScaleBarInfo,
    SliceInfo,
    ViewColorInfo,
    ViewExportOverlaysPayload,
    ViewHoverRequest,
    ViewHoverResponse,
    ViewImageResponse,
    ViewMtfAnalyzeRequest,
    ViewQaWaterAnalyzeRequest,
    ViewQaWaterAnalyzeResponse,
    ViewTransformPayload,
    ViewMtfAnalyzeResponse,
    ViewOperationRequest,
    ViewSetSizeRequest,
    WindowInfo,
)
from app.services.dicom_cache import CachedDicom, dicom_cache
from app.services.hover_mapping import map_normalized_canvas_to_image_row_col
from app.services.layered_renderer import RenderContext, layered_renderer
from app.services.measurement_utils import build_measurement_metrics, clamp_point_to_image
from app.services.mtf import MtfAnalyzer
from app.services.pseudocolor import DEFAULT_PSEUDOCOLOR_PRESET, apply_pseudocolor, normalize_pseudocolor_preset
from app.services.render_layers.render_context import CornerInfoOverlay, MprCrosshairOverlay, OrientationOverlay
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer
from app.services.view_registry import view_registry
from app.services.viewer_operation_handlers import OperationRenderOutcome, handle_view_operation
from app.services.volume_render_config import (
    create_default_volume_render_config,
    normalize_volume_preset_name,
    normalize_volume_render_config,
)
from app.services.volume_rendering import VolumeRenderRequest, vtk_volume_renderer


logger = get_logger(__name__)

CROSSHAIR_HIT_RADIUS = 12.0
MEASUREMENT_TOOL_TYPES = {"line", "rect", "ellipse", "angle"}


@dataclass(frozen=True)
class RenderedImageResult:
    meta: ViewImageResponse
    image_bytes: bytes


@dataclass(frozen=True)
class ExportedFileResult:
    file_bytes: bytes
    file_name: str
    media_type: str


@dataclass(frozen=True)
class RenderPlan:
    render_view: ViewRecord
    render_ratio: float


class ViewerService:
    def __init__(self) -> None:
        self._volume_cache: dict[str, np.ndarray] = {}
        self._series_patient_transform_cache: dict[str, dict[str, object] | None] = {}
        self._logger = logger

    @staticmethod
    def _is_mpr_view_type(view_type: str) -> bool:
        return view_type in {"MPR", "AX", "COR", "SAG"}

    @staticmethod
    def _is_3d_view_type(view_type: str) -> bool:
        return view_type == "3D"

    def set_view_size(self, payload: ViewSetSizeRequest) -> OperationAcceptedResponse:
        if payload.op_type != VIEW_OP_TYPE_SET_SIZE:
            raise HTTPException(status_code=400, detail="opType must be setSize")

        view = view_registry.get(payload.view_id)
        view.width = payload.size.width
        view.height = payload.size.height
        logger.info(
            "set view size view_id=%s width=%s height=%s",
            view.view_id,
            view.width,
            view.height,
        )

        if not view.is_initialized:
            if self._is_mpr_view_type(view.view_type):
                self._initialize_mpr_viewport(view)
            elif self._is_3d_view_type(view.view_type):
                self._initialize_3d_viewport(view)
            else:
                self._initialize_viewport(view)
            view.is_initialized = True

        return OperationAcceptedResponse(message="View size updated", viewId=view.view_id)

    def render_view_by_id(
        self,
        view_id: str,
        *,
        image_format: ImageFormat = "png",
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        view = view_registry.get(view_id)
        return self._render_by_view_type(view, image_format=image_format, fast_preview=fast_preview)

    def export_view_by_id(
        self,
        view_id: str,
        export_format: str,
        *,
        overlays: ViewExportOverlaysPayload | None = None,
    ) -> ExportedFileResult:
        view = view_registry.get(view_id)
        safe_view_type = str(view.view_type or "view").lower()

        if export_format == "png":
            rendered = self._render_by_view_type(view, image_format="png", fast_preview=False)
            if overlays and (overlays.annotations or overlays.measurements):
                try:
                    image = Image.open(io.BytesIO(rendered.image_bytes)).convert("RGB")
                    image = self._apply_export_overlays(image, overlays)
                    rendered_bytes = self._encode_image(image, "png")
                except Exception as exc:  # pragma: no cover - defensive
                    raise HTTPException(status_code=500, detail="Failed to render export overlays") from exc
            else:
                rendered_bytes = rendered.image_bytes
            return ExportedFileResult(
                file_bytes=rendered_bytes,
                file_name=f"{view.view_id}-{safe_view_type}.png",
                media_type="image/png",
            )
        if export_format != "dicom":
            raise HTTPException(status_code=400, detail="Unsupported export format")

        rendered = self._render_by_view_type(view, image_format="png", fast_preview=False)
        try:
            image = Image.open(io.BytesIO(rendered.image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="Failed to decode rendered image for DICOM export") from exc

        if overlays and (overlays.annotations or overlays.measurements):
            image = self._apply_export_overlays(image, overlays)

        reference_dataset = self._get_export_reference_dataset(view)
        dicom_bytes = self._build_secondary_capture_dicom_bytes(view, image, reference_dataset)
        return ExportedFileResult(
            file_bytes=dicom_bytes,
            file_name=f"{view.view_id}-{safe_view_type}.dcm",
            media_type="application/dicom",
        )

    def _apply_export_overlays(self, image: Image.Image, overlays: ViewExportOverlaysPayload) -> Image.Image:
        canvas = image.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        width, height = canvas.size

        for measurement in overlays.measurements:
            points = tuple((point.x * width, point.y * height) for point in measurement.points)
            self._draw_export_measurement(draw, font, measurement.tool_type, points, measurement.label_lines, width, height)

        for annotation in overlays.annotations:
            points = tuple((point.x * width, point.y * height) for point in annotation.points)
            self._draw_export_annotation(draw, font, points, annotation.text, annotation.color, annotation.size, width, height)

        return canvas.convert("RGB")

    def _draw_export_measurement(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        tool_type: str,
        points: tuple[tuple[float, float], ...],
        label_lines: list[str],
        width: int,
        height: int,
    ) -> None:
        if not points:
            return

        if tool_type == "line" and len(points) >= 2:
            self._draw_export_polyline(draw, points[:2])
        elif tool_type == "rect" and len(points) >= 2:
            left, right = sorted((points[0][0], points[1][0]))
            top, bottom = sorted((points[0][1], points[1][1]))
            draw.rectangle((left, top, right, bottom), outline=(3, 15, 24, 235), width=5)
            draw.rectangle((left, top, right, bottom), outline=(85, 231, 255, 255), width=2)
        elif tool_type == "ellipse" and len(points) >= 2:
            left, right = sorted((points[0][0], points[1][0]))
            top, bottom = sorted((points[0][1], points[1][1]))
            draw.ellipse((left, top, right, bottom), outline=(3, 15, 24, 235), width=5)
            draw.ellipse((left, top, right, bottom), outline=(85, 231, 255, 255), width=2)
        elif tool_type == "angle" and len(points) >= 2:
            self._draw_export_polyline(draw, points[:2])
            if len(points) >= 3:
                self._draw_export_polyline(draw, points[1:3])
        else:
            return

        if label_lines:
            anchor = points[1] if len(points) >= 2 else points[0]
            self._draw_export_label(draw, font, label_lines, anchor[0] + 12, anchor[1] - 32, width, height)

    @staticmethod
    def _draw_export_polyline(draw: ImageDraw.ImageDraw, points: tuple[tuple[float, float], ...]) -> None:
        draw.line(points, fill=(3, 15, 24, 235), width=5, joint="curve")
        draw.line(points, fill=(85, 231, 255, 255), width=2, joint="curve")

    def _draw_export_annotation(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        points: tuple[tuple[float, float], ...],
        text: str,
        color: str,
        size: str,
        width: int,
        height: int,
    ) -> None:
        if len(points) < 2:
            return

        stroke = self._parse_export_color(color)
        stroke_width = 3 if size == "lg" else 2
        draw.line(points[:2], fill=stroke, width=stroke_width)
        self._draw_export_arrow_head(draw, points[0], points[1], stroke, stroke_width * 3)

        visible_text = text.strip()
        if visible_text:
            self._draw_export_label(draw, font, [visible_text], points[0][0] + 12, points[0][1] - 30, width, height, text_fill=stroke)

    @staticmethod
    def _draw_export_arrow_head(
        draw: ImageDraw.ImageDraw,
        start: tuple[float, float],
        end: tuple[float, float],
        fill: tuple[int, int, int, int],
        size: int,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            return

        ux = dx / length
        uy = dy / length
        back_x = end[0] - ux * size * 2.8
        back_y = end[1] - uy * size * 2.8
        perp_x = -uy * size
        perp_y = ux * size
        draw.polygon(
            (
                end,
                (back_x + perp_x, back_y + perp_y),
                (back_x - perp_x, back_y - perp_y),
            ),
            fill=fill,
        )

    @staticmethod
    def _draw_export_label(
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        lines: list[str],
        x: float,
        y: float,
        width: int,
        height: int,
        *,
        text_fill: tuple[int, int, int, int] = (235, 245, 255, 255),
    ) -> None:
        visible_lines = [line.strip() for line in lines if line.strip()]
        if not visible_lines:
            return

        padding_x = 8
        padding_y = 6
        line_gap = 3
        line_sizes = [draw.textbbox((0, 0), line, font=font) for line in visible_lines]
        text_width = max((bbox[2] - bbox[0]) for bbox in line_sizes)
        text_height = sum((bbox[3] - bbox[1]) for bbox in line_sizes) + max(0, len(visible_lines) - 1) * line_gap
        left = max(6, min(width - text_width - padding_x * 2 - 6, int(round(x))))
        top = max(6, min(height - text_height - padding_y * 2 - 6, int(round(y))))
        right = left + text_width + padding_x * 2
        bottom = top + text_height + padding_y * 2

        draw.rounded_rectangle((left, top, right, bottom), radius=7, fill=(7, 16, 28, 232), outline=(108, 201, 255, 188), width=1)
        cursor_y = top + padding_y
        for index, line in enumerate(visible_lines):
            bbox = line_sizes[index]
            draw.text((left + padding_x, cursor_y), line, fill=text_fill, font=font)
            cursor_y += (bbox[3] - bbox[1]) + line_gap

    @staticmethod
    def _parse_export_color(value: str) -> tuple[int, int, int, int]:
        hex_value = value.strip().lstrip("#")
        if len(hex_value) == 3:
            hex_value = "".join(char * 2 for char in hex_value)
        if len(hex_value) != 6:
            return (255, 209, 102, 255)
        try:
            red = int(hex_value[0:2], 16)
            green = int(hex_value[2:4], 16)
            blue = int(hex_value[4:6], 16)
        except ValueError:
            return (255, 209, 102, 255)
        return (red, green, blue, 255)

    def handle_view_operation(self, payload: ViewOperationRequest) -> OperationRenderOutcome:
        return handle_view_operation(self, payload)

    def handle_view_hover(self, payload: ViewHoverRequest) -> ViewHoverResponse:
        view = view_registry.get(payload.view_id)
        row, col = self._resolve_hover_row_col(view, payload.x, payload.y)
        return ViewHoverResponse(viewId=view.view_id, row=row, col=col)

    def get_series_corner_info(self, payload: CornerInfoRequest) -> CornerInfoResponse:
        series = series_registry.get(payload.series_id)
        _, reference_cached = self._get_reference_instance_and_cache(series)
        overlay = self._build_series_corner_info_overlay(
            series,
            reference_cached.dataset if reference_cached is not None else None,
        )
        return CornerInfoResponse(cornerInfo=self._serialize_corner_info_overlay(overlay))

    def analyze_mtf(self, payload: ViewMtfAnalyzeRequest) -> ViewMtfAnalyzeResponse:
        view = view_registry.get(payload.view_id)
        if view.view_type not in {"Stack", "MPR", "AX", "COR", "SAG"}:
            raise HTTPException(status_code=400, detail="MTF analysis is only available for 2D views")
        if len(payload.points) < 2:
            raise HTTPException(status_code=400, detail="MTF analysis requires two ROI points")

        image_points = tuple(
            self._resolve_normalized_point_to_image_point(view, point.x, point.y)
            for point in payload.points[:2]
        )
        source_pixels, spacing_xy, _ = self._resolve_measurement_source_context(view)
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

    def analyze_qa_water(self, payload: ViewQaWaterAnalyzeRequest) -> ViewQaWaterAnalyzeResponse:
        view = view_registry.get(payload.view_id)
        if view.view_type not in {"Stack", "MPR", "AX", "COR", "SAG"}:
            raise HTTPException(status_code=400, detail="Water phantom QA is only available for 2D views")

        source_pixels, spacing_xy, _ = self._resolve_measurement_source_context(view)
        image_height = int(source_pixels.shape[0])
        image_width = int(source_pixels.shape[1])
        image_width_for_transform, image_height_for_transform, image_transform, canvas_width, canvas_height = self._build_hover_mapping_context(view)
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
                message="当前视图尚未准备好，无法进行水模 QA 分析。",
            )

        detected = self._detect_water_phantom_geometry(source_pixels)
        if detected is None:
            return ViewQaWaterAnalyzeResponse(
                viewId=view.view_id,
                viewportKey=payload.viewport_key,
                rois=[],
                status="error",
                message="未检测到水模轮廓，请确认当前图像包含完整水模并调整窗宽窗位后重试。",
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

        air_x, air_y = self._resolve_qa_water_air_roi_center(
            center_x,
            center_y,
            phantom_radius,
            air_roi_radius,
            image_width,
            image_height,
        )

        roi_sources = [
            (roi_id, label, "water", x, y, water_roi_radius)
            for roi_id, label, x, y in water_positions
        ]
        roi_sources.append(("air", "Air", "air", air_x, air_y, air_roi_radius))
        rois = [
            self._build_qa_water_roi_payload(
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
        metrics = self._build_qa_water_metrics(
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
    def _resolve_qa_water_air_roi_center(
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

    def _build_qa_water_metrics(
        self,
        source_pixels: np.ndarray,
        roi_sources: list[tuple[str, str, str, float, float, float]],
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
            self._build_qa_water_roi_stats_payload(
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
    def _build_qa_water_roi_stats_payload(
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
        height, width = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        best: tuple[int, int, int, int, int, float, float] | None = None

        for start_y in range(height):
            for start_x in range(width):
                if visited[start_y, start_x] or not mask[start_y, start_x]:
                    continue

                stack = [(start_x, start_y)]
                visited[start_y, start_x] = True
                area = 0
                min_x = width
                max_x = 0
                min_y = height
                max_y = 0
                sum_x = 0.0
                sum_y = 0.0

                while stack:
                    x, y = stack.pop()
                    area += 1
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    sum_x += float(x)
                    sum_y += float(y)

                    for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                            continue
                        if visited[next_y, next_x] or not mask[next_y, next_x]:
                            continue
                        visited[next_y, next_x] = True
                        stack.append((next_x, next_y))

                if best is None or area > best[0]:
                    best = (area, min_x, max_x, min_y, max_y, sum_x, sum_y)

        return best

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
    def _build_qa_water_roi_payload(
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

    def _resolve_hover_row_col(self, view: ViewRecord, normalized_x: float, normalized_y: float) -> tuple[int, int]:
        if not view.width or not view.height or self._is_3d_view_type(view.view_type):
            return (0, 0)

        image_width, image_height, image_transform, canvas_width, canvas_height = self._build_hover_mapping_context(view)
        return map_normalized_canvas_to_image_row_col(
            normalized_x,
            normalized_y,
            image_width=image_width,
            image_height=image_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            image_transform=image_transform,
        )

    def _build_hover_mapping_context(self, view: ViewRecord) -> tuple[int, int, Any, int, int]:
        """Prepare the source-image dimensions and inverse transform used for hover lookup."""

        image_width, image_height = self._get_hover_source_dimensions(view)
        pixel_aspect_x = 1.0
        pixel_aspect_y = 1.0
        if self._is_mpr_view_type(view.view_type):
            series = series_registry.get(view.series_id)
            pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy(series, self._resolve_mpr_viewport(view))
        render_plan = self._build_render_plan_for_shape(
            view,
            image_height,
            image_width,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=image_width,
            image_height=image_height,
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        return (
            image_width,
            image_height,
            image_transform,
            render_plan.render_view.width or 0,
            render_plan.render_view.height or 0,
        )

    def _get_hover_source_dimensions(self, view: ViewRecord) -> tuple[int, int]:
        if self._is_mpr_view_type(view.view_type):
            series = series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            if not view.is_initialized:
                self._initialize_mpr_viewport(view)
                view.is_initialized = True
            target_viewport = self._resolve_mpr_viewport(view)
            plane_pixels, _, _ = self._extract_mpr_plane(view, volume, target_viewport)
            return (int(plane_pixels.shape[1]), int(plane_pixels.shape[0]))

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            return (0, 0)
        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        return (int(cached.source_pixels.shape[1]), int(cached.source_pixels.shape[0]))

    def _resolve_normalized_point_to_image_point(
        self,
        view: ViewRecord,
        normalized_x: float,
        normalized_y: float,
    ) -> MeasurementPoint:
        image_width, image_height, image_transform, canvas_width, canvas_height = self._build_hover_mapping_context(view)
        if image_width <= 0 or image_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
            raise HTTPException(status_code=400, detail="View is not ready for measurement")

        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))
        max_canvas_x = max(float(canvas_width) - 1e-6, 0.0)
        max_canvas_y = max(float(canvas_height) - 1e-6, 0.0)
        canvas_x = min(max(x * float(canvas_width), 0.0), max_canvas_x)
        canvas_y = min(max(y * float(canvas_height), 0.0), max_canvas_y)

        affine_matrix, offset = image_transform.inverse_components()
        source_point = affine_matrix @ np.asarray([canvas_x, canvas_y], dtype=np.float64) + offset
        clamped = clamp_point_to_image(
            MeasurementPoint(x=float(source_point[0]), y=float(source_point[1])),
            image_width=image_width,
            image_height=image_height,
        )
        return clamped

    def _resolve_measurement_source_context(
        self,
        view: ViewRecord,
    ) -> tuple[np.ndarray, tuple[float, float] | None, MeasurementSliceContext]:
        if self._is_mpr_view_type(view.view_type):
            series = series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            target_viewport = self._resolve_mpr_viewport(view)
            plane_pixels, current_index, _ = self._extract_mpr_plane(view, volume, target_viewport)
            return (
                plane_pixels,
                self._get_mpr_spacing_xy(series, target_viewport),
                MeasurementSliceContext(kind="mpr", slice_index=current_index),
            )

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")
        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        return (
            cached.source_pixels,
            self._get_stack_spacing_xy(cached.dataset),
            MeasurementSliceContext(kind="stack", slice_index=view.current_index, sop_instance_uid=instance.sop_instance_uid),
        )

    @staticmethod
    def _resolve_measurement_tool_type(payload: ViewOperationRequest) -> str | None:
        tool_type = str(payload.sub_op_type or "").strip().lower()
        return tool_type if tool_type in MEASUREMENT_TOOL_TYPES else None

    def _resolve_measurement_image_points(
        self,
        view: ViewRecord,
        payload: ViewOperationRequest,
    ) -> tuple[MeasurementPoint, ...]:
        return tuple(
            self._resolve_normalized_point_to_image_point(view, point.x, point.y)
            for point in (payload.points or [])
        )

    @staticmethod
    def _is_empty_measurement(tool_type: str, points: tuple[MeasurementPoint, ...]) -> bool:
        if tool_type == "angle" or len(points) < 2:
            return False
        start, end = points[:2]
        return abs(end.x - start.x) < 1e-3 and abs(end.y - start.y) < 1e-3

    @staticmethod
    def _serialize_measurement_metrics(metrics) -> dict[str, float | str | None]:
        return {
            "length": metrics.length,
            "width": metrics.width,
            "height": metrics.height,
            "area": metrics.area,
            "angleDegrees": metrics.angle_degrees,
            "mean": metrics.mean,
            "sd": metrics.standard_deviation,
            "min": metrics.minimum,
            "max": metrics.maximum,
            "unit": metrics.unit,
            "areaUnit": metrics.area_unit,
        }

    def _build_measurement_preview_payload(
        self,
        *,
        view: ViewRecord,
        viewport_key: str,
        tool_type: str,
        slice_index: int,
        label_lines: tuple[str, ...] | list[str] = (),
        metrics=None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "viewId": view.view_id,
            "viewportKey": viewport_key,
            "toolType": tool_type,
            "labelLines": list(label_lines),
            "sliceIndex": slice_index,
        }
        if metrics is not None:
            payload["metrics"] = self._serialize_measurement_metrics(metrics)
        return payload

    def _build_measurement_preview(self, view: ViewRecord, payload: ViewOperationRequest) -> dict[str, object] | None:
        tool_type = self._resolve_measurement_tool_type(payload)
        if tool_type is None or not payload.points:
            return None

        image_points = self._resolve_measurement_image_points(view, payload)
        source_pixels, spacing_xy, slice_context = self._resolve_measurement_source_context(view)
        viewport_key = payload.viewport_key or self._resolve_measurement_viewport_key(view)

        if tool_type == "angle" and len(image_points) < 3:
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=slice_context.slice_index,
            )

        expected_points = 3 if tool_type == "angle" else 2
        if len(image_points) != expected_points:
            return None

        if self._is_empty_measurement(tool_type, image_points):
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=slice_context.slice_index,
            )

        metrics, label_lines = build_measurement_metrics(tool_type, image_points, source_pixels, spacing_xy)
        return self._build_measurement_preview_payload(
            view=view,
            viewport_key=viewport_key,
            tool_type=tool_type,
            slice_index=slice_context.slice_index,
            label_lines=label_lines,
            metrics=metrics,
        )

    def _handle_measurement(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        tool_type = self._resolve_measurement_tool_type(payload)
        if tool_type is None:
            raise HTTPException(status_code=400, detail="Unsupported measurement tool type")
        if not payload.points:
            raise HTTPException(status_code=400, detail="Measurement points are required")

        expected_points = 3 if tool_type == "angle" else 2
        if len(payload.points) != expected_points:
            return False

        image_points = self._resolve_measurement_image_points(view, payload)

        if self._is_empty_measurement(tool_type, image_points):
            return False

        source_pixels, spacing_xy, slice_context = self._resolve_measurement_source_context(view)
        metrics, label_lines = build_measurement_metrics(tool_type, image_points, source_pixels, spacing_xy)

        label_anchor = image_points[1] if tool_type != "angle" else image_points[1]
        measurement_id = str(payload.measurement_id or "").strip() or str(uuid4())
        next_measurement = MeasurementRecord(
            measurement_id=measurement_id,
            tool_type=tool_type,
            points=image_points,
            slice_context=slice_context,
            metrics=metrics,
            label_anchor=label_anchor,
            label_lines=label_lines,
        )
        existing_index = next(
            (index for index, measurement in enumerate(view.measurements) if measurement.measurement_id == measurement_id),
            None,
        )
        if existing_index is None:
            view.measurements.append(next_measurement)
        else:
            view.measurements[existing_index] = next_measurement
        view.is_initialized = True
        return True

    @staticmethod
    def _delete_measurement(view: ViewRecord, measurement_id: str | None) -> bool:
        target_measurement_id = str(measurement_id or "").strip()
        if not target_measurement_id:
            return False

        existing_count = len(view.measurements)
        if not existing_count:
            return False

        view.measurements = [
            measurement for measurement in view.measurements if measurement.measurement_id != target_measurement_id
        ]
        if len(view.measurements) == existing_count:
            return False

        view.is_initialized = True
        return True

    def _build_visible_measurements(self, view: ViewRecord) -> tuple[MeasurementRecord, ...]:
        if not view.measurements:
            return ()

        current_slice = self._resolve_current_measurement_slice_index(view)
        visible: list[MeasurementRecord] = []
        for measurement in view.measurements:
            if measurement.slice_context.kind == "stack":
                if not self._is_mpr_view_type(view.view_type) and measurement.slice_context.slice_index == current_slice:
                    visible.append(measurement)
                continue
            if self._is_mpr_view_type(view.view_type) and measurement.slice_context.slice_index == current_slice:
                visible.append(measurement)
        return tuple(visible)

    @staticmethod
    def _serialize_measurements(
        measurements: tuple[MeasurementRecord, ...],
        *,
        image_transform: Any,
        canvas_width: int,
        canvas_height: int,
    ) -> list[MeasurementOverlayPayload]:
        if canvas_width <= 0 or canvas_height <= 0:
            return []

        matrix = image_transform.matrix
        width = max(float(canvas_width), 1.0)
        height = max(float(canvas_height), 1.0)

        def serialize_point(point: MeasurementPoint) -> dict[str, float]:
            projected = matrix @ np.asarray([point.x, point.y, 1.0], dtype=np.float64)
            return {
                "x": max(0.0, min(1.0, float(projected[0]) / width)),
                "y": max(0.0, min(1.0, float(projected[1]) / height)),
            }

        return [
            MeasurementOverlayPayload(
                measurementId=measurement.measurement_id,
                toolType=measurement.tool_type,
                points=[serialize_point(point) for point in measurement.points],
                labelLines=list(measurement.label_lines),
            )
            for measurement in measurements
        ]

    def _resolve_current_measurement_slice_index(self, view: ViewRecord) -> int:
        if not self._is_mpr_view_type(view.view_type):
            return int(view.current_index)
        target_viewport = self._resolve_mpr_viewport(view)
        if target_viewport == MPR_VIEWPORT_CORONAL:
            return int(view.mpr_coronal_index)
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
            return int(view.mpr_sagittal_index)
        return int(view.mpr_axial_index)

    def _resolve_measurement_viewport_key(self, view: ViewRecord) -> str:
        if not self._is_mpr_view_type(view.view_type):
            return "single"
        return self._resolve_mpr_viewport(view)

    @staticmethod
    def _get_stack_spacing_xy(dataset: Dataset | None) -> tuple[float, float] | None:
        pixel_spacing = getattr(dataset, "PixelSpacing", None) if dataset is not None else None
        if pixel_spacing is None or len(pixel_spacing) < 2:
            return None
        try:
            row_spacing = max(abs(float(pixel_spacing[0])), 1e-6)
            col_spacing = max(abs(float(pixel_spacing[1])), 1e-6)
        except (TypeError, ValueError):
            return None
        return (col_spacing, row_spacing)

    def _get_mpr_spacing_xy(self, series: SeriesRecord, viewport_key: str) -> tuple[float, float] | None:
        spacing_x, spacing_y, spacing_z = self._get_3d_spacing_xyz(series)
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return (spacing_x, spacing_z)
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return (spacing_y, spacing_z)
        return (spacing_x, spacing_y)

    def _get_mpr_display_aspect_xy(self, series: SeriesRecord, viewport_key: str) -> tuple[float, float]:
        spacing_xy = self._get_mpr_spacing_xy(series, viewport_key)
        if spacing_xy is None:
            return (1.0, 1.0)
        return (
            max(abs(float(spacing_xy[0])), 1e-6),
            max(abs(float(spacing_xy[1])), 1e-6),
        )

    def _render_by_view_type(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if self._is_mpr_view_type(view.view_type):
            return self._render_mpr_view(view, image_format=image_format, fast_preview=fast_preview)
        if self._is_3d_view_type(view.view_type):
            return self._render_3d_view(view, image_format=image_format, fast_preview=fast_preview)
        return self._render_view(view, image_format=image_format, fast_preview=fast_preview)

    def _handle_scroll(self, view: ViewRecord, series: SeriesRecord, scroll: int) -> None:
        if not self._is_mpr_view_type(view.view_type):
            next_index = view.current_index + scroll
            view.current_index = max(0, min(next_index, len(series.instances) - 1))
            return

        volume = self._get_series_volume(series)
        depth, height, width = volume.shape
        target_viewport = self._resolve_mpr_viewport(view)
        if target_viewport == MPR_VIEWPORT_CORONAL:
            view.mpr_coronal_index = max(0, min(view.mpr_coronal_index + scroll, height - 1))
        elif target_viewport == MPR_VIEWPORT_SAGITTAL:
            view.mpr_sagittal_index = max(0, min(view.mpr_sagittal_index + scroll, width - 1))
        else:
            view.mpr_axial_index = max(0, min(view.mpr_axial_index + scroll, depth - 1))
        view.is_initialized = True

    def _initialize_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        image_height, image_width = cached.source_pixels.shape[:2]
        view.zoom = viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
        )
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_degrees = 0
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        view.window_width = cached.window_width or self._derive_default_window_width(cached)
        view.window_center = cached.window_center or self._derive_default_window_center(cached)
        self._reset_drag_state(view)
        logger.info(
            "viewport initialized view_id=%s image_width=%s image_height=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            image_width,
            image_height,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    def _initialize_mpr_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        depth, height, width = volume.shape
        view.mpr_axial_index = depth // 2
        view.mpr_coronal_index = height // 2
        view.mpr_sagittal_index = width // 2
        view.current_index = view.mpr_axial_index
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_degrees = 0
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        if view.view_group is not None:
            view.view_group.mpr_mip = self._create_default_mpr_mip_state()

        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            view.window_width = cached.window_width or self._derive_default_window_width(cached)
            view.window_center = cached.window_center or self._derive_default_window_center(cached)
        else:
            pixel_min = float(np.min(volume))
            pixel_max = float(np.max(volume))
            view.window_width = max(WINDOW_WIDTH_MIN, pixel_max - pixel_min)
            view.window_center = (pixel_max + pixel_min) / 2.0

        plane_pixels, _, _ = self._extract_mpr_plane(view, volume)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy(series, self._resolve_mpr_viewport(view))
        view.zoom = viewport_transformer.calculate_contain_zoom(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=view.width,
            canvas_height=view.height,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        self._reset_drag_state(view)
        logger.info(
            "mpr viewport initialized view_id=%s volume=%s axial=%s coronal=%s sagittal=%s zoom=%.4f",
            view.view_id,
            volume.shape,
            view.mpr_axial_index,
            view.mpr_coronal_index,
            view.mpr_sagittal_index,
            view.zoom,
        )

    def _initialize_3d_viewport(self, view: ViewRecord) -> None:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        view.current_index = max(0, min(volume.shape[0] // 2, len(series.instances) - 1)) if series.instances else 0

        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            view.window_width = cached.window_width or self._derive_default_window_width(cached)
            view.window_center = cached.window_center or self._derive_default_window_center(cached)
        else:
            pixel_min = float(np.min(volume))
            pixel_max = float(np.max(volume))
            view.window_width = max(WINDOW_WIDTH_MIN, pixel_max - pixel_min)
            view.window_center = (pixel_max + pixel_min) / 2.0

        view.zoom = 1.0
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.rotation_quaternion = vtk_volume_renderer.get_default_rotation_quaternion()
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        view.volume_preset = "aaa"
        view.volume_render_config = create_default_volume_render_config("aaa")
        self._reset_drag_state(view)
        logger.info(
            "3d viewport initialized view_id=%s volume=%s zoom=%.4f ww=%s wl=%s",
            view.view_id,
            volume.shape,
            view.zoom,
            view.window_width,
            view.window_center,
        )

    def _reset_view(self, view: ViewRecord) -> None:
        view.rotation_degrees = 0
        view.hor_flip = False
        view.ver_flip = False

        if self._is_mpr_view_type(view.view_type):
            self._initialize_mpr_viewport(view)
        elif self._is_3d_view_type(view.view_type):
            self._initialize_3d_viewport(view)
        else:
            self._initialize_viewport(view)

        view.is_initialized = True

    def _build_volume_render_request(
        self,
        view: ViewRecord,
        *,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        fast_preview: bool,
    ) -> VolumeRenderRequest:
        """Build the shared VTK request payload used by 3D render and drag paths."""

        return VolumeRenderRequest(
            view_id=view.view_id,
            volume=volume,
            spacing_xyz=spacing_xyz,
            canvas_width=view.width or 0,
            canvas_height=view.height or 0,
            window_width=float(view.window_width or WINDOW_WIDTH_MIN),
            window_center=float(view.window_center or 0.0),
            zoom=float(view.zoom),
            offset_x=float(view.offset_x),
            offset_y=float(view.offset_y),
            rotation_quaternion=tuple(float(value) for value in view.rotation_quaternion),
            volume_preset=str(view.volume_preset or "aaa"),
            volume_config=view.volume_render_config,
            fast_preview=fast_preview,
        )

    def _render_3d_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        if not view.is_initialized:
            self._initialize_3d_viewport(view)
            view.is_initialized = True

        spacing_xyz = self._get_3d_spacing_xyz(series)
        image = vtk_volume_renderer.render(
            self._build_volume_render_request(
                view,
                volume=volume,
                spacing_xyz=spacing_xyz,
                fast_preview=fast_preview,
            )
        )

        corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            None,
            current_index=view.current_index,
            total_slices=max(1, volume.shape[0]),
            viewport_label="3D VR",
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=max(1, volume.shape[0])),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                cornerInfo=self._serialize_corner_info_overlay(corner_info),
                orientation=self._build_3d_orientation_overlay(view),
                transform=self._build_view_transform_payload(view),
                volumePreset=str(view.volume_preset or "aaa"),
                volumeConfig=view.volume_render_config,
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    def _render_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        render_plan = self._build_render_plan_for_shape(view, *cached.source_pixels.shape[:2])
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=cached.source_pixels.shape[1],
            image_height=cached.source_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
        )
        scale_bar = self._build_scale_bar_info(
            render_plan.render_view,
            image_transform,
            self._get_stack_spacing_xy(cached.dataset),
        )
        slice_corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            cached.dataset,
            current_index=view.current_index,
            total_slices=len(series.instances),
            viewport_label="Stack",
        )
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=cached.source_pixels,
            pixel_min=cached.pixel_min,
            pixel_max=cached.pixel_max,
            instance=instance,
            cached=cached,
            image_transform=image_transform,
            measurements=self._build_visible_measurements(view),
            corner_info=None,
            orientation=None,
        )
        visible_measurements = self._build_visible_measurements(view)

        if fast_preview:
            image = self._render_fast_preview(context)
        else:
            image = layered_renderer.render(context)

        logger.debug(
            "render completed view_id=%s index=%s viewport=%sx%s render=%sx%s ratio=%.4f zoom=%.4f ww=%s wl=%s image_format=%s fast_window=%s",
            view.view_id,
            view.current_index,
            view.width,
            view.height,
            render_plan.render_view.width,
            render_plan.render_view.height,
            render_plan.render_ratio,
            view.zoom,
            view.window_width,
            view.window_center,
            image_format,
            fast_preview,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=view.current_index, total=len(series.instances)),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info),
                measurements=self._serialize_measurements(
                    visible_measurements,
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=self._serialize_orientation_overlay(
                    self._build_stack_orientation_overlay(render_plan.render_view, cached.dataset)
                ),
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    def _render_mpr_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
    ) -> RenderedImageResult:
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        if not view.is_initialized:
            self._initialize_mpr_viewport(view)
            view.is_initialized = True

        target_viewport = self._resolve_mpr_viewport(view)
        plane_pixels, current, total = self._extract_mpr_plane(view, volume, target_viewport)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy(series, target_viewport)
        render_plan = self._build_render_plan_for_shape(
            view,
            *plane_pixels.shape[:2],
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        scale_bar = self._build_scale_bar_info(
            render_plan.render_view,
            image_transform,
            self._get_mpr_spacing_xy(series, target_viewport),
        )
        plane_min = float(np.min(plane_pixels))
        plane_max = float(np.max(plane_pixels))
        mpr_crosshair_overlay = self._build_mpr_crosshair_overlay(
            render_plan.render_view,
            volume.shape,
            plane_pixels.shape,
            image_transform,
        )
        reference_instance, reference_cached = self._get_reference_instance_and_cache(series)
        slice_corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            reference_cached.dataset if reference_cached is not None else None,
            current_index=current,
            total_slices=total,
            viewport_label=self._build_mpr_viewport_label(target_viewport),
        )
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=plane_pixels,
            pixel_min=plane_min,
            pixel_max=plane_max,
            image_transform=image_transform,
            instance=reference_instance,
            cached=reference_cached,
            mpr_viewport=target_viewport,
            measurements=self._build_visible_measurements(view),
            mpr_crosshair=None,
            corner_info=None,
            orientation=None,
        )
        visible_measurements = self._build_visible_measurements(view)
        if fast_preview:
            image = self._render_fast_mpr_preview(context)
        else:
            image = layered_renderer.render(context)

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=current, total=total),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                mprMipConfig=self._serialize_mpr_mip_config(view.mpr_mip),
                mpr_crosshair=self._build_mpr_crosshair_info(mpr_crosshair_overlay),
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info),
                measurements=self._serialize_measurements(
                    visible_measurements,
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=self._serialize_orientation_overlay(
                    self._build_mpr_orientation_overlay(render_plan.render_view, target_viewport)
                ),
            ),
            image_bytes=self._encode_image(image, image_format),
        )

    @staticmethod
    def _render_fast_mpr_preview(context: RenderContext) -> Image.Image:
        return ViewerService._render_fast_preview(context)

    @staticmethod
    def _render_fast_preview(context: RenderContext) -> Image.Image:
        image = ViewerService._render_fast_base_image(
            source_pixels=context.source_pixels,
            pixel_min=context.pixel_min,
            pixel_max=context.pixel_max,
            render_view=context.view,
            image_transform=context.image_transform,
        ).convert("RGBA")
        return layered_renderer.composite_overlays(image, context)

    @staticmethod
    def _render_fast_base_image(
        source_pixels: np.ndarray,
        pixel_min: float,
        pixel_max: float,
        render_view: ViewRecord,
        image_transform,
    ) -> Image.Image:
        base_pixels = ViewerService._window_array(
            source_pixels,
            render_view.window_width,
            render_view.window_center,
            pixel_min=pixel_min,
            pixel_max=pixel_max,
        )
        transformed = viewport_transformer.apply_affine_array(
            base_pixels,
            render_view.width or 0,
            render_view.height or 0,
            image_transform,
            order=1,
            cval=0.0,
        )
        if render_view.pseudocolor_preset != DEFAULT_PSEUDOCOLOR_PRESET:
            transformed = apply_pseudocolor(transformed, render_view.pseudocolor_preset)
            return Image.fromarray(transformed, mode="RGB")
        return Image.fromarray(transformed, mode="L")

    def _build_render_plan_for_shape(
        self,
        view: ViewRecord,
        image_height: int,
        image_width: int,
        *,
        pixel_aspect_x: float = 1.0,
        pixel_aspect_y: float = 1.0,
    ) -> RenderPlan:
        render_ratio = self._resolve_render_ratio_for_shape(
            view,
            image_height,
            image_width,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        if render_ratio >= 0.999:
            return RenderPlan(render_view=view, render_ratio=1.0)

        render_width = max(1, int(round((view.width or 1) * render_ratio)))
        render_height = max(1, int(round((view.height or 1) * render_ratio)))
        scaled_transform = replace(
            view.transform,
            zoom=view.zoom * render_ratio,
            offset_x=view.offset_x * render_ratio,
            offset_y=view.offset_y * render_ratio,
        )
        render_view = replace(
            view,
            width=render_width,
            height=render_height,
            transform=scaled_transform,
        )
        return RenderPlan(render_view=render_view, render_ratio=render_ratio)

    @staticmethod
    def _resolve_render_ratio_for_shape(
        view: ViewRecord,
        image_height: int,
        image_width: int,
        *,
        pixel_aspect_x: float = 1.0,
        pixel_aspect_y: float = 1.0,
    ) -> float:
        if not view.width or not view.height:
            return 1.0

        physical_width = image_width * max(abs(float(pixel_aspect_x)), 1e-6)
        physical_height = image_height * max(abs(float(pixel_aspect_y)), 1e-6)
        if view.width <= physical_width or view.height <= physical_height:
            return 1.0

        contain_zoom = viewport_transformer.calculate_contain_zoom(
            image_width=image_width,
            image_height=image_height,
            canvas_width=view.width,
            canvas_height=view.height,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        if view.zoom > contain_zoom:
            return 1.0

        width_ratio = physical_width / view.width
        height_ratio = physical_height / view.height
        return max(width_ratio, height_ratio)

    @staticmethod
    def _get_mpr_plane_shape(volume_shape: tuple[int, int, int], viewport_key: str) -> tuple[int, int]:
        depth, height, width = volume_shape
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return depth, width
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return depth, height
        return height, width

    @staticmethod
    def _create_default_mpr_mip_state() -> MprMipState:
        return MprMipState()

    @staticmethod
    def _serialize_mpr_mip_config(state: MprMipState) -> MprMipConfig:
        return MprMipConfig(
            enabled=bool(state.enabled),
            algorithm=str(state.algorithm or "maximum"),
            viewports={
                viewport_key: MprMipViewportConfig(thickness=max(1, int(viewport_state.thickness)))
                for viewport_key, viewport_state in state.viewports.items()
            },
        )

    def _handle_mpr_mip_config(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or payload.mpr_mip_config is None:
            return False

        incoming = payload.mpr_mip_config
        current_state = view.mpr_mip
        next_viewports = dict(current_state.viewports)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL):
            next_config = incoming.viewports.get(viewport_key)
            if next_config is None:
                next_viewports[viewport_key] = current_state.viewports.get(viewport_key, MprMipViewportState())
                continue
            next_viewports[viewport_key] = MprMipViewportState(thickness=max(1, int(next_config.thickness)))

        next_state = MprMipState(
            enabled=bool(incoming.enabled),
            algorithm=str(incoming.algorithm or "maximum"),
            viewports=next_viewports,
        )
        if view.view_group is not None:
            view.view_group.mpr_mip = next_state
        return True

    def _extract_mpr_plane(
        self,
        view: ViewRecord,
        volume: np.ndarray,
        viewport_key: str | None = None,
    ) -> tuple[np.ndarray, int, int]:
        depth, height, width = volume.shape
        target_viewport = viewport_key or self._resolve_mpr_viewport(view)
        mip_state = view.mpr_mip

        def reduce_slab(slab: np.ndarray, axis: int) -> np.ndarray:
            if not mip_state.enabled:
                center_index = slab.shape[axis] // 2
                return np.take(slab, indices=center_index, axis=axis)
            algorithm = str(mip_state.algorithm or "maximum")
            if algorithm == "minimum":
                return np.min(slab, axis=axis)
            if algorithm == "average":
                return np.mean(slab, axis=axis)
            if algorithm == "sum":
                return np.sum(slab, axis=axis)
            return np.max(slab, axis=axis)

        def slab_bounds(center_index: int, total_size: int) -> tuple[int, int]:
            thickness = max(1, int(mip_state.viewports.get(target_viewport, MprMipViewportState()).thickness))
            half_before = (thickness - 1) // 2
            half_after = thickness // 2
            start = max(0, center_index - half_before)
            end = min(total_size, center_index + half_after + 1)
            return start, end

        if target_viewport == MPR_VIEWPORT_CORONAL:
            index = max(0, min(view.mpr_coronal_index, height - 1))
            start, end = slab_bounds(index, height)
            slab = volume[:, start:end, :]
            plane = np.flipud(reduce_slab(slab, axis=1))
            return plane.astype(np.float32), index, height
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
            index = max(0, min(view.mpr_sagittal_index, width - 1))
            start, end = slab_bounds(index, width)
            slab = volume[:, :, start:end]
            plane = np.flipud(reduce_slab(slab, axis=2))
            return plane.astype(np.float32), index, width
        index = max(0, min(view.mpr_axial_index, depth - 1))
        view.current_index = index
        start, end = slab_bounds(index, depth)
        slab = volume[start:end, :, :]
        plane = reduce_slab(slab, axis=0)
        return plane.astype(np.float32), index, depth

    @staticmethod
    def _clamp_3d_zoom(zoom: float) -> float:
        return min(max(float(zoom), ZOOM_MIN_3D), ZOOM_MAX_3D)

    def _get_3d_spacing_xyz(self, series: SeriesRecord) -> tuple[float, float, float]:
        transform = self._get_series_patient_transform(series)
        if transform is not None:
            axis_vectors = transform.get("axis_vectors")
            if isinstance(axis_vectors, tuple) and len(axis_vectors) == 3:
                spacing = tuple(max(float(np.linalg.norm(axis_vectors[index])), 1e-3) for index in (2, 1, 0))
                return spacing

        reference_instance, reference_cached = self._get_reference_instance_and_cache(series)
        dataset = reference_cached.dataset if reference_cached is not None else None
        pixel_spacing = getattr(dataset, "PixelSpacing", None) if dataset is not None else None
        slice_spacing = self._estimate_slice_spacing([], np.array([1.0, 0.0, 0.0], dtype=np.float64), dataset)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            try:
                row_spacing = max(abs(float(pixel_spacing[0])), 1e-3)
                col_spacing = max(abs(float(pixel_spacing[1])), 1e-3)
                return (col_spacing, row_spacing, max(slice_spacing, 1e-3))
            except (TypeError, ValueError):
                pass
        return (1.0, 1.0, 1.0)

    def _get_series_volume(self, series: SeriesRecord) -> np.ndarray:
        cached_volume = self._volume_cache.get(series.series_id)
        if cached_volume is not None:
            return cached_volume

        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
            dataset = cached.dataset
            orientation = self._get_dataset_orientation(dataset)
            position = self._get_dataset_position(dataset)
            slice_entries.append((cached.source_pixels, orientation, position))

        if not slice_entries:
            raise HTTPException(status_code=400, detail="Series does not contain readable pixel data")

        first_shape = slice_entries[0][0].shape
        if any(item[0].shape != first_shape for item in slice_entries):
            raise HTTPException(status_code=400, detail="MPR requires a series with consistent slice dimensions")

        volume = self._build_standardized_volume(slice_entries)
        self._volume_cache[series.series_id] = volume
        return volume

    @staticmethod
    def _get_dataset_orientation(dataset) -> np.ndarray | None:
        value = getattr(dataset, "ImageOrientationPatient", None)
        if value is None or len(value) < 6:
            return None
        try:
            orientation = np.asarray([float(item) for item in value[:6]], dtype=np.float64)
        except (TypeError, ValueError):
            self._series_patient_transform_cache[series.series_id] = None
            return None
        return orientation if np.all(np.isfinite(orientation)) else None

    @staticmethod
    def _get_dataset_position(dataset) -> np.ndarray | None:
        value = getattr(dataset, "ImagePositionPatient", None)
        if value is None or len(value) < 3:
            return None
        try:
            position = np.asarray([float(item) for item in value[:3]], dtype=np.float64)
        except (TypeError, ValueError):
            self._series_patient_transform_cache[series.series_id] = None
            return None
        return position if np.all(np.isfinite(position)) else None

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    def _build_standardized_volume(
        self,
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> np.ndarray:
        orientation = next((item[1] for item in slice_entries if item[1] is not None), None)
        if orientation is None:
            return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

        slice_direction = self._normalize_vector(np.cross(row_direction, column_direction))
        if slice_direction is None:
            return np.stack([item[0] for item in slice_entries], axis=0).astype(np.float32)

        positions = [item[2] for item in slice_entries]
        if any(position is None for position in positions):
            ordered_entries = slice_entries
        else:
            ordered_entries = sorted(
                slice_entries,
                key=lambda item: float(np.dot(item[2], slice_direction)) if item[2] is not None else 0.0,
            )

        raw_volume = np.stack([item[0] for item in ordered_entries], axis=0).astype(np.float32)
        raw_axis_vectors = (slice_direction, column_direction, row_direction)
        patient_axes: list[int] = []
        axis_signs: list[int] = []

        for vector in raw_axis_vectors:
            patient_axis = int(np.argmax(np.abs(vector)))
            if patient_axis in patient_axes:
                logger.warning("falling back to non-standardized volume because orientation axes are not orthogonal enough")
                return raw_volume
            patient_axes.append(patient_axis)
            axis_signs.append(1 if vector[patient_axis] >= 0 else -1)

        transpose_order = [patient_axes.index(2), patient_axes.index(1), patient_axes.index(0)]
        canonical_signs = [
            axis_signs[patient_axes.index(2)],
            axis_signs[patient_axes.index(1)],
            axis_signs[patient_axes.index(0)],
        ]
        volume = np.transpose(raw_volume, axes=transpose_order)
        for axis, sign in enumerate(canonical_signs):
            if sign < 0:
                volume = np.flip(volume, axis=axis)

        logger.info(
            "standardized MPR volume shape=%s raw_axes=%s canonical_signs=%s row_dir=%s col_dir=%s slice_dir=%s",
            volume.shape,
            patient_axes,
            canonical_signs,
            np.round(row_direction, 4).tolist(),
            np.round(column_direction, 4).tolist(),
            np.round(slice_direction, 4).tolist(),
        )
        return volume.astype(np.float32, copy=False)

    def _handle_drag_pan(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_offset_x = view.offset_x
            view.drag_origin_offset_y = view.offset_y
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_offset_x = view.drag_origin_offset_x if view.drag_origin_offset_x is not None else view.offset_x
            base_offset_y = view.drag_origin_offset_y if view.drag_origin_offset_y is not None else view.offset_y
            view.offset_x = float(base_offset_x) + float(payload.x or 0.0)
            view.offset_y = float(base_offset_y) + float(payload.y or 0.0)
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_offset_x = None
            view.drag_origin_offset_y = None

    def _handle_drag_rotate_3d(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type not in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}:
            return
        if payload.x is None or payload.y is None or not view.width or not view.height:
            return

        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_arcball_x = float(payload.x)
            view.drag_origin_arcball_y = float(payload.y)
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_arcball_x = None
            view.drag_origin_arcball_y = None
            return

        previous_x = view.drag_origin_arcball_x
        previous_y = view.drag_origin_arcball_y
        current_x = float(payload.x)
        current_y = float(payload.y)
        view.drag_origin_arcball_x = current_x
        view.drag_origin_arcball_y = current_y
        if previous_x is None or previous_y is None:
            return

        delta_x_pixels = (current_x - previous_x) * float(view.width)
        delta_y_pixels = (current_y - previous_y) * float(view.height)
        if abs(delta_x_pixels) < 0.01 and abs(delta_y_pixels) < 0.01:
            return

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        spacing_xyz = self._get_3d_spacing_xyz(series)
        view.rotation_quaternion = vtk_volume_renderer.apply_trackball_camera_delta(
            self._build_volume_render_request(
                view,
                volume=volume,
                spacing_xyz=spacing_xyz,
                fast_preview=True,
            ),
            delta_x_pixels=delta_x_pixels,
            delta_y_pixels=delta_y_pixels,
        )
        view.is_initialized = True

    def _handle_volume_config(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        view.volume_render_config = normalize_volume_render_config(payload.volume_config, view.volume_preset)
        view.volume_preset = str(view.volume_render_config.get("preset", view.volume_preset or "aaa"))
        view.is_initialized = True

    def _handle_volume_preset(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return

        view.volume_preset = normalize_volume_preset_name(payload.sub_op_type or "aaa")
        view.volume_render_config = create_default_volume_render_config(view.volume_preset)
        view.is_initialized = True

    def _handle_drag_zoom(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_zoom = view.zoom
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_zoom = view.drag_origin_zoom if view.drag_origin_zoom is not None else view.zoom
            delta_y = float(payload.y or 0.0)
            zoom_sensitivity = ZOOM_DRAG_SENSITIVITY_3D if self._is_3d_view_type(view.view_type) else ZOOM_DRAG_SENSITIVITY
            zoom_factor = 1.0 - delta_y * zoom_sensitivity
            zoom_factor = max(ZOOM_DRAG_FACTOR_MIN, zoom_factor)
            next_zoom = viewport_transformer.clamp_zoom(float(base_zoom) * zoom_factor)
            if self._is_3d_view_type(view.view_type):
                next_zoom = self._clamp_3d_zoom(next_zoom)
            view.zoom = next_zoom
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_zoom = None

    def _handle_drag_window(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_window_width = view.window_width
            view.drag_origin_window_center = view.window_center
            view.drag_origin_volume_render_config = None
            return

        if payload.action_type == DRAG_ACTION_MOVE:
            base_ww = view.drag_origin_window_width if view.drag_origin_window_width is not None else view.window_width
            base_wl = view.drag_origin_window_center if view.drag_origin_window_center is not None else view.window_center
            base_ww = float(base_ww or 0.0)
            base_wl = float(base_wl or 0.0)
            delta_x = float(payload.x or 0.0)
            delta_y = float(payload.y or 0.0)
            view.window_width = base_ww + delta_x * WINDOW_DRAG_SENSITIVITY
            view.window_center = base_wl - delta_y * WINDOW_DRAG_SENSITIVITY
            view.is_initialized = True
            return

        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_window_width = None
            view.drag_origin_window_center = None
            view.drag_origin_volume_render_config = None

    @staticmethod
    def _handle_pseudocolor(view: ViewRecord, payload: ViewOperationRequest) -> bool:
        next_preset = normalize_pseudocolor_preset(payload.pseudocolor_preset)
        if view.pseudocolor_preset == next_preset:
            return False
        view.pseudocolor_preset = next_preset
        return True

    def _get_mpr_group_views(self, view: ViewRecord) -> list[ViewRecord]:
        if view.view_group is None:
            return [view]
        group_views = view_registry.list_view_group(view.view_group.group_id)
        return group_views or [view]

    def _handle_mpr_crosshair(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if payload.x is None or payload.y is None:
            return False
        if not self._is_mpr_view_type(view.view_type):
            return False
        if not view.width or not view.height:
            raise HTTPException(status_code=400, detail="View size has not been set")

        volume = self._get_series_volume(series_registry.get(view.series_id))
        target_viewport = self._resolve_mpr_viewport(view)
        plane_shape = self._get_mpr_plane_shape(volume.shape, target_viewport)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy(series_registry.get(view.series_id), target_viewport)
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=plane_shape[1],
            image_height=plane_shape[0],
            canvas_width=view.width,
            canvas_height=view.height,
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        crosshair_info = self._build_mpr_crosshair_info(
            self._build_mpr_crosshair_overlay(view, volume.shape, plane_shape, image_transform)
        )

        if payload.action_type == DRAG_ACTION_START:
            view.mpr_crosshair_drag_active = True
            return False

        if payload.action_type == DRAG_ACTION_END:
            was_dragging = view.mpr_crosshair_drag_active
            view.mpr_crosshair_drag_active = False
            return was_dragging

        if payload.action_type != DRAG_ACTION_MOVE or not view.mpr_crosshair_drag_active:
            return False

        canvas_width = float(view.width or 0)
        canvas_height = float(view.height or 0)
        if canvas_width <= 0 or canvas_height <= 0:
            return False

        max_canvas_x = max(canvas_width - 1e-6, 0.0)
        max_canvas_y = max(canvas_height - 1e-6, 0.0)
        canvas_x = min(max(float(payload.x) * canvas_width, 0.0), max_canvas_x)
        canvas_y = min(max(float(payload.y) * canvas_height, 0.0), max_canvas_y)
        image_x, image_y = self._canvas_to_image_coordinates(image_transform, canvas_x, canvas_y)
        depth, height, width = volume.shape
        previous_indices = (view.mpr_axial_index, view.mpr_coronal_index, view.mpr_sagittal_index)

        def nearest_index(value: float, size: int) -> int:
            return max(0, min(int(np.round(value - 0.5)), size - 1))

        if target_viewport == MPR_VIEWPORT_CORONAL:
            view.mpr_sagittal_index = nearest_index(image_x, width)
            view.mpr_axial_index = max(0, min(depth - 1 - nearest_index(image_y, depth), depth - 1))
        elif target_viewport == MPR_VIEWPORT_SAGITTAL:
            view.mpr_coronal_index = nearest_index(image_x, height)
            view.mpr_axial_index = max(0, min(depth - 1 - nearest_index(image_y, depth), depth - 1))
        else:
            view.mpr_sagittal_index = nearest_index(image_x, width)
            view.mpr_coronal_index = nearest_index(image_y, height)

        current_indices = (view.mpr_axial_index, view.mpr_coronal_index, view.mpr_sagittal_index)
        if current_indices == previous_indices:
            return False

        view.current_index = view.mpr_axial_index
        view.is_initialized = True
        return True

    @staticmethod
    def _build_scale_bar_info(
        render_view: ViewRecord,
        image_transform,
        spacing_xy: tuple[float, float] | None,
    ) -> ScaleBarInfo | None:
        if spacing_xy is None or not render_view.width or render_view.width <= 0:
            return None

        spacing_x = max(abs(float(spacing_xy[0])), 1e-6)
        spacing_y = max(abs(float(spacing_xy[1])), 1e-6)
        inverse = np.linalg.inv(image_transform.matrix)
        image_dx = float(inverse[0, 0])
        image_dy = float(inverse[1, 0])
        mm_per_canvas_pixel = float(np.hypot(image_dx * spacing_x, image_dy * spacing_y))
        if not np.isfinite(mm_per_canvas_pixel) or mm_per_canvas_pixel <= 0.0:
            return None

        canvas_width = float(render_view.width)
        selected_length_mm = 100.0
        selected_length_px = selected_length_mm / mm_per_canvas_pixel
        if (
            not np.isfinite(selected_length_px)
            or selected_length_px <= 0.0
            or selected_length_px > canvas_width * 0.8
        ):
            return None

        return ScaleBarInfo(
            lengthNorm=float(selected_length_px) / canvas_width,
            label="10 cm",
        )

    def _get_export_reference_dataset(self, view: ViewRecord) -> Dataset | None:
        series = series_registry.get(view.series_id)
        if self._is_mpr_view_type(view.view_type) or self._is_3d_view_type(view.view_type):
            _, cached = self._get_reference_instance_and_cache(series)
            return cached.dataset if cached is not None else None

        if 0 <= view.current_index < len(series.instances):
            instance = series.instances[view.current_index]
            if instance.sop_instance_uid:
                cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
                return cached.dataset

        _, cached = self._get_reference_instance_and_cache(series)
        return cached.dataset if cached is not None else None

    @staticmethod
    def _build_secondary_capture_dicom_bytes(view: ViewRecord, image: Image.Image, reference_dataset: Dataset | None) -> bytes:
        now = datetime.now()
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        dataset = Dataset()
        dataset.file_meta = file_meta
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False

        if reference_dataset is not None:
            for attribute in (
                "PatientName",
                "PatientID",
                "PatientBirthDate",
                "PatientSex",
                "StudyInstanceUID",
                "StudyID",
                "AccessionNumber",
                "StudyDate",
                "StudyTime",
                "ReferringPhysicianName",
                "InstitutionName",
                "Manufacturer",
            ):
                value = getattr(reference_dataset, attribute, None)
                if value not in (None, ""):
                    setattr(dataset, attribute, value)

        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = generate_uid()
        dataset.Modality = "OT"
        dataset.SeriesNumber = 999
        dataset.InstanceNumber = 1
        dataset.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
        dataset.ConversionType = "WSD"
        dataset.SeriesDescription = f"Exported {view.view_type}"
        dataset.ContentDate = now.strftime("%Y%m%d")
        dataset.ContentTime = now.strftime("%H%M%S")
        dataset.InstanceCreationDate = dataset.ContentDate
        dataset.InstanceCreationTime = dataset.ContentTime
        dataset.BurnedInAnnotation = "YES"
        dataset.SpecificCharacterSet = "ISO_IR 192"

        rgb_image = image.convert("RGB")
        rows, cols = rgb_image.height, rgb_image.width
        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = "RGB"
        dataset.PlanarConfiguration = 0
        dataset.Rows = rows
        dataset.Columns = cols
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.PixelData = rgb_image.tobytes()

        output = io.BytesIO()
        dcmwrite(output, dataset, write_like_original=False)
        return output.getvalue()

    @staticmethod
    def _build_mpr_crosshair_info(overlay: MprCrosshairOverlay) -> MprCrosshairInfo | None:
        if overlay.center_x is None or overlay.center_y is None:
            return None

        normalized_radius = (
            CROSSHAIR_HIT_RADIUS / float(min(overlay.width, overlay.height))
            if min(overlay.width, overlay.height) > 0
            else 0.0
        )
        return MprCrosshairInfo(
            centerX=(
                float(overlay.center_x) / float(overlay.width)
                if overlay.width > 0
                else 0.0
            ),
            centerY=(
                float(overlay.center_y) / float(overlay.height)
                if overlay.height > 0
                else 0.0
            ),
            hitRadius=normalized_radius,
            horizontalPosition=(
                float(overlay.horizontal_position) / float(overlay.height)
                if overlay.horizontal_position is not None and overlay.height > 0
                else None
            ),
            verticalPosition=(
                float(overlay.vertical_position) / float(overlay.width)
                if overlay.vertical_position is not None and overlay.width > 0
                else None
            ),
        )

    @staticmethod
    def _is_point_near_mpr_crosshair_center(
        crosshair_info: MprCrosshairInfo | None,
        canvas_x: float,
        canvas_y: float,
    ) -> bool:
        if crosshair_info is None:
            return False

        delta_x = canvas_x - crosshair_info.center_x
        delta_y = canvas_y - crosshair_info.center_y
        return delta_x * delta_x + delta_y * delta_y <= crosshair_info.hit_radius * crosshair_info.hit_radius

    @staticmethod
    def _canvas_to_image_coordinates(image_transform, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        inverse = np.linalg.inv(image_transform.matrix)
        point = inverse @ np.array([canvas_x, canvas_y, 1.0], dtype=np.float64)
        return float(point[0]), float(point[1])

    @staticmethod
    def _build_mpr_crosshair_overlay(
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        plane_shape: tuple[int, int],
        image_transform,
    ) -> MprCrosshairOverlay:
        depth, _, _ = volume_shape
        plane_height, plane_width = plane_shape
        canvas_width = view.width or plane_width
        canvas_height = view.height or plane_height
        target_viewport = ViewerService._resolve_mpr_viewport(view)
        is_active = view.mpr_active_viewport == target_viewport
        line_alpha = 255

        def with_alpha(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
            return rgb[0], rgb[1], rgb[2], alpha

        axial_color = with_alpha((34, 197, 94), line_alpha)
        coronal_color = with_alpha((59, 130, 246), line_alpha)
        sagittal_color = with_alpha((239, 68, 68), line_alpha)

        def image_to_canvas(image_x: float, image_y: float) -> tuple[float, float]:
            point = image_transform.matrix @ np.array([image_x, image_y, 1.0], dtype=np.float64)
            return float(point[0]), float(point[1])

        top_left_x, top_left_y = image_to_canvas(0.0, 0.0)
        top_right_x, top_right_y = image_to_canvas(float(plane_width), 0.0)
        bottom_left_x, bottom_left_y = image_to_canvas(0.0, float(plane_height))
        bottom_right_x, bottom_right_y = image_to_canvas(float(plane_width), float(plane_height))
        image_left = min(top_left_x, top_right_x, bottom_left_x, bottom_right_x)
        image_top = min(top_left_y, top_right_y, bottom_left_y, bottom_right_y)
        image_right = max(top_left_x, top_right_x, bottom_left_x, bottom_right_x)
        image_bottom = max(top_left_y, top_right_y, bottom_left_y, bottom_right_y)
        image_width = image_right - image_left
        image_height = image_bottom - image_top

        if target_viewport == MPR_VIEWPORT_CORONAL:
            center_x, center_y = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, float(depth - 1 - view.mpr_axial_index) + 0.5)
            _, horizontal_position = image_to_canvas(0.0, float(depth - 1 - view.mpr_axial_index) + 0.5)
            vertical_position, _ = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, 0.0)
            return MprCrosshairOverlay(
                width=canvas_width,
                height=canvas_height,
                image_left=image_left,
                image_top=image_top,
                image_width=image_width,
                image_height=image_height,
                horizontal_position=horizontal_position,
                horizontal_color=axial_color,
                vertical_position=vertical_position,
                vertical_color=sagittal_color,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
            center_x, center_y = image_to_canvas(float(view.mpr_coronal_index) + 0.5, float(depth - 1 - view.mpr_axial_index) + 0.5)
            _, horizontal_position = image_to_canvas(0.0, float(depth - 1 - view.mpr_axial_index) + 0.5)
            vertical_position, _ = image_to_canvas(float(view.mpr_coronal_index) + 0.5, 0.0)
            return MprCrosshairOverlay(
                width=canvas_width,
                height=canvas_height,
                image_left=image_left,
                image_top=image_top,
                image_width=image_width,
                image_height=image_height,
                horizontal_position=horizontal_position,
                horizontal_color=axial_color,
                vertical_position=vertical_position,
                vertical_color=coronal_color,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
        center_x, center_y = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, float(view.mpr_coronal_index) + 0.5)
        _, horizontal_position = image_to_canvas(0.0, float(view.mpr_coronal_index) + 0.5)
        vertical_position, _ = image_to_canvas(float(view.mpr_sagittal_index) + 0.5, 0.0)
        return MprCrosshairOverlay(
            width=canvas_width,
            height=canvas_height,
            image_left=image_left,
            image_top=image_top,
            image_width=image_width,
            image_height=image_height,
            horizontal_position=horizontal_position,
            horizontal_color=coronal_color,
            vertical_position=vertical_position,
            vertical_color=sagittal_color,
            center_x=center_x,
            center_y=center_y,
            is_active=is_active,
        )

    @staticmethod
    def _get_reference_instance_and_cache(series: SeriesRecord) -> tuple[InstanceRecord | None, CachedDicom | None]:
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            return instance, dicom_cache.get(instance.sop_instance_uid, instance.path)
        return None, None

    def _build_series_corner_info_overlay(
        self,
        series: SeriesRecord,
        dataset: Dataset | None,
    ) -> CornerInfoOverlay:
        manufacturer = self._safe_text(getattr(dataset, "Manufacturer", None))
        manufacturer_model = self._safe_text(getattr(dataset, "ManufacturerModelName", None))
        station_name = self._safe_text(getattr(dataset, "StationName", None))
        institution_name = self._safe_text(getattr(dataset, "InstitutionName", None))
        study_description = self._safe_text(getattr(dataset, "StudyDescription", None))
        exam_text = self._first_non_empty(
            study_description,
            self._safe_text(getattr(dataset, "StudyID", None)),
            self._safe_text(series.series_description),
        )
        series_number = self._safe_text(getattr(dataset, "SeriesNumber", None))
        patient_name = self._safe_text(getattr(dataset, "PatientName", None))
        patient_id = self._first_non_empty(self._safe_text(getattr(dataset, "PatientID", None)), self._safe_text(series.patient_id))
        patient_sex = self._safe_text(getattr(dataset, "PatientSex", None))
        patient_age = self._safe_text(getattr(dataset, "PatientAge", None))
        acquisition_date = self._first_non_empty(
            self._format_dicom_date(getattr(dataset, "AcquisitionDate", None)),
            self._format_dicom_date(getattr(dataset, "ContentDate", None)),
            self._format_dicom_date(getattr(dataset, "StudyDate", None)),
        )
        acquisition_time = self._first_non_empty(
            self._format_dicom_time(getattr(dataset, "AcquisitionTime", None)),
            self._format_dicom_time(getattr(dataset, "ContentTime", None)),
            self._format_dicom_time(getattr(dataset, "StudyTime", None)),
        )
        kv = self._format_number(getattr(dataset, "KVP", None), suffix="kV")
        ma = self._format_number(getattr(dataset, "XRayTubeCurrent", None), suffix="mA")
        thickness = self._format_number(getattr(dataset, "SliceThickness", None), suffix="mm")

        vendor_line = self._join_non_empty(" / ", manufacturer, manufacturer_model)
        patient_meta = self._join_non_empty(" ", patient_id, self._join_non_empty(" / ", patient_sex, patient_age))
        technique_parts = [part for part in (kv, ma) if part]

        top_left = tuple(
            line
            for line in (
                vendor_line,
                station_name,
                institution_name,
                exam_text,
                f"Se: {series_number}" if series_number else None,
            )
            if line
        )
        top_right = tuple(
            line
            for line in (
                patient_name,
                patient_meta,
            )
            if line
        )
        bottom_left = tuple(
            line
            for line in (
                " ".join(technique_parts) if technique_parts else None,
                thickness,
                self._join_non_empty(" ", acquisition_date, acquisition_time),
            )
            if line
        )
        return CornerInfoOverlay(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=tuple(),
        )

    def _build_slice_corner_info_overlay(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        dataset: Dataset | None,
        *,
        current_index: int,
        total_slices: int,
        viewport_label: str,
    ) -> CornerInfoOverlay:
        zoom = self._format_number(view.zoom, precision=2, suffix="x")
        physical_location = self._build_physical_location_label(view, series, dataset, current_index, viewport_label)
        top_left = tuple(
            line
            for line in (
                self._join_non_empty("  ", viewport_label, physical_location),
                f"Im: {current_index + 1}/{total_slices}" if total_slices > 0 else None,
            )
            if line
        )
        top_right = tuple()
        bottom_left = tuple(
            line
            for line in (
                self._build_window_label(view.window_width, view.window_center),
            )
            if line
        )
        bottom_right = tuple(
            line
            for line in (
                f"Zoom:{zoom}" if zoom else None,
                f"X:{int(round(view.offset_x))} Y:{int(round(view.offset_y))}",
            )
            if line
        )
        return CornerInfoOverlay(
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
        )

    @staticmethod
    def _serialize_corner_info_overlay(overlay: CornerInfoOverlay) -> CornerInfoPayload:
        return CornerInfoPayload(
            topLeft=list(overlay.top_left),
            topRight=list(overlay.top_right),
            bottomLeft=list(overlay.bottom_left),
            bottomRight=list(overlay.bottom_right),
        )

    @staticmethod
    def _serialize_orientation_overlay(overlay: OrientationOverlay | None) -> OrientationInfo:
        return OrientationInfo(
            top=overlay.top if overlay is not None else None,
            right=overlay.right if overlay is not None else None,
            bottom=overlay.bottom if overlay is not None else None,
            left=overlay.left if overlay is not None else None,
            volumeQuaternion=getattr(overlay, "volume_quaternion", None) if overlay is not None else None,
        )

    def _build_3d_orientation_overlay(self, view: ViewRecord) -> OrientationInfo:
        quaternion = self._normalize_quaternion(tuple(float(value) for value in view.rotation_quaternion))
        return OrientationInfo(
            top=None,
            right=None,
            bottom=None,
            left=None,
            volumeQuaternion=quaternion,
        )

    @staticmethod
    def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        vector = np.asarray(quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        vector /= norm
        return tuple(float(value) for value in vector)

    def _build_physical_location_label(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        dataset: Dataset | None,
        current_index: int,
        viewport_label: str,
    ) -> str | None:
        label = viewport_label.lower()
        if self._is_mpr_view_type(view.view_type):
            transform = self._get_series_patient_transform(series)
            if transform is not None:
                shape = transform["shape"]
                x_index = max(0, min(view.mpr_sagittal_index, int(shape[2]) - 1))
                y_index = max(0, min(view.mpr_coronal_index, int(shape[1]) - 1))
                z_index = max(0, min(view.mpr_axial_index, int(shape[0]) - 1))
                patient_point = (
                    transform["origin"]
                    + transform["axis_vectors"][0] * float(z_index)
                    + transform["axis_vectors"][1] * float(y_index)
                    + transform["axis_vectors"][2] * float(x_index)
                )
                if label.startswith("cor"):
                    return self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A")
                if label.startswith("sag"):
                    return self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R")
                if label.startswith("ax"):
                    return self._format_oriented_mm(float(patient_point[2]), positive="S", negative="I")
                return self._join_non_empty(
                    " ",
                    self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R"),
                    self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A"),
                    self._format_oriented_mm(float(patient_point[2]), positive="S", negative="I"),
                )

        position = self._get_dataset_position(dataset)
        if position is None:
            return None
        if label.startswith("stack") or label.startswith("ax"):
            return self._format_oriented_mm(float(position[2]), positive="S", negative="I")
        if label.startswith("cor"):
            return self._format_oriented_mm(float(position[1]), positive="P", negative="A")
        if label.startswith("sag"):
            return self._format_oriented_mm(float(position[0]), positive="L", negative="R")
        return self._join_non_empty(
            " ",
            self._format_oriented_mm(float(position[0]), positive="L", negative="R"),
            self._format_oriented_mm(float(position[1]), positive="P", negative="A"),
            self._format_oriented_mm(float(position[2]), positive="S", negative="I"),
        )

    def _get_series_patient_transform(self, series: SeriesRecord) -> dict[str, object] | None:
        cached_transform = self._series_patient_transform_cache.get(series.series_id, Ellipsis)
        if cached_transform is not Ellipsis:
            return cached_transform

        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Dataset]] = []
        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
            dataset = cached.dataset
            slice_entries.append((
                cached.source_pixels,
                self._get_dataset_orientation(dataset),
                self._get_dataset_position(dataset),
                dataset,
            ))

        if not slice_entries:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        orientation = next((item[1] for item in slice_entries if item[1] is not None), None)
        if orientation is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        slice_direction = self._normalize_vector(np.cross(row_direction, column_direction))
        if slice_direction is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        positions = [item[2] for item in slice_entries]
        if any(position is None for position in positions):
            ordered_entries = slice_entries
        else:
            ordered_entries = sorted(
                slice_entries,
                key=lambda item: float(np.dot(item[2], slice_direction)) if item[2] is not None else 0.0,
            )

        first_dataset = ordered_entries[0][3]
        pixel_spacing = getattr(first_dataset, "PixelSpacing", None)
        if pixel_spacing is None or len(pixel_spacing) < 2:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        try:
            row_spacing = abs(float(pixel_spacing[0]))
            col_spacing = abs(float(pixel_spacing[1]))
        except (TypeError, ValueError):
            self._series_patient_transform_cache[series.series_id] = None
            return None

        ordered_positions = [item[2] for item in ordered_entries if item[2] is not None]
        slice_spacing = self._estimate_slice_spacing(ordered_positions, slice_direction, first_dataset)

        raw_axis_vectors = (slice_direction, column_direction, row_direction)
        raw_axis_steps = (slice_spacing, row_spacing, col_spacing)
        raw_lengths = (
            len(ordered_entries),
            int(getattr(first_dataset, "Rows", 0) or 0),
            int(getattr(first_dataset, "Columns", 0) or 0),
        )
        if any(length <= 0 for length in raw_lengths):
            return None

        patient_axes: list[int] = []
        axis_signs: list[int] = []
        for vector in raw_axis_vectors:
            patient_axis = int(np.argmax(np.abs(vector)))
            if patient_axis in patient_axes:
                self._series_patient_transform_cache[series.series_id] = None
                return None
            patient_axes.append(patient_axis)
            axis_signs.append(1 if vector[patient_axis] >= 0 else -1)

        transpose_order = [patient_axes.index(2), patient_axes.index(1), patient_axes.index(0)]
        canonical_signs = [
            axis_signs[patient_axes.index(2)],
            axis_signs[patient_axes.index(1)],
            axis_signs[patient_axes.index(0)],
        ]

        origin = np.asarray(ordered_entries[0][2], dtype=np.float64)
        for canonical_axis, raw_axis in enumerate(transpose_order):
            if canonical_signs[canonical_axis] < 0:
                origin = origin + raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(raw_lengths[raw_axis] - 1)

        axis_vectors = tuple(
            raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(canonical_signs[canonical_axis])
            for canonical_axis, raw_axis in enumerate(transpose_order)
        )
        shape = tuple(raw_lengths[raw_axis] for raw_axis in transpose_order)
        result = {
            "origin": origin,
            "axis_vectors": axis_vectors,
            "shape": shape,
        }
        self._series_patient_transform_cache[series.series_id] = result
        return result

    @staticmethod
    def _estimate_slice_spacing(
        positions: list[np.ndarray],
        slice_direction: np.ndarray,
        dataset: Dataset | None,
    ) -> float:
        if len(positions) >= 2:
            projected = sorted(float(np.dot(position, slice_direction)) for position in positions)
            diffs = [abs(projected[index] - projected[index - 1]) for index in range(1, len(projected))]
            diffs = [diff for diff in diffs if diff > 1e-6]
            if diffs:
                return float(np.median(diffs))
        slice_thickness = getattr(dataset, "SliceThickness", None) if dataset is not None else None
        try:
            thickness = abs(float(slice_thickness))
            if thickness > 0:
                return thickness
        except (TypeError, ValueError):
            pass
        return 1.0

    @staticmethod
    def _format_oriented_mm(value: float, *, positive: str, negative: str) -> str:
        orientation = positive if float(value) >= 0 else negative
        magnitude = abs(float(value))
        return f"{orientation} {magnitude:.2f}mm"

    @staticmethod
    def _rotate_screen_axes(
        x_vector: np.ndarray,
        y_vector: np.ndarray,
        rotation_degrees: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_rotation = viewport_transformer.normalize_rotation_degrees(rotation_degrees)
        if normalized_rotation == 90:
            return y_vector, -x_vector
        if normalized_rotation == 180:
            return -x_vector, -y_vector
        if normalized_rotation == 270:
            return -y_vector, x_vector
        return x_vector, y_vector

    @staticmethod
    def _build_view_transform_payload(view: ViewRecord) -> ViewTransformPayload:
        return ViewTransformPayload(
            rotationDegrees=viewport_transformer.normalize_rotation_degrees(view.rotation_degrees),
            horFlip=bool(view.hor_flip),
            verFlip=bool(view.ver_flip),
        )

    def _build_stack_orientation_overlay(self, view: ViewRecord, dataset: Dataset | None) -> OrientationOverlay | None:
        orientation = self._get_dataset_orientation(dataset)
        if orientation is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        x_vector = row_direction * (-1.0 if view.hor_flip else 1.0)
        y_vector = column_direction * (-1.0 if view.ver_flip else 1.0)
        x_vector, y_vector = self._rotate_screen_axes(x_vector, y_vector, view.rotation_degrees)
        return OrientationOverlay(
            top=self._orientation_text_for_vector(-y_vector),
            right=self._orientation_text_for_vector(x_vector),
            bottom=self._orientation_text_for_vector(y_vector),
            left=self._orientation_text_for_vector(-x_vector),
        )

    def _build_mpr_orientation_overlay(self, view: ViewRecord, viewport_key: str) -> OrientationOverlay:
        if viewport_key == MPR_VIEWPORT_CORONAL:
            x_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            y_vector = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        elif viewport_key == MPR_VIEWPORT_SAGITTAL:
            x_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            y_vector = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            x_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            y_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

        if view.hor_flip:
            x_vector = -x_vector
        if view.ver_flip:
            y_vector = -y_vector
        x_vector, y_vector = self._rotate_screen_axes(x_vector, y_vector, view.rotation_degrees)

        return OrientationOverlay(
            top=self._orientation_text_for_vector(-y_vector),
            right=self._orientation_text_for_vector(x_vector),
            bottom=self._orientation_text_for_vector(y_vector),
            left=self._orientation_text_for_vector(-x_vector),
        )

    @staticmethod
    def _build_mpr_viewport_label(viewport_key: str) -> str:
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return "CORONAL"
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return "SAGITTAL"
        return "AXIAL"

    @staticmethod
    def _build_window_label(window_width: float | None, window_center: float | None) -> str | None:
        ww = ViewerService._format_number(window_width, precision=0)
        wl = ViewerService._format_number(window_center, precision=0)
        if ww is None and wl is None:
            return None
        return f"W: {ww or '-'} L: {wl or '-'}"

    @staticmethod
    def _safe_text(value) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _first_non_empty(*values: str | None) -> str | None:
        for value in values:
            if value:
                return value
        return None

    @staticmethod
    def _join_non_empty(separator: str, *values: str | None) -> str | None:
        parts = [value for value in values if value]
        if not parts:
            return None
        return separator.join(parts)

    @staticmethod
    def _format_number(value, *, precision: int = 1, suffix: str = "") -> str | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            return f"{text}{suffix}" if text else None
        if precision <= 0:
            rendered = str(int(round(numeric)))
        else:
            rendered = f"{numeric:.{precision}f}".rstrip("0").rstrip(".")
        return f"{rendered}{suffix}"

    @staticmethod
    def _format_dicom_date(value) -> str | None:
        text = ViewerService._safe_text(value)
        if not text or len(text) != 8 or not text.isdigit():
            return text
        return f"{text[:4]}.{text[4:6]}.{text[6:8]}"

    @staticmethod
    def _format_dicom_time(value) -> str | None:
        text = ViewerService._safe_text(value)
        if not text:
            return None
        digits = ''.join(ch for ch in text if ch.isdigit())
        if len(digits) < 6:
            return text
        return f"{digits[:2]}:{digits[2:4]}:{digits[4:6]}"

    @staticmethod
    def _orientation_text_for_vector(vector: np.ndarray | None) -> str | None:
        if vector is None:
            return None
        axis_map = (
            (0, "L", "R"),
            (1, "P", "A"),
            (2, "S", "I"),
        )
        components: list[tuple[float, str]] = []
        for axis_index, positive_label, negative_label in axis_map:
            component = float(vector[axis_index])
            magnitude = abs(component)
            if magnitude < 0.2:
                continue
            label = positive_label if component >= 0 else negative_label
            components.append((magnitude, label))
        if not components:
            return None
        components.sort(key=lambda item: item[0], reverse=True)
        return ''.join(label for _, label in components[:3])

    @staticmethod
    def _window_array(
        pixels: np.ndarray,
        window_width: float | None,
        window_center: float | None,
        *,
        pixel_min: float | None = None,
        pixel_max: float | None = None,
    ) -> np.ndarray:
        lower_bound = float(np.min(pixels)) if pixel_min is None else float(pixel_min)
        upper_bound = float(np.max(pixels)) if pixel_max is None else float(pixel_max)

        if window_width is not None and window_width > 0 and window_center is not None:
            lower = window_center - window_width / 2.0
            upper = window_center + window_width / 2.0
        else:
            lower = lower_bound
            upper = upper_bound

        scale = upper - lower
        if scale <= 0:
            return np.zeros(pixels.shape, dtype=np.uint8)

        normalized = np.asarray(pixels, dtype=np.float32).copy()
        np.clip(normalized, lower, upper, out=normalized)
        normalized -= lower
        normalized *= 255.0 / scale
        return normalized.astype(np.uint8, copy=False)

    @staticmethod
    def _resolve_mpr_viewport(view: ViewRecord) -> str:
        if view.view_type == "COR":
            return MPR_VIEWPORT_CORONAL
        if view.view_type == "SAG":
            return MPR_VIEWPORT_SAGITTAL
        return MPR_VIEWPORT_AXIAL

    @staticmethod
    def _derive_default_window_width(cached: CachedDicom) -> float:
        return max(WINDOW_WIDTH_MIN, cached.pixel_max - cached.pixel_min)


    @staticmethod
    def _derive_default_window_center(cached: CachedDicom) -> float:
        return (cached.pixel_max + cached.pixel_min) / 2.0

    @staticmethod
    def _reset_drag_state(view: ViewRecord) -> None:
        view.drag_origin_zoom = None
        view.drag_origin_offset_x = None
        view.drag_origin_offset_y = None
        view.drag_origin_window_width = None
        view.drag_origin_window_center = None
        view.drag_origin_rotation_quaternion = None
        view.drag_origin_arcball_x = None
        view.drag_origin_arcball_y = None

    @staticmethod
    def _encode_image(image: Image.Image, image_format: ImageFormat) -> bytes:
        output = io.BytesIO()
        if image_format == "jpeg":
            image.convert("RGB").save(output, format="JPEG", quality=20)
        else:
            image.save(output, format="PNG")
        return output.getvalue()


viewer_service = ViewerService()

