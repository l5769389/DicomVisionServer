import io
import hashlib
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import import_module
from time import perf_counter
from typing import Any, Callable
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
from app.models.viewer import (
    InstanceRecord,
    MprCursorRecord,
    MprFrameState,
    MprMipState,
    MprMipViewportState,
    MprObliquePlaneState,
    MprRotationDragRecord,
    PresentationAnnotationRecord,
    PresentationMeasurementRecord,
    SeriesRecord,
    ViewGroupRecord,
    ViewRecord,
)
from app.schemas.dicom import CornerInfoPayload, CornerInfoRequest, CornerInfoResponse
from app.schemas.view import (
    ImageFormat,
    AnnotationOverlayPayload,
    MprCrosshairInfo,
    MprCursorInfo,
    MprFrameInfo,
    MprMipConfig,
    MprMipViewportConfig,
    MprPlaneInfo,
    MeasurementOverlayPayload,
    OperationAcceptedResponse,
    OrientationInfo,
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
from app.services.dicom_geometry import (
    build_standardized_volume,
    get_dataset_orientation,
    get_dataset_position,
    get_standardized_axis_mapping,
    normalize_vector,
)
from app.services.hover_mapping import map_normalized_canvas_to_image_row_col
from app.services.layered_renderer import RenderContext, layered_renderer
from app.services.measurement_geometry import build_smooth_path_points
from app.services.measurement_rules import get_measurement_point_requirement, has_required_measurement_points
from app.services.measurement_utils import build_measurement_metrics
from app.services.mpr import (
    MipConfig as ResliceMipConfig,
    MprCursorState,
    DEFAULT_MPR_CONVENTION,
    OutputShapePolicy,
    PlanePose,
    VolumeGeometry,
    axis_angle_rotation_matrix,
    build_geometry_from_patient_transform,
    build_identity_geometry,
    create_default_cursor,
    cursor_to_legacy_frame,
    derive_plane_pose,
    ijk_to_world_point,
    legacy_frame_to_cursor,
    orthonormalize_matrix,
    reslice_plane,
    spacing_along_world_direction,
    translate_cursor,
    world_to_ijk_point,
)
from app.services import mpr_geometry
from app.services.dicom_gsps_export_service import build_gsps_dicom_bytes
from app.services.dicom_sr_export_service import build_measurement_sr_dicom_bytes
from app.services.mpr_geometry import VolumePatientTransform
from app.services.mtf_analysis_service import MtfAnalysisService
from app.services.pseudocolor import DEFAULT_PSEUDOCOLOR_PRESET, apply_pseudocolor, normalize_pseudocolor_preset
from app.services.render_layers.render_context import CornerInfoOverlay, MprCrosshairOverlay, OrientationOverlay
from app.services.representative_slice_selector import (
    build_representative_sample_indexes,
    score_representative_pixels,
)
from app.services.series_volume_cache import SeriesVolumeCache
from app.services.series_registry import series_registry
from app.services.viewport_transformer import viewport_transformer
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry
from app.services.viewer_operation_handlers import OperationRenderOutcome, handle_view_operation
from app.services.viewer_render_dispatch import render_by_view_type
from app.services.viewer_render_guards import ensure_view_size
from app.services.water_phantom_qa_service import WaterPhantomQaService
from app.services.volume_render_config import (
    create_default_volume_render_config,
    normalize_volume_preset_name,
    normalize_volume_render_config,
)
from app.services.surface_render_config import create_default_surface_render_config, normalize_surface_render_config
from app.services.volume_rendering.contracts import SurfaceRenderRequest, VolumeRenderRequest


logger = get_logger(__name__)

CROSSHAIR_HIT_RADIUS = 12.0
MEASUREMENT_TOOL_TYPES = {"line", "rect", "ellipse", "angle", "curve", "freeform"}
VOLUME_CACHE_MAX_BYTES = 1024 * 1024 * 1024
FAST_PREVIEW_JPEG_QUALITY = 20
PNG_COMPRESS_LEVEL = 1
MPR_FAST_PREVIEW_SCALE = 0.33
MPR_FAST_PREVIEW_MIN_SIDE = 96
MPR_PLANE_CACHE_MAX_ITEMS = 48
FAST_BASE_PIXELS_CACHE_MAX_ITEMS = 64
MPR_CROSSHAIR_MODE_ORTHOGONAL = "orthogonal"
MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE = "double-oblique"
MPR_CROSSHAIR_MODES = {
    MPR_CROSSHAIR_MODE_ORTHOGONAL,
    MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE,
}


class _LazyRendererProxy:
    def __init__(self, module_name: str, renderer_name: str) -> None:
        self._module_name = module_name
        self._renderer_name = renderer_name
        self._target: Any | None = None

    def _resolve(self) -> Any:
        if self._target is None:
            module = import_module(self._module_name)
            self._target = getattr(module, self._renderer_name)
        return self._target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_module_name", "_renderer_name", "_target"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._resolve(), name, value)


vtk_volume_renderer = _LazyRendererProxy(
    "app.services.volume_rendering.vtk_volume_renderer",
    "vtk_volume_renderer",
)
vtk_surface_renderer = _LazyRendererProxy(
    "app.services.volume_rendering.vtk_surface_renderer",
    "vtk_surface_renderer",
)


def _get_vtk_volume_renderer():
    return vtk_volume_renderer


def _get_vtk_surface_renderer():
    return vtk_surface_renderer


@dataclass(frozen=True)
class RenderedImageResult:
    meta: ViewImageResponse
    image_bytes: bytes


@dataclass(frozen=True)
class MprPoseContext:
    geometry: VolumeGeometry
    cursor: MprCursorState
    poses: dict[str, PlanePose]


@dataclass(frozen=True)
class ExportedFileResult:
    file_bytes: bytes
    file_name: str
    media_type: str


@dataclass(frozen=True)
class RenderPlan:
    render_view: ViewRecord
    render_ratio: float


ViewRenderProgressCallback = Callable[[dict[str, object]], None]


class ViewerService:
    def __init__(self) -> None:
        self._series_patient_transform_cache: dict[str, VolumePatientTransform | None] = {}
        self._series_volume_geometry_cache: dict[str, VolumeGeometry] = {}
        self._series_representative_slice_cache: dict[str, tuple[int, int]] = {}
        self._series_volume_cache = SeriesVolumeCache(
            max_bytes=VOLUME_CACHE_MAX_BYTES,
            on_evict=self._handle_series_volume_cache_evict,
        )
        self._mpr_plane_cache: OrderedDict[tuple[object, ...], tuple[np.ndarray, int, int]] = OrderedDict()
        self._fast_base_pixels_cache: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._mtf_analysis_service = MtfAnalysisService(self)
        self._water_phantom_qa_service = WaterPhantomQaService(self)
        self._logger = logger

    @staticmethod
    def _is_mpr_view_type(view_type: str) -> bool:
        return view_type in {"MPR", "AX", "COR", "SAG"}

    @staticmethod
    def _is_3d_view_type(view_type: str) -> bool:
        return view_type == "3D"

    def set_view_size(
        self,
        payload: ViewSetSizeRequest,
        workspace_id: str | None = None,
    ) -> OperationAcceptedResponse:
        if payload.op_type != VIEW_OP_TYPE_SET_SIZE:
            raise HTTPException(status_code=400, detail="opType must be setSize")

        view = view_registry.get(payload.view_id, workspace_id=workspace_id)
        view.width = payload.size.width
        view.height = payload.size.height
        logger.info(
            "set view size view_id=%s width=%s height=%s",
            view.view_id,
            view.width,
            view.height,
        )

        if not view.is_initialized:
            if not (self._is_mpr_view_type(view.view_type) or self._is_3d_view_type(view.view_type)):
                self._initialize_viewport(view)
                view.is_initialized = True

        return OperationAcceptedResponse(message="View size updated", viewId=view.view_id)

    def render_view_by_id(
        self,
        view_id: str,
        *,
        image_format: ImageFormat = "png",
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
        workspace_id: str | None = None,
    ) -> RenderedImageResult:
        view = view_registry.get(view_id, workspace_id=workspace_id)
        if self._is_mpr_view_type(view.view_type):
            view = self._snapshot_mpr_view_for_render(view)
        return self._render_by_view_type(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
        )

    def _snapshot_mpr_view_for_render(self, view: ViewRecord) -> ViewRecord:
        ensure_view_size(view)
        if not view.is_initialized:
            self._initialize_mpr_viewport(view)
            view.is_initialized = True
        return deepcopy(view)

    def close_view_by_id(self, view_id: str, workspace_id: str | None = None) -> OperationAcceptedResponse:
        view = view_registry.delete(view_id, workspace_id=workspace_id)
        if self._is_3d_view_type(view.view_type):
            _get_vtk_volume_renderer().drop_session(view.view_id)
            _get_vtk_surface_renderer().drop_session(view.view_id)
        group = view.view_group
        if group is not None and not view_registry.list_view_group(group.group_id, workspace_id=workspace_id):
            view_group_registry.delete(group.group_id)
        return OperationAcceptedResponse(message="View closed", viewId=view.view_id)

    def export_view_by_id(
        self,
        view_id: str,
        export_format: str,
        *,
        overlays: ViewExportOverlaysPayload | None = None,
        workspace_id: str | None = None,
    ) -> ExportedFileResult:
        view = view_registry.get(view_id, workspace_id=workspace_id)
        safe_view_type = str(view.view_type or "view").lower()

        if export_format == "dicom-sr":
            reference_dataset = self._get_export_reference_dataset(view)
            dicom_sr_bytes = build_measurement_sr_dicom_bytes(view, overlays, reference_dataset)
            return ExportedFileResult(
                file_bytes=dicom_sr_bytes,
                file_name=f"{view.view_id}-{safe_view_type}-measurements-sr.dcm",
                media_type="application/dicom",
            )

        if export_format == "dicom-gsps":
            reference_dataset = self._get_export_reference_dataset(view)
            gsps_bytes = build_gsps_dicom_bytes(view, overlays, reference_dataset)
            return ExportedFileResult(
                file_bytes=gsps_bytes,
                file_name=f"{view.view_id}-{safe_view_type}-presentation-state.dcm",
                media_type="application/dicom",
            )

        if export_format == "png":
            rendered = self._render_by_view_type(view, image_format="png", fast_preview=False)
            if overlays and (overlays.annotations or overlays.measurements):
                try:
                    image = Image.open(io.BytesIO(rendered.image_bytes)).convert("RGB")
                    image = self._apply_export_overlays(image, overlays)
                    rendered_bytes = self._encode_image(image, "png", fast_preview=False)
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
        elif tool_type == "curve" and len(points) >= 2:
            self._draw_export_polyline(draw, build_smooth_path_points(points))
        elif tool_type == "freeform" and len(points) >= 3:
            self._draw_export_polyline(draw, build_smooth_path_points(points, close_path=True))
        else:
            return

        if label_lines:
            anchor = points[-1] if tool_type == "curve" else points[1] if len(points) >= 2 else points[0]
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

    def handle_view_operation(
        self,
        payload: ViewOperationRequest,
        workspace_id: str | None = None,
    ) -> OperationRenderOutcome:
        return handle_view_operation(self, payload, workspace_id=workspace_id)

    def handle_view_hover(
        self,
        payload: ViewHoverRequest,
        workspace_id: str | None = None,
    ) -> ViewHoverResponse:
        view = view_registry.get(payload.view_id, workspace_id=workspace_id)
        row, col = self._resolve_hover_row_col(view, payload.x, payload.y)
        return ViewHoverResponse(viewId=view.view_id, row=row, col=col)

    def get_series_corner_info(
        self,
        payload: CornerInfoRequest,
        workspace_id: str | None = None,
    ) -> CornerInfoResponse:
        series = series_registry.get(payload.series_id, workspace_id=workspace_id)
        _, reference_cached = self._get_reference_instance_and_cache(series)
        overlay = self._build_series_corner_info_overlay(
            series,
            reference_cached.dataset if reference_cached is not None else None,
        )
        return CornerInfoResponse(cornerInfo=self._serialize_corner_info_overlay(overlay))

    def analyze_mtf(
        self,
        payload: ViewMtfAnalyzeRequest,
        workspace_id: str | None = None,
    ) -> ViewMtfAnalyzeResponse:
        view_registry.get(payload.view_id, workspace_id=workspace_id)
        return self._mtf_analysis_service.analyze(payload)

    def analyze_qa_water(
        self,
        payload: ViewQaWaterAnalyzeRequest,
        workspace_id: str | None = None,
    ) -> ViewQaWaterAnalyzeResponse:
        view_registry.get(payload.view_id, workspace_id=workspace_id)
        return self._water_phantom_qa_service.analyze(payload)

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
            target_viewport = self._resolve_mpr_viewport(view)
            volume = self._get_series_volume(series)
            pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
            pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(
                pose_context.poses[target_viewport]
            )
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
        return MeasurementPoint(x=float(source_point[0]), y=float(source_point[1]))

    def _resolve_measurement_source_context(
        self,
        view: ViewRecord,
    ) -> tuple[np.ndarray, tuple[float, float] | None, MeasurementSliceContext]:
        if self._is_mpr_view_type(view.view_type):
            series = series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            target_viewport = self._resolve_mpr_viewport(view)
            plane_pixels, current_index, _ = self._extract_mpr_plane(view, volume, target_viewport)
            pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
            return (
                plane_pixels,
                self._get_mpr_spacing_xy_from_pose(pose_context.poses[target_viewport]),
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
        if tool_type in {"curve", "freeform"}:
            return len(points) < get_measurement_point_requirement(tool_type).min_points
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

        if tool_type == "angle" and len(image_points) < get_measurement_point_requirement(tool_type).min_points:
            return self._build_measurement_preview_payload(
                view=view,
                viewport_key=viewport_key,
                tool_type=tool_type,
                slice_index=slice_context.slice_index,
            )

        if not has_required_measurement_points(tool_type, len(image_points)):
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

        if not has_required_measurement_points(tool_type, len(payload.points)):
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

    @staticmethod
    def _clear_measurements(view: ViewRecord) -> bool:
        if not view.measurements:
            return False

        view.measurements = []
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
        measurements: tuple[Any, ...],
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

    def _build_visible_presentation_measurements(
        self,
        series: SeriesRecord,
        instance: InstanceRecord,
    ) -> tuple[PresentationMeasurementRecord, ...]:
        if not instance.sop_instance_uid:
            return ()

        presentation_states = series.presentation_states_by_sop_uid.get(str(instance.sop_instance_uid), [])
        return tuple(
            measurement
            for presentation_state in presentation_states
            for measurement in presentation_state.measurements
        )

    def _build_visible_presentation_annotations(
        self,
        series: SeriesRecord,
        instance: InstanceRecord,
    ) -> tuple[PresentationAnnotationRecord, ...]:
        if not instance.sop_instance_uid:
            return ()

        presentation_states = series.presentation_states_by_sop_uid.get(str(instance.sop_instance_uid), [])
        return tuple(
            annotation
            for presentation_state in presentation_states
            for annotation in presentation_state.annotations
        )

    @staticmethod
    def _serialize_annotations(
        annotations: tuple[PresentationAnnotationRecord, ...],
        *,
        image_transform: Any,
        canvas_width: int,
        canvas_height: int,
    ) -> list[AnnotationOverlayPayload]:
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
            AnnotationOverlayPayload(
                annotationId=annotation.annotation_id,
                toolType=annotation.tool_type,
                points=[serialize_point(point) for point in annotation.points],
                text=annotation.text,
                color=annotation.color,
                size=annotation.size,
            )
            for annotation in annotations
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

    def _get_mpr_spacing_xy(
        self,
        series: SeriesRecord,
        viewport_key: str,
        plane_state: MprObliquePlaneState | None = None,
    ) -> tuple[float, float] | None:
        if plane_state is not None:
            transform = self._get_series_patient_transform(series)
            if transform is not None:
                return (
                    transform.spacing_for_direction(plane_state.col),
                    transform.spacing_for_direction(plane_state.row),
                )
        spacing_x, spacing_y, spacing_z = self._get_3d_spacing_xyz(series)
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return (spacing_x, spacing_z)
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return (spacing_y, spacing_z)
        return (spacing_x, spacing_y)

    def _get_mpr_display_aspect_xy(
        self,
        series: SeriesRecord,
        viewport_key: str,
        plane_state: MprObliquePlaneState | None = None,
    ) -> tuple[float, float]:
        spacing_xy = self._get_mpr_spacing_xy(series, viewport_key, plane_state)
        if spacing_xy is None:
            return (1.0, 1.0)
        return (
            max(abs(float(spacing_xy[0])), 1e-6),
            max(abs(float(spacing_xy[1])), 1e-6),
        )

    @staticmethod
    def _get_mpr_spacing_xy_from_pose(plane_pose: PlanePose) -> tuple[float, float]:
        return (
            max(abs(float(plane_pose.pixel_spacing_col_mm)), 1e-6),
            max(abs(float(plane_pose.pixel_spacing_row_mm)), 1e-6),
        )

    @staticmethod
    def _get_mpr_display_aspect_xy_from_pose(plane_pose: PlanePose) -> tuple[float, float]:
        return ViewerService._get_mpr_spacing_xy_from_pose(plane_pose)

    def _render_by_view_type(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> RenderedImageResult:
        return render_by_view_type(
            self,
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
        )

    def _emit_render_progress(
        self,
        progress_callback: ViewRenderProgressCallback | None,
        phase: str,
        *,
        progress_percent: int | float | None = None,
        loaded_count: int | None = None,
        total_count: int | None = None,
    ) -> None:
        if progress_callback is None:
            return

        payload: dict[str, object] = {"phase": phase}
        if progress_percent is not None:
            payload["progressPercent"] = max(0, min(100, int(round(float(progress_percent)))))
        if loaded_count is not None:
            payload["loadedCount"] = max(0, int(loaded_count))
        if total_count is not None:
            payload["totalCount"] = max(0, int(total_count))

        try:
            progress_callback(payload)
        except Exception:
            logger.debug("render progress callback failed", exc_info=True)

    def _handle_scroll(self, view: ViewRecord, series: SeriesRecord, scroll: int) -> None:
        if not self._is_mpr_view_type(view.view_type):
            next_index = view.current_index + scroll
            view.current_index = max(0, min(next_index, len(series.instances) - 1))
            return

        volume = self._get_series_volume(series)
        target_viewport = self._resolve_mpr_viewport(view)
        if view.view_group is not None:
            group = view.view_group
            pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
            plane_pose = pose_context.poses[target_viewport]
            delta_world = (
                np.asarray(plane_pose.normal_world, dtype=np.float64)
                * spacing_along_world_direction(pose_context.geometry, plane_pose.normal_world)
                * float(scroll)
            )
            next_cursor = translate_cursor(pose_context.cursor, delta_world, pose_context.geometry)
            self._sync_group_from_mpr_cursor(group, next_cursor, pose_context.geometry, volume.shape)
        else:
            depth, height, width = volume.shape
            if target_viewport == MPR_VIEWPORT_CORONAL:
                view.mpr_coronal_index = max(0, min(view.mpr_coronal_index + scroll, height - 1))
            elif target_viewport == MPR_VIEWPORT_SAGITTAL:
                view.mpr_sagittal_index = max(0, min(view.mpr_sagittal_index + scroll, width - 1))
            else:
                view.mpr_axial_index = max(0, min(view.mpr_axial_index + scroll, depth - 1))
        view.is_initialized = True

    def _initialize_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        view.current_index = self._resolve_representative_stack_index(series)
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
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        if view.view_group is not None:
            if view.view_group.mpr_cursor is None:
                self._reset_mpr_group_geometry(view.view_group, volume.shape, series=series)
        else:
            depth, height, width = volume.shape
            view.mpr_axial_index = depth // 2
            view.mpr_coronal_index = height // 2
            view.mpr_sagittal_index = width // 2
        self._reset_mpr_view_display_state(view)
        self._reset_mpr_view_window(view, series, volume)
        self._fit_mpr_view_to_plane(view, series, volume)
        logger.info(
            "mpr viewport initialized view_id=%s volume=%s axial=%s coronal=%s sagittal=%s zoom=%.4f",
            view.view_id,
            volume.shape,
            view.mpr_axial_index,
            view.mpr_coronal_index,
            view.mpr_sagittal_index,
            view.zoom,
        )

    def _sync_mpr_state_from_source_view(
        self,
        target_view: ViewRecord,
        source_view_id: str,
        workspace_id: str | None = None,
    ) -> bool:
        if not self._is_mpr_view_type(target_view.view_type) or target_view.view_group is None:
            return False

        source_view = (
            view_registry.get(source_view_id)
            if workspace_id is None
            else view_registry.get(source_view_id, workspace_id=workspace_id)
        )
        if not self._is_mpr_view_type(source_view.view_type) or source_view.view_group is None:
            return False
        if source_view.view_group.group_id == target_view.view_group.group_id:
            return False

        source_series = (
            series_registry.get(source_view.series_id)
            if workspace_id is None
            else series_registry.get(source_view.series_id, workspace_id=workspace_id)
        )
        target_series = (
            series_registry.get(target_view.series_id)
            if workspace_id is None
            else series_registry.get(target_view.series_id, workspace_id=workspace_id)
        )
        logger.info(
            "mpr state sync source_view_id=%s source_series_id=%s target_view_id=%s target_series_id=%s",
            source_view.view_id,
            source_view.series_id,
            target_view.view_id,
            target_view.series_id,
        )
        source_volume = self._get_series_volume(source_series)
        target_volume = self._get_series_volume(target_series)
        source_context = self._build_mpr_pose_context(source_view, source_volume.shape, series=source_series)
        target_geometry = self._get_series_volume_geometry(target_series, target_volume.shape)
        source_group = source_view.view_group
        target_group = target_view.view_group

        target_group.active_viewport = source_group.active_viewport
        target_group.crosshair_drag_active = False
        target_group.crosshair_drag_origin_center = None
        target_group.crosshair_drag_origin_image = None
        target_group.rotation_drag = None
        target_group.mpr_crosshair_angles = deepcopy(source_group.mpr_crosshair_angles)
        target_group.mpr_crosshair_mode = self._normalize_mpr_crosshair_mode(source_group.mpr_crosshair_mode)
        target_group.mpr_independent_plane_normals = deepcopy(source_group.mpr_independent_plane_normals)
        target_group.mpr_mip = deepcopy(source_group.mpr_mip)
        target_group.mpr_use_display_basis_for_cursor_offsets = bool(source_group.mpr_use_display_basis_for_cursor_offsets)
        target_group.mpr_model_rotation_world = deepcopy(source_group.mpr_model_rotation_world)
        target_group.mpr_model_rotation_pivot_world = deepcopy(source_group.mpr_model_rotation_pivot_world)
        self._sync_group_from_mpr_cursor(target_group, source_context.cursor, target_geometry, target_volume.shape)
        if target_view.width and target_view.height:
            target_view.is_initialized = True
        return True

    def _initialize_3d_viewport(self, view: ViewRecord) -> None:
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        view.current_index = self._resolve_representative_stack_index(series)

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
        view.rotation_quaternion = _get_vtk_volume_renderer().get_default_rotation_quaternion()
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        view.volume_preset = "bone"
        view.volume_render_config = create_default_volume_render_config("bone")
        view.render_3d_mode = "volume"
        view.surface_render_config = create_default_surface_render_config("bone")
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
        if self._is_mpr_view_type(view.view_type):
            self._reset_mpr_view_group(view)
        elif self._is_3d_view_type(view.view_type):
            view.rotation_degrees = 0
            view.hor_flip = False
            view.ver_flip = False
            self._initialize_3d_viewport(view)
        else:
            view.rotation_degrees = 0
            view.hor_flip = False
            view.ver_flip = False
            self._initialize_viewport(view)

        view.is_initialized = True

    def _reset_mpr_view_group(self, view: ViewRecord) -> None:
        group_views = self._get_mpr_group_views(view)
        group = view.view_group
        if group is not None:
            series = series_registry.get(view.series_id)
            volume = self._get_series_volume(series)
            self._reset_mpr_group_geometry(group, volume.shape, series=series)
        else:
            series = None
            volume = None

        for group_view in group_views:
            if group is not None and series is not None and volume is not None:
                self._reset_mpr_view_display_state(group_view)
                self._reset_mpr_view_window(group_view, series, volume)
                self._fit_mpr_view_to_plane(group_view, series, volume)
            else:
                self._initialize_mpr_viewport(group_view)
            group_view.is_initialized = True

    def _reset_mpr_group_geometry(
        self,
        group: ViewGroupRecord,
        volume_shape: tuple[int, int, int],
        *,
        series: SeriesRecord | None = None,
    ) -> None:
        group.active_viewport = MPR_VIEWPORT_AXIAL
        group.crosshair_drag_active = False
        group.crosshair_drag_origin_center = None
        group.crosshair_drag_origin_image = None
        group.rotation_drag = None
        group.mpr_crosshair_angles.clear()
        group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_ORTHOGONAL
        group.mpr_independent_plane_normals.clear()
        group.mpr_mip = self._create_default_mpr_mip_state()
        group.mpr_use_display_basis_for_cursor_offsets = False
        self._set_mpr_model_rotation_matrix(group, np.eye(3, dtype=np.float64))
        group.mpr_model_rotation_pivot_world = None
        default_frame = self._build_default_mpr_frame_state(volume_shape)
        geometry = self._get_series_volume_geometry(series, volume_shape) if series is not None else build_identity_geometry(volume_shape)
        default_cursor = legacy_frame_to_cursor(default_frame, geometry, reference_center=default_frame.center)
        self._sync_group_from_mpr_cursor(group, default_cursor, geometry, volume_shape)
        self._reset_mpr_rotation_state(group)

    def _reset_mpr_view_display_state(self, view: ViewRecord) -> None:
        view.current_index = view.mpr_axial_index
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.zoom = 1.0
        view.rotation_degrees = 0
        view.hor_flip = False
        view.ver_flip = False
        view.pseudocolor_preset = DEFAULT_PSEUDOCOLOR_PRESET
        self._reset_drag_state(view)

    def _reset_mpr_view_window(self, view: ViewRecord, series: SeriesRecord, volume: np.ndarray) -> None:
        first_instance = next((instance for instance in series.instances if instance.sop_instance_uid), None)
        if first_instance is not None and first_instance.sop_instance_uid:
            cached = dicom_cache.get(first_instance.sop_instance_uid, first_instance.path)
            view.window_width = cached.window_width or self._derive_default_window_width(cached)
            view.window_center = cached.window_center or self._derive_default_window_center(cached)
            return
        pixel_min = float(np.min(volume))
        pixel_max = float(np.max(volume))
        view.window_width = max(WINDOW_WIDTH_MIN, pixel_max - pixel_min)
        view.window_center = (pixel_max + pixel_min) / 2.0

    def _fit_mpr_view_to_plane(self, view: ViewRecord, series: SeriesRecord, volume: np.ndarray) -> None:
        plane_pixels, _, _ = self._extract_mpr_plane(view, volume)
        target_viewport = self._resolve_mpr_viewport(view)
        pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(
            pose_context.poses[target_viewport]
        )
        view.zoom = viewport_transformer.calculate_contain_zoom(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=view.width or plane_pixels.shape[1],
            canvas_height=view.height or plane_pixels.shape[0],
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )

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
            volume_preset=str(view.volume_preset or "bone"),
            volume_config=view.volume_render_config,
            fast_preview=fast_preview,
        )

    def _build_surface_render_request(
        self,
        view: ViewRecord,
        *,
        volume: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        fast_preview: bool,
    ) -> SurfaceRenderRequest:
        """Build the shared VTK request payload used by 3D surface render and drag paths."""

        return SurfaceRenderRequest(
            view_id=view.view_id,
            volume=volume,
            spacing_xyz=spacing_xyz,
            canvas_width=view.width or 0,
            canvas_height=view.height or 0,
            zoom=float(view.zoom),
            offset_x=float(view.offset_x),
            offset_y=float(view.offset_y),
            rotation_quaternion=tuple(float(value) for value in view.rotation_quaternion),
            surface_config=view.surface_render_config,
            fast_preview=fast_preview,
        )

    def _render_3d_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> RenderedImageResult:
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        self._emit_render_progress(progress_callback, "volume", progress_percent=6)
        volume = self._get_series_volume(series, progress_callback=progress_callback)
        if not view.is_initialized:
            self._emit_render_progress(progress_callback, "initialize", progress_percent=72)
            self._initialize_3d_viewport(view)
            view.is_initialized = True

        spacing_xyz = self._get_3d_spacing_xyz(series)
        self._emit_render_progress(progress_callback, "render", progress_percent=82)
        render_3d_mode = self._normalize_render_3d_mode(view.render_3d_mode)
        if render_3d_mode == "surface":
            surface_request = self._build_surface_render_request(
                view,
                volume=volume,
                spacing_xyz=spacing_xyz,
                fast_preview=fast_preview,
            )
            image = _get_vtk_surface_renderer().render(surface_request)
            if not fast_preview:
                self._warm_surface_preview_session(surface_request)
            viewport_label = "3D SR"
        else:
            image = _get_vtk_volume_renderer().render(
                self._build_volume_render_request(
                    view,
                    volume=volume,
                    spacing_xyz=spacing_xyz,
                    fast_preview=fast_preview,
                )
            )
            viewport_label = "3D VR"

        corner_info = self._build_slice_corner_info_overlay(
            view,
            series,
            None,
            current_index=view.current_index,
            total_slices=max(1, volume.shape[0]),
            viewport_label=viewport_label,
        )

        self._emit_render_progress(progress_callback, "encode", progress_percent=96)
        image_bytes = self._encode_image(image, image_format, fast_preview=fast_preview)

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
                volumePreset=str(view.volume_preset or "bone"),
                volumeConfig=view.volume_render_config,
                render3dMode=render_3d_mode,
                surfaceConfig=view.surface_render_config,
            ),
            image_bytes=image_bytes,
        )

    def _warm_surface_preview_session(self, request: SurfaceRenderRequest) -> None:
        try:
            _get_vtk_surface_renderer().warm_preview_session(request)
        except Exception:
            logger.debug("failed to schedule surface preview warmup view_id=%s", request.view_id, exc_info=True)

    def _render_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
        metadata_mode: str = "full",
    ) -> RenderedImageResult:
        render_started_at = perf_counter()
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        instance = series.instances[view.current_index]
        if not instance.sop_instance_uid:
            raise HTTPException(status_code=400, detail="DICOM instance does not contain SOPInstanceUID")

        cache_started_at = perf_counter()
        cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
        cache_ms = (perf_counter() - cache_started_at) * 1000.0
        metadata_started_at = perf_counter()
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
        include_stack_overlay_payloads = not (fast_preview and metadata_mode == "stack-preview-lite")
        visible_measurements = self._build_visible_measurements(view) if include_stack_overlay_payloads else ()
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=cached.source_pixels,
            pixel_min=cached.pixel_min,
            pixel_max=cached.pixel_max,
            instance=instance,
            cached=cached,
            image_transform=image_transform,
            measurements=visible_measurements,
            corner_info=None,
            orientation=None,
        )
        visible_presentation_measurements = (
            self._build_visible_presentation_measurements(series, instance)
            if include_stack_overlay_payloads
            else ()
        )
        visible_presentation_annotations = (
            self._build_visible_presentation_annotations(series, instance)
            if include_stack_overlay_payloads
            else ()
        )
        metadata_ms = (perf_counter() - metadata_started_at) * 1000.0

        image_started_at = perf_counter()
        if fast_preview:
            image = self._render_fast_preview(context)
        else:
            image = layered_renderer.render(context)
        image_ms = (perf_counter() - image_started_at) * 1000.0

        encode_started_at = perf_counter()
        image_bytes = self._encode_image(image, image_format, fast_preview=fast_preview)
        encode_ms = (perf_counter() - encode_started_at) * 1000.0

        logger.debug(
            "stack render timing view_id=%s index=%s fast_preview=%s image_format=%s viewport=%sx%s render=%sx%s ratio=%.4f zoom=%.4f ww=%s wl=%s cache_ms=%.1f metadata_ms=%.1f image_ms=%.1f encode_ms=%.1f total_ms=%.1f",
            view.view_id,
            view.current_index,
            fast_preview,
            image_format,
            view.width,
            view.height,
            render_plan.render_view.width,
            render_plan.render_view.height,
            render_plan.render_ratio,
            view.zoom,
            view.window_width,
            view.window_center,
            cache_ms,
            metadata_ms,
            image_ms,
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
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
                measurements=[] if not include_stack_overlay_payloads else self._serialize_measurements(
                    (*visible_measurements, *visible_presentation_measurements),
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                annotations=[] if not include_stack_overlay_payloads else self._serialize_annotations(
                    visible_presentation_annotations,
                    image_transform=image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=self._serialize_orientation_overlay(
                    self._build_stack_orientation_overlay(render_plan.render_view, cached.dataset)
                ),
            ),
            image_bytes=image_bytes,
        )

    def _render_mpr_view(
        self,
        view: ViewRecord,
        image_format: ImageFormat = "png",
        *,
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> RenderedImageResult:
        render_started_at = perf_counter()
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        self._emit_render_progress(progress_callback, "volume", progress_percent=6)
        volume_started_at = perf_counter()
        volume = self._get_series_volume(series, progress_callback=progress_callback)
        volume_ms = (perf_counter() - volume_started_at) * 1000.0
        if not view.is_initialized:
            self._emit_render_progress(progress_callback, "initialize", progress_percent=72)
            self._initialize_mpr_viewport(view)
            view.is_initialized = True

        target_viewport = self._resolve_mpr_viewport(view)
        self._emit_render_progress(progress_callback, "render", progress_percent=82)
        preview_plane_shape = (
            self._get_mpr_fast_preview_plane_shape(
                volume.shape,
                target_viewport,
                viewport_size=(view.height or 0, view.width or 0),
            )
            if fast_preview and not fast_preview_full_resolution
            else None
        )
        reslice_started_at = perf_counter()
        plane_pixels, current, total = self._extract_mpr_plane(
            view,
            volume,
            target_viewport,
            output_shape=preview_plane_shape,
            interpolation_order=0 if fast_preview and not fast_preview_full_resolution else 1,
        )
        reslice_ms = (perf_counter() - reslice_started_at) * 1000.0
        metadata_started_at = perf_counter()
        payload_pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
        target_plane_pose = payload_pose_context.poses[target_viewport]
        plane_state = self._plane_state_from_pose(target_plane_pose) if view.view_group is not None else None
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(target_plane_pose)
        full_plane_height, full_plane_width = target_plane_pose.output_shape
        source_plane_height, source_plane_width = plane_pixels.shape[:2]
        render_pixel_aspect_x = pixel_aspect_x * float(full_plane_width) / float(max(1, source_plane_width))
        render_pixel_aspect_y = pixel_aspect_y * float(full_plane_height) / float(max(1, source_plane_height))
        render_plan = self._build_render_plan_for_shape(
            view,
            *plane_pixels.shape[:2],
            pixel_aspect_x=render_pixel_aspect_x,
            pixel_aspect_y=render_pixel_aspect_y,
        )
        render_image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=plane_pixels.shape[1],
            image_height=plane_pixels.shape[0],
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=render_pixel_aspect_x,
            pixel_aspect_y=render_pixel_aspect_y,
        )
        metadata_image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=full_plane_width,
            image_height=full_plane_height,
            canvas_width=render_plan.render_view.width or 0,
            canvas_height=render_plan.render_view.height or 0,
            view=render_plan.render_view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        scale_bar = self._build_scale_bar_info(
            render_plan.render_view,
            metadata_image_transform,
            self._get_mpr_spacing_xy_from_pose(target_plane_pose),
        )
        plane_min = float(np.min(plane_pixels))
        plane_max = float(np.max(plane_pixels))
        mpr_crosshair_overlay = self._build_mpr_crosshair_overlay(
            render_plan.render_view,
            volume.shape,
            target_plane_pose.output_shape,
            metadata_image_transform,
        )
        include_static_preview_metadata = not (fast_preview and metadata_mode == "mpr-pan-zoom-preview")
        reference_instance, reference_cached = (
            (None, None)
            if fast_preview
            else self._get_reference_instance_and_cache(series)
        )
        slice_corner_info = (
            None
            if not include_static_preview_metadata
            else self._build_slice_corner_info_overlay(
                view,
                series,
                reference_cached.dataset if reference_cached is not None else None,
                current_index=current,
                total_slices=total,
                viewport_label=self._build_mpr_viewport_label(target_viewport, plane_state),
                plane_state=plane_state,
                plane_pose=target_plane_pose,
                cursor=payload_pose_context.cursor,
            )
        )
        visible_measurements = [] if fast_preview else self._build_visible_measurements(view)
        context = RenderContext(
            view=render_plan.render_view,
            source_pixels=plane_pixels,
            pixel_min=plane_min,
            pixel_max=plane_max,
            image_transform=render_image_transform,
            instance=reference_instance,
            cached=reference_cached,
            mpr_viewport=target_viewport,
            measurements=visible_measurements,
            mpr_crosshair=None,
            corner_info=None,
            orientation=None,
        )
        metadata_ms = (perf_counter() - metadata_started_at) * 1000.0
        image_started_at = perf_counter()
        if fast_preview:
            image = self._render_fast_mpr_preview(
                context,
                order=1 if fast_preview_full_resolution else 0,
            )
        else:
            image = layered_renderer.render(context)
        image_ms = (perf_counter() - image_started_at) * 1000.0

        self._emit_render_progress(progress_callback, "encode", progress_percent=96)
        encode_started_at = perf_counter()
        image_bytes = self._encode_image(image, image_format, fast_preview=fast_preview)
        encode_ms = (perf_counter() - encode_started_at) * 1000.0
        logger.debug(
            "mpr render timing view_id=%s viewport=%s fast_preview=%s source_shape=%s full_shape=%s volume_ms=%.1f reslice_ms=%.1f metadata_ms=%.1f image_ms=%.1f encode_ms=%.1f total_ms=%.1f",
            view.view_id,
            target_viewport,
            fast_preview,
            plane_pixels.shape,
            target_plane_pose.output_shape,
            volume_ms,
            reslice_ms,
            metadata_ms,
            image_ms,
            encode_ms,
            (perf_counter() - render_started_at) * 1000.0,
        )

        return RenderedImageResult(
            meta=ViewImageResponse(
                slice_info=SliceInfo(current=current, total=total),
                window_info=WindowInfo(ww=view.window_width, wl=view.window_center),
                imageFormat=image_format,
                viewId=view.view_id,
                color=ViewColorInfo(pseudocolorPreset=view.pseudocolor_preset),
                mprFrame=self._build_mpr_frame_payload(payload_pose_context.cursor, payload_pose_context.geometry),
                mprCursor=self._build_mpr_cursor_payload(payload_pose_context.cursor),
                mprRevision=self._get_mpr_revision(view.view_group),
                mprPlane=self._build_mpr_plane_payload(
                    view,
                    target_viewport,
                    plane_pose=target_plane_pose,
                ),
                mprMipConfig=self._serialize_mpr_mip_config(view.mpr_mip),
                mprCrosshairMode=self._get_mpr_crosshair_mode(view.view_group),
                mpr_crosshair=self._build_mpr_crosshair_info(mpr_crosshair_overlay),
                scaleBar=scale_bar,
                cornerInfo=self._serialize_corner_info_overlay(slice_corner_info) if slice_corner_info is not None else None,
                measurements=[] if fast_preview else self._serialize_measurements(
                    visible_measurements,
                    image_transform=metadata_image_transform,
                    canvas_width=render_plan.render_view.width or 0,
                    canvas_height=render_plan.render_view.height or 0,
                ),
                transform=self._build_view_transform_payload(view),
                orientation=None if not include_static_preview_metadata else self._serialize_orientation_overlay(
                    self._build_mpr_orientation_overlay(
                        render_plan.render_view,
                        target_viewport,
                        plane_state,
                        plane_pose=target_plane_pose,
                    )
                ),
            ),
            image_bytes=image_bytes,
        )

    def _render_fast_mpr_preview(self, context: RenderContext, *, order: int = 0) -> Image.Image:
        return self._render_cached_fast_base_image(context, order=order)

    def _render_fast_preview(self, context: RenderContext) -> Image.Image:
        image = self._render_cached_fast_base_image(context)
        if not layered_renderer._has_overlay_content(context):
            return image
        return layered_renderer.composite_overlays(image.convert("RGBA"), context)

    def _render_cached_fast_base_image(self, context: RenderContext, *, order: int = 1) -> Image.Image:
        base_pixels = self._get_cached_fast_base_pixels(context)
        transformed = viewport_transformer.apply_affine_array(
            base_pixels,
            context.view.width or 0,
            context.view.height or 0,
            context.image_transform,
            order=order,
            cval=0.0,
        )
        if context.view.pseudocolor_preset != DEFAULT_PSEUDOCOLOR_PRESET:
            transformed = apply_pseudocolor(transformed, context.view.pseudocolor_preset)
            return Image.fromarray(transformed)
        return Image.fromarray(transformed)

    def _get_cached_fast_base_pixels(self, context: RenderContext) -> np.ndarray:
        cache_key = self._build_fast_base_pixels_cache_key(context)
        cached = self._fast_base_pixels_cache.get(cache_key)
        if cached is not None:
            self._fast_base_pixels_cache.move_to_end(cache_key)
            return cached

        base_pixels = self._window_array(
            context.source_pixels,
            context.view.window_width,
            context.view.window_center,
            pixel_min=context.pixel_min,
            pixel_max=context.pixel_max,
        )
        self._fast_base_pixels_cache[cache_key] = base_pixels
        self._fast_base_pixels_cache.move_to_end(cache_key)
        while len(self._fast_base_pixels_cache) > FAST_BASE_PIXELS_CACHE_MAX_ITEMS:
            self._fast_base_pixels_cache.popitem(last=False)
        return base_pixels

    @staticmethod
    def _build_fast_base_pixels_cache_key(context: RenderContext) -> tuple[object, ...]:
        return (
            id(context.source_pixels),
            tuple(context.source_pixels.shape),
            str(context.source_pixels.dtype),
            float(context.pixel_min),
            float(context.pixel_max),
            None if context.view.window_width is None else float(context.view.window_width),
            None if context.view.window_center is None else float(context.view.window_center),
        )

    @staticmethod
    def _render_fast_base_image(
        source_pixels: np.ndarray,
        pixel_min: float,
        pixel_max: float,
        render_view: ViewRecord,
        image_transform,
        *,
        order: int = 1,
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
            order=order,
            cval=0.0,
        )
        if render_view.pseudocolor_preset != DEFAULT_PSEUDOCOLOR_PRESET:
            transformed = apply_pseudocolor(transformed, render_view.pseudocolor_preset)
            return Image.fromarray(transformed)
        return Image.fromarray(transformed)

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
    def _get_mpr_fast_preview_plane_shape(
        volume_shape: tuple[int, int, int],
        viewport_key: str,
        viewport_size: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        full_height, full_width = ViewerService._get_mpr_plane_shape(volume_shape, viewport_key)
        viewport_height = int(viewport_size[0]) if viewport_size is not None else 0
        viewport_width = int(viewport_size[1]) if viewport_size is not None else 0

        def preview_dimension(value: int, viewport_value: int) -> int:
            if value <= MPR_FAST_PREVIEW_MIN_SIDE:
                return max(1, int(value))
            volume_scaled = max(MPR_FAST_PREVIEW_MIN_SIDE, int(round(float(value) * MPR_FAST_PREVIEW_SCALE)))
            if viewport_value > 0:
                viewport_scaled = max(
                    MPR_FAST_PREVIEW_MIN_SIDE,
                    int(round(float(viewport_value) * MPR_FAST_PREVIEW_SCALE)),
                )
                volume_scaled = min(volume_scaled, viewport_scaled)
            return min(
                int(value),
                volume_scaled,
            )

        return preview_dimension(full_height, viewport_height), preview_dimension(full_width, viewport_width)

    @staticmethod
    def _create_default_mpr_mip_state() -> MprMipState:
        return MprMipState()

    @staticmethod
    def _normalize_mpr_crosshair_mode(value: object) -> str:
        mode = str(value or "").strip().lower()
        return mode if mode in MPR_CROSSHAIR_MODES else MPR_CROSSHAIR_MODE_ORTHOGONAL

    @staticmethod
    def _get_mpr_crosshair_mode(group: ViewGroupRecord | None) -> str:
        return ViewerService._normalize_mpr_crosshair_mode(
            group.mpr_crosshair_mode if group is not None else MPR_CROSSHAIR_MODE_ORTHOGONAL
        )

    @staticmethod
    def _get_mpr_revision(group: ViewGroupRecord | None) -> int | None:
        return int(group.mpr_revision) if group is not None else None

    @staticmethod
    def _bump_mpr_revision(group: ViewGroupRecord | None) -> int | None:
        if group is None:
            return None
        group.mpr_revision = max(0, int(group.mpr_revision)) + 1
        return group.mpr_revision

    @staticmethod
    def _normalize_plane_normal_record(value: object) -> tuple[float, float, float] | None:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if vector.shape != (3,):
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-6:
            return None
        return tuple(float(component) for component in vector / norm)

    def _get_independent_plane_normal_overrides(
        self,
        group: ViewGroupRecord | None,
    ) -> dict[str, tuple[float, float, float]]:
        if self._get_mpr_crosshair_mode(group) != MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE or group is None:
            return {}
        return {
            viewport_key: normal
            for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
            if (normal := self._normalize_plane_normal_record(group.mpr_independent_plane_normals.get(viewport_key))) is not None
        }

    def _derive_mpr_plane_pose(
        self,
        cursor: MprCursorState,
        viewport_key: str,
        geometry: VolumeGeometry,
        shape_policy: OutputShapePolicy,
        normal_overrides: dict[str, tuple[float, float, float]] | None = None,
        use_display_basis_for_cursor_offsets: bool = False,
    ) -> PlanePose:
        return derive_plane_pose(
            cursor,
            viewport_key,
            geometry,
            shape_policy,
            normal_world_override=(normal_overrides or {}).get(viewport_key),
            use_display_basis_for_cursor_offsets=use_display_basis_for_cursor_offsets,
        )

    def _build_mpr_plane_poses(
        self,
        cursor: MprCursorState,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
        *,
        normal_overrides: dict[str, tuple[float, float, float]] | None = None,
        use_display_basis_for_cursor_offsets: bool = False,
    ) -> dict[str, PlanePose]:
        shape_policy = OutputShapePolicy(
            viewport_shapes={
                viewport_key: self._get_mpr_plane_shape(volume_shape, viewport_key)
                for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
            }
        )
        return {
            viewport_key: self._derive_mpr_plane_pose(
                cursor,
                viewport_key,
                geometry,
                shape_policy,
                normal_overrides,
                use_display_basis_for_cursor_offsets=use_display_basis_for_cursor_offsets,
            )
            for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
        }

    @staticmethod
    def _normal_records_from_poses(poses: dict[str, PlanePose]) -> dict[str, tuple[float, float, float]]:
        return {
            viewport_key: tuple(float(value) for value in mpr_geometry.normalize_oblique_vector(
                pose.normal_world,
                fallback=(1.0, 0.0, 0.0),
            ))
            for viewport_key, pose in poses.items()
        }

    @staticmethod
    def _serialize_mpr_mip_config(state: MprMipState) -> MprMipConfig:
        return MprMipConfig(
            enabled=bool(state.enabled),
            algorithm=str(state.algorithm or "maximum"),
            viewports={
                viewport_key: MprMipViewportConfig(thickness=max(0, min(100, int(viewport_state.thickness))))
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
            next_viewports[viewport_key] = MprMipViewportState(thickness=max(0, min(100, int(next_config.thickness))))

        next_state = MprMipState(
            enabled=bool(incoming.enabled),
            algorithm=str(incoming.algorithm or "maximum"),
            viewports=next_viewports,
        )
        if view.view_group is not None:
            view.view_group.mpr_mip = next_state
        return True

    def _handle_mpr_crosshair_mode(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.mpr_crosshair_mode is None:
            return False
        next_mode = self._normalize_mpr_crosshair_mode(payload.mpr_crosshair_mode)
        group = view.view_group
        current_mode = self._get_mpr_crosshair_mode(group)
        if next_mode == current_mode:
            return False

        series = series_registry.get(view.series_id)
        volume_shape = self._get_series_volume(series).shape
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        group.active_viewport = self._resolve_mpr_viewport(view)
        group.rotation_drag = None

        if next_mode == MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE:
            group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE
            self._ensure_mpr_independent_plane_normals(group, pose_context.poses)
            group.mpr_crosshair_angles.clear()
            self._ensure_mpr_crosshair_angle_cache(group, pose_context.poses)
            view.is_initialized = True
            return True

        self._reorthogonalize_mpr_group_from_pose_context(group, pose_context, volume_shape)
        group.mpr_crosshair_mode = MPR_CROSSHAIR_MODE_ORTHOGONAL
        group.mpr_independent_plane_normals.clear()
        group.mpr_crosshair_angles.clear()
        group.rotation_drag = None
        view.is_initialized = True
        return True

    def _ensure_mpr_independent_plane_normals(
        self,
        group: ViewGroupRecord,
        poses: dict[str, PlanePose],
    ) -> None:
        next_normals = self._normal_records_from_poses(poses)
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL):
            existing_normal = self._normalize_plane_normal_record(group.mpr_independent_plane_normals.get(viewport_key))
            if existing_normal is not None:
                next_normals[viewport_key] = existing_normal
        group.mpr_independent_plane_normals = next_normals

    def _reorthogonalize_mpr_group_from_pose_context(
        self,
        group: ViewGroupRecord,
        pose_context: MprPoseContext,
        volume_shape: tuple[int, int, int],
    ) -> None:
        active_viewport = (
            group.active_viewport
            if group.active_viewport in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL)
            else MPR_VIEWPORT_AXIAL
        )
        active_plane = pose_context.poses[active_viewport]
        active_normal = np.asarray(active_plane.normal_world, dtype=np.float64)
        horizontal_angle, _ = self._get_mpr_visible_crosshair_line_angles(
            group,
            pose_context.poses,
            active_viewport,
        )
        horizontal_line_world = mpr_geometry.direction_from_screen_angle(
            np.asarray(active_plane.row_world, dtype=np.float64),
            np.asarray(active_plane.col_world, dtype=np.float64),
            horizontal_angle,
        )
        vertical_line_world = mpr_geometry.direction_from_screen_angle(
            np.asarray(active_plane.row_world, dtype=np.float64),
            np.asarray(active_plane.col_world, dtype=np.float64),
            horizontal_angle + float(np.pi / 2.0),
        )

        normal_updates: dict[str, np.ndarray] = {
            active_viewport: active_normal,
        }
        for line, line_world in (("horizontal", horizontal_line_world), ("vertical", vertical_line_world)):
            target_viewport = self._resolve_mpr_oblique_target_viewport(active_viewport, line)
            target_plane = pose_context.poses[target_viewport]
            next_normal = mpr_geometry.normalize_oblique_vector(
                np.cross(line_world, active_normal),
                fallback=tuple(target_plane.normal_world),
            )
            if float(np.dot(next_normal, np.asarray(target_plane.normal_world, dtype=np.float64))) < 0.0:
                next_normal = -next_normal
            normal_updates[target_viewport] = next_normal

        next_cursor = self._replace_mpr_cursor_plane_normals(pose_context.cursor, normal_updates)
        self._sync_group_from_mpr_cursor(group, next_cursor, pose_context.geometry, volume_shape)

    def _extract_mpr_plane(
        self,
        view: ViewRecord,
        volume: np.ndarray,
        viewport_key: str | None = None,
        output_shape: tuple[int, int] | None = None,
        interpolation_order: int = 1,
    ) -> tuple[np.ndarray, int, int]:
        target_viewport = viewport_key or self._resolve_mpr_viewport(view)
        full_plane_shape = self._get_mpr_plane_shape(volume.shape, target_viewport)
        effective_output_shape = tuple(int(value) for value in output_shape) if output_shape is not None else full_plane_shape
        cache_key = self._get_mpr_plane_cache_key(
            view,
            target_viewport,
            effective_output_shape,
            interpolation_order,
        )
        cached_plane = self._mpr_plane_cache.get(cache_key)
        if cached_plane is not None:
            self._mpr_plane_cache.move_to_end(cache_key)
            plane_pixels, current, total = cached_plane
            if target_viewport == MPR_VIEWPORT_AXIAL:
                view.current_index = current
            return plane_pixels, current, total

        try:
            series = series_registry.get(view.series_id)
        except Exception:
            series = None
        geometry = self._get_series_volume_geometry(series, volume.shape) if series is not None else build_identity_geometry(volume.shape)
        cursor = self._get_mpr_cursor_state(view, geometry, volume.shape)
        plane_pose = self._derive_mpr_plane_pose(
            cursor,
            target_viewport,
            geometry,
            OutputShapePolicy(viewport_shapes={target_viewport: full_plane_shape}),
            self._get_independent_plane_normal_overrides(view.view_group),
            use_display_basis_for_cursor_offsets=self._should_use_mpr_display_basis_for_cursor_offsets(view.view_group),
        )
        if output_shape is not None and tuple(output_shape) != full_plane_shape:
            sample_height = max(1, int(output_shape[0]))
            sample_width = max(1, int(output_shape[1]))
            plane_pose = replace(
                plane_pose,
                output_shape=(sample_height, sample_width),
                pixel_spacing_row_mm=float(plane_pose.pixel_spacing_row_mm) * float(full_plane_shape[0]) / float(sample_height),
                pixel_spacing_col_mm=float(plane_pose.pixel_spacing_col_mm) * float(full_plane_shape[1]) / float(sample_width),
            )
        sampling_geometry = self._build_mpr_model_sampling_geometry(
            view,
            geometry,
            pivot_world=cursor.center_world,
        )
        mip_config = self._build_reslice_mip_config(view.mpr_mip, target_viewport)
        if output_shape is not None and mip_config.enabled:
            mip_config = replace(mip_config, max_samples=3)
        plane = reslice_plane(
            volume,
            sampling_geometry,
            plane_pose,
            mip_config,
            interpolation_order=interpolation_order,
        )
        current, total = self._get_mpr_viewport_index_info(view, volume.shape, target_viewport, cursor=cursor, geometry=geometry)
        if target_viewport == MPR_VIEWPORT_AXIAL:
            view.current_index = current
        plane_pixels = plane.astype(np.float32, copy=False)
        self._store_mpr_plane_cache(cache_key, plane_pixels, current, total)
        return plane_pixels, current, total

    def _get_mpr_plane_cache_key(
        self,
        view: ViewRecord,
        viewport_key: str,
        output_shape: tuple[int, int],
        interpolation_order: int,
    ) -> tuple[object, ...]:
        group = view.view_group
        mip_state = view.mpr_mip.viewports.get(viewport_key, MprMipViewportState())
        model_rotation = (
            tuple(tuple(float(value) for value in row) for row in group.mpr_model_rotation_world)
            if group is not None
            else None
        )
        independent_normals = (
            tuple(
                (key, tuple(float(value) for value in group.mpr_independent_plane_normals[key]))
                for key in sorted(group.mpr_independent_plane_normals)
            )
            if group is not None
            else None
        )
        return (
            view.workspace_id,
            view.series_id,
            group.group_id if group is not None else view.view_id,
            self._get_mpr_revision(group),
            self._should_use_mpr_display_basis_for_cursor_offsets(group),
            None if group is not None else int(view.mpr_axial_index),
            None if group is not None else int(view.mpr_coronal_index),
            None if group is not None else int(view.mpr_sagittal_index),
            viewport_key,
            int(output_shape[0]),
            int(output_shape[1]),
            int(interpolation_order),
            bool(view.mpr_mip.enabled),
            str(view.mpr_mip.algorithm or "maximum"),
            max(0, min(100, int(mip_state.thickness))),
            model_rotation,
            independent_normals,
        )

    def _store_mpr_plane_cache(
        self,
        cache_key: tuple[object, ...],
        plane_pixels: np.ndarray,
        current: int,
        total: int,
    ) -> None:
        self._mpr_plane_cache[cache_key] = (plane_pixels, int(current), int(total))
        self._mpr_plane_cache.move_to_end(cache_key)
        while len(self._mpr_plane_cache) > MPR_PLANE_CACHE_MAX_ITEMS:
            self._mpr_plane_cache.popitem(last=False)

    def _extract_oblique_mpr_plane(
        self,
        view: ViewRecord,
        volume: np.ndarray,
        viewport_key: str,
        plane_state: MprObliquePlaneState,
    ) -> tuple[np.ndarray, int, int]:
        del plane_state
        return self._extract_mpr_plane(view, volume, viewport_key)

    def _build_mpr_model_sampling_geometry(
        self,
        view: ViewRecord,
        geometry: VolumeGeometry,
        *,
        pivot_world: np.ndarray,
    ) -> VolumeGeometry:
        group = view.view_group
        if group is None:
            return geometry

        rotation_world = self._get_mpr_model_rotation_matrix(group)
        if np.allclose(rotation_world, np.eye(3, dtype=np.float64), atol=1e-8):
            return geometry

        if group.mpr_model_rotation_pivot_world is None:
            self._set_mpr_model_rotation_pivot_world(group, pivot_world)
        pivot = self._get_mpr_model_rotation_pivot_world(group, pivot_world)
        inverse_rotation = rotation_world.T
        inverse_model_transform = np.eye(4, dtype=np.float64)
        inverse_model_transform[:3, :3] = inverse_rotation
        inverse_model_transform[:3, 3] = pivot - inverse_rotation @ pivot
        world_to_ijk = np.asarray(geometry.world_to_ijk, dtype=np.float64) @ inverse_model_transform
        return VolumeGeometry(
            shape_ijk=geometry.shape_ijk,
            ijk_to_world=np.linalg.inv(world_to_ijk),
            world_to_ijk=world_to_ijk,
            spacing_hint_mm=geometry.spacing_hint_mm,
        )

    @staticmethod
    def _get_mpr_model_rotation_matrix(group: ViewGroupRecord) -> np.ndarray:
        matrix = np.asarray(group.mpr_model_rotation_world, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            return np.eye(3, dtype=np.float64)
        return orthonormalize_matrix(matrix)

    @staticmethod
    def _get_mpr_model_rotation_pivot_world(group: ViewGroupRecord, fallback_world: np.ndarray) -> np.ndarray:
        if group.mpr_model_rotation_pivot_world is not None:
            pivot = np.asarray(group.mpr_model_rotation_pivot_world, dtype=np.float64)
            if pivot.shape == (3,) and np.all(np.isfinite(pivot)):
                return pivot
        return np.asarray(fallback_world, dtype=np.float64)

    @staticmethod
    def _set_mpr_model_rotation_pivot_world(group: ViewGroupRecord, pivot_world: np.ndarray) -> None:
        pivot = np.asarray(pivot_world, dtype=np.float64)
        if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
            return
        group.mpr_model_rotation_pivot_world = tuple(float(value) for value in pivot)

    @staticmethod
    def _set_mpr_model_rotation_matrix(
        group: ViewGroupRecord,
        matrix: np.ndarray,
        *,
        pivot_world: np.ndarray | None = None,
    ) -> None:
        normalized = orthonormalize_matrix(np.asarray(matrix, dtype=np.float64))
        group.mpr_model_rotation_world = tuple(
            tuple(float(value) for value in normalized[row_index])
            for row_index in range(3)
        )
        if np.allclose(normalized, np.eye(3, dtype=np.float64), atol=1e-8):
            group.mpr_model_rotation_pivot_world = None
        elif pivot_world is not None and group.mpr_model_rotation_pivot_world is None:
            ViewerService._set_mpr_model_rotation_pivot_world(group, pivot_world)

    @staticmethod
    def _get_mpr_model_source_direction(group: ViewGroupRecord | None, direction_world: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction_world, dtype=np.float64)
        if group is None:
            return direction
        return ViewerService._get_mpr_model_rotation_matrix(group).T @ direction

    @staticmethod
    def _should_apply_mpr_model_rotation_to_plane_labels(
        group: ViewGroupRecord | None,
        plane_pose: PlanePose | None,
    ) -> bool:
        if group is None or plane_pose is None:
            return False
        rotation = ViewerService._get_mpr_model_rotation_matrix(group)
        if np.allclose(rotation, np.eye(3, dtype=np.float64), atol=1e-8):
            return False
        normal = mpr_geometry.normalize_oblique_vector(
            np.asarray(plane_pose.normal_world, dtype=np.float64),
            fallback=(1.0, 0.0, 0.0),
        )
        return not np.allclose(rotation @ normal, normal, atol=1e-6)

    @staticmethod
    def _normalize_oblique_vector(
        value: tuple[float, float, float] | np.ndarray,
        *,
        fallback: tuple[float, float, float],
    ) -> np.ndarray:
        return mpr_geometry.normalize_oblique_vector(value, fallback=fallback)

    def _build_default_mpr_frame_state(self, volume_shape: tuple[int, int, int]) -> MprFrameState:
        return mpr_geometry.default_mpr_frame_state(volume_shape)

    def _ensure_mpr_reference_center(
        self,
        group: ViewGroupRecord,
        volume_shape: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        if group.mpr_reference_center is None:
            group.mpr_reference_center = tuple(
                float(value)
                for value in self._build_default_mpr_frame_state(volume_shape).center
            )
        return group.mpr_reference_center

    @staticmethod
    def _reset_mpr_rotation_state(group: ViewGroupRecord) -> None:
        group.rotation_drag = None

    @staticmethod
    def _get_mpr_viewport_index_info(
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        viewport_key: str,
        *,
        cursor: MprCursorState | None = None,
        geometry: VolumeGeometry | None = None,
    ) -> tuple[int, int]:
        depth, height, width = volume_shape
        if view.view_group is not None and cursor is not None and geometry is not None:
            center = world_to_ijk_point(geometry, cursor.center_world)
            if viewport_key == MPR_VIEWPORT_CORONAL:
                return max(0, min(int(np.round(center[1])), height - 1)), height
            if viewport_key == MPR_VIEWPORT_SAGITTAL:
                return max(0, min(int(np.round(center[2])), width - 1)), width
            return max(0, min(int(np.round(center[0])), depth - 1)), depth
        if view.view_group is not None:
            if viewport_key == MPR_VIEWPORT_CORONAL:
                return max(0, min(view.view_group.coronal_index, height - 1)), height
            if viewport_key == MPR_VIEWPORT_SAGITTAL:
                return max(0, min(view.view_group.sagittal_index, width - 1)), width
            return max(0, min(view.view_group.axial_index, depth - 1)), depth
        if viewport_key == MPR_VIEWPORT_CORONAL:
            return max(0, min(view.mpr_coronal_index, height - 1)), height
        if viewport_key == MPR_VIEWPORT_SAGITTAL:
            return max(0, min(view.mpr_sagittal_index, width - 1)), width
        return max(0, min(view.mpr_axial_index, depth - 1)), depth

    @staticmethod
    def _clamp_3d_zoom(zoom: float) -> float:
        return min(max(float(zoom), ZOOM_MIN_3D), ZOOM_MAX_3D)

    @staticmethod
    def _normalize_render_3d_mode(value: object) -> str:
        return "surface" if str(value or "").strip().lower() == "surface" else "volume"

    def _resolve_representative_stack_index(self, series: SeriesRecord) -> int:
        instance_count = len(series.instances)
        if instance_count <= 1:
            return 0

        cached_entry = self._series_representative_slice_cache.get(series.series_id)
        if cached_entry is not None and cached_entry[0] == instance_count:
            return max(0, min(int(cached_entry[1]), instance_count - 1))

        sample_indexes = build_representative_sample_indexes(instance_count)
        midpoint = (instance_count - 1) / 2.0
        best_index = int(round(midpoint))
        best_score = -1.0
        readable_indexes: list[int] = []

        for index in sample_indexes:
            instance = series.instances[index]
            if not instance.sop_instance_uid:
                continue
            readable_indexes.append(index)
            try:
                cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
            except HTTPException:
                readable_indexes.pop()
                continue

            score = score_representative_pixels(cached.source_pixels)
            if score > best_score or (abs(score - best_score) <= 1e-6 and abs(index - midpoint) < abs(best_index - midpoint)):
                best_score = score
                best_index = index

        if best_score <= 1e-6 and readable_indexes:
            best_index = min(readable_indexes, key=lambda index: abs(index - midpoint))

        best_index = max(0, min(best_index, instance_count - 1))
        self._series_representative_slice_cache[series.series_id] = (instance_count, best_index)
        logger.info(
            "representative stack slice resolved series_id=%s index=%s total=%s score=%.4f",
            series.series_id,
            best_index,
            instance_count,
            max(best_score, 0.0),
        )
        return best_index

    def _get_3d_spacing_xyz(self, series: SeriesRecord) -> tuple[float, float, float]:
        transform = self._get_series_patient_transform(series)
        if transform is not None:
            return transform.spacing_xyz()

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

    def _get_series_volume(
        self,
        series: SeriesRecord,
        *,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> np.ndarray:
        volume_cache_key = self._build_series_volume_cache_key(series)
        cached_volume = self._get_cached_series_volume(volume_cache_key)
        if cached_volume is not None:
            self._emit_render_progress(
                progress_callback,
                "volume",
                progress_percent=70,
                loaded_count=len(series.instances),
                total_count=len(series.instances),
            )
            return cached_volume

        build_lock = self._get_series_volume_build_lock(volume_cache_key)
        if build_lock.locked():
            self._emit_render_progress(progress_callback, "waiting", progress_percent=8)

        with build_lock:
            cached_volume = self._get_cached_series_volume(volume_cache_key)
            if cached_volume is not None:
                self._emit_render_progress(
                    progress_callback,
                    "volume",
                    progress_percent=70,
                    loaded_count=len(series.instances),
                    total_count=len(series.instances),
                )
                return cached_volume

            started_at = perf_counter()
            volume = self._build_series_volume(series, progress_callback=progress_callback)
            stored_volume = self._store_series_volume(volume_cache_key, volume)
            self._emit_render_progress(
                progress_callback,
                "volume",
                progress_percent=70,
                loaded_count=len(series.instances),
                total_count=len(series.instances),
            )
            logger.info(
                "series volume built series_id=%s cache_key=%s shape=%s bytes=%s elapsed_ms=%.1f",
                series.series_id,
                volume_cache_key,
                stored_volume.shape,
                int(stored_volume.nbytes),
                (perf_counter() - started_at) * 1000.0,
            )
            return stored_volume

    @staticmethod
    def _build_series_volume_cache_key(series: SeriesRecord) -> str:
        cached_key = getattr(series, "volume_cache_key", None)
        if cached_key:
            return str(cached_key)

        content_keys = [
            dicom_cache.build_instance_content_key(instance.sop_instance_uid, instance.path)
            for instance in series.instances
            if instance.sop_instance_uid
        ]
        digest = hashlib.sha256("\n".join(content_keys).encode("utf-8")).hexdigest()
        volume_cache_key = f"volume::{digest}"
        try:
            series.volume_cache_key = volume_cache_key
        except Exception:
            pass
        return volume_cache_key

    def _build_series_volume(
        self,
        series: SeriesRecord,
        *,
        progress_callback: ViewRenderProgressCallback | None = None,
    ) -> np.ndarray:
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        readable_total = sum(1 for instance in series.instances if instance.sop_instance_uid)
        loaded_count = 0
        last_progress_percent = -1

        for instance in series.instances:
            if not instance.sop_instance_uid:
                continue
            cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
            dataset = cached.dataset
            orientation = self._get_dataset_orientation(dataset)
            position = self._get_dataset_position(dataset)
            slice_entries.append((cached.source_pixels, orientation, position))
            loaded_count += 1

            if readable_total:
                progress_percent = 10 + int((loaded_count / readable_total) * 55)
                if progress_percent != last_progress_percent:
                    self._emit_render_progress(
                        progress_callback,
                        "volume",
                        progress_percent=progress_percent,
                        loaded_count=loaded_count,
                        total_count=readable_total,
                    )
                    last_progress_percent = progress_percent

        if not slice_entries:
            raise HTTPException(status_code=400, detail="Series does not contain readable pixel data")

        first_shape = slice_entries[0][0].shape
        if any(item[0].shape != first_shape for item in slice_entries):
            raise HTTPException(status_code=400, detail="MPR requires a series with consistent slice dimensions")

        self._emit_render_progress(
            progress_callback,
            "normalize",
            progress_percent=66,
            loaded_count=loaded_count,
            total_count=readable_total,
        )
        return self._build_standardized_volume(slice_entries)

    def _get_series_volume_build_lock(self, series_id: str) -> Any:
        return self._series_volume_cache.get_build_lock(series_id)

    def _get_cached_series_volume(self, series_id: str) -> np.ndarray | None:
        return self._series_volume_cache.get(series_id)

    def _store_series_volume(self, series_id: str, volume: np.ndarray) -> np.ndarray:
        return self._series_volume_cache.store(series_id, volume)

    def _handle_series_volume_cache_evict(self, series_id: str, volume: np.ndarray) -> None:
        self._series_volume_geometry_cache.pop(series_id, None)
        self._series_patient_transform_cache.pop(series_id, None)
        self._series_representative_slice_cache.pop(series_id, None)
        logger.debug("volume cache evict series_id=%s bytes=%s", series_id, int(volume.nbytes))

    def get_volume_cache_stats(self) -> dict[str, int]:
        return self._series_volume_cache.stats()

    @staticmethod
    def _get_dataset_orientation(dataset) -> np.ndarray | None:
        return get_dataset_orientation(dataset)

    @staticmethod
    def _get_dataset_position(dataset) -> np.ndarray | None:
        return get_dataset_position(dataset)

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
        return normalize_vector(vector)

    def _build_standardized_volume(
        self,
        slice_entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> np.ndarray:
        return build_standardized_volume(slice_entries, logger=self._logger)

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

    def _handle_mpr_model_rotate_3d(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.action_type not in {DRAG_ACTION_START, DRAG_ACTION_MOVE, DRAG_ACTION_END}:
            return False
        if payload.x is None or payload.y is None or not view.width or not view.height:
            if payload.action_type == DRAG_ACTION_END:
                was_dragging = view.drag_origin_arcball_x is not None
                view.drag_origin_arcball_x = None
                view.drag_origin_arcball_y = None
                return was_dragging
            return False

        group = view.view_group
        series = series_registry.get(view.series_id)
        volume_shape = self._get_series_volume(series).shape
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        active_viewport = self._resolve_mpr_viewport(view)
        active_plane = pose_context.poses[active_viewport]
        plane_shape = active_plane.output_shape
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(active_plane)
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=int(plane_shape[1]),
            image_height=int(plane_shape[0]),
            canvas_width=int(view.width or 0),
            canvas_height=int(view.height or 0),
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        pointer_angle_rad = self._resolve_mpr_rotation_pointer_angle(
            view,
            active_plane,
            image_transform,
            float(payload.x),
            float(payload.y),
        )
        group.active_viewport = active_viewport

        if payload.action_type == DRAG_ACTION_START:
            view.drag_origin_arcball_x = pointer_angle_rad
            view.drag_origin_arcball_y = None
            if group.mpr_model_rotation_pivot_world is None:
                self._set_mpr_model_rotation_pivot_world(group, active_plane.cursor_center_world)
            return False

        previous_angle_rad = view.drag_origin_arcball_x
        if payload.action_type == DRAG_ACTION_END:
            view.drag_origin_arcball_x = None
            view.drag_origin_arcball_y = None
        elif pointer_angle_rad is not None:
            view.drag_origin_arcball_x = pointer_angle_rad

        if previous_angle_rad is None:
            if pointer_angle_rad is not None and payload.action_type != DRAG_ACTION_END:
                view.drag_origin_arcball_x = pointer_angle_rad
            return False
        if pointer_angle_rad is None:
            return payload.action_type == DRAG_ACTION_END

        delta_angle_rad = self._normalize_screen_full_turn_delta(
            float(pointer_angle_rad) - float(previous_angle_rad)
        )
        if abs(delta_angle_rad) < 1e-6:
            return payload.action_type == DRAG_ACTION_END

        self._apply_mpr_model_rotation_delta(
            view.view_group,
            active_plane,
            screen_angle_delta_rad=delta_angle_rad,
        )
        view.is_initialized = True
        return True

    def _apply_mpr_model_rotation_delta(
        self,
        group: ViewGroupRecord,
        active_plane: PlanePose,
        *,
        screen_angle_delta_rad: float,
    ) -> None:
        rotation_axis_world = mpr_geometry.normalize_oblique_vector(
            np.asarray(active_plane.normal_world, dtype=np.float64),
            fallback=(1.0, 0.0, 0.0),
        )
        delta_rotation = axis_angle_rotation_matrix(rotation_axis_world, float(screen_angle_delta_rad))
        self._set_mpr_model_rotation_matrix(
            group,
            delta_rotation @ self._get_mpr_model_rotation_matrix(group),
            pivot_world=active_plane.cursor_center_world,
        )

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
        if self._normalize_render_3d_mode(view.render_3d_mode) == "surface":
            view.rotation_quaternion = _get_vtk_surface_renderer().apply_trackball_camera_delta(
                self._build_surface_render_request(
                    view,
                    volume=volume,
                    spacing_xyz=spacing_xyz,
                    fast_preview=True,
                ),
                delta_x_pixels=delta_x_pixels,
                delta_y_pixels=delta_y_pixels,
            )
        else:
            view.rotation_quaternion = _get_vtk_volume_renderer().apply_trackball_camera_delta(
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
        view.volume_preset = str(view.volume_render_config.get("preset", view.volume_preset or "bone"))
        view.is_initialized = True

    def _handle_volume_preset(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return

        view.volume_preset = normalize_volume_preset_name(payload.sub_op_type or "bone")
        view.volume_render_config = create_default_volume_render_config(view.volume_preset)
        view.is_initialized = True

    def _handle_render_3d_mode(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        view.render_3d_mode = self._normalize_render_3d_mode(payload.render_3d_mode or payload.sub_op_type)
        if view.surface_render_config is None:
            view.surface_render_config = create_default_surface_render_config("bone")
        view.is_initialized = True

    def _handle_surface_config(self, view: ViewRecord, payload: ViewOperationRequest) -> None:
        if not self._is_3d_view_type(view.view_type):
            return
        view.surface_render_config = normalize_surface_render_config(payload.surface_config, "bone")
        view.render_3d_mode = "surface"
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
        ensure_view_size(view)

        series = series_registry.get(view.series_id)
        volume = self._get_series_volume(series)
        target_viewport = self._resolve_mpr_viewport(view)
        pose_context = self._build_mpr_pose_context(view, volume.shape, series=series)
        active_plane = pose_context.poses[target_viewport]
        plane_shape = active_plane.output_shape
        canvas_width = max(float(view.width or 0), 1.0)
        canvas_height = max(float(view.height or 0), 1.0)
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(active_plane)
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=int(plane_shape[1]),
            image_height=int(plane_shape[0]),
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )

        def payload_to_plane_image_point() -> tuple[float, float]:
            canvas_x = min(max(float(payload.x or 0.0), 0.0), 1.0) * canvas_width
            canvas_y = min(max(float(payload.y or 0.0), 0.0), 1.0) * canvas_height
            return self._canvas_to_image_coordinates(image_transform, canvas_x, canvas_y)

        if payload.action_type == DRAG_ACTION_START:
            view.mpr_crosshair_drag_active = True
            if view.view_group is not None:
                origin_center_ijk = world_to_ijk_point(pose_context.geometry, pose_context.cursor.center_world)
                view.view_group.crosshair_drag_origin_center = tuple(float(value) for value in origin_center_ijk)
                if payload.x is not None and payload.y is not None:
                    view.view_group.crosshair_drag_origin_image = payload_to_plane_image_point()
                else:
                    view.view_group.crosshair_drag_origin_image = None
            return False

        is_drag_end = payload.action_type == DRAG_ACTION_END
        was_dragging = view.mpr_crosshair_drag_active
        if (payload.action_type != DRAG_ACTION_MOVE and not is_drag_end) or not was_dragging:
            return False

        image_x, image_y = payload_to_plane_image_point()
        depth, height, width = volume.shape
        if view.view_group is not None:
            previous_center = tuple(float(value) for value in world_to_ijk_point(pose_context.geometry, pose_context.cursor.center_world))
            next_center_world = self._resolve_mpr_center_from_image_point(
                view.view_group,
                pose_context.poses[target_viewport],
                pose_context.geometry,
                image_x,
                image_y,
            )
            next_center = world_to_ijk_point(pose_context.geometry, next_center_world)
        else:
            previous_center = (float(view.mpr_axial_index), float(view.mpr_coronal_index), float(view.mpr_sagittal_index))
            next_center = np.array(previous_center, dtype=np.float64)
            if target_viewport == MPR_VIEWPORT_CORONAL:
                next_center[2] = float(max(0.0, min(image_x - 0.5, width - 1)))
                next_center[0] = float(max(0.0, min(depth - image_y - 0.5, depth - 1)))
            elif target_viewport == MPR_VIEWPORT_SAGITTAL:
                next_center[1] = float(max(0.0, min(image_x - 0.5, height - 1)))
                next_center[0] = float(max(0.0, min(depth - image_y - 0.5, depth - 1)))
            else:
                next_center[2] = float(max(0.0, min(image_x - 0.5, width - 1)))
                next_center[1] = float(max(0.0, min(image_y - 0.5, height - 1)))

        center_changed = not np.allclose(next_center, np.asarray(previous_center, dtype=np.float64), atol=1e-6)

        if center_changed:
            if view.view_group is not None:
                next_cursor = replace(pose_context.cursor, center_world=np.asarray(next_center_world, dtype=np.float64))
                self._sync_group_from_mpr_cursor(view.view_group, next_cursor, pose_context.geometry, volume.shape)
                view.view_group.mpr_use_display_basis_for_cursor_offsets = True
            else:
                view.mpr_axial_index = int(np.round(next_center[0]))
                view.mpr_coronal_index = int(np.round(next_center[1]))
                view.mpr_sagittal_index = int(np.round(next_center[2]))
            view.current_index = view.mpr_axial_index
            view.is_initialized = True

        if is_drag_end:
            view.mpr_crosshair_drag_active = False
            if view.view_group is not None:
                view.view_group.crosshair_drag_origin_center = None
                view.view_group.crosshair_drag_origin_image = None
            return was_dragging or center_changed

        return center_changed

    def _resolve_mpr_center_from_image_point(
        self,
        group: ViewGroupRecord,
        plane_pose: PlanePose,
        geometry: VolumeGeometry,
        image_x: float,
        image_y: float,
    ) -> np.ndarray:
        origin_center = np.asarray(
            group.crosshair_drag_origin_center or world_to_ijk_point(geometry, plane_pose.cursor_center_world),
            dtype=np.float64,
        )
        origin_center_world = ijk_to_world_point(geometry, origin_center)
        origin_image_x, origin_image_y = group.crosshair_drag_origin_image or self._project_world_point_to_plane_image(
            plane_pose,
            origin_center_world,
        )
        row_offset_mm = (float(image_y) - float(origin_image_y)) * float(plane_pose.pixel_spacing_row_mm)
        col_offset_mm = (float(image_x) - float(origin_image_x)) * float(plane_pose.pixel_spacing_col_mm)
        next_center_world = (
            origin_center_world
            + np.asarray(plane_pose.row_world, dtype=np.float64) * row_offset_mm
            + np.asarray(plane_pose.col_world, dtype=np.float64) * col_offset_mm
        )
        next_center_ijk = world_to_ijk_point(geometry, next_center_world)
        clamped_center_ijk = np.array(
            [
                max(0.0, min(float(next_center_ijk[0]), geometry.shape_ijk[0] - 1)),
                max(0.0, min(float(next_center_ijk[1]), geometry.shape_ijk[1] - 1)),
                max(0.0, min(float(next_center_ijk[2]), geometry.shape_ijk[2] - 1)),
            ],
            dtype=np.float64,
        )
        return ijk_to_world_point(geometry, clamped_center_ijk)

    def _handle_mpr_oblique(self, view: ViewRecord, payload: ViewOperationRequest) -> bool:
        if not self._is_mpr_view_type(view.view_type) or view.view_group is None:
            return False
        if payload.line not in {"horizontal", "vertical"}:
            return False
        if payload.x is None or payload.y is None:
            if payload.action_type == DRAG_ACTION_END:
                was_dragging = view.view_group.rotation_drag is not None
                view.view_group.rotation_drag = None
                return was_dragging
            return False

        group = view.view_group
        series = series_registry.get(view.series_id)
        volume_shape = self._get_series_volume(series).shape
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        self._ensure_mpr_reference_center(group, volume_shape)
        active_viewport = self._resolve_mpr_viewport(view)
        group.active_viewport = active_viewport
        active_plane = pose_context.poses[active_viewport]
        plane_shape = active_plane.output_shape
        pixel_aspect_x, pixel_aspect_y = self._get_mpr_display_aspect_xy_from_pose(active_plane)
        image_transform = viewport_transformer.build_image_to_canvas_transform(
            image_width=int(plane_shape[1]),
            image_height=int(plane_shape[0]),
            canvas_width=int(view.width or 0),
            canvas_height=int(view.height or 0),
            view=view,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        pointer_angle_rad = self._resolve_mpr_rotation_pointer_angle(
            view,
            active_plane,
            image_transform,
            float(payload.x),
            float(payload.y),
        )
        if pointer_angle_rad is None:
            if payload.action_type == DRAG_ACTION_END:
                was_dragging = group.rotation_drag is not None
                group.rotation_drag = None
                return was_dragging
            return False

        if payload.action_type == DRAG_ACTION_START:
            if self._get_mpr_crosshair_mode(group) == MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE:
                self._ensure_mpr_independent_plane_normals(group, pose_context.poses)
            self._ensure_mpr_crosshair_angle_cache(group, pose_context.poses)
            start_horizontal_angle, start_vertical_angle = self._get_mpr_visible_crosshair_line_angles(
                group,
                pose_context.poses,
                active_viewport,
            )
            group.rotation_drag = MprRotationDragRecord(
                viewport=active_viewport,
                line=payload.line,
                start_cursor=self._serialize_mpr_cursor_record(pose_context.cursor),
                start_pointer_angle_rad=pointer_angle_rad,
                start_line_angle_rad=start_horizontal_angle if payload.line == "horizontal" else start_vertical_angle,
                start_independent_plane_normals=deepcopy(group.mpr_independent_plane_normals),
            )
            return False

        if payload.action_type == DRAG_ACTION_END:
            was_dragging = group.rotation_drag is not None
            if was_dragging and group.rotation_drag is not None:
                self._apply_mpr_rotation_pointer_drag(
                    group,
                    group.rotation_drag,
                    pointer_angle_rad,
                    pose_context.geometry,
                    volume_shape,
                )
            group.rotation_drag = None
            return was_dragging

        if payload.action_type != DRAG_ACTION_MOVE or group.rotation_drag is None:
            return False

        self._apply_mpr_rotation_pointer_drag(
            group,
            group.rotation_drag,
            pointer_angle_rad,
            pose_context.geometry,
            volume_shape,
        )
        view.is_initialized = True
        return True

    def _resolve_mpr_rotation_pointer_angle(
        self,
        view: ViewRecord,
        active_plane: PlanePose,
        image_transform,
        normalized_x: float,
        normalized_y: float,
    ) -> float | None:
        canvas_width = float(view.width or 0)
        canvas_height = float(view.height or 0)
        if canvas_width <= 0.0 or canvas_height <= 0.0:
            return None
        canvas_x = min(max(float(normalized_x) * canvas_width, 0.0), max(canvas_width - 1e-6, 0.0))
        canvas_y = min(max(float(normalized_y) * canvas_height, 0.0), max(canvas_height - 1e-6, 0.0))
        center_image_x, center_image_y = self._project_world_point_to_plane_image(
            active_plane,
            active_plane.cursor_center_world,
        )
        center_canvas = image_transform.matrix @ np.array([center_image_x, center_image_y, 1.0], dtype=np.float64)
        delta_x = canvas_x - float(center_canvas[0])
        delta_y = canvas_y - float(center_canvas[1])
        if float(np.hypot(delta_x, delta_y)) <= 1e-6:
            return None
        return float(np.arctan2(delta_y, delta_x))

    def _apply_mpr_rotation_pointer_drag(
        self,
        group: ViewGroupRecord,
        drag: MprRotationDragRecord,
        pointer_angle_rad: float,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ) -> None:
        if self._get_mpr_crosshair_mode(group) == MPR_CROSSHAIR_MODE_DOUBLE_OBLIQUE:
            self._apply_mpr_double_oblique_rotation_pointer_drag(
                group,
                drag,
                pointer_angle_rad,
                geometry,
                volume_shape,
            )
            return

        start_cursor = self._deserialize_mpr_cursor_record(drag.start_cursor)
        start_poses = self._build_mpr_plane_poses(start_cursor, geometry, volume_shape)
        start_active_plane = start_poses[drag.viewport]
        active_normal = np.asarray(start_active_plane.normal_world, dtype=np.float64)
        active_row = np.asarray(start_active_plane.row_world, dtype=np.float64)
        active_col = np.asarray(start_active_plane.col_world, dtype=np.float64)
        target_line_angle_rad = float(drag.start_line_angle_rad) + self._normalize_screen_full_turn_delta(
            float(pointer_angle_rad) - float(drag.start_pointer_angle_rad)
        )
        self._set_mpr_visible_crosshair_line_angles(group, drag.viewport, drag.line, target_line_angle_rad)
        target_line_world = mpr_geometry.direction_from_screen_angle(
            active_row,
            active_col,
            target_line_angle_rad,
        )
        perpendicular_line_world = mpr_geometry.direction_from_screen_angle(
            active_row,
            active_col,
            target_line_angle_rad
            + (float(np.pi / 2.0) if drag.line == "horizontal" else -float(np.pi / 2.0)),
        )

        line_directions = {
            drag.line: target_line_world,
            self._resolve_perpendicular_crosshair_line(drag.line): perpendicular_line_world,
        }
        normal_updates: dict[str, np.ndarray] = {}
        for line, line_world in line_directions.items():
            target_viewport = self._resolve_mpr_oblique_target_viewport(drag.viewport, line)
            start_target_plane = start_poses[target_viewport]
            next_target_normal = mpr_geometry.normalize_oblique_vector(
                np.cross(line_world, active_normal),
                fallback=tuple(start_target_plane.normal_world),
            )
            if float(np.dot(next_target_normal, np.asarray(start_target_plane.normal_world, dtype=np.float64))) < 0.0:
                next_target_normal = -next_target_normal
            normal_updates[target_viewport] = next_target_normal

        next_cursor = self._replace_mpr_cursor_plane_normals(start_cursor, normal_updates)
        self._sync_group_from_mpr_cursor(group, next_cursor, geometry, volume_shape)

    def _apply_mpr_double_oblique_rotation_pointer_drag(
        self,
        group: ViewGroupRecord,
        drag: MprRotationDragRecord,
        pointer_angle_rad: float,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ) -> None:
        start_cursor = self._deserialize_mpr_cursor_record(drag.start_cursor)
        start_poses = self._build_mpr_plane_poses(
            start_cursor,
            geometry,
            volume_shape,
            normal_overrides=drag.start_independent_plane_normals,
        )
        start_active_plane = start_poses[drag.viewport]
        active_normal = np.asarray(start_active_plane.normal_world, dtype=np.float64)
        active_row = np.asarray(start_active_plane.row_world, dtype=np.float64)
        active_col = np.asarray(start_active_plane.col_world, dtype=np.float64)
        target_line_angle_rad = float(drag.start_line_angle_rad) + self._normalize_screen_full_turn_delta(
            float(pointer_angle_rad) - float(drag.start_pointer_angle_rad)
        )
        self._set_mpr_independent_visible_crosshair_line_angle(group, drag.viewport, drag.line, target_line_angle_rad)
        target_line_world = mpr_geometry.direction_from_screen_angle(
            active_row,
            active_col,
            target_line_angle_rad,
        )
        target_viewport = self._resolve_mpr_oblique_target_viewport(drag.viewport, drag.line)
        start_target_plane = start_poses[target_viewport]
        next_target_normal = mpr_geometry.normalize_oblique_vector(
            np.cross(target_line_world, active_normal),
            fallback=tuple(start_target_plane.normal_world),
        )
        if float(np.dot(next_target_normal, np.asarray(start_target_plane.normal_world, dtype=np.float64))) < 0.0:
            next_target_normal = -next_target_normal

        next_normals = self._normal_records_from_poses(start_poses)
        next_normals[target_viewport] = tuple(float(value) for value in next_target_normal)
        group.mpr_independent_plane_normals = next_normals

    @staticmethod
    def _replace_mpr_cursor_plane_normals(
        cursor: MprCursorState,
        normal_updates: dict[str, np.ndarray],
    ) -> MprCursorState:
        orientation = np.asarray(cursor.orientation_world, dtype=np.float64).copy()
        for viewport_key, normal_world in normal_updates.items():
            convention = DEFAULT_MPR_CONVENTION.get(viewport_key, DEFAULT_MPR_CONVENTION[MPR_VIEWPORT_AXIAL])
            normalized_normal = mpr_geometry.normalize_oblique_vector(
                normal_world,
                fallback=tuple(orientation[:, convention.normal_axis_index]),
            )
            orientation[:, convention.normal_axis_index] = normalized_normal / float(convention.normal_sign)
        return replace(cursor, orientation_world=orientation)

    @staticmethod
    def _resolve_perpendicular_crosshair_line(line: str) -> str:
        return "vertical" if line == "horizontal" else "horizontal"

    @staticmethod
    def _normalize_screen_half_turn_angle(angle_rad: float) -> float:
        return mpr_geometry.normalize_screen_half_turn_angle(angle_rad)

    def _ensure_mpr_crosshair_angle_cache(
        self,
        group: ViewGroupRecord,
        poses: dict[str, PlanePose],
    ) -> None:
        for viewport_key in (MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL):
            if viewport_key in group.mpr_crosshair_angles:
                continue
            group.mpr_crosshair_angles[viewport_key] = self._get_mpr_crosshair_line_angles_from_poses(
                poses,
                viewport_key,
            )

    def _get_mpr_visible_crosshair_line_angles(
        self,
        group: ViewGroupRecord | None,
        poses: dict[str, PlanePose],
        viewport_key: str,
    ) -> tuple[float, float]:
        cached_angles = group.mpr_crosshair_angles.get(viewport_key) if group is not None else None
        if cached_angles is not None:
            return (
                self._normalize_screen_half_turn_angle(float(cached_angles[0])),
                self._normalize_screen_half_turn_angle(float(cached_angles[1])),
            )
        return self._get_mpr_crosshair_line_angles_from_poses(poses, viewport_key)

    def _set_mpr_visible_crosshair_line_angles(
        self,
        group: ViewGroupRecord,
        viewport_key: str,
        line: str,
        line_angle_rad: float,
    ) -> None:
        if line == "horizontal":
            horizontal_angle = self._normalize_screen_half_turn_angle(line_angle_rad)
            vertical_angle = self._normalize_screen_half_turn_angle(line_angle_rad + float(np.pi / 2.0))
        else:
            vertical_angle = self._normalize_screen_half_turn_angle(line_angle_rad)
            horizontal_angle = self._normalize_screen_half_turn_angle(line_angle_rad - float(np.pi / 2.0))
        group.mpr_crosshair_angles[viewport_key] = (horizontal_angle, vertical_angle)

    def _set_mpr_independent_visible_crosshair_line_angle(
        self,
        group: ViewGroupRecord,
        viewport_key: str,
        line: str,
        line_angle_rad: float,
    ) -> None:
        cached_angles = group.mpr_crosshair_angles.get(viewport_key) or (0.0, float(np.pi / 2.0))
        if line == "horizontal":
            group.mpr_crosshair_angles[viewport_key] = (
                self._normalize_screen_half_turn_angle(line_angle_rad),
                self._normalize_screen_half_turn_angle(float(cached_angles[1])),
            )
            return

        group.mpr_crosshair_angles[viewport_key] = (
            self._normalize_screen_half_turn_angle(float(cached_angles[0])),
            self._normalize_screen_half_turn_angle(line_angle_rad),
        )

    @staticmethod
    def _normalize_screen_full_turn_delta(angle_rad: float) -> float:
        full_turn = float(np.pi * 2.0)
        delta = (float(angle_rad) + float(np.pi)) % full_turn - float(np.pi)
        if delta <= -float(np.pi):
            delta += full_turn
        return delta

    def _get_mpr_display_basis(
        self,
        viewport_key: str,
        normal_dir: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return mpr_geometry.get_mpr_display_basis(viewport_key, normal_dir)

    @staticmethod
    def _resolve_mpr_oblique_target_viewport(active_viewport: str, line: str) -> str:
        if active_viewport == MPR_VIEWPORT_CORONAL:
            return MPR_VIEWPORT_AXIAL if line == "horizontal" else MPR_VIEWPORT_SAGITTAL
        if active_viewport == MPR_VIEWPORT_SAGITTAL:
            return MPR_VIEWPORT_AXIAL if line == "horizontal" else MPR_VIEWPORT_CORONAL
        return MPR_VIEWPORT_CORONAL if line == "horizontal" else MPR_VIEWPORT_SAGITTAL

    @staticmethod
    def _default_mpr_oblique_plane(viewport_key: str) -> MprObliquePlaneState:
        return mpr_geometry.default_mpr_oblique_plane(viewport_key)

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

        selected_length_mm = 100.0
        selected_length_px = selected_length_mm / mm_per_canvas_pixel
        if not np.isfinite(selected_length_px) or selected_length_px <= 0.0:
            return None

        return ScaleBarInfo(
            lengthNorm=float(selected_length_px) / float(render_view.width),
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

        canvas_width = float(overlay.width)
        canvas_height = float(overlay.height)
        min_canvas_dimension = min(canvas_width, canvas_height)
        normalized_radius = (
            CROSSHAIR_HIT_RADIUS / min_canvas_dimension
            if min_canvas_dimension > 0
            else 0.0
        )
        return MprCrosshairInfo(
            centerX=(
                float(overlay.center_x) / canvas_width
                if canvas_width > 0
                else 0.0
            ),
            centerY=(
                float(overlay.center_y) / canvas_height
                if canvas_height > 0
                else 0.0
            ),
            hitRadius=normalized_radius,
            horizontalPosition=(
                float(overlay.horizontal_position) / canvas_height
                if overlay.horizontal_position is not None and canvas_height > 0
                else None
            ),
            verticalPosition=(
                float(overlay.vertical_position) / canvas_width
                if overlay.vertical_position is not None and canvas_width > 0
                else None
            ),
            horizontalAngleRad=float(overlay.horizontal_angle_rad),
            verticalAngleRad=float(overlay.vertical_angle_rad),
            horizontalSlabOffsetX=(
                float(overlay.horizontal_slab_offset_x) / canvas_width
                if overlay.horizontal_slab_offset_x is not None and canvas_width > 0
                else None
            ),
            horizontalSlabOffsetY=(
                float(overlay.horizontal_slab_offset_y) / canvas_height
                if overlay.horizontal_slab_offset_y is not None and canvas_height > 0
                else None
            ),
            verticalSlabOffsetX=(
                float(overlay.vertical_slab_offset_x) / canvas_width
                if overlay.vertical_slab_offset_x is not None and canvas_width > 0
                else None
            ),
            verticalSlabOffsetY=(
                float(overlay.vertical_slab_offset_y) / canvas_height
                if overlay.vertical_slab_offset_y is not None and canvas_height > 0
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
    def _resolve_mpr_slab_offset_canvas(
        plane_pose,
        target_pose,
        thickness_mm: float,
        center_image_x: float,
        center_image_y: float,
        center_canvas_x: float,
        center_canvas_y: float,
        image_to_canvas,
    ) -> tuple[float | None, float | None]:
        active_normal = np.asarray(plane_pose.normal_world, dtype=np.float64)
        target_normal = np.asarray(target_pose.normal_world, dtype=np.float64)
        projected_normal = target_normal - float(np.dot(target_normal, active_normal)) * active_normal
        projected_norm = float(np.linalg.norm(projected_normal))
        if not np.isfinite(projected_norm) or projected_norm <= 1e-6:
            return None, None

        offset_world = projected_normal / projected_norm * (float(thickness_mm) / 2.0 / projected_norm)
        offset_image_x = float(np.dot(offset_world, plane_pose.col_world)) / max(float(plane_pose.pixel_spacing_col_mm), 1e-6)
        offset_image_y = float(np.dot(offset_world, plane_pose.row_world)) / max(float(plane_pose.pixel_spacing_row_mm), 1e-6)
        offset_canvas_x, offset_canvas_y = image_to_canvas(
            center_image_x + offset_image_x,
            center_image_y + offset_image_y,
        )
        return float(offset_canvas_x - center_canvas_x), float(offset_canvas_y - center_canvas_y)

    def _build_mpr_crosshair_overlay(
        self,
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        plane_shape: tuple[int, int],
        image_transform,
    ) -> MprCrosshairOverlay:
        plane_height, plane_width = plane_shape
        canvas_width = view.width or plane_width
        canvas_height = view.height or plane_height
        target_viewport = self._resolve_mpr_viewport(view)
        is_active = view.mpr_active_viewport == target_viewport
        line_alpha = 255
        try:
            series = series_registry.get(view.series_id)
        except Exception:
            series = None
        pose_context = self._build_mpr_pose_context(view, volume_shape, series=series)
        plane_pose = pose_context.poses[target_viewport]
        horizontal_angle, vertical_angle = self._get_mpr_visible_crosshair_line_angles(
            view.view_group,
            pose_context.poses,
            target_viewport,
        )
        center_image_x, center_image_y = self._project_world_point_to_plane_image(plane_pose, pose_context.cursor.center_world)

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
        center_x, center_y = image_to_canvas(center_image_x, center_image_y)
        horizontal_position = None
        vertical_position = None
        if not plane_pose.is_oblique:
            _, horizontal_position = image_to_canvas(0.0, center_image_y)
            vertical_position, _ = image_to_canvas(center_image_x, 0.0)

        def slab_offset_for_line(line: str) -> tuple[float | None, float | None]:
            if not view.mpr_mip.enabled:
                return None, None
            target_line_viewport = self._resolve_mpr_oblique_target_viewport(target_viewport, line)
            viewport_mip = view.mpr_mip.viewports.get(target_line_viewport, MprMipViewportState())
            target_pose = pose_context.poses.get(target_line_viewport)
            if target_pose is None:
                return None, None
            configured_thickness_mm = float(viewport_mip.thickness)
            thickness_mm = (
                max(1e-6, float(spacing_along_world_direction(pose_context.geometry, target_pose.normal_world)))
                if configured_thickness_mm <= 0.0
                else configured_thickness_mm
            )
            return self._resolve_mpr_slab_offset_canvas(
                plane_pose,
                target_pose,
                thickness_mm,
                center_image_x,
                center_image_y,
                center_x,
                center_y,
                image_to_canvas,
            )

        horizontal_slab_offset_x, horizontal_slab_offset_y = slab_offset_for_line("horizontal")
        vertical_slab_offset_x, vertical_slab_offset_y = slab_offset_for_line("vertical")

        if target_viewport == MPR_VIEWPORT_CORONAL:
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
                horizontal_angle_rad=horizontal_angle,
                vertical_angle_rad=vertical_angle,
                horizontal_slab_offset_x=horizontal_slab_offset_x,
                horizontal_slab_offset_y=horizontal_slab_offset_y,
                vertical_slab_offset_x=vertical_slab_offset_x,
                vertical_slab_offset_y=vertical_slab_offset_y,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
        if target_viewport == MPR_VIEWPORT_SAGITTAL:
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
                horizontal_angle_rad=horizontal_angle,
                vertical_angle_rad=vertical_angle,
                horizontal_slab_offset_x=horizontal_slab_offset_x,
                horizontal_slab_offset_y=horizontal_slab_offset_y,
                vertical_slab_offset_x=vertical_slab_offset_x,
                vertical_slab_offset_y=vertical_slab_offset_y,
                center_x=center_x,
                center_y=center_y,
                is_active=is_active,
            )
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
            horizontal_angle_rad=horizontal_angle,
            vertical_angle_rad=vertical_angle,
            horizontal_slab_offset_x=horizontal_slab_offset_x,
            horizontal_slab_offset_y=horizontal_slab_offset_y,
            vertical_slab_offset_x=vertical_slab_offset_x,
            vertical_slab_offset_y=vertical_slab_offset_y,
            center_x=center_x,
            center_y=center_y,
            is_active=is_active,
        )

    def _get_mpr_crosshair_line_angles_from_poses(
        self,
        poses: dict[str, PlanePose],
        viewport_key: str,
    ) -> tuple[float, float]:
        active_pose = poses[viewport_key]

        def line_angle(line: str, fallback: float) -> float:
            target_viewport = self._resolve_mpr_oblique_target_viewport(viewport_key, line)
            target_pose = poses[target_viewport]
            line_world = self._normalize_oblique_vector(
                np.cross(active_pose.normal_world, target_pose.normal_world),
                fallback=tuple(active_pose.col_world if line == "horizontal" else active_pose.row_world),
            )
            col_component = float(np.dot(line_world, active_pose.col_world))
            row_component = float(np.dot(line_world, active_pose.row_world))
            magnitude = float(np.hypot(col_component, row_component))
            if not np.isfinite(magnitude) or magnitude <= 1e-8:
                return fallback
            return self._normalize_screen_half_turn_angle(float(np.arctan2(row_component, col_component)))

        return (
            line_angle("horizontal", 0.0),
            line_angle("vertical", float(np.pi / 2.0)),
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
        plane_state: MprObliquePlaneState | None = None,
        plane_pose: PlanePose | None = None,
        cursor: MprCursorState | None = None,
    ) -> CornerInfoOverlay:
        zoom = self._format_number(view.zoom, precision=2, suffix="x")
        physical_location = self._build_physical_location_label(
            view,
            series,
            dataset,
            current_index,
            viewport_label,
            plane_state=plane_state,
            plane_pose=plane_pose,
            cursor=cursor,
        )
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
        *,
        plane_state: MprObliquePlaneState | None = None,
        plane_pose: PlanePose | None = None,
        cursor: MprCursorState | None = None,
    ) -> str | None:
        label = viewport_label.lower()
        if label.startswith("oblique "):
            label = label.removeprefix("oblique ").strip()
        if self._is_mpr_view_type(view.view_type):
            transform = self._get_series_patient_transform(series)
            if plane_pose is not None and cursor is not None and plane_pose.is_oblique:
                return self._format_mpr_plane_pose_physical_location(
                    cursor,
                    plane_pose,
                    transform,
                )
            if cursor is not None:
                try:
                    geometry = self._get_series_volume_geometry(series, self._get_series_volume(series).shape)
                    frame_center = world_to_ijk_point(geometry, cursor.center_world)
                except Exception:
                    frame_center = np.asarray(cursor.center_world, dtype=np.float64)
            else:
                frame_center = np.asarray(
                    [float(view.mpr_axial_index), float(view.mpr_coronal_index), float(view.mpr_sagittal_index)],
                    dtype=np.float64,
                )
            if transform is not None:
                patient_point = transform.clamped_point_to_patient(frame_center)
                return self._format_standard_physical_location(label, patient_point)

        position = self._get_dataset_position(dataset)
        if position is None:
            return None
        return self._format_standard_physical_location(label, position)

    def _format_standard_physical_location(self, label: str, patient_point: np.ndarray) -> str | None:
        if label.startswith("stack") or label.startswith("ax"):
            return self._format_oriented_mm(float(patient_point[2]), positive="I", negative="S")
        if label.startswith("cor"):
            return self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A")
        if label.startswith("sag"):
            return self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R")
        return self._join_non_empty(
            " ",
            self._format_oriented_mm(float(patient_point[0]), positive="L", negative="R"),
            self._format_oriented_mm(float(patient_point[1]), positive="P", negative="A"),
            self._format_oriented_mm(float(patient_point[2]), positive="S", negative="I"),
        )

    def _format_mpr_plane_pose_physical_location(
        self,
        cursor: MprCursorState,
        plane_pose: PlanePose,
        transform: VolumePatientTransform | None,
    ) -> str | None:
        delta_world = np.asarray(cursor.center_world, dtype=np.float64) - np.asarray(cursor.reference_center_world, dtype=np.float64)
        normal_world = np.asarray(plane_pose.normal_world, dtype=np.float64)
        if transform is not None:
            distance_vector = delta_world
            direction_vector = mpr_geometry.normalize_patient_vector(
                normal_world,
                fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            )
        else:
            distance_vector = np.asarray([delta_world[2], delta_world[1], delta_world[0]], dtype=np.float64)
            direction_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(normal_world)

        signed_distance = float(np.dot(distance_vector, direction_vector))
        if abs(signed_distance) < 0.005:
            signed_distance = 0.0
        label = self._dominant_orientation_text_for_vector(direction_vector if signed_distance >= 0.0 else -direction_vector)
        if not label:
            return None
        magnitude = self._format_number(abs(signed_distance), precision=2, suffix="mm") or "0mm"
        return f"{label} {magnitude}"

    def _get_mpr_reference_center(
        self,
        view: ViewRecord,
        series: SeriesRecord,
        fallback_center: np.ndarray,
    ) -> np.ndarray:
        group = view.view_group
        if group is not None and group.mpr_reference_center is not None:
            return np.asarray(group.mpr_reference_center, dtype=np.float64)
        if group is not None:
            try:
                reference_center = self._ensure_mpr_reference_center(group, self._get_series_volume(series).shape)
                return np.asarray(reference_center, dtype=np.float64)
            except Exception:
                pass
        return np.asarray(fallback_center, dtype=np.float64)

    def _get_series_patient_transform(self, series: SeriesRecord) -> VolumePatientTransform | None:
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

        axis_mapping = get_standardized_axis_mapping(orientation)
        if axis_mapping is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        positions = [item[2] for item in slice_entries]
        if any(position is None for position in positions):
            ordered_entries = slice_entries
        else:
            ordered_entries = sorted(
                slice_entries,
                key=lambda item: float(np.dot(item[2], axis_mapping.slice_direction)) if item[2] is not None else 0.0,
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
        slice_spacing = self._estimate_slice_spacing(ordered_positions, axis_mapping.slice_direction, first_dataset)

        raw_axis_vectors = (axis_mapping.slice_direction, axis_mapping.column_direction, axis_mapping.row_direction)
        raw_axis_steps = (slice_spacing, row_spacing, col_spacing)
        raw_lengths = (
            len(ordered_entries),
            int(getattr(first_dataset, "Rows", 0) or 0),
            int(getattr(first_dataset, "Columns", 0) or 0),
        )
        if any(length <= 0 for length in raw_lengths):
            self._series_patient_transform_cache[series.series_id] = None
            return None

        if ordered_entries[0][2] is None:
            self._series_patient_transform_cache[series.series_id] = None
            return None

        origin = np.asarray(ordered_entries[0][2], dtype=np.float64)
        for canonical_axis, raw_axis in enumerate(axis_mapping.transpose_order):
            if axis_mapping.canonical_signs[canonical_axis] < 0:
                origin = origin + raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(raw_lengths[raw_axis] - 1)

        axis_vectors = tuple(
            raw_axis_vectors[raw_axis] * raw_axis_steps[raw_axis] * float(axis_mapping.canonical_signs[canonical_axis])
            for canonical_axis, raw_axis in enumerate(axis_mapping.transpose_order)
        )
        shape = tuple(raw_lengths[raw_axis] for raw_axis in axis_mapping.transpose_order)
        result = VolumePatientTransform(origin=origin, axis_vectors=axis_vectors, shape=shape)
        self._series_patient_transform_cache[series.series_id] = result
        return result

    def _get_series_volume_geometry(self, series: SeriesRecord, volume_shape: tuple[int, int, int]) -> VolumeGeometry:
        cached_geometry = self._series_volume_geometry_cache.get(series.series_id)
        normalized_shape = tuple(int(value) for value in volume_shape)
        if cached_geometry is not None and cached_geometry.shape_ijk == normalized_shape:
            return cached_geometry

        transform = self._get_series_patient_transform(series)
        geometry = build_geometry_from_patient_transform(transform) if transform is not None else build_identity_geometry(normalized_shape)
        if geometry.shape_ijk != normalized_shape:
            geometry = build_identity_geometry(normalized_shape)
        self._series_volume_geometry_cache[series.series_id] = geometry
        return geometry

    @staticmethod
    def _build_fallback_mpr_frame(view: ViewRecord) -> MprFrameState:
        return MprFrameState(
            center=(
                float(view.mpr_axial_index),
                float(view.mpr_coronal_index),
                float(view.mpr_sagittal_index),
            ),
            axis_slice=(1.0, 0.0, 0.0),
            axis_row=(0.0, 1.0, 0.0),
            axis_col=(0.0, 0.0, 1.0),
        )

    def _get_mpr_cursor_state(
        self,
        view: ViewRecord,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ):
        if view.view_group is None:
            frame = self._build_fallback_mpr_frame(view)
            return legacy_frame_to_cursor(frame, geometry, reference_center=frame.center)

        group = view.view_group
        if group.mpr_cursor is not None:
            return self._deserialize_mpr_cursor_record(group.mpr_cursor)

        reference_center = self._ensure_mpr_reference_center(group, volume_shape)
        cursor = create_default_cursor(geometry)
        center_ijk = np.asarray(
            [
                float(max(0, min(group.axial_index, volume_shape[0] - 1))),
                float(max(0, min(group.coronal_index, volume_shape[1] - 1))),
                float(max(0, min(group.sagittal_index, volume_shape[2] - 1))),
            ],
            dtype=np.float64,
        )
        cursor = replace(
            cursor,
            center_world=ijk_to_world_point(geometry, center_ijk),
            reference_center_world=ijk_to_world_point(geometry, reference_center),
        )
        group.mpr_cursor = self._serialize_mpr_cursor_record(cursor)
        return cursor

    @staticmethod
    def _should_use_mpr_display_basis_for_cursor_offsets(group: ViewGroupRecord | None) -> bool:
        return bool(
            group is not None
            and (
                group.crosshair_drag_active
                or group.mpr_use_display_basis_for_cursor_offsets
            )
        )

    @staticmethod
    def _build_reslice_mip_config(mip_state: MprMipState, viewport_key: str) -> ResliceMipConfig:
        viewport_config = mip_state.viewports.get(viewport_key, MprMipViewportState())
        return ResliceMipConfig(
            enabled=bool(mip_state.enabled),
            algorithm=str(mip_state.algorithm or "maximum"),
            thickness=max(0, min(100, int(viewport_config.thickness))),
        )

    @staticmethod
    def _serialize_mpr_cursor_record(cursor: MprCursorState) -> MprCursorRecord:
        orientation = np.asarray(cursor.orientation_world, dtype=np.float64)
        return MprCursorRecord(
            center_world=tuple(float(value) for value in np.asarray(cursor.center_world, dtype=np.float64)),
            reference_center_world=tuple(float(value) for value in np.asarray(cursor.reference_center_world, dtype=np.float64)),
            orientation_world=tuple(
                tuple(float(value) for value in orientation[:, column_index])
                for column_index in range(orientation.shape[1])
            ),
            linked_to_volume_rotation=bool(cursor.linked_to_volume_rotation),
        )

    @staticmethod
    def _deserialize_mpr_cursor_record(record: MprCursorRecord) -> MprCursorState:
        orientation_columns = [
            np.asarray(column, dtype=np.float64)
            for column in record.orientation_world
        ]
        if len(orientation_columns) != 3:
            orientation_columns = [
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            ]
        return MprCursorState(
            center_world=np.asarray(record.center_world, dtype=np.float64),
            reference_center_world=np.asarray(record.reference_center_world, dtype=np.float64),
            orientation_world=np.column_stack(orientation_columns),
            linked_to_volume_rotation=bool(record.linked_to_volume_rotation),
        )

    def _build_mpr_pose_context(
        self,
        view: ViewRecord,
        volume_shape: tuple[int, int, int],
        *,
        series: SeriesRecord | None = None,
    ) -> MprPoseContext:
        normalized_shape = tuple(int(value) for value in volume_shape)
        geometry = (
            self._get_series_volume_geometry(series, normalized_shape)
            if series is not None
            else build_identity_geometry(normalized_shape)
        )
        cursor = self._get_mpr_cursor_state(view, geometry, normalized_shape)
        return MprPoseContext(
            geometry=geometry,
            cursor=cursor,
            poses=self._build_mpr_plane_poses(
                cursor,
                geometry,
                normalized_shape,
                normal_overrides=self._get_independent_plane_normal_overrides(view.view_group),
                use_display_basis_for_cursor_offsets=self._should_use_mpr_display_basis_for_cursor_offsets(view.view_group),
            ),
        )

    @staticmethod
    def _project_world_point_to_plane_image(plane_pose: PlanePose, point_world: np.ndarray) -> tuple[float, float]:
        delta_world = np.asarray(point_world, dtype=np.float64) - np.asarray(plane_pose.center_world, dtype=np.float64)
        image_y = (
            float(np.dot(delta_world, plane_pose.row_world)) / max(float(plane_pose.pixel_spacing_row_mm), 1e-6)
            + float(plane_pose.output_shape[0]) / 2.0
        )
        image_x = (
            float(np.dot(delta_world, plane_pose.col_world)) / max(float(plane_pose.pixel_spacing_col_mm), 1e-6)
            + float(plane_pose.output_shape[1]) / 2.0
        )
        return image_x, image_y

    def _sync_group_from_mpr_cursor(
        self,
        group: ViewGroupRecord,
        cursor: MprCursorState,
        geometry: VolumeGeometry,
        volume_shape: tuple[int, int, int],
    ) -> None:
        group.mpr_reference_center = tuple(
            float(value)
            for value in world_to_ijk_point(geometry, cursor.reference_center_world)
        )
        group.mpr_cursor = self._serialize_mpr_cursor_record(cursor)
        center_ijk = world_to_ijk_point(geometry, cursor.center_world)
        group.axial_index = int(max(0, min(int(np.round(center_ijk[0])), volume_shape[0] - 1)))
        group.coronal_index = int(max(0, min(int(np.round(center_ijk[1])), volume_shape[1] - 1)))
        group.sagittal_index = int(max(0, min(int(np.round(center_ijk[2])), volume_shape[2] - 1)))

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

    def _format_projected_physical_location(
        self,
        patient_point: np.ndarray,
        patient_normal: np.ndarray,
        *,
        origin_point: np.ndarray | None = None,
        orientation_vector: np.ndarray | None = None,
    ) -> str | None:
        normal = mpr_geometry.normalize_patient_vector(patient_normal, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        point = np.asarray(patient_point, dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64) if origin_point is None else np.asarray(origin_point, dtype=np.float64)
        orientation_source = normal if orientation_vector is None else mpr_geometry.normalize_patient_vector(
            orientation_vector,
            fallback=normal,
        )
        if float(np.dot(normal, orientation_source)) < 0.0:
            normal = -normal
        distance = float(np.dot(point - origin, normal))
        if abs(distance) < 0.005:
            distance = 0.0
        orientation = self._dominant_orientation_text_for_vector(orientation_source if distance >= 0.0 else -orientation_source)
        if not orientation:
            return None
        magnitude = self._format_number(abs(distance), precision=2, suffix="mm") or "0mm"
        return f"{orientation} {magnitude}"

    @staticmethod
    def _resolve_mpr_directed_line_angle(current_row: np.ndarray, current_col: np.ndarray, line_dir: np.ndarray) -> float | None:
        col_component = float(np.dot(line_dir, current_col))
        row_component = float(np.dot(line_dir, current_row))
        if not np.isfinite(col_component) or not np.isfinite(row_component):
            return None
        magnitude = float(np.hypot(col_component, row_component))
        if magnitude <= 1e-8:
            return None
        angle = float(np.arctan2(row_component, col_component))
        if angle < 0.0:
            angle += float(np.pi * 2.0)
        return angle

    @staticmethod
    def _dominant_orientation_text_for_vector(vector: np.ndarray | None) -> str | None:
        return ViewerService._orientation_text_for_vector(
            vector,
            minimum_magnitude=1e-4,
            max_components=1,
            axis_priority=(1, 0, 2),
        )

    @staticmethod
    def _mpr_oblique_orientation_text_for_vector(vector: np.ndarray | None) -> str | None:
        return ViewerService._orientation_text_for_vector(
            vector,
            minimum_magnitude=0.2,
            max_components=2,
            axis_priority=(1, 0, 2),
        )

    @staticmethod
    def _orientation_text_for_vector(
        vector: np.ndarray | None,
        *,
        minimum_magnitude: float = 0.2,
        max_components: int = 3,
        axis_priority: tuple[int, int, int] = (0, 1, 2),
    ) -> str | None:
        if vector is None:
            return None
        axis_map = (
            (0, "L", "R", axis_priority[0]),
            (1, "P", "A", axis_priority[1]),
            (2, "S", "I", axis_priority[2]),
        )
        components: list[tuple[float, int, str]] = []
        for axis_index, positive_label, negative_label, priority in axis_map:
            component = float(vector[axis_index])
            magnitude = abs(component)
            if magnitude < minimum_magnitude:
                continue
            label = positive_label if component >= 0 else negative_label
            components.append((magnitude, priority, label))
        if not components:
            return None
        components.sort(key=lambda item: (-item[0], item[1]))
        return ''.join(label for _, _, label in components[:max(1, max_components)])

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
            zoom=float(view.zoom),
            offsetX=float(view.offset_x),
            offsetY=float(view.offset_y),
        )

    @staticmethod
    def _build_mpr_frame_payload(cursor: MprCursorState | None, geometry: VolumeGeometry | None) -> MprFrameInfo | None:
        if cursor is None or geometry is None:
            return None
        frame = cursor_to_legacy_frame(cursor, geometry)
        return MprFrameInfo(
            center=tuple(float(value) for value in frame.center),
            axisSlice=tuple(float(value) for value in frame.axis_slice),
            axisRow=tuple(float(value) for value in frame.axis_row),
            axisCol=tuple(float(value) for value in frame.axis_col),
        )

    @staticmethod
    def _vector_payload(vector: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.asarray(vector, dtype=np.float64))

    @staticmethod
    def _build_mpr_cursor_payload(cursor: MprCursorState | None) -> MprCursorInfo | None:
        if cursor is None:
            return None
        orientation = np.asarray(cursor.orientation_world, dtype=np.float64)
        return MprCursorInfo(
            centerWorld=ViewerService._vector_payload(cursor.center_world),
            referenceCenterWorld=ViewerService._vector_payload(cursor.reference_center_world),
            orientationWorld=tuple(
                tuple(float(value) for value in orientation[row_index, :3])
                for row_index in range(3)
            ),
            linkedToVolumeRotation=bool(cursor.linked_to_volume_rotation),
        )

    @staticmethod
    def _plane_state_from_pose(plane_pose: PlanePose) -> MprObliquePlaneState:
        return MprObliquePlaneState(
            row=ViewerService._vector_payload(plane_pose.row_world),
            col=ViewerService._vector_payload(plane_pose.col_world),
            normal=ViewerService._vector_payload(plane_pose.normal_world),
            is_oblique=bool(plane_pose.is_oblique),
        )

    def _build_mpr_plane_payload(
        self,
        view: ViewRecord,
        viewport_key: str,
        *,
        plane_pose: PlanePose | None = None,
    ) -> MprPlaneInfo | None:
        if view.view_group is None:
            return None
        plane = self._plane_state_from_pose(plane_pose) if plane_pose is not None else self._default_mpr_oblique_plane(viewport_key)
        center_world = plane_pose.center_world if plane_pose is not None else (0.0, 0.0, 0.0)
        cursor_center_world = plane_pose.cursor_center_world if plane_pose is not None else center_world
        row_world = plane_pose.row_world if plane_pose is not None else plane.row
        col_world = plane_pose.col_world if plane_pose is not None else plane.col
        normal_world = plane_pose.normal_world if plane_pose is not None else plane.normal
        output_shape = plane_pose.output_shape if plane_pose is not None else (0, 0)
        return MprPlaneInfo(
            viewport=viewport_key,
            centerWorld=self._vector_payload(center_world),
            cursorCenterWorld=self._vector_payload(cursor_center_world),
            rowWorld=self._vector_payload(row_world),
            colWorld=self._vector_payload(col_world),
            normalWorld=self._vector_payload(normal_world),
            pixelSpacingRowMm=float(plane_pose.pixel_spacing_row_mm) if plane_pose is not None else 1.0,
            pixelSpacingColMm=float(plane_pose.pixel_spacing_col_mm) if plane_pose is not None else 1.0,
            outputShape=(int(output_shape[0]), int(output_shape[1])),
            row=tuple(float(value) for value in plane.row),
            col=tuple(float(value) for value in plane.col),
            normal=tuple(float(value) for value in plane.normal),
            isOblique=bool(plane_pose.is_oblique if plane_pose is not None else plane.is_oblique),
        )

    def _build_stack_orientation_overlay(self, view: ViewRecord, dataset: Dataset | None) -> OrientationOverlay | None:
        orientation = self._get_dataset_orientation(dataset)
        if orientation is None:
            return None

        row_direction = self._normalize_vector(orientation[:3])
        column_direction = self._normalize_vector(orientation[3:6])
        if row_direction is None or column_direction is None:
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

    def _build_mpr_orientation_overlay(
        self,
        view: ViewRecord,
        viewport_key: str,
        plane_state: MprObliquePlaneState | None = None,
        *,
        plane_pose: PlanePose | None = None,
    ) -> OrientationOverlay:
        resolved_plane = plane_state or self._default_mpr_oblique_plane(viewport_key)
        try:
            series = series_registry.get(view.series_id)
        except Exception:
            series = None
        transform = self._get_series_patient_transform(series) if series is not None else None
        use_model_label_directions = self._should_apply_mpr_model_rotation_to_plane_labels(
            view.view_group,
            plane_pose,
        )
        if plane_pose is not None and transform is not None:
            col_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.col_world)
                if use_model_label_directions
                else plane_pose.col_world
            )
            row_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.row_world)
                if use_model_label_directions
                else plane_pose.row_world
            )
            x_vector = mpr_geometry.normalize_patient_vector(
                col_world,
                fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            )
            y_vector = mpr_geometry.normalize_patient_vector(
                row_world,
                fallback=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            )
        elif plane_pose is not None:
            col_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.col_world)
                if use_model_label_directions
                else plane_pose.col_world
            )
            row_world = (
                self._get_mpr_model_source_direction(view.view_group, plane_pose.row_world)
                if use_model_label_directions
                else plane_pose.row_world
            )
            x_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(col_world)
            y_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(row_world)
        elif transform is not None:
            x_vector = mpr_geometry.normalize_patient_vector(
                transform.direction_step_to_patient(resolved_plane.col),
                fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            )
            y_vector = mpr_geometry.normalize_patient_vector(
                transform.direction_step_to_patient(resolved_plane.row),
                fallback=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            )
        else:
            x_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(resolved_plane.col)
            y_vector = mpr_geometry.fallback_volume_direction_to_patient_vector(resolved_plane.row)

        if view.hor_flip:
            x_vector = -x_vector
        if view.ver_flip:
            y_vector = -y_vector
        x_vector, y_vector = self._rotate_screen_axes(x_vector, y_vector, view.rotation_degrees)

        orientation_text = (
            self._mpr_oblique_orientation_text_for_vector
            if use_model_label_directions or ((plane_pose is not None and plane_pose.is_oblique) or resolved_plane.is_oblique)
            else self._dominant_orientation_text_for_vector
        )

        return OrientationOverlay(
            top=orientation_text(-y_vector),
            right=orientation_text(x_vector),
            bottom=orientation_text(y_vector),
            left=orientation_text(-x_vector),
        )

    def _resolve_mpr_orientation_screen_axes(
        self,
        view: ViewRecord,
        normal_vector: np.ndarray,
        plane_state: MprObliquePlaneState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        series = None
        try:
            series = series_registry.get(view.series_id)
        except Exception:
            series = None
        transform = self._get_series_patient_transform(series) if series is not None else None
        if plane_state is not None and transform is not None:
            return (
                mpr_geometry.volume_direction_to_patient_vector(plane_state.col, transform),
                mpr_geometry.volume_direction_to_patient_vector(plane_state.row, transform),
            )
        return mpr_geometry.resolve_mpr_orientation_screen_axes(normal_vector, transform)

    @staticmethod
    def _build_mpr_viewport_label(viewport_key: str, plane_state: MprObliquePlaneState | None = None) -> str:
        if viewport_key == MPR_VIEWPORT_CORONAL:
            label = "CORONAL"
        elif viewport_key == MPR_VIEWPORT_SAGITTAL:
            label = "SAGITTAL"
        else:
            label = "AXIAL"
        if plane_state is not None and plane_state.is_oblique:
            return f"OBLIQUE {label}"
        return label

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
    def _window_array(
        pixels: np.ndarray,
        window_width: float | None,
        window_center: float | None,
        *,
        pixel_min: float | None = None,
        pixel_max: float | None = None,
    ) -> np.ndarray:
        if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            color_pixels = pixels[..., :3]
            if color_pixels.dtype == np.uint8:
                return color_pixels
            return np.clip(color_pixels, 0, 255).astype(np.uint8)

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
    def _encode_image(image: Image.Image, image_format: ImageFormat, *, fast_preview: bool = False) -> bytes:
        output = io.BytesIO()
        if image_format == "jpeg":
            # JPEG is only used for transient interaction previews. Settled frames
            # stay PNG so overlays and measurements align with lossless pixels.
            image.convert("RGB").save(output, format="JPEG", quality=FAST_PREVIEW_JPEG_QUALITY)
        else:
            # PNG is lossless at every compression level. Keep all viewer PNG
            # frames at a low compression level to reduce encode latency and
            # avoid final-frame tail spikes during interaction.
            image.save(
                output,
                format="PNG",
                compress_level=PNG_COMPRESS_LEVEL,
                optimize=False,
            )
        return output.getvalue()


viewer_service = ViewerService()
