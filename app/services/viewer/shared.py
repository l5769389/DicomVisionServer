import sys
import io
import hashlib
import json
import zipfile
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
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
from scipy import ndimage

from app.core import (
    DRAG_ACTION_END,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_START,
    FUSION_PANE_CT_AXIAL,
    FUSION_PANE_OVERLAY_AXIAL,
    FUSION_PANE_PET_AXIAL,
    FUSION_PANE_PET_CORONAL_MIP,
    MPR_VIEWPORT_AXIAL,
    MPR_VIEWPORT_CORONAL,
    MPR_VIEWPORT_SAGITTAL,
    VIEW_OP_TYPE_CROSSHAIR,
    VIEW_OP_TYPE_PAN,
    VIEW_OP_TYPE_PET_CONFIG,
    VIEW_OP_TYPE_SET_SIZE,
    VIEW_OP_TYPE_WINDOW,
    VIEW_OP_TYPE_ZOOM,
    VIEW_OP_TYPE_ROTATE_3D,
    VIEW_OP_TYPE_VOLUME_CONFIG,
    VIEW_OP_TYPE_MPR_MIP_CONFIG,
    WINDOW_DRAG_SENSITIVITY,
    WINDOW_DRAG_MIN_SENSITIVITY,
    WINDOW_DRAG_REFERENCE_WIDTH,
    WINDOW_WIDTH_MIN,
    ZOOM_DRAG_FACTOR_MIN,
    ZOOM_DRAG_LOG_SENSITIVITY,
    ZOOM_DRAG_SENSITIVITY,
    ZOOM_DRAG_SENSITIVITY_3D,
    ZOOM_MAX_3D,
    ZOOM_MIN_3D,
    VIEW_OP_TYPE_SCROLL,
)
from app.core.logging import get_logger
from app.models.measurement import DrawingScope, MeasurementPoint, MeasurementRecord, MeasurementSliceContext
from app.models.viewer import (
    AnnotationRecord,
    FusionRegistrationState,
    InstanceRecord,
    MprCursorRecord,
    MprFrameState,
    MprMipState,
    MprMipViewportState,
    MprObliquePlaneState,
    MprRotationDragRecord,
    MprSegmentationState,
    MprThresholdRegionBoxState,
    MprThresholdRegionState,
    MprThresholdRegionStatsState,
    MprVoiSphereState,
    MprVoiSphereStatsState,
    MprSegmentationVoiBoxState,
    PresentationAnnotationRecord,
    PresentationMeasurementRecord,
    SeriesRecord,
    ViewGroupRecord,
    ViewRecord,
)
from app.schemas.dicom import CornerInfoPayload, CornerInfoRequest, CornerInfoResponse
from app.schemas.view import (
    FusionRegistrationArtifactExportRequest,
    FusionRegistrationExportRequest,
    FusionRegistrationExportResponse,
    ImageFormat,
    AnnotationOverlayPayload,
    MprCrosshairInfo,
    MprCursorInfo,
    MprFrameInfo,
    MprMipConfig,
    MprMipViewportConfig,
    MprPlaneInfo,
    MprSegmentationConfig,
    MprSegmentationOverlay,
    MprSegmentationOverlayRect,
    MprSegmentationOverlayRegion,
    MprSegmentationOverlaySamples,
    MprThresholdRegion,
    MprThresholdRegionBox,
    MprThresholdRegionStats,
    MprVoiSphere,
    MprVoiSphereStats,
    MprSegmentationVoiBox,
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
    FusionCompositeInfo,
    FusionInfo,
    FusionCompositeLayerInfo,
    PetInfo,
    FusionProjectionInfo,
    FusionRegistrationInfo,
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
from app.services.viewport_transformer import AffineTransform, viewport_transformer
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry
from app.services.viewer_operation_handlers import OperationRenderOutcome, handle_view_operation
from app.services.viewer_render_dispatch import render_by_view_type
from app.services.viewer_fusion import (
    FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET,
    FUSION_VIEW_TYPES,
    FUSION_VIEW_TYPE_TO_PANE_ROLE,
    FusionSourceProjection,
    build_fusion_axial_display_plane,
    build_ct_axial_plane,
    image_from_pixels,
    render_fusion_pixels,
    transform_pet_sampling_plane,
)
from app.services.viewer_render_guards import ensure_view_size
from app.services.water_phantom_qa_service import WaterPhantomQaService
from app.services.volume_render_config import (
    build_volume_intensity_stats,
    create_adaptive_volume_render_config,
    create_default_volume_render_config,
    normalize_volume_preset_name,
    normalize_volume_render_config,
    select_default_volume_preset,
)
from app.services.surface_render_config import (
    create_adaptive_surface_render_config,
    create_default_surface_render_config,
    normalize_surface_preset_name,
    normalize_surface_render_config,
)
from app.services.volume_rendering.camera_math import (
    anatomical_orientation_quaternion,
    apply_direct_model_trackball_control_points_to_quaternion,
    quaternion_to_rotation_matrix,
    resolve_direct_model_trackball_control_point,
)
from app.services.volume_rendering.contracts import SurfaceRenderRequest, VolumeRenderRequest


logger = get_logger(__name__)

FUSION_PET_UNIT_SOURCE = "source"
FUSION_PET_UNIT_KBQML = "kBqml"
FUSION_PET_UNIT_SUV_BW = "SUVbw"
FUSION_PET_UNIT_SUV_BSA = "SUVbsa"
FUSION_PET_UNIT_SUL = "SUL"
FUSION_PET_UNIT_PERCENT_ID_G = "percentIDg"
MPR_SEGMENTATION_OVERLAY_SAMPLE_LIMIT = 120_000
FUSION_PET_STANDALONE_BACKGROUND_CVAL = 255.0
PET_STANDALONE_PSEUDOCOLOR_PRESET = FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET
MPR_SEGMENTATION_OVERLAY_PREVIEW_SAMPLE_LIMIT = 12_000
FUSION_PET_UNIT_LABELS: dict[str, str] = {
    FUSION_PET_UNIT_SOURCE: "source",
    FUSION_PET_UNIT_KBQML: "kBq/ml (uptake)",
    FUSION_PET_UNIT_SUV_BSA: "cm2/ml (SUVbsa)",
    FUSION_PET_UNIT_SUV_BW: "g/ml (SUVbw)",
    FUSION_PET_UNIT_SUL: "g/ml* (SUL)",
    FUSION_PET_UNIT_PERCENT_ID_G: "%ID/g",
}
# SUV fusion display starts from a reference-viewer preset, not from PET intensity percentiles.
FUSION_DEFAULT_SUV_WINDOW_MIN = 0.0
FUSION_DEFAULT_SUV_WINDOW_MAX = 4.49


@dataclass(frozen=True)
class FusionPetDisplayVolume:
    volume: np.ndarray
    unit: str
    unit_label: str
    source_units: str | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class FusionRegistrationPreviewDrag:
    group_id: str
    origin_registration: FusionRegistrationState
    sub_op_type: str
    delta_x: float
    delta_y: float
    pivot_x: float
    pivot_y: float
    rotation_delta_degrees: float


@dataclass(frozen=True)
class FusionRegistrationCanvasMapping:
    col_mm_from_canvas: tuple[float, float, float]
    row_mm_from_canvas: tuple[float, float, float]


@dataclass(frozen=True)
class FusionRegistrationOverlayRenderFrame:
    plane: PlanePose
    cache_key: tuple[object, ...]
    canvas_mapping: FusionRegistrationCanvasMapping | None = None
    pet_center_canvas: tuple[float, float] | None = None


@dataclass(frozen=True)
class FusionRegistrationPetLayerCacheEntry:
    image: Image.Image
    slice_index: int
    slice_total: int
    pet_unit_label: str
    canvas_mapping: FusionRegistrationCanvasMapping | None = None
    overlay_frame: FusionRegistrationOverlayRenderFrame | None = None
    pet_center_canvas: tuple[float, float] | None = None


CROSSHAIR_HIT_RADIUS = 12.0
MEASUREMENT_TOOL_TYPES = {"line", "rect", "ellipse", "angle", "curve", "freeform"}
VOLUME_CACHE_MAX_BYTES = 1024 * 1024 * 1024
FAST_PREVIEW_JPEG_QUALITY = 20
WEBP_PREVIEW_QUALITY = 80
WEBP_PREVIEW_METHOD = 0
PNG_COMPRESS_LEVEL = 1
MPR_FAST_PREVIEW_SCALE = 0.33
MPR_FAST_PREVIEW_MIN_SIDE = 96
VOLUME_FAST_PREVIEW_MAX_DIMENSION = 720
MPR_PLANE_CACHE_MAX_ITEMS = 48
FAST_BASE_PIXELS_CACHE_MAX_ITEMS = 64
FUSION_REGISTRATION_PET_LAYER_CACHE_MAX_ITEMS = 32
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
    extra_image_bytes: dict[str, bytes] = field(default_factory=dict)
    performance_timings: dict[str, float | str] = field(default_factory=dict)
    raw_image: Image.Image | None = None


@dataclass(frozen=True)
class MprPoseContext:
    geometry: VolumeGeometry
    cursor: MprCursorState
    poses: dict[str, PlanePose]


@dataclass(frozen=True)
class MprThresholdPlaneGrid:
    row_grid_mm: np.ndarray
    col_grid_mm: np.ndarray
    center_world: np.ndarray
    row_world: np.ndarray
    col_world: np.ndarray


@dataclass(frozen=True)
class MprThresholdPlaneMask:
    region_id: str
    mask: np.ndarray
    color: str


@dataclass(frozen=True)
class ExportedFileResult:
    file_bytes: bytes
    file_name: str
    media_type: str
    extra_headers: dict[str, str] | None = None


@dataclass(frozen=True)
class FusionRegistrationExportContext:
    group: ViewGroupRecord
    ct_series: SeriesRecord
    pet_series: SeriesRecord
    ct_volume: np.ndarray
    pet_volume: np.ndarray
    ct_geometry: VolumeGeometry
    pet_geometry: VolumeGeometry
    pet_display: FusionPetDisplayVolume
    series_description: str


@dataclass(frozen=True)
class RenderPlan:
    render_view: ViewRecord
    render_ratio: float


@dataclass(frozen=True)
class VolumePreviewRenderPlan:
    width: int
    height: int
    ratio: float


ViewRenderProgressCallback = Callable[[dict[str, object]], None]


class _ViewerServiceCompatibilityProxy:
    """Resolve patch-sensitive dependencies through the legacy facade module."""

    def __getattr__(self, name: str) -> Any:
        facade = sys.modules.get("app.services.viewer_service")
        if facade is not None and name in facade.__dict__:
            return facade.__dict__[name]
        return globals()[name]


compat = _ViewerServiceCompatibilityProxy()

# Re-export private helpers as well; the facade and mixins intentionally share this namespace.
__all__ = [name for name in globals() if not name.startswith("__")]
