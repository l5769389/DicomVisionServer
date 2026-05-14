from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.dicom import CornerInfoPayload


ViewType = Literal["Stack", "MPR", "3D", "AX", "COR", "SAG"]
ImageFormat = Literal["png", "jpeg"]
ExportFormat = Literal["png", "dicom"]
ViewSetSizeOperationType = Literal["setSize"]
ViewOperationType = Literal["scroll", "crosshair", "pan", "zoom", "window", "pseudocolor", "transform2d", "rotate3d", "reset", "volumePreset", "volumeConfig", "mprMipConfig", "mprOblique", "mprStateSync", "measurement"]
ViewActionType = Literal["start", "move", "end", "delete"]
VolumeBlendMode = Literal["composite", "mip"]
VolumeInterpolationMode = Literal["nearest", "linear", "cubic"]
MprMipAlgorithm = Literal["maximum", "minimum", "average", "sum"]
MprCrosshairLine = Literal["horizontal", "vertical"]


class ViewCreateRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered series ID to view.")
    view_type: ViewType = Field(alias="viewType", description="Requested view type: Stack, MPR, 3D, or a concrete MPR viewport.")
    view_group_key: str | None = Field(
        default=None,
        alias="viewGroupKey",
        description="Optional key for sharing MPR state across AX/COR/SAG views or 4D phases.",
    )
    four_d_phase_index: int | None = Field(
        default=None,
        alias="fourDPhaseIndex",
        description="Optional 4D phase index represented by this view.",
    )

    model_config = {"populate_by_name": True}


class ViewCreateResponse(BaseModel):
    view_id: str = Field(alias="viewId")

    model_config = {"populate_by_name": True}


class ViewCloseRequest(BaseModel):
    view_id: str = Field(alias="viewId")

    model_config = {"populate_by_name": True}


class OperationAcceptedResponse(BaseModel):
    success: bool = True
    message: str
    view_id: str = Field(alias="viewId")

    model_config = {"populate_by_name": True}


class ViewSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ViewSetSizeRequest(BaseModel):
    op_type: ViewSetSizeOperationType = Field(alias="opType", description="Operation name, always setSize for this endpoint.")
    size: ViewSize = Field(description="Current frontend viewport size in CSS pixels.")
    view_id: str = Field(alias="viewId", description="Server-side view ID returned by /view/create.")

    model_config = {"populate_by_name": True}


class SliceInfo(BaseModel):
    current: int
    total: int


class WindowInfo(BaseModel):
    ww: float | None = None
    wl: float | None = None


class ViewColorInfo(BaseModel):
    pseudocolor_preset: str = Field(alias="pseudocolorPreset")

    model_config = {"populate_by_name": True}


class MprCrosshairInfo(BaseModel):
    center_x: float = Field(alias="centerX")
    center_y: float = Field(alias="centerY")
    hit_radius: float = Field(alias="hitRadius")
    horizontal_position: float | None = Field(default=None, alias="horizontalPosition")
    vertical_position: float | None = Field(default=None, alias="verticalPosition")
    horizontal_angle_rad: float | None = Field(default=None, alias="horizontalAngleRad")
    vertical_angle_rad: float | None = Field(default=None, alias="verticalAngleRad")

    model_config = {"populate_by_name": True}


class MprFrameInfo(BaseModel):
    center: tuple[float, float, float]
    axis_slice: tuple[float, float, float] = Field(alias="axisSlice")
    axis_row: tuple[float, float, float] = Field(alias="axisRow")
    axis_col: tuple[float, float, float] = Field(alias="axisCol")

    model_config = {"populate_by_name": True}


class MprCursorInfo(BaseModel):
    center_world: tuple[float, float, float] = Field(alias="centerWorld")
    reference_center_world: tuple[float, float, float] = Field(alias="referenceCenterWorld")
    orientation_world: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = Field(alias="orientationWorld")
    linked_to_volume_rotation: bool = Field(default=False, alias="linkedToVolumeRotation")

    model_config = {"populate_by_name": True}


class MprPlaneInfo(BaseModel):
    viewport: str
    center_world: tuple[float, float, float] = Field(alias="centerWorld")
    cursor_center_world: tuple[float, float, float] = Field(alias="cursorCenterWorld")
    row_world: tuple[float, float, float] = Field(alias="rowWorld")
    col_world: tuple[float, float, float] = Field(alias="colWorld")
    normal_world: tuple[float, float, float] = Field(alias="normalWorld")
    pixel_spacing_row_mm: float = Field(alias="pixelSpacingRowMm")
    pixel_spacing_col_mm: float = Field(alias="pixelSpacingColMm")
    output_shape: tuple[int, int] = Field(alias="outputShape")
    row: tuple[float, float, float]
    col: tuple[float, float, float]
    normal: tuple[float, float, float]
    is_oblique: bool = Field(alias="isOblique")

    model_config = {"populate_by_name": True}


class ViewExportPointPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class ViewExportMeasurementOverlayPayload(BaseModel):
    measurement_id: str = Field(alias="measurementId")
    tool_type: str = Field(alias="toolType")
    points: list[ViewExportPointPayload]
    label_lines: list[str] = Field(alias="labelLines", default_factory=list)

    model_config = {"populate_by_name": True}


class ViewExportAnnotationOverlayPayload(BaseModel):
    annotation_id: str = Field(alias="annotationId")
    tool_type: str = Field(alias="toolType")
    points: list[ViewExportPointPayload]
    text: str = ""
    color: str = "#ffd166"
    size: str = "md"

    model_config = {"populate_by_name": True}


class ViewExportOverlaysPayload(BaseModel):
    annotations: list[ViewExportAnnotationOverlayPayload] = Field(default_factory=list)
    measurements: list[ViewExportMeasurementOverlayPayload] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ViewExportRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="Server-side view ID to render for export.")
    export_format: ExportFormat = Field(alias="exportFormat", description="Export container format: png or dicom.")
    overlays: ViewExportOverlaysPayload = Field(
        default_factory=ViewExportOverlaysPayload,
        description="Frontend overlay geometry to burn into the exported image.",
    )
    overlay_mode: str | None = Field(default=None, alias="overlayMode", description="Reserved overlay rendering mode.")
    preserve_source_dicom: bool = Field(
        default=True,
        alias="preserveSourceDicom",
        description="When possible, copy patient/study metadata into Secondary Capture exports.",
    )

    model_config = {"populate_by_name": True}


class ScaleBarInfo(BaseModel):
    length_norm: float = Field(alias="lengthNorm")
    label: str

    model_config = {"populate_by_name": True}


class OrientationInfo(BaseModel):
    top: str | None = None
    right: str | None = None
    bottom: str | None = None
    left: str | None = None
    volume_quaternion: tuple[float, float, float, float] | None = Field(default=None, alias="volumeQuaternion")

    model_config = {"populate_by_name": True}


class VolumeLayerConfig(BaseModel):
    key: str
    label: str
    enabled: bool = True
    ww: float
    wl: float
    opacity: float = Field(ge=0.0, le=1.0)
    color_start: str = Field(alias="colorStart")
    color_end: str = Field(alias="colorEnd")

    model_config = {"populate_by_name": True}


class VolumeLightingConfig(BaseModel):
    shading: bool = True
    interpolation: VolumeInterpolationMode = "linear"
    ambient: float = Field(default=0.18, ge=0.0, le=1.0)
    diffuse: float = Field(default=0.82, ge=0.0, le=1.0)
    specular: float = Field(default=0.12, ge=0.0, le=1.0)
    roughness: float = Field(default=0.85, ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class VolumeRenderConfig(BaseModel):
    preset: str
    blend_mode: VolumeBlendMode = Field(alias="blendMode")
    layers: list[VolumeLayerConfig]
    lighting: VolumeLightingConfig = Field(default_factory=VolumeLightingConfig)

    model_config = {"populate_by_name": True}


class ViewTransformPayload(BaseModel):
    rotation_degrees: int = Field(default=0, alias="rotationDegrees")
    hor_flip: bool = Field(default=False, alias="horFlip")
    ver_flip: bool = Field(default=False, alias="verFlip")

    model_config = {"populate_by_name": True}


class MprMipViewportConfig(BaseModel):
    thickness: int = Field(default=12, ge=1, le=512)


class MprMipConfig(BaseModel):
    enabled: bool = False
    algorithm: MprMipAlgorithm = "maximum"
    viewports: dict[str, MprMipViewportConfig] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ViewImageResponse(BaseModel):
    slice_info: SliceInfo = Field(alias="slice_info")
    window_info: WindowInfo = Field(alias="window_info")
    image_format: ImageFormat = Field(alias="imageFormat")
    view_id: str = Field(alias="viewId")
    mpr_crosshair: MprCrosshairInfo | None = Field(default=None, alias="mpr_crosshair")
    mpr_cursor: MprCursorInfo | None = Field(default=None, alias="mprCursor")
    mpr_frame: MprFrameInfo | None = Field(default=None, alias="mprFrame")
    mpr_plane: MprPlaneInfo | None = Field(default=None, alias="mprPlane")
    scale_bar: ScaleBarInfo | None = Field(default=None, alias="scaleBar")
    corner_info: CornerInfoPayload | None = Field(default=None, alias="cornerInfo")
    measurements: list["MeasurementOverlayPayload"] = Field(default_factory=list)
    orientation: OrientationInfo | None = None
    transform: ViewTransformPayload | None = None
    color: ViewColorInfo | None = None
    mpr_mip_config: MprMipConfig | None = Field(default=None, alias="mprMipConfig")
    volume_preset: str | None = Field(default=None, alias="volumePreset")
    volume_config: VolumeRenderConfig | None = Field(default=None, alias="volumeConfig")

    model_config = {"populate_by_name": True}


class ViewHoverRequest(BaseModel):
    view_id: str = Field(alias="viewId")
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class ViewHoverResponse(BaseModel):
    view_id: str = Field(alias="viewId")
    row: int
    col: int

    model_config = {"populate_by_name": True}


class MeasurementPointPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class MeasurementOverlayPayload(BaseModel):
    measurement_id: str = Field(alias="measurementId")
    tool_type: str = Field(alias="toolType")
    points: list[MeasurementPointPayload]
    label_lines: list[str] = Field(alias="labelLines", default_factory=list)

    model_config = {"populate_by_name": True}


class ViewOperationRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="Server-side view ID that receives the interaction.")
    op_type: ViewOperationType = Field(alias="opType", description="Interaction type such as scroll, window, crosshair, or measurement.")
    measurement_id: str | None = Field(default=None, alias="measurementId")
    viewport_key: str | None = Field(default=None, alias="viewportKey")
    sub_op_type: str | None = Field(default=None, alias="subOpType")
    action_type: ViewActionType | None = Field(default=None, alias="actionType")
    x: float | None = None
    y: float | None = None
    line: MprCrosshairLine | None = None
    points: list[MeasurementPointPayload] | None = Field(default=None, description="Normalized points used by measurement and ROI operations.")
    zoom: float | None = None
    delta: int | None = None
    ww: float | None = None
    wl: float | None = None
    pseudocolor_preset: str | None = Field(default=None, alias="pseudocolorPreset")
    mpr_mip_config: MprMipConfig | None = Field(default=None, alias="mprMipConfig")
    source_view_id: str | None = Field(default=None, alias="sourceViewId")
    rotation_degrees: int | None = Field(default=None, alias="rotationDegrees")
    hor_flip: bool | None = Field(default=None, alias="hor_flip")
    ver_flip: bool | None = Field(default=None, alias="ver_flip")
    volume_config: VolumeRenderConfig | None = Field(default=None, alias="volumeConfig", description="3D transfer-function and lighting settings.")

    model_config = {"populate_by_name": True}


class MtfMetricsPayload(BaseModel):
    mtf50: float | None = None
    mtf10: float | None = None
    fwhm_w: float | None = Field(default=None, alias="fwhmW")
    fwhm_h: float | None = Field(default=None, alias="fwhmH")
    peak_value: float | None = Field(default=None, alias="peakValue")
    sample_count: int | None = Field(default=None, alias="sampleCount")
    unit: str | None = None

    model_config = {"populate_by_name": True}


class MtfCurvePointPayload(BaseModel):
    frequency: float
    value: float


class ViewMtfAnalyzeRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="2D view ID to analyze.")
    viewport_key: str = Field(alias="viewportKey", description="Frontend viewport key used to route analysis results.")
    points: list[MeasurementPointPayload] = Field(description="Normalized ROI points from the frontend MTF overlay.")

    model_config = {"populate_by_name": True}


class ViewMtfAnalyzeResponse(BaseModel):
    view_id: str = Field(alias="viewId")
    viewport_key: str = Field(alias="viewportKey")
    points: list[MeasurementPointPayload]
    metrics: MtfMetricsPayload
    curve: list[MtfCurvePointPayload]
    is_placeholder: bool = Field(default=True, alias="isPlaceholder")

    model_config = {"populate_by_name": True}


QaWaterRoiKind = Literal["water", "air"]


class QaWaterRoiPayload(BaseModel):
    id: str
    label: str
    kind: QaWaterRoiKind
    center: MeasurementPointPayload
    radius: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class QaWaterAccuracyMetricsPayload(BaseModel):
    center_mean: float = Field(alias="centerMean")
    deviation_hu: float = Field(alias="deviationHu")
    target_hu: float = Field(default=0.0, alias="targetHu")
    unit: str = "HU"

    model_config = {"populate_by_name": True}


class QaWaterRoiStatsPayload(BaseModel):
    id: str
    label: str
    kind: QaWaterRoiKind
    area: float
    width: float
    height: float
    mean: float
    std_dev: float = Field(alias="stdDev")
    sample_count: int = Field(alias="sampleCount")
    deviation_from_center: float | None = Field(default=None, alias="deviationFromCenter")
    size_unit: str = Field(alias="sizeUnit")
    area_unit: str = Field(alias="areaUnit")
    unit: str = "HU"

    model_config = {"populate_by_name": True}


class QaWaterUniformityMetricsPayload(BaseModel):
    center_mean: float = Field(alias="centerMean")
    max_deviation: float = Field(alias="maxDeviation")
    peripheral_means: list[float] = Field(alias="peripheralMeans")
    roi_stats: list[QaWaterRoiStatsPayload] = Field(default_factory=list, alias="roiStats")
    unit: str = "HU"

    model_config = {"populate_by_name": True}


class QaWaterNoiseMetricsPayload(BaseModel):
    std_dev: float = Field(alias="stdDev")
    unit: str = "HU"

    model_config = {"populate_by_name": True}


class QaWaterMetricsPayload(BaseModel):
    accuracy: QaWaterAccuracyMetricsPayload | None = None
    uniformity: QaWaterUniformityMetricsPayload | None = None
    noise: QaWaterNoiseMetricsPayload | None = None

    model_config = {"populate_by_name": True}


class ViewQaWaterAnalyzeRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="2D view ID to analyze.")
    viewport_key: str = Field(alias="viewportKey", description="Frontend viewport key used to route QA results.")
    metrics: list[str] = Field(
        default_factory=list,
        description="Optional metric subset. Empty means the service may return all supported water phantom metrics.",
    )

    model_config = {"populate_by_name": True}


class ViewQaWaterAnalyzeResponse(BaseModel):
    view_id: str = Field(alias="viewId")
    viewport_key: str = Field(alias="viewportKey")
    rois: list[QaWaterRoiPayload]
    metrics: QaWaterMetricsPayload = Field(default_factory=QaWaterMetricsPayload)
    status: Literal["ready", "error"] = "ready"
    message: str | None = None

    model_config = {"populate_by_name": True}

