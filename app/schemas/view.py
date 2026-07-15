from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator

from app.schemas.dicom import CornerInfoPayload


ViewType = Literal[
    "Stack",
    "MPR",
    "3D",
    "PET",
    "AX",
    "COR",
    "SAG",
    "FusionCTAxial",
    "FusionPETAxial",
    "FusionOverlayAxial",
    "FusionPETCoronalMip",
]
ImageFormat = Literal["png", "jpeg", "webp"]
VIEW_IMAGE_FORMATS = {"png", "jpeg", "webp"}


def normalize_image_format(value: Any, default: ImageFormat = "webp") -> ImageFormat:
    text = str(value or default).strip().lower()
    if text == "jpg":
        text = "jpeg"
    if text in VIEW_IMAGE_FORMATS:
        return cast(ImageFormat, text)
    return default
ExportFormat = Literal["png", "dicom", "dicom-sr", "dicom-gsps"]
FusionRegistrationExportMode = Literal["newDicom", "br"]
ViewSetSizeOperationType = Literal["setSize"]
ViewOperationType = Literal["scroll", "crosshair", "pan", "zoom", "window", "pseudocolor", "transform2d", "rotate3d", "reset", "volumePreset", "volumeConfig", "render3dMode", "surfaceConfig", "volumeRenderOptions", "volumeClip", "mprMipConfig", "mprSegmentation", "mprOblique", "mprCrosshairMode", "mprStateSync", "measurement", "annotation", "fusionRegistration", "fusionConfig", "petConfig"]
ViewActionType = Literal["start", "move", "end", "delete"]
VolumeBlendMode = Literal["composite", "mip"]
VolumeInterpolationMode = Literal["nearest", "linear", "cubic"]
Render3DMode = Literal["volume", "surface"]
VolumeClipMode = Literal["inside", "outside"]
MprMipAlgorithm = Literal["maximum", "minimum", "average", "sum"]
MprCrosshairLine = Literal["horizontal", "vertical"]
MprCrosshairMode = Literal["orthogonal", "double-oblique"]


class ViewCreateRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered series ID to view.")
    view_type: ViewType = Field(alias="viewType", description="Requested view type: Stack, MPR, 3D, or a concrete MPR viewport.")
    secondary_series_id: str | None = Field(
        default=None,
        alias="secondarySeriesId",
        description="Optional second series ID for paired views such as PET/CT fusion.",
    )
    fusion_pane_role: str | None = Field(
        default=None,
        alias="fusionPaneRole",
        description="Optional frontend pane role for PET/CT fusion routing.",
    )
    view_group_key: str | None = Field(
        default=None,
        alias="viewGroupKey",
        description="Optional key for sharing MPR or fusion state across related views.",
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
    image_format: ImageFormat = Field(default="webp", alias="imageFormat")

    @field_validator("image_format", mode="before")
    @classmethod
    def _normalize_image_format(cls, value: Any) -> ImageFormat:
        return normalize_image_format(value)

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
    horizontal_slab_offset_x: float | None = Field(default=None, alias="horizontalSlabOffsetX")
    horizontal_slab_offset_y: float | None = Field(default=None, alias="horizontalSlabOffsetY")
    vertical_slab_offset_x: float | None = Field(default=None, alias="verticalSlabOffsetX")
    vertical_slab_offset_y: float | None = Field(default=None, alias="verticalSlabOffsetY")

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
    pixel_spacing_normal_mm: float = Field(default=1.0, alias="pixelSpacingNormalMm")
    output_shape: tuple[int, int] = Field(alias="outputShape")
    row: tuple[float, float, float]
    col: tuple[float, float, float]
    normal: tuple[float, float, float]
    image_to_canvas_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = Field(default=None, alias="imageToCanvasMatrix")
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
    export_format: ExportFormat = Field(alias="exportFormat", description="Export container format: png, dicom, dicom-sr, or dicom-gsps.")
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


class FusionRegistrationExportRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="Overlay fusion view ID whose registration state should be exported.")
    mode: FusionRegistrationExportMode = Field(description="Registration export mode: new derived DICOM series or .br sidecar.")
    series_description: str | None = Field(default=None, alias="seriesDescription")
    output_directory: str = Field(alias="outputDirectory")

    model_config = {"populate_by_name": True}


class FusionRegistrationArtifactExportRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="Overlay fusion view ID whose registration state should be exported.")
    mode: FusionRegistrationExportMode = Field(description="Registration export mode: new derived DICOM series zip or .br sidecar.")
    series_description: str | None = Field(default=None, alias="seriesDescription")

    model_config = {"populate_by_name": True}


class FusionRegistrationExportResponse(BaseModel):
    mode: FusionRegistrationExportMode
    directory_path: str = Field(alias="directoryPath")
    file_path: str | None = Field(default=None, alias="filePath")
    file_count: int = Field(alias="fileCount")
    series_description: str = Field(alias="seriesDescription")
    pet_unit: str = Field(alias="petUnit")
    pet_unit_label: str = Field(alias="petUnitLabel")

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


class SurfaceRenderConfig(BaseModel):
    preset: str = "bone"
    iso_value: float = Field(default=240.0, ge=-2000.0, le=4000.0, alias="isoValue")
    smoothing: float = Field(default=0.28, ge=0.0, le=1.0)
    decimation: float = Field(default=0.18, ge=0.0, le=0.9)
    color: str = "#f3eadb"
    ambient: float = Field(default=0.2, ge=0.0, le=1.0)
    diffuse: float = Field(default=0.76, ge=0.0, le=1.0)
    specular: float = Field(default=0.36, ge=0.0, le=1.0)
    roughness: float = Field(default=0.34, ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class VolumeClipPointPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class VolumeClipState(BaseModel):
    mode: VolumeClipMode
    points: list[VolumeClipPointPayload] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class VolumeRenderOptions(BaseModel):
    remove_bed: bool = Field(default=False, alias="removeBed")
    clip: VolumeClipState | None = None

    model_config = {"populate_by_name": True}


class ViewTransformPayload(BaseModel):
    rotation_degrees: int = Field(default=0, alias="rotationDegrees")
    hor_flip: bool = Field(default=False, alias="horFlip")
    ver_flip: bool = Field(default=False, alias="verFlip")
    zoom: float = 1.0
    offset_x: float = Field(default=0.0, alias="offsetX")
    offset_y: float = Field(default=0.0, alias="offsetY")

    model_config = {"populate_by_name": True}


class MprMipViewportConfig(BaseModel):
    thickness: int = Field(default=10, ge=0, le=100)


class MprMipConfig(BaseModel):
    enabled: bool = False
    algorithm: MprMipAlgorithm = "maximum"
    viewports: dict[str, MprMipViewportConfig] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class MprSegmentationVoiBox(BaseModel):
    x_min: float = Field(default=0.0, ge=0.0, le=1.0, alias="xMin")
    x_max: float = Field(default=1.0, ge=0.0, le=1.0, alias="xMax")
    y_min: float = Field(default=0.0, ge=0.0, le=1.0, alias="yMin")
    y_max: float = Field(default=1.0, ge=0.0, le=1.0, alias="yMax")
    z_min: float = Field(default=0.0, ge=0.0, le=1.0, alias="zMin")
    z_max: float = Field(default=1.0, ge=0.0, le=1.0, alias="zMax")

    model_config = {"populate_by_name": True}


class MprThresholdRegionStats(BaseModel):
    hu_mean: float | None = Field(default=None, alias="huMean")
    hu_min: float | None = Field(default=None, alias="huMin")
    hu_max: float | None = Field(default=None, alias="huMax")
    hu_std_dev: float | None = Field(default=None, alias="huStdDev")
    volume_cm3: float = Field(default=0.0, ge=0.0, alias="volumeCm3")
    sample_count: int = Field(default=0, ge=0, alias="sampleCount")
    effective_threshold_hu: float | None = Field(default=None, alias="effectiveThresholdHu")

    model_config = {"populate_by_name": True}


class MprThresholdRegionBox(BaseModel):
    center_world: tuple[float, float, float] = Field(alias="centerWorld")
    row_world: tuple[float, float, float] = Field(alias="rowWorld")
    col_world: tuple[float, float, float] = Field(alias="colWorld")
    normal_world: tuple[float, float, float] = Field(alias="normalWorld")
    width_mm: float = Field(default=1.0, gt=0.0, alias="widthMm")
    height_mm: float = Field(default=1.0, gt=0.0, alias="heightMm")
    depth_mm: float = Field(default=1.0, gt=0.0, alias="depthMm")
    source_viewport: str = Field(default="mpr-ax", alias="sourceViewport")

    model_config = {"populate_by_name": True}


class MprThresholdRegion(BaseModel):
    id: str
    enabled: bool = True
    label: str = ""
    threshold_hu: float = Field(default=300.0, ge=-1024.0, le=3071.0, alias="thresholdHu")
    threshold_mode: str = Field(default="hu", alias="thresholdMode")
    threshold_percentile: float = Field(default=80.0, ge=0.0, le=100.0, alias="thresholdPercentile")
    color: str = "#ff4df8"
    box: MprThresholdRegionBox
    stats: MprThresholdRegionStats | None = None

    model_config = {"populate_by_name": True}


class MprVoiSphereStats(BaseModel):
    hu_mean: float | None = Field(default=None, alias="huMean")
    hu_min: float | None = Field(default=None, alias="huMin")
    hu_max: float | None = Field(default=None, alias="huMax")
    hu_std_dev: float | None = Field(default=None, alias="huStdDev")
    volume_cm3: float = Field(default=0.0, ge=0.0, alias="volumeCm3")
    sample_count: int = Field(default=0, ge=0, alias="sampleCount")

    model_config = {"populate_by_name": True}


class MprVoiSphere(BaseModel):
    id: str | None = None
    label: str = ""
    enabled: bool = True
    center_world: tuple[float, float, float] = Field(alias="centerWorld")
    radius_mm: float = Field(default=10.0, gt=0.0, alias="radiusMm")
    color: str = "#22d3ee"
    stats: MprVoiSphereStats | None = None

    model_config = {"populate_by_name": True}


class MprSegmentationConfig(BaseModel):
    enabled: bool = False
    client_revision: int = Field(default=0, ge=0, alias="clientRevision")
    selected_region_id: str | None = Field(default=None, alias="selectedRegionId")
    selected_voi: bool = Field(default=False, alias="selectedVoi")
    selected_voi_id: str | None = Field(default=None, alias="selectedVoiId")
    threshold_regions: list[MprThresholdRegion] = Field(default_factory=list, alias="thresholdRegions")
    voi_spheres: list[MprVoiSphere] = Field(default_factory=list, alias="voiSpheres")
    voi_sphere: MprVoiSphere | None = Field(default=None, alias="voiSphere")
    lower_hu: float | None = Field(default=None, ge=-1024.0, le=3071.0, alias="lowerHu")
    upper_hu: float | None = Field(default=None, ge=-1024.0, le=3071.0, alias="upperHu")
    opacity: float = Field(default=0.45, ge=0.0, le=1.0)
    color: str = "#ff4df8"
    voi_box: MprSegmentationVoiBox | None = Field(default=None, alias="voiBox")

    model_config = {"populate_by_name": True}


class MprSegmentationOverlayRect(BaseModel):
    x_min: float = Field(default=0.0, ge=0.0, le=1.0, alias="xMin")
    y_min: float = Field(default=0.0, ge=0.0, le=1.0, alias="yMin")
    x_max: float = Field(default=1.0, ge=0.0, le=1.0, alias="xMax")
    y_max: float = Field(default=1.0, ge=0.0, le=1.0, alias="yMax")

    model_config = {"populate_by_name": True}


class MprSegmentationOverlaySamples(BaseModel):
    points: list[float] = Field(default_factory=list)
    total_count: int = Field(default=0, ge=0, alias="totalCount")
    sampled_count: int = Field(default=0, ge=0, alias="sampledCount")

    model_config = {"populate_by_name": True}


class MprSegmentationOverlayRegion(BaseModel):
    region_id: str = Field(alias="regionId")
    visible: bool = False
    rect: MprSegmentationOverlayRect | None = None
    sample_revision: int = Field(default=0, ge=0, alias="sampleRevision")
    samples: MprSegmentationOverlaySamples | None = None

    model_config = {"populate_by_name": True}


class MprSegmentationOverlay(BaseModel):
    regions: list[MprSegmentationOverlayRegion] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class FusionRegistrationInfo(BaseModel):
    translate_row_mm: float = Field(default=0.0, alias="translateRowMm")
    translate_col_mm: float = Field(default=0.0, alias="translateColMm")
    rotation_degrees: float = Field(default=0.0, alias="rotationDegrees")
    saved: bool = False

    model_config = {"populate_by_name": True}


class FusionInfo(BaseModel):
    pane_role: str = Field(alias="paneRole")
    ct_series_id: str = Field(alias="ctSeriesId")
    pet_series_id: str = Field(alias="petSeriesId")
    pet_pseudocolor_preset: str = Field(alias="petPseudocolorPreset")
    pet_unit: str = Field(default="SUVbw", alias="petUnit")
    pet_unit_label: str = Field(default="g/ml (SUVbw)", alias="petUnitLabel")
    pet_window_min: float | None = Field(default=None, alias="petWindowMin")
    pet_window_max: float | None = Field(default=None, alias="petWindowMax")
    alpha: float
    revision: int
    registration: FusionRegistrationInfo

    model_config = {"populate_by_name": True}


class FusionCompositeLayerInfo(BaseModel):
    key: str
    role: str
    image_format: str = Field(default="webp", alias="imageFormat")

    model_config = {"populate_by_name": True}


class FusionCompositeInfo(BaseModel):
    mode: str = "ctPetLayers"
    revision: int
    alpha: float
    registration: FusionRegistrationInfo
    width: int
    height: int
    layers: list[FusionCompositeLayerInfo]
    primary_image_unchanged: bool = Field(default=False, alias="primaryImageUnchanged")

    model_config = {"populate_by_name": True}


class PetInfo(BaseModel):
    series_id: str = Field(alias="seriesId")
    pet_unit: str = Field(default="SUVbw", alias="petUnit")
    pet_unit_label: str = Field(default="g/ml (SUVbw)", alias="petUnitLabel")
    pet_window_min: float | None = Field(default=None, alias="petWindowMin")
    pet_window_max: float | None = Field(default=None, alias="petWindowMax")
    pseudocolor_preset: str = Field(default="bwinverse", alias="pseudocolorPreset")

    model_config = {"populate_by_name": True}


class FusionProjectionInfo(BaseModel):
    pane_role: str = Field(alias="paneRole")
    reference_world: tuple[float, float, float] = Field(alias="referenceWorld")
    reference_x: float = Field(alias="referenceX")
    reference_y: float = Field(alias="referenceY")
    normalized_to_world_origin: tuple[float, float, float] = Field(alias="normalizedToWorldOrigin")
    normalized_to_world_x: tuple[float, float, float] = Field(alias="normalizedToWorldX")
    normalized_to_world_y: tuple[float, float, float] = Field(alias="normalizedToWorldY")
    world_to_normalized_x: tuple[float, float, float, float] = Field(alias="worldToNormalizedX")
    world_to_normalized_y: tuple[float, float, float, float] = Field(alias="worldToNormalizedY")

    model_config = {"populate_by_name": True}


class ViewImageResponse(BaseModel):
    slice_info: SliceInfo = Field(alias="slice_info")
    window_info: WindowInfo = Field(alias="window_info")
    image_format: ImageFormat = Field(alias="imageFormat")
    view_id: str = Field(alias="viewId")
    mpr_crosshair: MprCrosshairInfo | None = Field(default=None, alias="mpr_crosshair")
    mpr_cursor: MprCursorInfo | None = Field(default=None, alias="mprCursor")
    mpr_frame: MprFrameInfo | None = Field(default=None, alias="mprFrame")
    mpr_revision: int | None = Field(default=None, alias="mprRevision")
    mpr_plane: MprPlaneInfo | None = Field(default=None, alias="mprPlane")
    scale_bar: ScaleBarInfo | None = Field(default=None, alias="scaleBar")
    corner_info: CornerInfoPayload | None = Field(default=None, alias="cornerInfo")
    measurements: list["MeasurementOverlayPayload"] = Field(default_factory=list)
    annotations: list["AnnotationOverlayPayload"] = Field(default_factory=list)
    orientation: OrientationInfo | None = None
    transform: ViewTransformPayload | None = None
    color: ViewColorInfo | None = None
    pet_info: PetInfo | None = Field(default=None, alias="petInfo")
    fusion_info: FusionInfo | None = Field(default=None, alias="fusionInfo")
    fusion_composite: FusionCompositeInfo | None = Field(default=None, alias="fusionComposite")
    fusion_projection: FusionProjectionInfo | None = Field(default=None, alias="fusionProjection")
    mpr_mip_config: MprMipConfig | None = Field(default=None, alias="mprMipConfig")
    mpr_segmentation_config: MprSegmentationConfig | None = Field(default=None, alias="mprSegmentationConfig")
    mpr_segmentation_overlay: MprSegmentationOverlay | None = Field(default=None, alias="mprSegmentationOverlay")
    mpr_crosshair_mode: MprCrosshairMode = Field(default="orthogonal", alias="mprCrosshairMode")
    volume_preset: str | None = Field(default=None, alias="volumePreset")
    volume_config: VolumeRenderConfig | None = Field(default=None, alias="volumeConfig")
    render_3d_mode: Render3DMode | None = Field(default=None, alias="render3dMode")
    surface_config: SurfaceRenderConfig | None = Field(default=None, alias="surfaceConfig")
    volume_render_options: VolumeRenderOptions | None = Field(default=None, alias="volumeRenderOptions")

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
    pixel_value: float | None = Field(default=None, alias="pixelValue")
    value_label: str | None = Field(default=None, alias="valueLabel")
    value_unit: str | None = Field(default=None, alias="valueUnit")
    display_text: str | None = Field(default=None, alias="displayText")

    model_config = {"populate_by_name": True}


class MeasurementPointPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class OverlayPointPayload(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class MeasurementOverlayPayload(BaseModel):
    measurement_id: str = Field(alias="measurementId")
    tool_type: str = Field(alias="toolType")
    points: list[OverlayPointPayload]
    label_lines: list[str] = Field(alias="labelLines", default_factory=list)
    scope: Literal["image", "series"] = "image"
    slice_index: int | None = Field(default=None, alias="sliceIndex")

    model_config = {"populate_by_name": True}


class AnnotationOverlayPayload(BaseModel):
    annotation_id: str = Field(alias="annotationId")
    tool_type: str = Field(alias="toolType")
    points: list[OverlayPointPayload]
    text: str = ""
    color: str = "#ffd166"
    size: str = "md"
    scope: Literal["image", "series"] = "image"
    slice_index: int | None = Field(default=None, alias="sliceIndex")

    model_config = {"populate_by_name": True}


class ViewOperationRequest(BaseModel):
    view_id: str = Field(alias="viewId", description="Server-side view ID that receives the interaction.")
    op_type: ViewOperationType = Field(alias="opType", description="Interaction type such as scroll, window, crosshair, or measurement.")
    image_format: ImageFormat = Field(default="webp", alias="imageFormat")
    measurement_id: str | None = Field(default=None, alias="measurementId")
    annotation_id: str | None = Field(default=None, alias="annotationId")
    viewport_key: str | None = Field(default=None, alias="viewportKey")
    sub_op_type: str | None = Field(default=None, alias="subOpType")
    action_type: ViewActionType | None = Field(default=None, alias="actionType")
    x: float | None = None
    y: float | None = None
    canvas_x: float | None = Field(default=None, alias="canvasX")
    canvas_y: float | None = Field(default=None, alias="canvasY")
    canvas_width: float | None = Field(default=None, alias="canvasWidth")
    canvas_height: float | None = Field(default=None, alias="canvasHeight")
    interaction_id: str | None = Field(default=None, alias="interactionId")
    anchor_x: float | None = Field(default=None, alias="anchorX")
    anchor_y: float | None = Field(default=None, alias="anchorY")
    current_x: float | None = Field(default=None, alias="currentX")
    current_y: float | None = Field(default=None, alias="currentY")
    pivot_x: float | None = Field(default=None, alias="pivotX")
    pivot_y: float | None = Field(default=None, alias="pivotY")
    rotation_delta_degrees: float | None = Field(default=None, alias="rotationDeltaDegrees")
    line: MprCrosshairLine | None = None
    points: list[MeasurementPointPayload] | None = Field(default=None, description="Normalized points used by measurement and ROI operations.")
    zoom: float | None = None
    delta: int | None = None
    ww: float | None = None
    wl: float | None = None
    pseudocolor_preset: str | None = Field(default=None, alias="pseudocolorPreset")
    fusion_alpha: float | None = Field(default=None, alias="fusionAlpha")
    fusion_manual_registration: bool | None = Field(default=None, alias="fusionManualRegistration")
    fusion_pet_unit: str | None = Field(default=None, alias="fusionPetUnit")
    fusion_pet_window_min: float | None = Field(default=None, alias="fusionPetWindowMin")
    fusion_pet_window_max: float | None = Field(default=None, alias="fusionPetWindowMax")
    pet_unit: str | None = Field(default=None, alias="petUnit")
    pet_window_min: float | None = Field(default=None, alias="petWindowMin")
    pet_window_max: float | None = Field(default=None, alias="petWindowMax")
    fusion_registration_file: dict[str, Any] | None = Field(default=None, alias="fusionRegistrationFile")
    mpr_mip_config: MprMipConfig | None = Field(default=None, alias="mprMipConfig")
    mpr_segmentation_config: MprSegmentationConfig | None = Field(default=None, alias="mprSegmentationConfig")
    mpr_crosshair_mode: MprCrosshairMode | None = Field(default=None, alias="mprCrosshairMode")
    tool_type: str | None = Field(default=None, alias="toolType")
    text: str | None = None
    color: str | None = None
    size: str | None = None
    scope: Literal["image", "series"] | None = None
    slice_index: int | None = Field(default=None, alias="sliceIndex")
    source_view_id: str | None = Field(default=None, alias="sourceViewId")
    rotation_degrees: int | None = Field(default=None, alias="rotationDegrees")
    hor_flip: bool | None = Field(default=None, alias="hor_flip")
    ver_flip: bool | None = Field(default=None, alias="ver_flip")
    volume_config: VolumeRenderConfig | None = Field(default=None, alias="volumeConfig", description="3D transfer-function and lighting settings.")
    render_3d_mode: Render3DMode | None = Field(default=None, alias="render3dMode", description="3D renderer mode: volume or surface.")
    surface_config: SurfaceRenderConfig | None = Field(default=None, alias="surfaceConfig", description="3D surface extraction and material settings.")
    volume_render_options: VolumeRenderOptions | None = Field(default=None, alias="volumeRenderOptions", description="3D render-time options such as bed removal and freeform clipping.")
    remove_bed: bool | None = Field(default=None, alias="removeBed", description="Compatibility shortcut for volumeRenderOptions.removeBed.")

    @field_validator("image_format", mode="before")
    @classmethod
    def _normalize_image_format(cls, value: Any) -> ImageFormat:
        return normalize_image_format(value)

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
