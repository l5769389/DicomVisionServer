from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.dicom import CornerInfoPayload


ViewType = Literal["Stack", "MPR", "3D", "AX", "COR", "SAG"]
ImageFormat = Literal["png", "jpeg"]
ViewSetSizeOperationType = Literal["setSize"]
ViewOperationType = Literal["scroll", "crosshair", "pan", "zoom", "window", "pseudocolor", "transform2d", "rotate3d", "reset", "volumePreset", "volumeConfig", "mprMipConfig", "measurement"]
ViewActionType = Literal["start", "move", "end", "delete"]
VolumeBlendMode = Literal["composite", "mip"]
VolumeInterpolationMode = Literal["nearest", "linear", "cubic"]
MprMipAlgorithm = Literal["maximum", "minimum", "average", "sum"]


class ViewCreateRequest(BaseModel):
    series_id: str = Field(alias="seriesId")
    view_type: ViewType = Field(alias="viewType")

    model_config = {"populate_by_name": True}


class ViewCreateResponse(BaseModel):
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
    op_type: ViewSetSizeOperationType = Field(alias="opType")
    size: ViewSize
    view_id: str = Field(alias="viewId")

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
    view_id: str = Field(alias="viewId")
    op_type: ViewOperationType = Field(alias="opType")
    measurement_id: str | None = Field(default=None, alias="measurementId")
    viewport_key: str | None = Field(default=None, alias="viewportKey")
    sub_op_type: str | None = Field(default=None, alias="subOpType")
    action_type: ViewActionType | None = Field(default=None, alias="actionType")
    x: float | None = None
    y: float | None = None
    points: list[MeasurementPointPayload] | None = None
    zoom: float | None = None
    delta: int | None = None
    pseudocolor_preset: str | None = Field(default=None, alias="pseudocolorPreset")
    mpr_mip_config: MprMipConfig | None = Field(default=None, alias="mprMipConfig")
    rotation_degrees: int | None = Field(default=None, alias="rotationDegrees")
    hor_flip: bool | None = Field(default=None, alias="hor_flip")
    ver_flip: bool | None = Field(default=None, alias="ver_flip")
    volume_config: VolumeRenderConfig | None = Field(default=None, alias="volumeConfig")

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
    view_id: str = Field(alias="viewId")
    viewport_key: str = Field(alias="viewportKey")
    points: list[MeasurementPointPayload]

    model_config = {"populate_by_name": True}


class ViewMtfAnalyzeResponse(BaseModel):
    view_id: str = Field(alias="viewId")
    viewport_key: str = Field(alias="viewportKey")
    points: list[MeasurementPointPayload]
    metrics: MtfMetricsPayload
    curve: list[MtfCurvePointPayload]
    is_placeholder: bool = Field(default=True, alias="isPlaceholder")

    model_config = {"populate_by_name": True}

