from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.dicom import CornerInfoPayload


ViewType = Literal["Stack", "MPR", "3D", "AX", "COR", "SAG"]
ImageFormat = Literal["png", "jpeg"]
ViewSetSizeOperationType = Literal["setSize"]
ViewOperationType = Literal["scroll", "crosshair", "pan", "zoom", "window", "rotate3d", "reset"]
ViewActionType = Literal["start", "move", "end"]


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


class MprCrosshairInfo(BaseModel):
    center_x: float = Field(alias="centerX")
    center_y: float = Field(alias="centerY")
    hit_radius: float = Field(alias="hitRadius")
    horizontal_position: float | None = Field(default=None, alias="horizontalPosition")
    vertical_position: float | None = Field(default=None, alias="verticalPosition")

    model_config = {"populate_by_name": True}


class OrientationInfo(BaseModel):
    top: str | None = None
    right: str | None = None
    bottom: str | None = None
    left: str | None = None
    volume_quaternion: tuple[float, float, float, float] | None = Field(default=None, alias="volumeQuaternion")

    model_config = {"populate_by_name": True}


class ViewImageResponse(BaseModel):
    slice_info: SliceInfo = Field(alias="slice_info")
    window_info: WindowInfo = Field(alias="window_info")
    image_format: ImageFormat = Field(alias="imageFormat")
    view_id: str = Field(alias="viewId")
    mpr_crosshair: MprCrosshairInfo | None = Field(default=None, alias="mpr_crosshair")
    corner_info: CornerInfoPayload | None = Field(default=None, alias="cornerInfo")
    orientation: OrientationInfo | None = None

    model_config = {"populate_by_name": True}


class ViewOperationRequest(BaseModel):
    view_id: str = Field(alias="viewId")
    op_type: ViewOperationType = Field(alias="opType")
    sub_op_type: str | None = Field(default=None, alias="subOpType")
    action_type: ViewActionType | None = Field(default=None, alias="actionType")
    x: float | None = None
    y: float | None = None
    zoom: float | None = None
    delta: int | None = None
    hor_flip: bool | None = Field(default=None, alias="hor_flip")
    ver_flip: bool | None = Field(default=None, alias="ver_flip")

    model_config = {"populate_by_name": True}
