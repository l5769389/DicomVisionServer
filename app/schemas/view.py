from typing import Literal

from pydantic import BaseModel, Field


ViewType = Literal["Stack", "MPR", "3D"]
ImageFormat = Literal["png", "jpeg"]


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
    op_type: str = Field(alias="opType")
    size: ViewSize
    view_id: str = Field(alias="viewId")

    model_config = {"populate_by_name": True}


class SliceInfo(BaseModel):
    current: int
    total: int


class WindowInfo(BaseModel):
    ww: float | None = None
    wl: float | None = None


class ViewImageResponse(BaseModel):
    slice_info: SliceInfo = Field(alias="slice_info")
    window_info: WindowInfo = Field(alias="window_info")
    image_format: ImageFormat = Field(alias="imageFormat")
    view_id: str = Field(alias="viewId")

    model_config = {"populate_by_name": True}


class ViewOperationRequest(BaseModel):
    view_id: str = Field(alias="viewId")
    op_type: str = Field(alias="opType")
    sub_op_type: str | None = Field(default=None, alias="subOpType")
    action_type: str | None = Field(default=None, alias="actionType")
    x: float | None = None
    y: float | None = None
    zoom: float | None = None
    scroll: int | None = None
    hor_flip: bool | None = Field(default=None, alias="hor_flip")
    ver_flip: bool | None = Field(default=None, alias="ver_flip")

    model_config = {"populate_by_name": True}
