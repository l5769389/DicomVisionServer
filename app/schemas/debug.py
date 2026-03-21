from pydantic import BaseModel, Field


class CacheStatsResponse(BaseModel):
    entries: int
    max_entries: int = Field(alias="maxEntries")

    model_config = {"populate_by_name": True}


class SeriesDebugResponse(BaseModel):
    series_id: str = Field(alias="seriesId")
    series_instance_uid: str | None = Field(default=None, alias="seriesInstanceUid")
    instance_count: int = Field(alias="instanceCount")
    folder_path: str = Field(alias="folderPath")

    model_config = {"populate_by_name": True}


class ViewDebugResponse(BaseModel):
    view_id: str = Field(alias="viewId")
    series_id: str = Field(alias="seriesId")
    view_type: str = Field(alias="viewType")
    width: int | None = None
    height: int | None = None
    current_index: int = Field(alias="currentIndex")
    zoom: float
    offset_x: float = Field(alias="offsetX")
    offset_y: float = Field(alias="offsetY")
    hor_flip: bool = Field(alias="horFlip")
    ver_flip: bool = Field(alias="verFlip")

    model_config = {"populate_by_name": True}


class RuntimeStateResponse(BaseModel):
    series: list[SeriesDebugResponse]
    views: list[ViewDebugResponse]
    cache: CacheStatsResponse
