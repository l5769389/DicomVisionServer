from pydantic import BaseModel, Field


class SeriesSummary(BaseModel):
    series_id: str = Field(alias="seriesId")
    series_instance_uid: str | None = Field(default=None, alias="seriesInstanceUid")
    study_instance_uid: str | None = Field(default=None, alias="studyInstanceUid")
    patient_id: str | None = Field(default=None, alias="patientId")
    modality: str | None = None
    series_description: str | None = Field(default=None, alias="seriesDescription")
    instance_count: int = Field(alias="instanceCount")
    width: int | None = None
    height: int | None = None
    folder_path: str = Field(alias="folderPath")

    model_config = {"populate_by_name": True}


class LoadFolderRequest(BaseModel):
    folder_path: str = Field(alias="folderPath")

    model_config = {"populate_by_name": True}


class LoadFolderResponse(BaseModel):
    series_id: str | None = Field(default=None, alias="seriesId")
    series_list: list[SeriesSummary] = Field(default_factory=list, alias="seriesList")

    model_config = {"populate_by_name": True}


class CornerInfoPayload(BaseModel):
    top_left: list[str] = Field(default_factory=list, alias="topLeft")
    top_right: list[str] = Field(default_factory=list, alias="topRight")
    bottom_left: list[str] = Field(default_factory=list, alias="bottomLeft")
    bottom_right: list[str] = Field(default_factory=list, alias="bottomRight")

    model_config = {"populate_by_name": True}


class CornerInfoRequest(BaseModel):
    series_id: str = Field(alias="seriesId")

    model_config = {"populate_by_name": True}


class CornerInfoResponse(BaseModel):
    corner_info: CornerInfoPayload = Field(alias="cornerInfo")

    model_config = {"populate_by_name": True}
