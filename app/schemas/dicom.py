from typing import Literal

from pydantic import BaseModel, Field


class FourDPhaseItem(BaseModel):
    phase_index: int = Field(alias="phaseIndex")
    label: str
    series_id: str | None = Field(default=None, alias="seriesId")
    image_src: str = Field(default="", alias="imageSrc")
    viewport_images: dict[str, str] = Field(default_factory=dict, alias="viewportImages")
    status: Literal["pending", "ready", "error"] = "pending"

    model_config = {"populate_by_name": True}


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
    is_four_d_series: bool = Field(default=False, alias="isFourDSeries")
    four_d_phase_count: int | None = Field(default=None, alias="fourDPhaseCount")
    four_d_phases: list[FourDPhaseItem] | None = Field(default=None, alias="fourDPhases")

    model_config = {"populate_by_name": True}


class LoadFolderRequest(BaseModel):
    folder_path: str = Field(alias="folderPath")

    model_config = {"populate_by_name": True}


class LoadFolderResponse(BaseModel):
    series_id: str | None = Field(default=None, alias="seriesId")
    series_list: list[SeriesSummary] = Field(default_factory=list, alias="seriesList")

    model_config = {"populate_by_name": True}


class LoadSampleResponse(LoadFolderResponse):
    sample_path: str = Field(alias="samplePath")

    model_config = {"populate_by_name": True}


class FourDPhasesRequest(BaseModel):
    series_id: str = Field(alias="seriesId")

    model_config = {"populate_by_name": True}


class FourDPhasesResponse(BaseModel):
    series_id: str = Field(alias="seriesId")
    is_four_d_series: bool = Field(default=False, alias="isFourDSeries")
    four_d_phase_count: int = Field(default=0, alias="fourDPhaseCount")
    four_d_phases: list[FourDPhaseItem] = Field(default_factory=list, alias="fourDPhases")

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


class DicomTagsRequest(BaseModel):
    series_id: str = Field(alias="seriesId")
    index: int = 0

    model_config = {"populate_by_name": True}


class DicomTagItem(BaseModel):
    tag: str
    keyword: str
    name: str
    vr: str
    value: str
    depth: int = 0


class DicomTagsResponse(BaseModel):
    series_id: str = Field(alias="seriesId")
    index: int
    total: int
    instance_number: int | None = Field(default=None, alias="instanceNumber")
    sop_instance_uid: str | None = Field(default=None, alias="sopInstanceUid")
    file_path: str | None = Field(default=None, alias="filePath")
    items: list[DicomTagItem] = Field(default_factory=list)

    model_config = {"populate_by_name": True}
