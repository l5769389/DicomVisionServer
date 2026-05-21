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


DicomCompatibilitySeverity = Literal["info", "warning", "error"]


class DicomCompatibilityIssue(BaseModel):
    code: str
    severity: DicomCompatibilitySeverity = "warning"
    title: str
    detail: str | None = None
    affected_instances: int = Field(default=0, alias="affectedInstances")

    model_config = {"populate_by_name": True}


class SeriesSummary(BaseModel):
    series_id: str = Field(alias="seriesId")
    series_instance_uid: str | None = Field(default=None, alias="seriesInstanceUid")
    study_instance_uid: str | None = Field(default=None, alias="studyInstanceUid")
    patient_id: str | None = Field(default=None, alias="patientId")
    patient_name: str | None = Field(default=None, alias="patientName")
    study_date: str | None = Field(default=None, alias="studyDate")
    study_description: str | None = Field(default=None, alias="studyDescription")
    accession_number: str | None = Field(default=None, alias="accessionNumber")
    modality: str | None = None
    series_description: str | None = Field(default=None, alias="seriesDescription")
    instance_count: int = Field(alias="instanceCount")
    width: int | None = None
    height: int | None = None
    thumbnail_src: str = Field(default="", alias="thumbnailSrc")
    thumbnail_url: str = Field(default="", alias="thumbnailUrl")
    folder_path: str = Field(alias="folderPath")
    is_four_d_series: bool = Field(default=False, alias="isFourDSeries")
    four_d_phase_count: int | None = Field(default=None, alias="fourDPhaseCount")
    four_d_phases: list[FourDPhaseItem] | None = Field(default=None, alias="fourDPhases")
    compatibility_issues: list[DicomCompatibilityIssue] = Field(default_factory=list, alias="compatibilityIssues")

    model_config = {"populate_by_name": True}


class LoadFolderRequest(BaseModel):
    folder_path: str = Field(alias="folderPath", description="Local file or folder path to scan for DICOM files.")

    model_config = {"populate_by_name": True}


class LoadFolderResponse(BaseModel):
    series_id: str | None = Field(default=None, alias="seriesId")
    series_list: list[SeriesSummary] = Field(default_factory=list, alias="seriesList")

    model_config = {"populate_by_name": True}


class LoadSampleResponse(LoadFolderResponse):
    sample_path: str = Field(alias="samplePath")

    model_config = {"populate_by_name": True}


class FourDPhasesRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered source series ID.")
    include_preview_images: bool = Field(
        default=False,
        alias="includePreviewImages",
        description="Whether to include preview image URLs in the phase manifest.",
    )
    preview_phase_index: int | None = Field(
        default=None,
        alias="previewPhaseIndex",
        description="Optional phase index to prioritize when generating preview metadata.",
    )

    model_config = {"populate_by_name": True}


class FourDPhasesResponse(BaseModel):
    series_id: str = Field(alias="seriesId")
    is_four_d_series: bool = Field(default=False, alias="isFourDSeries")
    four_d_phase_count: int = Field(default=0, alias="fourDPhaseCount")
    four_d_phases: list[FourDPhaseItem] = Field(default_factory=list, alias="fourDPhases")

    model_config = {"populate_by_name": True}


class FourDPlaybackStartRequest(BaseModel):
    tab_key: str = Field(alias="tabKey")
    phase_index: int = Field(alias="phaseIndex")
    phase_count: int = Field(alias="phaseCount")
    fps: int

    model_config = {"populate_by_name": True}


class FourDPlaybackStopRequest(BaseModel):
    tab_key: str = Field(alias="tabKey")

    model_config = {"populate_by_name": True}


class FourDPlaybackFpsRequest(BaseModel):
    tab_key: str = Field(alias="tabKey")
    fps: int

    model_config = {"populate_by_name": True}


class FourDPlaybackPhaseEvent(BaseModel):
    tab_key: str = Field(alias="tabKey")
    phase_index: int = Field(alias="phaseIndex")

    model_config = {"populate_by_name": True}


class FourDPlaybackStateEvent(BaseModel):
    tab_key: str = Field(alias="tabKey")
    is_playing: bool = Field(alias="isPlaying")
    fps: int | None = None
    phase_index: int | None = Field(default=None, alias="phaseIndex")

    model_config = {"populate_by_name": True}


class CornerInfoPayload(BaseModel):
    top_left: list[str] = Field(default_factory=list, alias="topLeft")
    top_right: list[str] = Field(default_factory=list, alias="topRight")
    bottom_left: list[str] = Field(default_factory=list, alias="bottomLeft")
    bottom_right: list[str] = Field(default_factory=list, alias="bottomRight")

    model_config = {"populate_by_name": True}


class CornerInfoRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered series ID used to build viewport corner lines.")

    model_config = {"populate_by_name": True}


class CornerInfoResponse(BaseModel):
    corner_info: CornerInfoPayload = Field(alias="cornerInfo")

    model_config = {"populate_by_name": True}


class DicomTagsRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered series ID.")
    index: int = Field(default=0, description="Zero-based instance index inside the series.")

    model_config = {"populate_by_name": True}


class DicomTagItem(BaseModel):
    tag: str
    keyword: str
    name: str
    vr: str
    value: str
    depth: int = 0
    tag_path: list[str] = Field(default_factory=list, alias="tagPath")

    model_config = {"populate_by_name": True}


class DicomTagsResponse(BaseModel):
    series_id: str = Field(alias="seriesId")
    index: int
    total: int
    instance_number: int | None = Field(default=None, alias="instanceNumber")
    sop_instance_uid: str | None = Field(default=None, alias="sopInstanceUid")
    file_path: str | None = Field(default=None, alias="filePath")
    items: list[DicomTagItem] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class DicomTagModifyRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered series ID.")
    index: int = Field(default=0, description="Zero-based instance index inside the series.")
    tag_path: list[str] = Field(alias="tagPath", description="Path returned by the DICOM tag API for the target tag.")
    value: str = Field(description="New DICOM tag value, using backslash separators for multi-value fields.")
    scope: Literal["current", "series"] = Field(
        default="current",
        description="Whether to write the current instance only or every instance in the series.",
    )

    model_config = {"populate_by_name": True}


DicomDeidentifyFieldKey = Literal[
    "patientIdentity",
    "patientDemographics",
    "datesAndTimes",
    "accessionInstitution",
    "physiciansOperators",
    "descriptions",
    "deviceInfo",
    "privateTags",
    "uids",
]


class DicomDeidentifyRequest(BaseModel):
    series_id: str = Field(alias="seriesId", description="Registered series ID.")
    field_keys: list[DicomDeidentifyFieldKey] = Field(
        default_factory=list,
        alias="fieldKeys",
        description="De-identification groups selected by the user.",
    )
    replacement_prefix: str = Field(
        default="ANON",
        alias="replacementPrefix",
        description="Short prefix used for replacement patient identifiers.",
    )

    model_config = {"populate_by_name": True}


DicomTagModifyJobState = Literal["pending", "running", "succeeded", "failed"]


class DicomTagModifyJobStatusResponse(BaseModel):
    job_id: str = Field(alias="jobId")
    status: DicomTagModifyJobState
    status_url: str = Field(alias="statusUrl")
    artifact_url: str | None = Field(default=None, alias="artifactUrl")
    error: str | None = None
    artifact_kind: Literal["dicom", "zip"] | None = Field(default=None, alias="artifactKind")
    file_name: str | None = Field(default=None, alias="fileName")
    media_type: str | None = Field(default=None, alias="mediaType")
    modified_count: int | None = Field(default=None, alias="modifiedCount")
    processed_count: int = Field(default=0, alias="processedCount")
    progress_percent: int = Field(default=0, alias="progressPercent")
    series_folder: str | None = Field(default=None, alias="seriesFolder")
    total_count: int = Field(default=0, alias="totalCount")
    created_at: str = Field(alias="createdAt")
    completed_at: str | None = Field(default=None, alias="completedAt")

    model_config = {"populate_by_name": True}
