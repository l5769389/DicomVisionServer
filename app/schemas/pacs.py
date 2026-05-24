from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.dicom import SeriesSummary


DicomwebAuthType = Literal["none", "basic", "bearer"]
DicomwebProfilePreset = Literal["orthanc", "dcm4chee", "custom"]
DimseQueryModel = Literal["study-root", "patient-root"]
PacsWadoDownloadJobState = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class PacsDicomwebProfile(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_url: str = Field(alias="baseUrl", min_length=1)
    qido_path: str = Field(default="/dicom-web", alias="qidoPath")
    wado_path: str = Field(default="/dicom-web", alias="wadoPath")
    auth_type: DicomwebAuthType = Field(default="none", alias="authType")
    username: str | None = None
    password: str | None = None
    bearer_token: str | None = Field(default=None, alias="bearerToken")
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=60.0, alias="timeoutSeconds")
    preset: DicomwebProfilePreset = "custom"

    model_config = {"populate_by_name": True}


class PacsDicomwebTestRequest(BaseModel):
    profile: PacsDicomwebProfile

    model_config = {"populate_by_name": True}


class PacsDicomwebTestResponse(BaseModel):
    ok: bool
    status_code: int | None = Field(default=None, alias="statusCode")
    message: str

    model_config = {"populate_by_name": True}


class PacsDimseProfile(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(default=104, ge=1, le=65535)
    called_ae_title: str = Field(default="ANY-SCP", alias="calledAeTitle", min_length=1, max_length=16)
    client_ae_title: str = Field(default="DICOMVISION", alias="clientAeTitle", min_length=1, max_length=16)
    query_model: DimseQueryModel = Field(default="study-root", alias="queryModel")
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=60.0, alias="timeoutSeconds")

    model_config = {"populate_by_name": True}


class PacsDimseTestRequest(BaseModel):
    profile: PacsDimseProfile

    model_config = {"populate_by_name": True}


class PacsDimseStudyQueryRequest(BaseModel):
    profile: PacsDimseProfile
    study_instance_uid: str | None = Field(default=None, alias="studyInstanceUid")
    patient_id: str | None = Field(default=None, alias="patientId")
    patient_name: str | None = Field(default=None, alias="patientName")
    accession_number: str | None = Field(default=None, alias="accessionNumber")
    study_description: str | None = Field(default=None, alias="studyDescription")
    study_date_from: str | None = Field(default=None, alias="studyDateFrom")
    study_date_to: str | None = Field(default=None, alias="studyDateTo")
    modality: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)

    model_config = {"populate_by_name": True}


class PacsDimseSeriesQueryRequest(BaseModel):
    profile: PacsDimseProfile
    study_instance_uid: str = Field(alias="studyInstanceUid", min_length=1)
    series_instance_uid: str | None = Field(default=None, alias="seriesInstanceUid")
    modality: str | None = None
    series_description: str | None = Field(default=None, alias="seriesDescription")
    body_part_examined: str | None = Field(default=None, alias="bodyPartExamined")
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    model_config = {"populate_by_name": True}


class PacsQidoStudyQueryRequest(BaseModel):
    profile: PacsDicomwebProfile
    study_instance_uid: str | None = Field(default=None, alias="studyInstanceUid")
    patient_id: str | None = Field(default=None, alias="patientId")
    patient_name: str | None = Field(default=None, alias="patientName")
    accession_number: str | None = Field(default=None, alias="accessionNumber")
    study_description: str | None = Field(default=None, alias="studyDescription")
    study_date_from: str | None = Field(default=None, alias="studyDateFrom")
    study_date_to: str | None = Field(default=None, alias="studyDateTo")
    modality: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)

    model_config = {"populate_by_name": True}


class PacsQidoSeriesQueryRequest(BaseModel):
    profile: PacsDicomwebProfile
    study_instance_uid: str = Field(alias="studyInstanceUid", min_length=1)
    series_instance_uid: str | None = Field(default=None, alias="seriesInstanceUid")
    modality: str | None = None
    series_description: str | None = Field(default=None, alias="seriesDescription")
    body_part_examined: str | None = Field(default=None, alias="bodyPartExamined")
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    model_config = {"populate_by_name": True}


class PacsStudyItem(BaseModel):
    study_instance_uid: str = Field(alias="studyInstanceUid")
    patient_name: str | None = Field(default=None, alias="patientName")
    patient_id: str | None = Field(default=None, alias="patientId")
    study_date: str | None = Field(default=None, alias="studyDate")
    study_time: str | None = Field(default=None, alias="studyTime")
    accession_number: str | None = Field(default=None, alias="accessionNumber")
    study_description: str | None = Field(default=None, alias="studyDescription")
    modalities_in_study: list[str] = Field(default_factory=list, alias="modalitiesInStudy")
    number_of_study_related_series: int | None = Field(default=None, alias="numberOfStudyRelatedSeries")
    number_of_study_related_instances: int | None = Field(default=None, alias="numberOfStudyRelatedInstances")
    raw: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class PacsSeriesItem(BaseModel):
    study_instance_uid: str = Field(alias="studyInstanceUid")
    series_instance_uid: str = Field(alias="seriesInstanceUid")
    series_number: str | None = Field(default=None, alias="seriesNumber")
    modality: str | None = None
    series_description: str | None = Field(default=None, alias="seriesDescription")
    body_part_examined: str | None = Field(default=None, alias="bodyPartExamined")
    number_of_series_related_instances: int | None = Field(default=None, alias="numberOfSeriesRelatedInstances")
    raw: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class PacsQidoStudyQueryResponse(BaseModel):
    items: list[PacsStudyItem] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class PacsQidoSeriesQueryResponse(BaseModel):
    items: list[PacsSeriesItem] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class PacsSeriesPreviewRequest(BaseModel):
    profile: PacsDicomwebProfile
    study_instance_uid: str = Field(alias="studyInstanceUid", min_length=1)
    series_instance_uid: str = Field(alias="seriesInstanceUid", min_length=1)
    thumbnail: bool = True

    model_config = {"populate_by_name": True}


class PacsSeriesPreviewResponse(BaseModel):
    study_instance_uid: str = Field(alias="studyInstanceUid")
    series_instance_uid: str = Field(alias="seriesInstanceUid")
    instance_count: int = Field(default=0, alias="instanceCount")
    rows: int | None = None
    columns: int | None = None
    number_of_frames: int | None = Field(default=None, alias="numberOfFrames")
    has_multi_frame_instances: bool = Field(default=False, alias="hasMultiFrameInstances")
    transfer_syntaxes: list[str] = Field(default_factory=list, alias="transferSyntaxes")
    is_compressed: bool = Field(default=False, alias="isCompressed")
    photometric_interpretations: list[str] = Field(default_factory=list, alias="photometricInterpretations")
    sop_instance_uid: str | None = Field(default=None, alias="sopInstanceUid")
    thumbnail_src: str | None = Field(default=None, alias="thumbnailSrc")
    thumbnail_error: str | None = Field(default=None, alias="thumbnailError")

    model_config = {"populate_by_name": True}


class PacsWadoSeriesDownloadRequest(BaseModel):
    profile: PacsDicomwebProfile
    study_instance_uid: str = Field(alias="studyInstanceUid", min_length=1)
    series_instance_uid: str = Field(alias="seriesInstanceUid", min_length=1)

    model_config = {"populate_by_name": True}


class PacsWadoSeriesDownloadJobStatusResponse(BaseModel):
    job_id: str = Field(alias="jobId")
    status: PacsWadoDownloadJobState
    status_url: str = Field(alias="statusUrl")
    error: str | None = None
    folder_path: str | None = Field(default=None, alias="folderPath")
    processed_count: int = Field(default=0, alias="processedCount")
    progress_percent: int = Field(default=0, alias="progressPercent")
    series_id: str | None = Field(default=None, alias="seriesId")
    series_list: list[SeriesSummary] = Field(default_factory=list, alias="seriesList")
    total_count: int = Field(default=0, alias="totalCount")
    created_at: str = Field(alias="createdAt")
    completed_at: str | None = Field(default=None, alias="completedAt")

    model_config = {"populate_by_name": True}
