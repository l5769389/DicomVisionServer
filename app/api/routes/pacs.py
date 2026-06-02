from fastapi import APIRouter, Depends, HTTPException, status

from app.api.workspace import get_request_workspace_id
from app.core.workspace import DEFAULT_WORKSPACE_ID
from app.schemas.pacs import (
    PacsDimseSeriesDownloadJobStatusResponse,
    PacsDimseSeriesDownloadRequest,
    PacsDimseSeriesQueryRequest,
    PacsDimseStudyQueryRequest,
    PacsDimseTestRequest,
    PacsDicomwebTestRequest,
    PacsDicomwebTestResponse,
    PacsQidoSeriesQueryRequest,
    PacsQidoSeriesQueryResponse,
    PacsQidoStudyQueryRequest,
    PacsQidoStudyQueryResponse,
    PacsSeriesPreviewRequest,
    PacsSeriesPreviewResponse,
    PacsWadoSeriesDownloadJobStatusResponse,
    PacsWadoSeriesDownloadRequest,
)
from app.services.pacs_dimse_job_service import pacs_dimse_download_job_service
from app.services.pacs_dimse_service import PacsDimseError, pacs_dimse_service
from app.services.pacs_dicomweb_service import PacsDicomwebError, pacs_dicomweb_service
from app.services.pacs_wado_job_service import pacs_wado_download_job_service

router = APIRouter(prefix="/pacs", tags=["pacs"])


def _pacs_gateway_error(exc: PacsDicomwebError) -> HTTPException:
    detail = {"message": str(exc), "statusCode": exc.status_code}
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


def _pacs_dimse_gateway_error(exc: PacsDimseError) -> HTTPException:
    detail = {"message": str(exc), "statusCode": None}
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


@router.post("/dicomweb/test", response_model=PacsDicomwebTestResponse)
def test_dicomweb_connection(payload: PacsDicomwebTestRequest) -> PacsDicomwebTestResponse:
    return pacs_dicomweb_service.test_connection(payload.profile)


@router.post("/dimse/test", response_model=PacsDicomwebTestResponse)
def test_dimse_connection(payload: PacsDimseTestRequest) -> PacsDicomwebTestResponse:
    return pacs_dimse_service.test_connection(payload.profile)


@router.post("/dimse/studies", response_model=PacsQidoStudyQueryResponse)
def query_dimse_studies(payload: PacsDimseStudyQueryRequest) -> PacsQidoStudyQueryResponse:
    try:
        return pacs_dimse_service.query_studies(payload)
    except PacsDimseError as exc:
        raise _pacs_dimse_gateway_error(exc) from exc


@router.post("/dimse/series", response_model=PacsQidoSeriesQueryResponse)
def query_dimse_series(payload: PacsDimseSeriesQueryRequest) -> PacsQidoSeriesQueryResponse:
    try:
        return pacs_dimse_service.query_series(payload)
    except PacsDimseError as exc:
        raise _pacs_dimse_gateway_error(exc) from exc


@router.post(
    "/dimse/downloadSeries/jobs",
    response_model=PacsDimseSeriesDownloadJobStatusResponse,
    summary="Start a PACS DIMSE series download job",
    description=(
        "Retrieves one DIMSE series through C-GET into the server cache, "
        "then registers the downloaded folder using the same loader as local files."
    ),
)
def create_dimse_series_download_job(
    payload: PacsDimseSeriesDownloadRequest,
    workspace_id: str = Depends(get_request_workspace_id),
) -> PacsDimseSeriesDownloadJobStatusResponse:
    if workspace_id == DEFAULT_WORKSPACE_ID:
        return pacs_dimse_download_job_service.create_job(payload)
    return pacs_dimse_download_job_service.create_job(payload, workspace_id=workspace_id)


@router.get(
    "/dimse/downloadSeries/jobs/{job_id}",
    response_model=PacsDimseSeriesDownloadJobStatusResponse,
    summary="Get PACS DIMSE series download job status",
)
def get_dimse_series_download_job(
    job_id: str,
    workspace_id: str = Depends(get_request_workspace_id),
) -> PacsDimseSeriesDownloadJobStatusResponse:
    return pacs_dimse_download_job_service.get_status(job_id, workspace_id=workspace_id)


@router.post(
    "/dimse/downloadSeries/jobs/{job_id}/cancel",
    response_model=PacsDimseSeriesDownloadJobStatusResponse,
    summary="Cancel a PACS DIMSE series download job",
)
def cancel_dimse_series_download_job(
    job_id: str,
    workspace_id: str = Depends(get_request_workspace_id),
) -> PacsDimseSeriesDownloadJobStatusResponse:
    return pacs_dimse_download_job_service.cancel_job(job_id, workspace_id=workspace_id)


@router.post("/dicomweb/studies", response_model=PacsQidoStudyQueryResponse)
def query_dicomweb_studies(payload: PacsQidoStudyQueryRequest) -> PacsQidoStudyQueryResponse:
    try:
        return pacs_dicomweb_service.query_studies(payload)
    except PacsDicomwebError as exc:
        raise _pacs_gateway_error(exc) from exc


@router.post("/dicomweb/series", response_model=PacsQidoSeriesQueryResponse)
def query_dicomweb_series(payload: PacsQidoSeriesQueryRequest) -> PacsQidoSeriesQueryResponse:
    try:
        return pacs_dicomweb_service.query_series(payload)
    except PacsDicomwebError as exc:
        raise _pacs_gateway_error(exc) from exc


@router.post("/dicomweb/seriesPreview", response_model=PacsSeriesPreviewResponse)
def preview_dicomweb_series(payload: PacsSeriesPreviewRequest) -> PacsSeriesPreviewResponse:
    try:
        return pacs_dicomweb_service.preview_series(payload)
    except PacsDicomwebError as exc:
        raise _pacs_gateway_error(exc) from exc


@router.post(
    "/dicomweb/downloadSeries/jobs",
    response_model=PacsWadoSeriesDownloadJobStatusResponse,
    summary="Start a PACS WADO series download job",
    description=(
        "Downloads one DICOMweb series through WADO-RS into the server cache, "
        "then registers the downloaded folder using the same loader as local files."
    ),
)
def create_dicomweb_series_download_job(
    payload: PacsWadoSeriesDownloadRequest,
    workspace_id: str = Depends(get_request_workspace_id),
) -> PacsWadoSeriesDownloadJobStatusResponse:
    if workspace_id == DEFAULT_WORKSPACE_ID:
        return pacs_wado_download_job_service.create_job(payload)
    return pacs_wado_download_job_service.create_job(payload, workspace_id=workspace_id)


@router.get(
    "/dicomweb/downloadSeries/jobs/{job_id}",
    response_model=PacsWadoSeriesDownloadJobStatusResponse,
    summary="Get PACS WADO series download job status",
)
def get_dicomweb_series_download_job(
    job_id: str,
    workspace_id: str = Depends(get_request_workspace_id),
) -> PacsWadoSeriesDownloadJobStatusResponse:
    return pacs_wado_download_job_service.get_status(job_id, workspace_id=workspace_id)


@router.post(
    "/dicomweb/downloadSeries/jobs/{job_id}/cancel",
    response_model=PacsWadoSeriesDownloadJobStatusResponse,
    summary="Cancel a PACS WADO series download job",
)
def cancel_dicomweb_series_download_job(
    job_id: str,
    workspace_id: str = Depends(get_request_workspace_id),
) -> PacsWadoSeriesDownloadJobStatusResponse:
    return pacs_wado_download_job_service.cancel_job(job_id, workspace_id=workspace_id)
