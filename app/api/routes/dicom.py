from fastapi import APIRouter, HTTPException, Response

from app.core.config import get_settings
from app.schemas.dicom import (
    CornerInfoRequest,
    CornerInfoResponse,
    DicomTagsRequest,
    DicomTagsResponse,
    FourDPhasesRequest,
    FourDPhasesResponse,
    LoadFolderRequest,
    LoadFolderResponse,
    LoadSampleResponse,
)
from app.services.dicom_tag_service import dicom_tag_service
from app.services.four_d_service import four_d_service
from app.services.series_registry import series_registry
from app.services.viewer_service import viewer_service

router = APIRouter(prefix="/dicom", tags=["dicom"])
settings = get_settings()


@router.post("/loadFolder", response_model=LoadFolderResponse)
def load_folder(payload: LoadFolderRequest) -> LoadFolderResponse:
    return series_registry.load_folder(payload)


@router.post("/loadSample", response_model=LoadSampleResponse)
def load_sample_folder() -> LoadSampleResponse:
    sample_path = settings.web_sample_dicom_path
    if not sample_path:
        raise HTTPException(status_code=400, detail="WEB_SAMPLE_DICOM_PATH is not configured")

    response = series_registry.load_folder(LoadFolderRequest(folderPath=sample_path))
    return LoadSampleResponse(
        seriesId=response.series_id,
        seriesList=response.series_list,
        samplePath=sample_path,
    )


@router.post("/cornerInfo", response_model=CornerInfoResponse)
def get_corner_info(payload: CornerInfoRequest) -> CornerInfoResponse:
    return viewer_service.get_series_corner_info(payload)


@router.get("/thumbnail")
def get_series_thumbnail(seriesId: str) -> Response:
    return Response(content=series_registry.get_series_thumbnail_png(seriesId), media_type="image/png")


@router.post("/fourD/phases", response_model=FourDPhasesResponse)
def get_four_d_phases(payload: FourDPhasesRequest) -> FourDPhasesResponse:
    series_registry.ensure_four_d_phase_series(payload.series_id)
    return four_d_service.get_four_d_phases(
        payload.series_id,
        series_registry.list_all(),
        include_preview_images=payload.include_preview_images,
        preview_phase_index=payload.preview_phase_index,
    )


@router.get("/fourD/preview")
def get_four_d_preview(seriesId: str, phaseIndex: int, viewportKey: str) -> Response:
    series_registry.ensure_four_d_phase_series(seriesId)
    return Response(
        content=four_d_service.get_four_d_preview_png(
            seriesId,
            series_registry.list_all(),
            phase_index=phaseIndex,
            viewport_key=viewportKey,
        ),
        media_type="image/png",
    )


@router.post("/tags", response_model=DicomTagsResponse)
def get_dicom_tags(payload: DicomTagsRequest) -> DicomTagsResponse:
    return dicom_tag_service.get_series_tags(payload)
