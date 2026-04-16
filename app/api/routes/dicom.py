from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.dicom import (
    CornerInfoRequest,
    CornerInfoResponse,
    DicomTagsRequest,
    DicomTagsResponse,
    LoadFolderRequest,
    LoadFolderResponse,
    LoadSampleResponse,
)
from app.services.dicom_tag_service import dicom_tag_service
from app.services.series_registry import series_registry
from app.services.viewer_service import viewer_service

router = APIRouter(prefix="/dicom", tags=["dicom"])
settings = get_settings()


@router.post("/loadFolder", response_model=LoadFolderResponse)
async def load_folder(payload: LoadFolderRequest) -> LoadFolderResponse:
    return series_registry.load_folder(payload)


@router.post("/loadSample", response_model=LoadSampleResponse)
async def load_sample_folder() -> LoadSampleResponse:
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
async def get_corner_info(payload: CornerInfoRequest) -> CornerInfoResponse:
    return viewer_service.get_series_corner_info(payload)


@router.post("/tags", response_model=DicomTagsResponse)
async def get_dicom_tags(payload: DicomTagsRequest) -> DicomTagsResponse:
    return dicom_tag_service.get_series_tags(payload)
