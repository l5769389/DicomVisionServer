from fastapi import APIRouter

from app.schemas.dicom import CornerInfoRequest, CornerInfoResponse, LoadFolderRequest, LoadFolderResponse
from app.services.series_registry import series_registry
from app.services.viewer_service import viewer_service

router = APIRouter(prefix="/dicom", tags=["dicom"])


@router.post("/loadFolder", response_model=LoadFolderResponse)
async def load_folder(payload: LoadFolderRequest) -> LoadFolderResponse:
    return series_registry.load_folder(payload)


@router.post("/cornerInfo", response_model=CornerInfoResponse)
async def get_corner_info(payload: CornerInfoRequest) -> CornerInfoResponse:
    return viewer_service.get_series_corner_info(payload)
