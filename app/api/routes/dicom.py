from fastapi import APIRouter

from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse
from app.services.series_registry import series_registry

router = APIRouter(prefix="/dicom", tags=["dicom"])


@router.post("/loadFolder", response_model=LoadFolderResponse)
async def load_folder(payload: LoadFolderRequest) -> LoadFolderResponse:
    return series_registry.load_folder(payload)
