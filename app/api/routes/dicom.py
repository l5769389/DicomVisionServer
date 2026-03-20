from fastapi import APIRouter

from app.schemas.dicom import DicomRenderRequest, DicomRenderResponse
from app.services.dicom_service import dicom_service

router = APIRouter(prefix="/dicom", tags=["dicom"])


@router.post("/render", response_model=DicomRenderResponse)
async def render_dicom(payload: DicomRenderRequest) -> DicomRenderResponse:
    return dicom_service.render_from_request(payload)
