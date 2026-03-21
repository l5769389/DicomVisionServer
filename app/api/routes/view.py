from fastapi import APIRouter

from app.schemas.view import ViewCreateRequest, ViewCreateResponse, ViewImageResponse, ViewSetSizeRequest
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service

router = APIRouter(prefix="/view", tags=["view"])


@router.post("/create", response_model=ViewCreateResponse)
async def create_view(payload: ViewCreateRequest) -> ViewCreateResponse:
    return view_registry.create(payload)


@router.post("/setSize", response_model=ViewImageResponse)
async def set_view_size(payload: ViewSetSizeRequest) -> ViewImageResponse:
    return viewer_service.set_view_size(payload)
