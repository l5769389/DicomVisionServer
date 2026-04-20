from fastapi import APIRouter, BackgroundTasks

from fastapi.responses import Response

from app.schemas.view import (
    OperationAcceptedResponse,
    ViewCreateRequest,
    ViewCreateResponse,
    ViewExportRequest,
    ViewMtfAnalyzeRequest,
    ViewMtfAnalyzeResponse,
    ViewQaWaterAnalyzeRequest,
    ViewQaWaterAnalyzeResponse,
    ViewSetSizeRequest,
)
from app.sockets.runtime import view_socket_hub
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service

router = APIRouter(prefix="/view", tags=["view"])


async def _emit_render_after_resize(view_id: str) -> None:
    try:
        await view_socket_hub.emit_render_for_view(view_id)
    except Exception as exc:
        await view_socket_hub.emit_error_for_view(view_id, getattr(exc, "detail", str(exc)))


@router.post("/create", response_model=ViewCreateResponse)
async def create_view(payload: ViewCreateRequest) -> ViewCreateResponse:
    return view_registry.create(payload)


@router.post("/setSize", response_model=OperationAcceptedResponse)
async def set_view_size(payload: ViewSetSizeRequest, background_tasks: BackgroundTasks) -> OperationAcceptedResponse:
    result = viewer_service.set_view_size(payload)
    background_tasks.add_task(_emit_render_after_resize, payload.view_id)
    return result


@router.post("/mtf/analyze", response_model=ViewMtfAnalyzeResponse)
async def analyze_mtf(payload: ViewMtfAnalyzeRequest) -> ViewMtfAnalyzeResponse:
    return viewer_service.analyze_mtf(payload)


@router.post("/qa/water/analyze", response_model=ViewQaWaterAnalyzeResponse)
async def analyze_qa_water(payload: ViewQaWaterAnalyzeRequest) -> ViewQaWaterAnalyzeResponse:
    return viewer_service.analyze_qa_water(payload)


@router.post("/export")
async def export_view(payload: ViewExportRequest) -> Response:
    exported = viewer_service.export_view_by_id(
        payload.view_id,
        payload.export_format,
        overlays=payload.overlays,
    )
    return Response(
        content=exported.file_bytes,
        media_type=exported.media_type,
        headers={"Content-Disposition": f'attachment; filename="{exported.file_name}"'},
    )
