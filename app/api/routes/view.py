from fastapi import APIRouter, BackgroundTasks

from fastapi.responses import Response
from urllib.parse import quote
from app.schemas.view import (
    OperationAcceptedResponse,
    ViewCloseRequest,
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


def _sanitize_attachment_filename(file_name: str) -> str:
    sanitized = file_name.replace("\r\n", "_").replace("\r", "_").replace("\n", "_").strip()
    return sanitized or "dicomvision-export"


def _build_ascii_attachment_fallback(file_name: str) -> str:
    fallback = "".join(
        character if 32 <= ord(character) < 127 and character not in {'"', "\\", ";"} else "_"
        for character in file_name
    ).strip(" ._")
    return fallback or "dicomvision-export"


def _build_attachment_headers(file_name: str) -> dict[str, str]:
    safe_file_name = _sanitize_attachment_filename(file_name)
    ascii_fallback = _build_ascii_attachment_fallback(safe_file_name)
    encoded_file_name = quote(safe_file_name, safe="")
    return {
        "Content-Disposition": f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded_file_name}"
    }


async def _emit_render_after_resize(view_id: str) -> None:
    try:
        await view_socket_hub.emit_render_for_view(view_id)
    except Exception as exc:
        await view_socket_hub.emit_error_for_view(view_id, getattr(exc, "detail", str(exc)))


@router.post("/create", response_model=ViewCreateResponse)
def create_view(payload: ViewCreateRequest) -> ViewCreateResponse:
    return view_registry.create(payload)


@router.post("/close", response_model=OperationAcceptedResponse)
def close_view(payload: ViewCloseRequest) -> OperationAcceptedResponse:
    result = viewer_service.close_view_by_id(payload.view_id)
    view_socket_hub.unbind_view(payload.view_id)
    return result


@router.post("/setSize", response_model=OperationAcceptedResponse)
def set_view_size(payload: ViewSetSizeRequest, background_tasks: BackgroundTasks) -> OperationAcceptedResponse:
    result = viewer_service.set_view_size(payload)
    background_tasks.add_task(_emit_render_after_resize, payload.view_id)
    return result


@router.post("/mtf/analyze", response_model=ViewMtfAnalyzeResponse)
def analyze_mtf(payload: ViewMtfAnalyzeRequest) -> ViewMtfAnalyzeResponse:
    return viewer_service.analyze_mtf(payload)


@router.post("/qa/water/analyze", response_model=ViewQaWaterAnalyzeResponse)
def analyze_qa_water(payload: ViewQaWaterAnalyzeRequest) -> ViewQaWaterAnalyzeResponse:
    return viewer_service.analyze_qa_water(payload)


@router.post("/export")
def export_view(payload: ViewExportRequest) -> Response:
    exported = viewer_service.export_view_by_id(
        payload.view_id,
        payload.export_format,
        overlays=payload.overlays,
    )
    return Response(
        content=exported.file_bytes,
        media_type=exported.media_type,
        headers=_build_attachment_headers(exported.file_name),
    )
