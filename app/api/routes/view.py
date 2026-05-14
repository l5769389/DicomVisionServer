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


@router.post(
    "/create",
    response_model=ViewCreateResponse,
    summary="Create a render view",
    description=(
        "Creates a server-side view state for Stack, MPR, 3D volume, or 4D phase rendering. "
        "MPR viewports can share a viewGroupKey so crosshair, window, and MIP state stay synchronized."
    ),
)
def create_view(payload: ViewCreateRequest) -> ViewCreateResponse:
    """Create the server-side state object that Socket events will operate on."""
    return view_registry.create(payload)


@router.post(
    "/close",
    response_model=OperationAcceptedResponse,
    summary="Close a render view",
    description="Releases a view, detaches Socket bindings, and drops view-owned resources such as VTK sessions.",
)
def close_view(payload: ViewCloseRequest) -> OperationAcceptedResponse:
    """Close a view and remove any realtime bindings for it."""
    result = viewer_service.close_view_by_id(payload.view_id)
    view_socket_hub.unbind_view(payload.view_id)
    return result


@router.post(
    "/setSize",
    response_model=OperationAcceptedResponse,
    summary="Set viewport size and render",
    description=(
        "Updates the server-side canvas size for a view. The HTTP response only acknowledges the size update; "
        "the rendered image is pushed asynchronously through Socket.IO as image_update."
    ),
)
def set_view_size(payload: ViewSetSizeRequest, background_tasks: BackgroundTasks) -> OperationAcceptedResponse:
    """Resize a view and schedule an image_update for connected clients."""
    result = viewer_service.set_view_size(payload)
    background_tasks.add_task(_emit_render_after_resize, payload.view_id)
    return result


@router.post(
    "/mtf/analyze",
    response_model=ViewMtfAnalyzeResponse,
    summary="Analyze MTF in a viewport ROI",
    description="Maps a frontend ROI into image coordinates and returns MTF metrics, curve points, and frequency units.",
)
def analyze_mtf(payload: ViewMtfAnalyzeRequest) -> ViewMtfAnalyzeResponse:
    """Run MTF analysis for an ROI drawn on a 2D viewport."""
    return viewer_service.analyze_mtf(payload)


@router.post(
    "/qa/water/analyze",
    response_model=ViewQaWaterAnalyzeResponse,
    summary="Analyze water phantom QA",
    description="Detects water phantom ROIs in the current 2D view and returns CT value, uniformity, and noise metrics.",
)
def analyze_qa_water(payload: ViewQaWaterAnalyzeRequest) -> ViewQaWaterAnalyzeResponse:
    """Run water phantom QA analysis for the active 2D view."""
    return viewer_service.analyze_qa_water(payload)


@router.post(
    "/export",
    summary="Export current view",
    description="Renders the current view state and returns a PNG or DICOM Secondary Capture attachment.",
)
def export_view(payload: ViewExportRequest) -> Response:
    """Render and package the current view for download/export."""
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
