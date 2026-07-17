from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.dicom import router as dicom_router
from app.api.routes.health import router as health_router
from app.api.routes.pacs import router as pacs_router
from app.api.routes.view import router as view_router
from app.api.routes.workspace import router as workspace_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.sockets.handlers import register_socket_handlers

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def application_lifespan(_app: FastAPI):
    from app.services.volume_rendering.diagnostics import collect_vtk_render_diagnostics
    from app.services.volume_rendering.gpu_render_process import (
        shutdown_gpu_render_process,
        start_gpu_render_process_if_enabled,
    )

    try:
        diagnostics = start_gpu_render_process_if_enabled()
        process_mode = diagnostics is not None
        if diagnostics is None:
            diagnostics = collect_vtk_render_diagnostics()
        logger.info(
            (
                "VTK render diagnostics process_mode=%s vtk=%s python=%s platform=%s "
                "opengl_vendor=%s opengl_renderer=%s opengl_version=%s mapper_mode=%s software_renderer=%s error=%s"
            ),
            process_mode,
            diagnostics.get("vtk"),
            diagnostics.get("python"),
            diagnostics.get("platform"),
            diagnostics.get("opengl_vendor"),
            diagnostics.get("opengl_renderer"),
            diagnostics.get("opengl_version"),
            diagnostics.get("mapper_mode"),
            diagnostics.get("software_renderer"),
            diagnostics.get("error"),
        )
        logger.debug("VTK OpenGL capabilities:\n%s", diagnostics.get("capabilities", ""))
        logger.info(
            (
                "3D transport configuration transport=%s codec=%s bitrate_bps=%s fps=%s "
                "initial_burst_frames=%s"
            ),
            settings.normalized_three_d_transport,
            settings.normalized_webrtc_video_codec,
            settings.normalized_webrtc_video_bitrate_bps,
            settings.normalized_webrtc_video_fps,
            settings.normalized_webrtc_initial_burst_frames,
        )
    except Exception:
        logger.exception("failed to initialize VTK render diagnostics")
    try:
        yield
    finally:
        shutdown_gpu_render_process()


fastapi_app = FastAPI(
    title=settings.app_name,
    version="3.1.2",
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    lifespan=application_lifespan,
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-DicomVision-Artifact-Kind",
        "X-DicomVision-File-Name",
        "X-DicomVision-Keyword",
        "X-DicomVision-Modified-Count",
        "X-DicomVision-Series-Folder",
        "X-DicomVision-Tag",
        "X-DicomVision-VR",
    ],
)

fastapi_app.include_router(health_router)
fastapi_app.include_router(dicom_router, prefix="/api/v1")
fastapi_app.include_router(pacs_router, prefix="/api/v1")
fastapi_app.include_router(view_router, prefix="/api/v1")
fastapi_app.include_router(workspace_router, prefix="/api/v1")

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.cors_origins,
)
register_socket_handlers(sio)
logger.info("application initialized env=%s port=%s", settings.app_env, settings.app_port)

app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=fastapi_app,
    socketio_path="/socket.io",
)
