import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.dicom import router as dicom_router
from app.api.routes.health import router as health_router
from app.api.routes.view import router as view_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.sockets.handlers import register_socket_handlers

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

fastapi_app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi_app.include_router(health_router)
fastapi_app.include_router(dicom_router, prefix="/api/v1")
fastapi_app.include_router(view_router, prefix="/api/v1")

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
