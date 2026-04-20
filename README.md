# DicomVision Server

[中文说明](./README.zh-CN.md)

DicomVision Server is the backend service for the DicomVision viewing system. It is responsible for DICOM series loading, viewport lifecycle management, 2D image rendering, MPR reconstruction, 3D volume rendering, and real-time image streaming to the frontend.

Companion frontend repository: [DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## Overview

DicomVision uses a frontend-backend medical imaging architecture:

- `DicomVisionServer`: FastAPI + Socket.IO backend for data loading, rendering, and interaction processing
- `DicomVisionClient`: Electron + Vue frontend for workflow orchestration, viewport interaction, and image presentation

The backend is the rendering core of the system. It handles image computation, view synchronization, and real-time event delivery, while the frontend focuses on user interaction and workspace management.

## Key Features

### DICOM Data Services

- Load local DICOM folders and discover readable series
- Maintain series metadata and runtime registries in memory
- Provide sample-data loading entry points for web deployment scenarios

### Viewport and Rendering Pipeline

- Create and manage `Stack`, `MPR`, and `3D` viewports
- Render 2D stack images and overlays
- Perform multi-planar reconstruction for orthogonal viewing workflows
- Run VTK-based 3D volume rendering on the backend

### Realtime Interaction

- Support low-latency interaction over Socket.IO
- Process pan, zoom, scroll, hover, reset, crosshair, and image operations
- Push rendered frames, overlay data, and acknowledgement events back to the client

### 3D Volume Capabilities

- Built-in volume presets such as `aaa`, `red`, `cardiac`, `muscle`, and `mip`
- Transfer-function normalization and application
- Blend mode, lighting, interpolation, opacity, color, and layer controls
- Fast preview path plus full-quality render path

## Architecture

The backend exposes two communication layers:

- HTTP API for coarse-grained actions such as loading folders, creating views, and setting viewport size
- Socket.IO for interactive commands and real-time render updates

Typical request flow:

1. The frontend calls `POST /api/v1/dicom/loadFolder` to register a folder.
2. The frontend calls `POST /api/v1/view/create` to create a viewport.
3. The viewport size is set through `POST /api/v1/view/setSize` or `set_view_size`.
4. The frontend binds a socket session through `bind_view`.
5. Interactive commands are sent through `view_operation`, `image_operation`, or `view_hover`.
6. The backend emits `image_update`, `hover_info`, `view_ack`, or error events.

Core backend responsibilities:

- register and index readable DICOM series
- manage lifecycle and state for `Stack`, `MPR`, and `3D` views
- render image frames and overlays
- synchronize multi-view interaction state
- normalize and apply 3D rendering configuration

## Tech Stack

- Python 3.13+
- FastAPI
- python-socketio
- pydicom
- NumPy
- SciPy
- Pillow
- VTK
- uv

## Repository Structure

```text
app/
  api/routes/              HTTP routes
  core/                    settings, constants, logging
  models/                  in-memory runtime models
  schemas/                 request and response schemas
  services/                DICOM processing, rendering, registries
  sockets/                 Socket.IO handlers and runtime hub
  utils/                   shared helpers
sample-data/               optional sample DICOM data for web deployment
tests/                     automated tests
run.py                     local startup entry
render.yaml                Render deployment manifest
pyproject.toml             project metadata and dependencies
```

## Core Modules

- `series_registry`: series discovery and metadata management
- `view_registry`: viewport instance lifecycle management
- `dicom_cache`: decoded pixel caching
- `viewer_service`: main interaction and rendering orchestration layer
- `view_socket_hub`: targeted socket binding and render delivery
- `app/services/volume_rendering/`: VTK-based volume rendering pipeline

## Quick Start

### Requirements

- Python 3.13 or newer
- A runtime environment compatible with VTK
- Read access to the DICOM folders that will be loaded

### Install Dependencies

```bash
uv sync
```

Install optional development dependencies when needed:

```bash
uv sync --extra dev
```

### Run the Server

```bash
uv run python run.py
```

Default service addresses:

- HTTP: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Socket.IO: `http://127.0.0.1:8000/socket.io`

## Configuration

Create a local `.env` file when needed. Common settings:

```env
APP_NAME=DicomVision Server
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=["*"]
WEB_SAMPLE_DICOM_PATH=
```

Key configuration items:

- `APP_ENV`: enables development-mode behavior such as reload
- `APP_HOST`: bind host for the backend service
- `APP_PORT`: listening port, default `8000`
- `CORS_ORIGINS`: allowed origins for HTTP and Socket.IO
- `WEB_SAMPLE_DICOM_PATH`: sample DICOM folder used by `POST /api/v1/dicom/loadSample`

## API and Realtime Events

Base HTTP path: `/api/v1`

### HTTP API

- `GET /health`
- `POST /dicom/loadFolder`
- `POST /dicom/loadSample`
- `POST /dicom/cornerInfo`
- `POST /view/create`
- `POST /view/setSize`

Use `/docs` for exact request and response schemas.

### Socket.IO Events

Client-to-server:

- `bind_view`
- `set_view_size`
- `view_operation`
- `image_operation`
- `view_hover`

Server-to-client:

- `connected`
- `view_bound`
- `view_ack`
- `image_update`
- `hover_info`
- `image_error`
- `render_error`

`image_update` carries rendered image payloads and viewport metadata.

## Frontend Integration

Companion frontend repository:

[https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

Recommended local startup order:

1. Start `DicomVisionServer`.
2. Start `DicomVisionClient`.
3. Load a DICOM folder from the client.
4. Create Stack, MPR, or 3D viewports and begin interaction.

The current frontend defaults to `http://127.0.0.1:8000` for both HTTP and Socket.IO in desktop development mode.

## Render Deployment

This repository includes `render.yaml` for deploying the backend to Render.

Recommended environment variables:

```env
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=10000
CORS_ORIGINS=["https://your-vercel-app.vercel.app"]
WEB_SAMPLE_DICOM_PATH=/opt/render/project/src/sample-data
```

Notes:

- `CORS_ORIGINS` is shared by FastAPI CORS and Socket.IO allowed origins
- `WEB_SAMPLE_DICOM_PATH` must point to a readable directory
- the web frontend can use `POST /api/v1/dicom/loadSample` without exposing local filesystem paths

## Testing

Run tests with:

```bash
uv run pytest
```

## Desktop Packaging

This repository builds the backend desktop bundle consumed by the Electron client installer.

Install PyInstaller when required:

```bash
uv run python -m pip install pyinstaller
```

Build the Windows backend bundle:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1
```

Default output:

```text
dist/
  DicomVisionServer/
    DicomVisionServer.exe
    ...
```

Override the output root if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1 -OutputRoot .\artifacts
```

The frontend installer flow can then consume this directory through `DICOM_VISION_SERVER_BUNDLE_PATH`.
