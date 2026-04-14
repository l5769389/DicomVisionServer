# DicomVision Server

[中文文档](./README.zh-CN.md)

DicomVision Server is the backend service for the DicomVision desktop viewer. It provides DICOM series loading, viewport lifecycle management, 2D image rendering, MPR reconstruction, 3D volume rendering, and real-time image updates over Socket.IO.

It is built with FastAPI and Python, and is designed to work with the Electron/Vue client in the companion repository at `D:\ct\git-repo\my\dicomVision\DicomVisionClient`.

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API and Realtime Events](#api-and-realtime-events)
- [3D Volume Rendering](#3d-volume-rendering)
- [Development Notes](#development-notes)
- [Testing](#testing)
- [Desktop Packaging](#desktop-packaging)
- [Frontend Integration](#frontend-integration)

## Overview

The backend exposes two communication channels:

- HTTP for coarse-grained resource operations such as loading a DICOM folder or creating a viewport
- Socket.IO for interactive operations such as pan, zoom, scroll, hover, crosshair updates, and render streaming

Typical lifecycle:

1. The client calls `POST /api/v1/dicom/loadFolder` to register a folder and obtain a `seriesId`.
2. The client calls `POST /api/v1/view/create` to create a viewport and obtain a `viewId`.
3. The client sets viewport dimensions through `POST /api/v1/view/setSize` or the `set_view_size` socket event.
4. The client binds a socket connection to a view through `bind_view`.
5. Interactive commands are sent through `view_operation` or `image_operation`.
6. The server emits `image_update`, `hover_info`, `view_ack`, or error events back to the client.

## Highlights

- FastAPI-based HTTP API for series loading and viewport creation
- Socket.IO event pipeline for low-latency viewport interaction and image refresh
- 2D stack rendering, MPR reconstruction, and VTK-based 3D volume rendering
- In-memory registries for series and view state management
- DICOM pixel caching to reduce repeated file reads
- Corner overlays, orientation overlays, crosshair synchronization, and hover coordinate mapping
- 3D preset and transfer-function configuration support

## Technology Stack

- Python 3.13+
- FastAPI
- python-socketio
- pydicom
- NumPy
- Pillow
- SciPy
- VTK
- uv

## Repository Structure

```text
app/
  api/routes/              HTTP routes
  core/                    settings, constants, logging
  models/                  in-memory domain models
  schemas/                 pydantic request/response models
  services/                rendering, registries, DICOM processing
  sockets/                 Socket.IO handlers and runtime hub
  utils/                   shared helpers
run.py                     local startup entry
pyproject.toml             dependencies and project metadata
.env.example               environment variable example
```

## Architecture

Core backend responsibilities:

- Register and index readable DICOM series from local folders
- Manage viewport state for `Stack`, `MPR`, and `3D`
- Render image frames and overlays for client consumption
- Keep multi-view interactions synchronized
- Normalize and apply 3D volume rendering presets and custom config

Core modules:

- `series_registry`: folder-level series discovery and metadata
- `view_registry`: viewport instances and lifecycle
- `dicom_cache`: decoded pixel cache
- `viewer_service`: main orchestration layer for rendering and interaction handling
- `view_socket_hub`: targeted socket binding and render emission

## Quick Start

### Requirements

- Python 3.13 or newer
- A VTK-compatible runtime environment
- Access to local DICOM folders on the machine where the backend runs

### Install Dependencies

```bash
uv sync
```

Install optional development dependencies if needed:

```bash
uv sync --extra dev
```

### Run the Server

```bash
uv run python run.py
```

The service starts on `http://127.0.0.1:8000` by default.

Available endpoints after startup:

- OpenAPI docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Socket.IO path: `http://127.0.0.1:8000/socket.io`

## Configuration

Create a local `.env` file from `.env.example` when required.

```env
APP_NAME=DicomVision Server
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=["*"]
```

Key settings:

- `APP_ENV`: enables auto-reload when set to `development`
- `APP_HOST`: bind address for Uvicorn
- `APP_PORT`: service port, default `8000`
- `CORS_ORIGINS`: allowed HTTP and Socket.IO origins

## API and Realtime Events

Base HTTP path: `/api/v1`

### HTTP API

- `GET /health`
- `POST /dicom/loadFolder`
  - Purpose: load a local folder and register readable DICOM series
- `POST /dicom/cornerInfo`
  - Purpose: resolve series-level corner overlay information
- `POST /view/create`
  - Purpose: create a viewport for `Stack`, `MPR`, or `3D`
- `POST /view/setSize`
  - Purpose: store viewport size and trigger rendering

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

`image_update` returns rendered viewport metadata together with encoded image bytes.

## 3D Volume Rendering

The VTK-based renderer lives under `app/services/volume_rendering/`.

Current 3D behavior includes:

- quaternion-driven camera rotation
- fast preview and full render paths
- configurable blend mode
- configurable lighting and interpolation
- multiple transfer-function layers
- built-in presets such as `aaa`, `red`, `cardiac`, `muscle`, and `mip`

3D configuration changes are normalized on the backend before being applied to a render session.

## Development Notes

- The current implementation favors backend-driven rendering over frontend image computation.
- Socket events are the primary path for low-latency interaction updates.
- The current client defaults to `http://127.0.0.1:8000` for both HTTP and Socket.IO.

## Testing

If tests are added or expanded:

```bash
uv run pytest
```

## Desktop Packaging

This repository is responsible for producing its own desktop backend bundle. The Electron client repository consumes that bundle and assembles the final installer.

Install `PyInstaller` into the local virtual environment when needed:

```bash
uv run python -m pip install pyinstaller
```

Build the Windows desktop bundle:

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

You can override the output root if your release pipeline needs a different staging location:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1 -OutputRoot .\artifacts
```

The companion Electron client can then stage this directory through `DICOM_VISION_SERVER_BUNDLE_PATH` and package it into the final desktop installer.

If the companion repository is located next to this one, you can also trigger the full release chain from `DicomVisionClient` with `npm run release:win`.

## Frontend Integration

The companion frontend project is located at:

`D:\ct\git-repo\my\dicomVision\DicomVisionClient`

Run the backend before launching the Electron client.
