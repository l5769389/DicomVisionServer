# DicomVision Server

[中文说明](./README.md)

DicomVision Server powers the backend side of DicomVision with DICOM series discovery, PACS DICOMweb/DIMSE query and series retrieval, thumbnails and tag reading, DICOM tag editing, DICOM de-identification export, Stack rendering and Stack Compare rendering, MPR and oblique MPR reconstruction, 4D phase preview and playback coordination, VTK-based 3D volume rendering, measurement calculation, MTF/FWHM analysis, water phantom QA, image export, and realtime image delivery to the frontend over Socket.IO.

## Version 1.2.0 Updates

- Supported the new client-side Stack Compare workflow by keeping independent backend views for the source and target series.
- Added PACS Browser backend support for DICOMweb QIDO/WADO and DIMSE C-ECHO, C-FIND, and C-GET series retrieval into the server cache.
- Ensured explicit pseudocolor operations re-render frames even when the requested preset matches the previous state, so Compare panes can reliably apply the configured default pseudocolor.
- Kept the asynchronous DICOM tag edit and de-identification artifact jobs with pollable progress and downloadable results.
- Kept desktop bundle packaging support for the Electron client release.

## Repositories

- Server: [https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- Client: [https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## Feature Overview

- **DICOM data services**: load local folders or single DICOM files, discover series, generate thumbnails, and read instance-level DICOM tags.
- **PACS integration**: query studies and series through DICOMweb or DIMSE, download DICOMweb WADO or DIMSE C-GET series into the server cache, and register retrieved data through the same local-folder loading pipeline.
- **Stack rendering**: render 2D images from viewport size, window/level, pseudocolor, rotation, flip, zoom, and pan state.
- **Stack Compare support**: maintain independent source/target Stack views for side-by-side comparison while accepting synchronized scroll, window, pseudocolor, zoom, pan, and transform operations from the client.
- **MPR and oblique MPR**: build standardized volumes for axial, coronal, and sagittal reconstruction, synchronized crosshair navigation, oblique rotation, and MIP configuration.
- **4D support**: detect phase groups, generate phase lists and preview images, and coordinate frontend playback through Socket.IO.
- **3D volume rendering**: render volumes through VTK with presets, transfer functions, lighting, interpolation, blend modes, and layer configuration.
- **Measurement and QA analysis**: calculate line, rectangle, ellipse, angle, curve, and freeform ROI metrics, plus MTF/FWHM and water phantom QA results.
- **Realtime interaction**: process scroll, window/level, zoom, pan, crosshair, oblique MPR, 3D rotation, hover, and measurement draft operations over Socket.IO.
- **DICOM metadata export**: generate modified tag copies and de-identified DICOM series artifacts without overwriting source files.
- **Background artifact jobs**: process long-running tag edits and de-identification exports asynchronously with pollable progress and downloadable artifacts.
- **Deployment and packaging**: deploy as the web backend on Render, or build a Windows desktop backend bundle consumed by the Electron client.

## Product Screenshots

Screenshots are maintained in the companion client repository. This backend README references those assets to show the full product experience.

| Stack viewing | MPR reconstruction |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/stack.png" alt="Stack viewing" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr.png" alt="MPR reconstruction" width="420"> |

| PACS data sources | PACS browser import |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pacs_dicom_import.png" alt="PACS DICOMweb and DIMSE profile setup" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pacs_dicom_import_1.png" alt="PACS Browser query and downloaded series import" width="420"> |

| Oblique MPR / crosshair rotation | 4D phase playback |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr_rotate.png" alt="Oblique MPR and crosshair rotation" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/4D.png" alt="4D phase playback" width="420"> |

| Measurement tools | DICOM tags |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/measure.png" alt="Measurement tools" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/dicomTags.png" alt="DICOM tags" width="420"> |

| Stack Compare | Batch DICOM tag editing |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/compare_stack.png" alt="Side-by-side Stack Compare" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/batch_modify_tags.png" alt="Batch DICOM tag editing" width="420"> |

| MTF analysis | FWHM result |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mtf.png" alt="MTF analysis" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mtf_fwhm.png" alt="FWHM result" width="420"> |

| Water phantom QA |
| --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/water_phantom_qa.png" alt="Water phantom QA" width="420"> |

| Drag-and-drop import | De-identification export |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/drag_import.png" alt="Drag-and-drop DICOM import" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/deIndentifyExport.png" alt="DICOM de-identification export" width="420"> |

## Architecture

The backend exposes two communication layers:

- HTTP API for loading data, reading tags, fetching thumbnails, creating and closing views, resizing views, analysis, and export.
- Socket.IO for low-latency interaction commands and realtime image updates to bound frontend sessions.

Typical flow:

1. The frontend calls `POST /api/v1/dicom/loadFolder`, `POST /api/v1/dicom/upload`, or `POST /api/v1/dicom/loadSample` to register data.
2. The backend scans DICOM files and stores series, instance, and phase metadata.
3. The frontend calls `POST /api/v1/view/create` to create a Stack, MPR, 3D, or other viewport.
4. The frontend binds a Socket.IO session to the viewport with `bind_view`.
5. The frontend sends `view_operation`, `image_operation`, `view_hover`, or 4D playback events.
6. The backend returns `image_update`, `hover_info`, `measurement_draft`, `view_ack`, 4D playback state, or error events.

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
- PyInstaller for desktop bundle builds

## Repository Structure

```text
app/
  api/routes/              HTTP routes
  core/                    settings, constants, logging
  models/                  in-memory runtime models
  schemas/                 request and response schemas
  services/                DICOM processing, rendering, analysis, and registries
  services/render_layers/  overlay rendering
  services/volume_rendering/ VTK volume rendering
  sockets/                 Socket.IO events and realtime delivery
  utils/                   shared helpers
sample-data/               optional sample DICOM data for web deployment
scripts/                   desktop bundle and API type generation scripts
tests/                     automated tests
run.py                     local startup entry
render.yaml                Render deployment manifest
pyproject.toml             project metadata and dependencies
```

## Quick Start

### Requirements

- Python 3.13 or newer
- A system environment compatible with VTK
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

Default addresses:

- HTTP: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Socket.IO: `http://127.0.0.1:8000/socket.io`

## Configuration

Create a local `.env` file when needed:

```env
APP_NAME=DicomVision Server
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=["*"]
EXPOSE_API_DOCS=
WEB_SAMPLE_DICOM_PATH=
WEB_UPLOAD_DICOM_ROOT=
WEB_UPLOAD_MAX_FILES=5000
WEB_UPLOAD_MAX_BYTES=2147483648
WEB_UPLOAD_MAX_AGE_SECONDS=1800
WEB_UPLOAD_CLEANUP_INTERVAL_SECONDS=1800
```

Key settings:

- `APP_ENV`: runtime environment, usually `production` for deployed services.
- `APP_HOST`: bind host.
- `APP_PORT`: listening port, default `8000`.
- `CORS_ORIGINS`: allowed frontend origins for HTTP and Socket.IO.
- `EXPOSE_API_DOCS`: set to `false` to hide `/docs`, `/redoc`, and `/openapi.json`; by default docs are hidden in production and exposed in development.
- `WEB_SAMPLE_DICOM_PATH`: server-side sample DICOM directory used by `POST /api/v1/dicom/loadSample`.
- `WEB_UPLOAD_DICOM_ROOT`: optional temporary storage root for browser-uploaded DICOM files.
- `WEB_UPLOAD_MAX_FILES` / `WEB_UPLOAD_MAX_BYTES`: upload limits for real web deployments.
- `WEB_UPLOAD_MAX_AGE_SECONDS`: maximum age for browser upload sessions before cleanup, default `1800`.
- `WEB_UPLOAD_CLEANUP_INTERVAL_SECONDS`: background upload cleanup interval, default `1800`.

## HTTP API

Base path: `/api/v1`

- `GET /health`
- `POST /dicom/loadFolder`
- `POST /dicom/upload`
- `POST /dicom/loadSample`
- `POST /dicom/cornerInfo`
- `GET /dicom/thumbnail`
- `POST /dicom/fourD/phases`
- `GET /dicom/fourD/preview`
- `POST /dicom/tags`
- `POST /dicom/modifyTag`
- `POST /dicom/modifyTag/jobs`
- `GET /dicom/modifyTag/jobs/{job_id}`
- `GET /dicom/modifyTag/jobs/{job_id}/artifact`
- `POST /dicom/deidentify`
- `POST /dicom/deidentify/jobs`
- `GET /dicom/deidentify/jobs/{job_id}`
- `GET /dicom/deidentify/jobs/{job_id}/artifact`
- `POST /pacs/dicomweb/test`
- `POST /pacs/dicomweb/studies`
- `POST /pacs/dicomweb/series`
- `POST /pacs/dicomweb/seriesPreview`
- `POST /pacs/dicomweb/downloadSeries/jobs`
- `GET /pacs/dicomweb/downloadSeries/jobs/{job_id}`
- `POST /pacs/dicomweb/downloadSeries/jobs/{job_id}/cancel`
- `POST /pacs/dimse/test`
- `POST /pacs/dimse/studies`
- `POST /pacs/dimse/series`
- `POST /pacs/dimse/downloadSeries/jobs`
- `GET /pacs/dimse/downloadSeries/jobs/{job_id}`
- `POST /pacs/dimse/downloadSeries/jobs/{job_id}/cancel`
- `POST /view/create`
- `POST /view/close`
- `POST /view/setSize`
- `POST /view/mtf/analyze`
- `POST /view/qa/water/analyze`
- `POST /view/export`

Use `/docs` for exact request and response schemas.

## Socket.IO Events

Client-to-server:

- `bind_view`
- `set_view_size`
- `view_operation`
- `image_operation`
- `view_hover`
- `four_d_playback_start`
- `four_d_playback_stop`
- `four_d_playback_fps`

Server-to-client:

- `connected`
- `view_bound`
- `view_ack`
- `image_update`
- `hover_info`
- `measurement_draft`
- `four_d_phase_index`
- `four_d_playback_state`
- `image_error`
- `render_error`

## Web Deployment

This repository includes `render.yaml` for deploying the backend to Render. Recommended environment variables:

```env
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=10000
CORS_ORIGINS=["https://your-vercel-app.vercel.app"]
WEB_SAMPLE_DICOM_PATH=/opt/render/project/src/sample-data
```

For web frontend deployment:

- Set frontend `VITE_BACKEND_ORIGIN` to the backend origin.
- Add the frontend domain to backend `CORS_ORIGINS`.
- If the web client should load server-side sample data, configure `WEB_SAMPLE_DICOM_PATH` and set frontend `VITE_WEB_APP_MODE=demo-web`.

## Fly.io Deployment

This repository includes `fly.toml`, `Dockerfile`, and `.dockerignore`. The Fly CLI is recommended because it prints full build and startup logs.

### 1. Install and sign in to Fly CLI

Windows PowerShell:

```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
fly auth login
```

### 2. Create or confirm the app name

The default app name in `fly.toml` is:

```toml
app = "dicomvision-server-l5769389"
```

Fly app names must be globally unique. If creation fails, change it to your own name:

```toml
app = "dicomvision-server-yourname"
```

Create the app:

```powershell
fly apps create dicomvision-server-l5769389
```

If the name is already taken, choose another name and update `fly.toml`.

### 3. Configure CORS origins

The current default is convenient for testing:

```toml
CORS_ORIGINS = "[\"*\"]"
```

For production, set your frontend origin:

```powershell
fly secrets set CORS_ORIGINS='["https://your-vercel-app.vercel.app"]'
```

If you use the bundled sample dataset, `fly.toml` already sets:

```toml
WEB_SAMPLE_DICOM_PATH = "/app/sample-data/test"
```

### 4. Deploy

```powershell
fly deploy
```

After deployment:

```powershell
fly status
fly logs
fly open
```

Health check URL:

```text
https://dicomvision-server-l5769389.fly.dev/health
```

### GitHub deploy page notes

When using Fly.io's GitHub Deploy page:

- Set `Working directory` to `./`
- Set `Config path` to `fly.toml` or `./fly.toml`, not `./`
- Set `Internal port` to `8000`
- Use at least `shared-cpu-1x / 2GB`
- Push `fly.toml`, `Dockerfile`, and `.dockerignore` to GitHub before deploying from the web UI
- If the page says `Failed to create app`, first check whether the app name is already taken, or whether the generated/configured app name should be changed to a unique lowercase name with hyphens

## Desktop Bundle

This repository can build the backend bundle consumed by the Electron desktop installers.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1
```

```bash
python3 scripts/build_desktop_bundle.py
```

Default output:

```text
dist/
  DicomVisionServer/
    DicomVisionServer.exe  # Windows
    DicomVisionServer      # macOS
    ...
```

Override the output root when needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1 -OutputRoot .\artifacts
```

```bash
python3 scripts/build_desktop_bundle.py --output-root ./artifacts
```

On macOS, the client `npm run release:mac` command calls `scripts/build_desktop_bundle.py` before packaging the Electron app. The client packaging flow can also consume an existing bundle through `DICOM_VISION_SERVER_BUNDLE_PATH`, `npm run release:win`, or `npm run release:mac`.

## Testing

```bash
uv run pytest
```
