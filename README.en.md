# DicomVision Server

[中文说明](./README.md)

DicomVision Server is the FastAPI + Socket.IO backend for DicomVision. It provides DICOM discovery, PACS query and retrieval, 2D/MPR/4D/3D rendering, PET/CT fusion, segmentation, measurement, QA, export, and the backend bundle consumed by the desktop app.

## v3.0.0 Backend Updates

- **PET/CT Fusion**: CT/PET/Fusion/PET MIP rendering, PET-only display, PET intensity ranges, fusion previews, manual registration, registration persistence, and Socket.IO interaction support.
- **MPR segmentation and VOI**: threshold segmentation, spherical VOI, overlay render intent, segmentation preview metadata, and sidecar-style data flow.
- **MPR/4D/playback**: improved MPR, 4D MPR, slice playback, phase synchronization, and viewport size update stability.
- **QA/MTF**: MTF/FWHM, water phantom QA, ROI metrics, and report data used by the v3.0.0 right-side result panels.
- **PACS and export**: DICOMweb/DIMSE query and download, tag modification, de-identification, DICOM SR/GSPS, and PNG/DICOM export support.
- **Desktop bundle**: Windows Server bundle builds for the Electron desktop installer.

## Repositories

- Server: [https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- Client: [https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## Capabilities

- Load DICOM folders, single files, browser uploads, and sample data.
- Serve thumbnails, corner info, DICOM tags, series, instances, and 4D phase metadata.
- PACS DICOMweb QIDO/WADO and DIMSE C-ECHO/C-FIND/C-GET.
- 2D, Compare, Layout, MPR, oblique MPR, MIP, 3D volume rendering, 4D phase, and PET/CT Fusion rendering.
- Measurement ROI metrics, MTF/FWHM, water phantom QA, realtime hover and draft interactions.
- DICOM tag edits, background jobs, de-identification, DICOM SR/GSPS, and image export.
- Socket.IO image updates, view acknowledgements, error events, and playback state synchronization.

## Product Screenshots

Screenshots are maintained in the companion Client repository.

| PET/CT Fusion | PET/CT manual registration |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pet_ct_fusion.png" alt="PET/CT Fusion" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pet_ct_fusion_registration.png" alt="PET/CT manual registration" width="420"> |

| MPR / oblique | Segmentation and VOI |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr_rotate.png" alt="MPR oblique rotation" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/segmentation_voi.png" alt="Segmentation and VOI" width="420"> |

| 4D | MTF/FWHM |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/4D.png" alt="4D phase playback" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mtf_fwhm_1.png" alt="MTF and FWHM" width="420"> |

| PACS Browser | Mobile PET/CT |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pacs_dicom_import_1.png" alt="PACS Browser" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mobile_pet_ct_fusion.png" alt="Mobile PET/CT Fusion" width="260"> |

## Quick Start

```bash
uv sync
uv run python run.py
```

Default endpoints:

- HTTP: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Socket.IO: `http://127.0.0.1:8000/socket.io`
- Health: `http://127.0.0.1:8000/health`

## Configuration

Common environment variables:

- `APP_ENV`: runtime environment, usually `production` for deployments.
- `APP_HOST` / `APP_PORT`: bind host and port.
- `CORS_ORIGINS`: allowed frontend origins as a JSON array string, for example `["http://localhost:5173"]`.
- `WEB_SAMPLE_DICOM_PATH`: server-side sample DICOM path for web demo mode.
- `WEB_UPLOAD_DICOM_ROOT`: temporary storage root for browser-uploaded DICOM files.
- `DICOMVISION_PACS_CACHE_ROOT`: PACS download cache directory.
- `DICOMVISION_PACS_CACHE_TTL_SECONDS`: PACS cache retention time.

## Common API

Base path: `/api/v1`

- `POST /dicom/loadFolder`
- `POST /dicom/upload`
- `POST /dicom/loadSample`
- `POST /dicom/tags`
- `POST /pacs/dicomweb/studies`
- `POST /pacs/dicomweb/series`
- `POST /pacs/dimse/studies`
- `POST /pacs/dimse/series`
- `POST /view/create`
- `POST /view/close`
- `POST /view/setSize`
- `POST /view/operation`
- `POST /view/export`
- `POST /view/mtf/analyze`
- `POST /view/qa/water/analyze`

Use `/docs` for exact request and response schemas.

## Desktop Bundle

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1
```

Cross-platform Python script:

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

The Client `npm run release:win` command builds this Server bundle and embeds it in the Windows desktop installer.

## Testing

```bash
uv run pytest
```
