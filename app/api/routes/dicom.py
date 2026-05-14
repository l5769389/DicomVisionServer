from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Response

from app.core.config import get_settings
from app.schemas.dicom import (
    CornerInfoRequest,
    CornerInfoResponse,
    DicomTagsRequest,
    DicomTagsResponse,
    DicomTagModifyRequest,
    FourDPhasesRequest,
    FourDPhasesResponse,
    LoadFolderRequest,
    LoadFolderResponse,
    LoadSampleResponse,
)
from app.services.dicom_tag_service import dicom_tag_service
from app.services.four_d_service import four_d_service
from app.services.series_registry import series_registry
from app.services.viewer_service import viewer_service

router = APIRouter(prefix="/dicom", tags=["dicom"])
settings = get_settings()


@router.post(
    "/loadFolder",
    response_model=LoadFolderResponse,
    summary="Scan a DICOM file or folder",
    description=(
        "Reads DICOM headers from a local file or folder, groups instances into series, "
        "registers them in memory, and returns lightweight series summaries. "
        "Pixel data is decoded later on demand by rendering APIs."
    ),
)
def load_folder(payload: LoadFolderRequest) -> LoadFolderResponse:
    """Register DICOM files without decoding all pixel data up front."""
    return series_registry.load_folder(payload)


@router.post(
    "/loadSample",
    response_model=LoadSampleResponse,
    summary="Load configured sample DICOM data",
    description="Loads the sample folder configured by WEB_SAMPLE_DICOM_PATH and returns the same series summary shape as loadFolder.",
)
def load_sample_folder() -> LoadSampleResponse:
    """Load the demo dataset used by the web preview mode."""
    sample_path = settings.web_sample_dicom_path
    if not sample_path:
        raise HTTPException(status_code=400, detail="WEB_SAMPLE_DICOM_PATH is not configured")

    response = series_registry.load_folder(LoadFolderRequest(folderPath=sample_path))
    return LoadSampleResponse(
        seriesId=response.series_id,
        seriesList=response.series_list,
        samplePath=sample_path,
    )


@router.post(
    "/cornerInfo",
    response_model=CornerInfoResponse,
    summary="Build series corner information",
    description="Returns patient, study, series, image, and display metadata used by viewport corner overlays.",
)
def get_corner_info(payload: CornerInfoRequest) -> CornerInfoResponse:
    """Return corner overlay metadata for the selected series."""
    return viewer_service.get_series_corner_info(payload)


@router.get(
    "/thumbnail",
    summary="Get a series thumbnail",
    description="Returns a PNG thumbnail generated from the middle slice of a registered series.",
)
def get_series_thumbnail(seriesId: str) -> Response:
    """Return a small PNG preview for sidebar series cards."""
    return Response(content=series_registry.get_series_thumbnail_png(seriesId), media_type="image/png")


@router.post(
    "/fourD/phases",
    response_model=FourDPhasesResponse,
    summary="Resolve 4D phase manifest",
    description=(
        "Detects phase partitions for a 4D-capable series and returns phase metadata. "
        "For single-series 4D data, this also materializes virtual phase series IDs."
    ),
)
def get_four_d_phases(payload: FourDPhasesRequest) -> FourDPhasesResponse:
    """Return the phase list needed by the frontend 4D timeline."""
    series_registry.ensure_four_d_phase_series(payload.series_id)
    return four_d_service.get_four_d_phases(
        payload.series_id,
        series_registry.list_all(),
        include_preview_images=payload.include_preview_images,
        preview_phase_index=payload.preview_phase_index,
    )


@router.get(
    "/fourD/preview",
    summary="Get a 4D phase preview",
    description="Returns a PNG preview for one phase and MPR viewport in a registered 4D series.",
)
def get_four_d_preview(seriesId: str, phaseIndex: int, viewportKey: str) -> Response:
    """Return a preview image for a phase selector item."""
    series_registry.ensure_four_d_phase_series(seriesId)
    return Response(
        content=four_d_service.get_four_d_preview_png(
            seriesId,
            series_registry.list_all(),
            phase_index=phaseIndex,
            viewport_key=viewportKey,
        ),
        media_type="image/png",
    )


@router.post(
    "/tags",
    response_model=DicomTagsResponse,
    summary="Read DICOM tags",
    description="Returns formatted DICOM tags for a specific instance index in a registered series.",
)
def get_dicom_tags(payload: DicomTagsRequest) -> DicomTagsResponse:
    """Return metadata rows for the DICOM tag viewer."""
    return dicom_tag_service.get_series_tags(payload)


@router.post(
    "/modifyTag",
    summary="Create modified DICOM tag download artifact",
    description=(
        "Updates one editable DICOM tag in the current instance or every instance in a registered series. "
        "The source files are never overwritten; the modified DICOM file or ZIP archive is returned to the client "
        "so desktop and web frontends can save it in the user's chosen location."
    ),
    responses={
        200: {
            "content": {
                "application/dicom": {},
                "application/zip": {},
            },
            "description": "A modified DICOM file for current scope, or a ZIP archive for series scope.",
        }
    },
)
def modify_dicom_tag(payload: DicomTagModifyRequest) -> Response:
    """Return modified DICOM bytes for the requested tag edit."""
    artifact = dicom_tag_service.modify_series_tag(payload)
    quoted_file_name = quote(artifact.file_name)
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"{artifact.file_name}\"; filename*=UTF-8''{quoted_file_name}",
            "X-DicomVision-Artifact-Kind": artifact.artifact_kind,
            "X-DicomVision-File-Name": artifact.file_name,
            "X-DicomVision-Keyword": artifact.keyword,
            "X-DicomVision-Modified-Count": str(artifact.modified_count),
            "X-DicomVision-Series-Folder": artifact.series_folder,
            "X-DicomVision-Tag": artifact.tag,
            "X-DicomVision-VR": artifact.vr,
        },
    )
