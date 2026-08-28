import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.schemas.view import ViewCreateRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry
from app.services.view_registry import view_registry


def _write_ct_dicom(
    path: Path,
    stored_pixels: np.ndarray,
    *,
    pixel_spacing: tuple[float, float],
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    instance_number: int = 1,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    pixels = np.asarray(stored_pixels, dtype=np.uint16)
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_instance_uid or generate_uid()
    dataset.SeriesInstanceUID = series_instance_uid or generate_uid()
    dataset.PatientName = "Medical^Analysis"
    dataset.PatientID = "MEDICAL-ANALYSIS"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Medical analysis API truth"
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    dataset.RescaleSlope = float(rescale_slope)
    dataset.RescaleIntercept = float(rescale_intercept)
    dataset.RescaleType = "HU"
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _register_stack_view(path: Path, *, workspace_id: str = "default"):
    loaded = series_registry.load_folder(
        LoadFolderRequest(folderPath=str(path)),
        workspace_id=workspace_id,
    )
    series_id = loaded.series_list[0].series_id
    created = view_registry.create(
        ViewCreateRequest(seriesId=series_id, viewType="Stack"),
        workspace_id=workspace_id,
    )
    view = view_registry.get(created.view_id, workspace_id=workspace_id)
    first_instance = series_registry.get(series_id, workspace_id=workspace_id).instances[0]
    dataset = dicom_cache.get(
        first_instance.sop_instance_uid,
        first_instance.path,
    ).dataset
    view.width = int(dataset.Columns)
    view.height = int(dataset.Rows)
    view.is_initialized = True
    return view


@pytest.fixture(autouse=True)
def _clear_medical_analysis_state():
    series_registry.clear()
    dicom_cache.clear()
    yield
    view_registry.delete_workspace("default")
    view_registry.delete_workspace("medical-a")
    view_registry.delete_workspace("medical-b")
    series_registry.clear()
    dicom_cache.clear()


def test_mtf_api_uses_real_dicom_pixel_spacing_for_frequency_and_fwhm(tmp_path: Path) -> None:
    size = 65
    sigma_pixels = 2.0
    spacing_mm = 0.5
    y_grid, x_grid = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    stored_pixels = np.rint(
        1000.0
        * np.exp(-((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2.0 * sigma_pixels**2))
    ).astype(np.uint16)
    dicom_path = tmp_path / "mtf-physical-truth.dcm"
    _write_ct_dicom(
        dicom_path,
        stored_pixels,
        pixel_spacing=(spacing_mm, spacing_mm),
    )
    view = _register_stack_view(dicom_path)

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    metrics = data["metrics"]
    expected_mtf50 = math.sqrt(math.log(2.0)) / (
        math.sqrt(2.0) * math.pi * sigma_pixels * spacing_mm
    )
    expected_mtf10 = math.sqrt(math.log(10.0)) / (
        math.sqrt(2.0) * math.pi * sigma_pixels * spacing_mm
    )
    expected_fwhm_mm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_pixels * spacing_mm

    assert data["viewId"] == view.view_id
    assert data["viewportKey"] == "single"
    assert data["isPlaceholder"] is False
    assert metrics["unit"] == "lp/mm"
    assert metrics["sampleCount"] == size * size
    assert metrics["peakValue"] == pytest.approx(1000.0)
    assert metrics["mtf50"] == pytest.approx(expected_mtf50, rel=0.04)
    assert metrics["mtf10"] == pytest.approx(expected_mtf10, rel=0.06)
    assert metrics["mtf50W"] == pytest.approx(expected_mtf50, rel=0.04)
    assert metrics["mtf10W"] == pytest.approx(expected_mtf10, rel=0.06)
    assert metrics["mtf50H"] == pytest.approx(expected_mtf50, rel=0.04)
    assert metrics["mtf10H"] == pytest.approx(expected_mtf10, rel=0.06)
    assert metrics["fwhmW"] == pytest.approx(expected_fwhm_mm, rel=0.04)
    assert metrics["fwhmH"] == pytest.approx(expected_fwhm_mm, rel=0.04)
    assert metrics["nyquistW"] == pytest.approx(1.0)
    assert metrics["nyquistH"] == pytest.approx(1.0)
    assert metrics["radialNyquist"] == pytest.approx(1.0)
    assert metrics["sourceSizeCorrected"] is False
    assert {warning["code"] for warning in data["qualityWarnings"]} == {"source-size-uncorrected"}
    assert data["curve"][0] == {"frequency": 0.0, "value": 1.0}
    assert all(
        data["curve"][index]["frequency"] <= data["curve"][index + 1]["frequency"]
        for index in range(len(data["curve"]) - 1)
    )


def test_mtf_api_uses_requested_source_slice_when_view_moves_before_analysis(tmp_path: Path) -> None:
    size = 33
    study_uid = generate_uid()
    series_uid = generate_uid()
    first_pixels = np.zeros((size, size), dtype=np.uint16)
    second_pixels = np.zeros((size, size), dtype=np.uint16)
    first_pixels[size // 2, size // 2] = 1000
    second_pixels[size // 2, size // 2] = 3000
    for index, pixels in enumerate((first_pixels, second_pixels), start=1):
        _write_ct_dicom(
            tmp_path / f"slice-{index}.dcm",
            pixels,
            pixel_spacing=(0.7, 0.7),
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            instance_number=index,
        )

    view = _register_stack_view(tmp_path)
    view.current_index = 1

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "sourceSliceIndex": 0,
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["peakValue"] == pytest.approx(1000.0)
    assert view.current_index == 1


def test_mtf_api_rejects_missing_requested_source_slice(tmp_path: Path) -> None:
    pixels = np.zeros((33, 33), dtype=np.uint16)
    pixels[16, 16] = 1000
    dicom_path = tmp_path / "mtf-source-slice.dcm"
    _write_ct_dicom(dicom_path, pixels, pixel_spacing=(0.7, 0.7))
    view = _register_stack_view(dicom_path)

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "sourceSliceIndex": 5,
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "mtf-source-slice-unavailable"


@pytest.mark.parametrize("view_type", ["MPR", "AX", "COR", "SAG"])
def test_mtf_api_rejects_interpolated_mpr_views(tmp_path: Path, view_type: str) -> None:
    pixels = np.zeros((33, 33), dtype=np.uint16)
    pixels[16, 16] = 1000
    dicom_path = tmp_path / f"mtf-{view_type.lower()}.dcm"
    _write_ct_dicom(dicom_path, pixels, pixel_spacing=(0.7, 0.7))
    view = _register_stack_view(dicom_path)
    view.view_type = view_type

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "mpr-ax",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "mtf-view-not-supported"
    assert "original 2D Stack or PET" in response.json()["detail"]["message"]


def test_mtf_api_auto_expands_small_roi_without_changing_returned_points(tmp_path: Path) -> None:
    pixels = np.zeros((33, 33), dtype=np.uint16)
    pixels[16, 16] = 1000
    dicom_path = tmp_path / "mtf-small-roi.dcm"
    _write_ct_dicom(dicom_path, pixels, pixel_spacing=(0.7, 0.7))
    view = _register_stack_view(dicom_path)
    points = [{"x": 0.48, "y": 0.48}, {"x": 0.52, "y": 0.52}]

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={"viewId": view.view_id, "viewportKey": "single", "points": points},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["points"] == points
    assert data["metrics"]["sampleCount"] == 81
    assert "roi-auto-expanded" in {warning["code"] for warning in data["qualityWarnings"]}


def test_mtf_api_returns_structured_reason_when_roi_has_no_point_source(tmp_path: Path) -> None:
    pixels = np.full((33, 33), 100, dtype=np.uint16)
    dicom_path = tmp_path / "mtf-no-source.dcm"
    _write_ct_dicom(dicom_path, pixels, pixel_spacing=(0.7, 0.7))
    view = _register_stack_view(dicom_path)

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "mtf-no-detectable-source"
    assert detail["message"]
    assert detail["suggestion"]


def test_mtf_api_accepts_original_pet_view(tmp_path: Path) -> None:
    pixels = np.zeros((33, 33), dtype=np.uint16)
    pixels[16, 16] = 1000
    dicom_path = tmp_path / "mtf-pet.dcm"
    _write_ct_dicom(dicom_path, pixels, pixel_spacing=(0.7, 0.7))
    view = _register_stack_view(dicom_path)
    view.view_type = "PET"

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["mtf50"] is None
    assert "mtf50-beyond-nyquist" in {
        warning["code"] for warning in response.json()["qualityWarnings"]
    }


def test_mtf_api_maps_dicom_row_column_spacing_to_h_w_metrics(tmp_path: Path) -> None:
    size = 81
    row_spacing, column_spacing = 0.8, 0.4
    physical_sigma = 1.6
    sigma_x = physical_sigma / column_spacing
    sigma_y = physical_sigma / row_spacing
    y_grid, x_grid = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    stored_pixels = np.rint(
        1000.0
        * np.exp(
            -(
                (x_grid - center) ** 2 / (2.0 * sigma_x**2)
                + (y_grid - center) ** 2 / (2.0 * sigma_y**2)
            )
        )
    ).astype(np.uint16)
    dicom_path = tmp_path / "mtf-anisotropic-spacing.dcm"
    _write_ct_dicom(
        dicom_path,
        stored_pixels,
        pixel_spacing=(row_spacing, column_spacing),
    )
    view = _register_stack_view(dicom_path)

    response = TestClient(fastapi_app).post(
        "/api/v1/view/mtf/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        },
    )

    assert response.status_code == 200
    metrics = response.json()["metrics"]
    expected_mtf50 = math.sqrt(math.log(2.0)) / (math.sqrt(2.0) * math.pi * physical_sigma)
    expected_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * physical_sigma
    assert metrics["mtf50W"] == pytest.approx(expected_mtf50, rel=0.02)
    assert metrics["mtf50H"] == pytest.approx(expected_mtf50, rel=0.02)
    assert metrics["fwhmW"] == pytest.approx(expected_fwhm, rel=0.03)
    assert metrics["fwhmH"] == pytest.approx(expected_fwhm, rel=0.03)
    assert metrics["nyquistW"] == pytest.approx(1.25)
    assert metrics["nyquistH"] == pytest.approx(0.625)


def test_water_qa_api_uses_rescaled_hu_and_anisotropic_physical_roi_size(tmp_path: Path) -> None:
    size = 256
    center = 128
    phantom_radius = 80
    stored_pixels = np.zeros((size, size), dtype=np.uint16)
    y_grid, x_grid = np.ogrid[:size, :size]
    water_mask = (x_grid - center) ** 2 + (y_grid - center) ** 2 <= phantom_radius**2
    stored_pixels[water_mask] = 1007
    dicom_path = tmp_path / "water-physical-truth.dcm"
    _write_ct_dicom(
        dicom_path,
        stored_pixels,
        pixel_spacing=(1.0, 0.5),
        rescale_intercept=-1000.0,
    )
    view = _register_stack_view(dicom_path)

    response = TestClient(fastapi_app).post(
        "/api/v1/view/qa/water/analyze",
        json={
            "viewId": view.view_id,
            "viewportKey": "single",
            "metrics": ["accuracy", "uniformity", "noise"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert len(data["rois"]) == 6
    assert {roi["kind"] for roi in data["rois"]} == {"water", "air"}
    assert data["metrics"]["accuracy"] == {
        "centerMean": 7.0,
        "deviationHu": 7.0,
        "targetHu": 0.0,
        "unit": "HU",
    }
    assert data["metrics"]["uniformity"]["peripheralMeans"] == [7.0, 7.0, 7.0, 7.0]
    assert data["metrics"]["uniformity"]["maxDeviation"] == 0.0
    assert data["metrics"]["noise"] == {"stdDev": 0.0, "unit": "HU"}

    center_stats = next(
        item for item in data["metrics"]["uniformity"]["roiStats"] if item["id"] == "center"
    )
    assert center_stats["mean"] == 7.0
    assert center_stats["sizeUnit"] == "mm"
    assert center_stats["areaUnit"] == "mm2"
    assert center_stats["height"] == pytest.approx(center_stats["width"] * 2.0, abs=0.02)
    assert center_stats["area"] == pytest.approx(
        math.pi * (center_stats["width"] / 2.0) * (center_stats["height"] / 2.0),
        abs=0.15,
    )


def test_water_qa_api_returns_structured_error_when_no_phantom_is_detected(tmp_path: Path) -> None:
    dicom_path = tmp_path / "uniform-no-phantom.dcm"
    _write_ct_dicom(
        dicom_path,
        np.full((64, 64), 1000, dtype=np.uint16),
        pixel_spacing=(1.0, 1.0),
        rescale_intercept=-1000.0,
    )
    view = _register_stack_view(dicom_path)

    response = TestClient(fastapi_app).post(
        "/api/v1/view/qa/water/analyze",
        json={"viewId": view.view_id, "viewportKey": "single", "metrics": ["accuracy"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "viewId": view.view_id,
        "viewportKey": "single",
        "rois": [],
        "metrics": {"accuracy": None, "uniformity": None, "noise": None},
        "status": "error",
        "message": "No water phantom contour was detected in the current image.",
    }


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        (
            "/api/v1/view/mtf/analyze",
            {"viewportKey": "single", "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]},
        ),
        (
            "/api/v1/view/qa/water/analyze",
            {"viewportKey": "single", "metrics": ["accuracy"]},
        ),
    ],
)
def test_medical_analysis_apis_reject_cross_workspace_view_access(
    tmp_path: Path,
    endpoint: str,
    payload: dict,
) -> None:
    dicom_path = tmp_path / "workspace-isolation.dcm"
    pixels = np.zeros((32, 32), dtype=np.uint16)
    pixels[16, 16] = 1000
    _write_ct_dicom(dicom_path, pixels, pixel_spacing=(1.0, 1.0))
    view = _register_stack_view(dicom_path, workspace_id="medical-a")

    response = TestClient(fastapi_app).post(
        endpoint,
        headers={"X-DicomVision-Workspace-Id": "medical-b"},
        json={"viewId": view.view_id, **payload},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "viewId not found"
