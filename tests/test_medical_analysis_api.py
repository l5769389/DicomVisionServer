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
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientName = "Medical^Analysis"
    dataset.PatientID = "MEDICAL-ANALYSIS"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Medical analysis API truth"
    dataset.InstanceNumber = 1
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
    dataset = dicom_cache.get(
        series_registry.get(series_id, workspace_id=workspace_id).instances[0].sop_instance_uid,
        path,
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
    assert metrics["fwhmW"] == pytest.approx(expected_fwhm_mm, rel=0.04)
    assert metrics["fwhmH"] == pytest.approx(expected_fwhm_mm, rel=0.04)
    assert data["curve"][0] == {"frequency": 0.0, "value": 1.0}
    assert all(
        data["curve"][index]["frequency"] <= data["curve"][index + 1]["frequency"]
        for index in range(len(data["curve"]) - 1)
    )
    assert all(
        data["curve"][index]["value"] >= data["curve"][index + 1]["value"]
        for index in range(len(data["curve"]) - 1)
    )


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
