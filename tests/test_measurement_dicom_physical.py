import math
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.schemas.dicom import LoadFolderRequest
from app.schemas.view import ViewCreateRequest, ViewOperationRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service


def _write_physical_ct_dicom(path: Path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientName = "Physical^Truth"
    dataset.PatientID = "PHYSICAL-TRUTH"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Physical measurement truth"
    dataset.InstanceNumber = 1
    dataset.Rows = 16
    dataset.Columns = 16
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15

    # DICOM PixelSpacing is [row spacing, column spacing]. The viewer must
    # therefore use 2.0 mm for a y delta and 0.5 mm for an x delta.
    dataset.PixelSpacing = [2.0, 0.5]
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1024.0
    dataset.RescaleType = "HU"
    dataset.PixelData = np.full((16, 16), 600, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


@pytest.fixture
def physical_ct_view(tmp_path: Path):
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "physical-truth.dcm"
    _write_physical_ct_dicom(dicom_path)

    loaded = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = loaded.series_list[0].series_id
    created = view_registry.create(ViewCreateRequest(seriesId=series_id, viewType="Stack"))
    view = view_registry.get(created.view_id)
    view.width = 16
    view.height = 16
    view.is_initialized = True

    try:
        yield view
    finally:
        view_registry._view_by_id.pop(view.view_id, None)
        series_registry.clear()
        dicom_cache.clear()


def test_stack_measurement_uses_dicom_physical_spacing_and_rescaled_ct_values(
    physical_ct_view,
    monkeypatch,
) -> None:
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: object())

    viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate(
            {
                "viewId": physical_ct_view.view_id,
                "opType": "measurement",
                "subOpType": "rect",
                "actionType": "end",
                "measurementId": "physical-rect",
                "points": [
                    {"x": 4 / 16, "y": 4 / 16},
                    {"x": 8 / 16, "y": 8 / 16},
                ],
            }
        )
    )

    [measurement] = physical_ct_view.measurements
    assert [(point.x, point.y) for point in measurement.points] == pytest.approx([(4.0, 4.0), (8.0, 8.0)])
    assert measurement.metrics.unit == "mm"
    assert measurement.metrics.area_unit == "mm2"
    assert measurement.metrics.width == pytest.approx(4 * 0.5)
    assert measurement.metrics.height == pytest.approx(4 * 2.0)
    assert measurement.metrics.area == pytest.approx(16.0)

    # Stored value 600 is converted to 600 * 2 - 1024 = 176 HU before
    # measurement statistics are calculated.
    assert measurement.metrics.mean == pytest.approx(176.0)
    assert measurement.metrics.standard_deviation == pytest.approx(0.0)
    assert measurement.metrics.minimum == pytest.approx(176.0)
    assert measurement.metrics.maximum == pytest.approx(176.0)
    assert measurement.label_lines[:3] == (
        "Size 2.0 * 8.0 mm",
        "Area 16.0 mm2",
        "Mean 176.0",
    )


def test_stack_line_measurement_preserves_physical_length_through_canvas_mapping(
    physical_ct_view,
    monkeypatch,
) -> None:
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: object())

    viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate(
            {
                "viewId": physical_ct_view.view_id,
                "opType": "measurement",
                "subOpType": "line",
                "actionType": "end",
                "measurementId": "physical-line",
                "points": [
                    {"x": 2 / 16, "y": 3 / 16},
                    {"x": 10 / 16, "y": 7 / 16},
                ],
            }
        )
    )

    [measurement] = physical_ct_view.measurements
    expected_length_mm = math.hypot(8 * 0.5, 4 * 2.0)
    assert measurement.metrics.unit == "mm"
    assert measurement.metrics.length == pytest.approx(expected_length_mm)
    assert measurement.label_lines == (f"{expected_length_mm:.1f} mm",)
