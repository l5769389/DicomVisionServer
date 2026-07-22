import math
from pathlib import Path

from fastapi import HTTPException
import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.models.measurement import MeasurementToolType
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
    dataset.Rows = 64
    dataset.Columns = 64
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
    dataset.PixelData = np.full((64, 64), 600, dtype=np.uint16).tobytes()
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
    view.width = 64
    view.height = 64
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
                    {"x": 4 / 64, "y": 4 / 64},
                    {"x": 8 / 64, "y": 8 / 64},
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
                    {"x": 2 / 64, "y": 3 / 64},
                    {"x": 10 / 64, "y": 7 / 64},
                ],
            }
        )
    )

    [measurement] = physical_ct_view.measurements
    expected_length_mm = math.hypot(8 * 0.5, 4 * 2.0)
    assert measurement.metrics.unit == "mm"
    assert measurement.metrics.length == pytest.approx(expected_length_mm)
    assert measurement.label_lines == (f"{expected_length_mm:.1f} mm",)


@pytest.mark.parametrize(
    ("tool_type", "expected_degrees", "expected_label"),
    [
        ("alignment-horizontal", math.degrees(math.atan2(2.0, 4.0)), "ΔH 26.6°"),
        ("alignment-vertical", math.degrees(math.atan2(4.0, 2.0)), "ΔV 63.4°"),
    ],
)
def test_alignment_measurement_uses_real_dicom_physical_axes(
    physical_ct_view,
    monkeypatch,
    tool_type: MeasurementToolType,
    expected_degrees: float,
    expected_label: str,
) -> None:
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: object())

    viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate(
            {
                "viewId": physical_ct_view.view_id,
                "opType": "measurement",
                "subOpType": tool_type,
                "actionType": "end",
                "measurementId": f"physical-{tool_type}",
                "points": [
                    {"x": 0 / 64, "y": 0 / 64},
                    {"x": 40 / 64, "y": 5 / 64},
                ],
            }
        )
    )

    [measurement] = physical_ct_view.measurements
    assert measurement.metrics.unit == "mm"
    assert measurement.metrics.length == pytest.approx(math.hypot(20.0, 10.0))
    assert measurement.metrics.angle_degrees == pytest.approx(expected_degrees)
    assert measurement.label_lines == (expected_label, "22.4 mm")


def test_alignment_measurement_never_emits_a_pseudo_physical_result_without_pixel_spacing(
    physical_ct_view,
    monkeypatch,
) -> None:
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: object())
    series = series_registry.get(physical_ct_view.series_id)
    instance = series.instances[physical_ct_view.current_index]
    cached = dicom_cache.get(instance.sop_instance_uid, instance.path)
    del cached.dataset.PixelSpacing

    request = ViewOperationRequest.model_validate(
        {
            "viewId": physical_ct_view.view_id,
            "opType": "measurement",
            "subOpType": "alignment-horizontal",
            "actionType": "end",
            "measurementId": "unavailable-spacing-alignment",
            "points": [
                {"x": 0.0, "y": 0.0},
                {"x": 40 / 64, "y": 5 / 64},
            ],
        }
    )

    preview = viewer_service._build_measurement_preview(
        physical_ct_view,
        request.model_copy(update={"action_type": "move"}),
    )
    assert preview is not None
    assert preview["labelLines"] == ["DICOM Pixel Spacing unavailable"]
    assert "metrics" not in preview

    viewer_service.handle_view_operation(request)

    [measurement] = physical_ct_view.measurements
    assert measurement.metrics.length is None
    assert measurement.metrics.angle_degrees is None
    assert measurement.label_lines == ("DICOM Pixel Spacing unavailable",)

    invalid_spacing_dataset = Dataset()
    invalid_spacing_dataset.PixelSpacing = [0, float("nan")]
    assert viewer_service._get_stack_spacing_xy(invalid_spacing_dataset) is None


@pytest.mark.parametrize(
    ("view_type", "modality"),
    [("AX", "CT"), ("Stack", "MR")],
)
def test_alignment_measurement_is_rejected_outside_an_ordinary_2d_ct_view(
    physical_ct_view,
    view_type: str,
    modality: str,
) -> None:
    physical_ct_view.view_type = view_type
    series_registry.get(physical_ct_view.series_id).modality = modality
    request = ViewOperationRequest.model_validate(
        {
            "viewId": physical_ct_view.view_id,
            "opType": "measurement",
            "subOpType": "alignment-horizontal",
            "actionType": "end",
            "measurementId": "unsupported-alignment",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 40 / 64, "y": 5 / 64}],
        }
    )

    with pytest.raises(HTTPException, match="Available only in 2D CT views") as exc_info:
        viewer_service._handle_measurement(physical_ct_view, request)

    assert exc_info.value.status_code == 400
    assert physical_ct_view.measurements == []
