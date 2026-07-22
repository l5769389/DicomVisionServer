from io import BytesIO

import pytest
from fastapi import HTTPException
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import CTImageStorage, EnhancedSRStorage, generate_uid

from app.models.viewer import ViewRecord
from app.schemas.view import (
    ViewExportMeasurementOverlayPayload,
    ViewExportOverlaysPayload,
    ViewExportPointPayload,
)
from app.services.dicom_sr_export_service import build_measurement_sr_dicom_bytes


def _reference_dataset() -> Dataset:
    dataset = Dataset()
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = generate_uid()
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = "PATIENT-1"
    dataset.PatientName = "Anonymous"
    dataset.Rows = 512
    dataset.Columns = 256
    return dataset


def test_measurement_sr_export_writes_structured_report_with_spatial_reference() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=128, height=128)
    overlays = ViewExportOverlaysPayload(
        measurements=[
            ViewExportMeasurementOverlayPayload(
                measurementId="m-1",
                toolType="line",
                points=[ViewExportPointPayload(x=0.0, y=0.0), ViewExportPointPayload(x=1.0, y=1.0)],
                labelLines=["Length: 12.5 mm"],
            )
        ]
    )

    exported = build_measurement_sr_dicom_bytes(view, overlays, _reference_dataset())
    dataset = dcmread(BytesIO(exported))

    assert dataset.SOPClassUID == EnhancedSRStorage
    assert dataset.Modality == "SR"
    assert dataset.PatientID == "PATIENT-1"
    assert dataset.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].ReferencedSOPSequence

    group = dataset.ContentSequence[0]
    assert group.ValueType == "CONTAINER"
    content_items = list(group.ContentSequence)
    scoord = next(item for item in content_items if item.ValueType == "SCOORD")
    assert scoord.GraphicType == "POLYLINE"
    assert list(scoord.GraphicData) == pytest.approx([0.0, 0.0, 255.0, 511.0])

    numeric_item = next(item for item in content_items if item.ValueType == "NUM")
    measured_value = numeric_item.MeasuredValueSequence[0]
    assert float(measured_value.NumericValue) == pytest.approx(12.5)
    assert measured_value.MeasurementUnitsCodeSequence[0].CodeValue == "mm"


def test_measurement_sr_export_requires_measurements() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")

    with pytest.raises(HTTPException) as exc_info:
        build_measurement_sr_dicom_bytes(
            view,
            ViewExportOverlaysPayload(measurements=[]),
            _reference_dataset(),
        )

    assert exc_info.value.status_code == 400


def test_alignment_angle_sr_export_preserves_degree_units() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=128, height=128)
    overlays = ViewExportOverlaysPayload(
        measurements=[
            ViewExportMeasurementOverlayPayload(
                measurementId="alignment-1",
                toolType="alignment-horizontal",
                points=[ViewExportPointPayload(x=0.1, y=0.4), ViewExportPointPayload(x=0.9, y=0.5)],
                labelLines=["ΔH 7.1°", "42.0 mm"],
            )
        ]
    )

    dataset = dcmread(BytesIO(build_measurement_sr_dicom_bytes(view, overlays, _reference_dataset())))
    numeric_items = [item for item in dataset.ContentSequence[0].ContentSequence if item.ValueType == "NUM"]
    angle_item = next(item for item in numeric_items if float(item.MeasuredValueSequence[0].NumericValue) == 7.1)

    unit = angle_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0]
    assert unit.CodeValue == "deg"
    assert unit.CodeMeaning == "degree"
