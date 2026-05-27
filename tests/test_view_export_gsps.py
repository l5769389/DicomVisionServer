from io import BytesIO

import pytest
from fastapi import HTTPException
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import CTImageStorage, GrayscaleSoftcopyPresentationStateStorage, generate_uid

from app.models.viewer import ViewRecord
from app.schemas.view import (
    ViewExportAnnotationOverlayPayload,
    ViewExportMeasurementOverlayPayload,
    ViewExportOverlaysPayload,
    ViewExportPointPayload,
)
from app.services.dicom_gsps_export_service import build_gsps_dicom_bytes


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


def test_gsps_export_writes_presentation_state_with_graphics_and_text() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=128, height=128)
    overlays = ViewExportOverlaysPayload(
        annotations=[
            ViewExportAnnotationOverlayPayload(
                annotationId="a-1",
                toolType="arrow",
                points=[ViewExportPointPayload(x=0.25, y=0.25), ViewExportPointPayload(x=0.5, y=0.5)],
                text="Note",
            )
        ],
        measurements=[
            ViewExportMeasurementOverlayPayload(
                measurementId="m-1",
                toolType="line",
                points=[ViewExportPointPayload(x=0.0, y=0.0), ViewExportPointPayload(x=1.0, y=1.0)],
                labelLines=["210.2 mm"],
            )
        ],
    )

    exported = build_gsps_dicom_bytes(view, overlays, _reference_dataset())
    dataset = dcmread(BytesIO(exported))

    assert dataset.SOPClassUID == GrayscaleSoftcopyPresentationStateStorage
    assert dataset.Modality == "PR"
    assert dataset.PatientID == "PATIENT-1"
    assert dataset.ReferencedSeriesSequence[0].ReferencedImageSequence[0].ReferencedSOPInstanceUID
    assert dataset.DisplayedAreaSelectionSequence[0].DisplayedAreaTopLeftHandCorner == [1, 1]
    assert dataset.DisplayedAreaSelectionSequence[0].DisplayedAreaBottomRightHandCorner == [256, 512]

    layer_names = {item.GraphicLayer for item in dataset.GraphicLayerSequence}
    assert {"MEASURE", "ANNOT"} <= layer_names

    measurement_item = next(item for item in dataset.GraphicAnnotationSequence if item.GraphicLayer == "MEASURE")
    graphic_object = measurement_item.GraphicObjectSequence[0]
    assert graphic_object.GraphicType == "POLYLINE"
    assert list(graphic_object.GraphicData) == pytest.approx([1.0, 1.0, 256.0, 512.0])
    assert measurement_item.TextObjectSequence[0].UnformattedTextValue == "210.2 mm"

    annotation_item = next(item for item in dataset.GraphicAnnotationSequence if item.GraphicLayer == "ANNOT")
    assert annotation_item.TextObjectSequence[0].UnformattedTextValue == "Note"


def test_gsps_export_requires_stack_view() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="MPR")
    overlays = ViewExportOverlaysPayload(
        measurements=[
            ViewExportMeasurementOverlayPayload(
                measurementId="m-1",
                toolType="line",
                points=[ViewExportPointPayload(x=0.0, y=0.0), ViewExportPointPayload(x=1.0, y=1.0)],
                labelLines=[],
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        build_gsps_dicom_bytes(view, overlays, _reference_dataset())

    assert exc_info.value.status_code == 400
