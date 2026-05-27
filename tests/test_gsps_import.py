from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.models.viewer import ViewRecord
from app.schemas.dicom import DicomTagsRequest, LoadFolderRequest
from app.schemas.view import (
    ViewExportMeasurementOverlayPayload,
    ViewExportOverlaysPayload,
    ViewExportPointPayload,
)
from app.services.dicom_cache import dicom_cache
from app.services.dicom_gsps_export_service import build_gsps_dicom_bytes
from app.services.dicom_gsps_import_service import is_gsps_dataset, parse_gsps_dataset
from app.services.dicom_tag_service import dicom_tag_service
from app.services.series_registry import series_registry


def _create_test_dicom(path: Path) -> Dataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientName = "GSPS^Tester"
    dataset.PatientID = "patient-001"
    dataset.StudyDate = "20260527"
    dataset.StudyTime = "101112"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesDescription = "GSPS Source"
    dataset.Modality = "OT"
    dataset.InstanceNumber = 1
    dataset.Rows = 512
    dataset.Columns = 256
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelData = np.array([[1, 2], [3, 4]], dtype=np.uint16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)
    return dataset


def _create_gsps(path: Path, reference_dataset: Dataset) -> None:
    overlays = ViewExportOverlaysPayload(
        measurements=[
            ViewExportMeasurementOverlayPayload(
                measurementId="m-1",
                toolType="line",
                points=[ViewExportPointPayload(x=0.0, y=0.0), ViewExportPointPayload(x=1.0, y=1.0)],
                labelLines=["210.2 mm"],
            )
        ]
    )
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    path.write_bytes(build_gsps_dicom_bytes(view, overlays, reference_dataset))


def test_parse_gsps_dataset_returns_measurements_for_referenced_image(tmp_path: Path) -> None:
    source_dataset = _create_test_dicom(tmp_path / "source.dcm")
    gsps_path = tmp_path / "source-presentation-state.dcm"
    _create_gsps(gsps_path, source_dataset)
    gsps_dataset = dcmread(str(gsps_path), stop_before_pixels=True)

    assert is_gsps_dataset(gsps_dataset)
    records = parse_gsps_dataset(gsps_dataset, gsps_path)

    assert len(records) == 1
    record = records[0]
    assert record.referenced_sop_instance_uid == source_dataset.SOPInstanceUID
    assert record.measurements[0].label_lines == ("210.2 mm",)
    assert record.measurements[0].points[0].x == 0
    assert record.measurements[0].points[1].x == 255
    assert record.measurements[0].points[1].y == 511


def test_load_folder_attaches_gsps_to_source_series_without_extra_series(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    source_dataset = _create_test_dicom(tmp_path / "source.dcm")
    _create_gsps(tmp_path / "source-presentation-state.dcm", source_dataset)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

    assert len(load_response.series_list) == 1
    series = series_registry.get(load_response.series_list[0].series_id)
    states = series.presentation_states_by_sop_uid[source_dataset.SOPInstanceUID]
    assert len(states) == 1
    assert states[0].measurements[0].label_lines == ("210.2 mm",)


def test_load_folder_registers_unattached_gsps_as_dicom_document_series(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    try:
        source_path = tmp_path / "source.dcm"
        source_dataset = _create_test_dicom(source_path)
        source_path.unlink()
        _create_gsps(tmp_path / "source-presentation-state.dcm", source_dataset)

        load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

        assert len(load_response.series_list) == 1
        summary = load_response.series_list[0]
        assert summary.modality == "PR"
        assert summary.is_image_series is False
        assert summary.standard_object_type == "DICOM_GSPS"
        assert summary.preferred_view_type == "Tag"
        assert summary.instance_count == 1
        assert summary.thumbnail_url == ""

        tags = dicom_tag_service.get_series_tags(DicomTagsRequest(seriesId=summary.series_id, index=0))
        assert tags.total == 1
        assert any(item.keyword == "GraphicAnnotationSequence" for item in tags.items)

        with pytest.raises(HTTPException) as exc_info:
            series_registry.get_series_thumbnail_png(summary.series_id)
        assert exc_info.value.status_code == 404
    finally:
        series_registry.clear()
        dicom_cache.clear()
