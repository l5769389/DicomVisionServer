from fastapi import HTTPException
import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.models.viewer import ViewRecord
from app.schemas.dicom import DicomTagsRequest, LoadFolderRequest
from app.schemas.view import (
    ViewExportMeasurementOverlayPayload,
    ViewExportOverlaysPayload,
    ViewExportPointPayload,
)
from app.services.dicom_cache import dicom_cache
from app.services.dicom_sr_export_service import build_measurement_sr_dicom_bytes
from app.services.dicom_tag_service import dicom_tag_service
from app.services.series_registry import series_registry


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


def _measurement_overlays() -> ViewExportOverlaysPayload:
    return ViewExportOverlaysPayload(
        measurements=[
            ViewExportMeasurementOverlayPayload(
                measurementId="m-1",
                toolType="line",
                points=[ViewExportPointPayload(x=0.0, y=0.0), ViewExportPointPayload(x=1.0, y=1.0)],
                labelLines=["Length: 12.5 mm"],
            )
        ]
    )


def _create_image_dicom(path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = "PATIENT-1"
    dataset.PatientName = "Anonymous"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "OT"
    dataset.SeriesDescription = "Image Series"
    dataset.InstanceNumber = 1
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelData = np.array([[1, 2], [3, 4]], dtype=np.uint16).tobytes()
    dataset.save_as(str(path), write_like_original=False)


def test_load_folder_registers_sr_as_dicom_document_series(tmp_path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    try:
        exported = build_measurement_sr_dicom_bytes(
            ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=128, height=128),
            _measurement_overlays(),
            _reference_dataset(),
        )
        sr_path = tmp_path / "measurement-report.dcm"
        sr_path.write_bytes(exported)

        response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

        assert len(response.series_list) == 1
        summary = response.series_list[0]
        assert summary.modality == "SR"
        assert summary.is_image_series is False
        assert summary.standard_object_type == "DICOM_SR"
        assert summary.preferred_view_type == "Tag"
        assert summary.instance_count == 1
        assert summary.thumbnail_url == ""
        assert summary.width is None
        assert summary.height is None

        tags = dicom_tag_service.get_series_tags(DicomTagsRequest(seriesId=summary.series_id, index=0))
        assert tags.total == 1
        assert any(item.keyword == "ContentSequence" for item in tags.items)

        with pytest.raises(HTTPException) as exc_info:
            series_registry.get_series_thumbnail_png(summary.series_id)
        assert exc_info.value.status_code == 404
    finally:
        series_registry.clear()
        dicom_cache.clear()


def test_load_folder_keeps_image_series_before_sr_documents(tmp_path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    try:
        _create_image_dicom(tmp_path / "image.dcm")
        exported = build_measurement_sr_dicom_bytes(
            ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=128, height=128),
            _measurement_overlays(),
            _reference_dataset(),
        )
        (tmp_path / "report.dcm").write_bytes(exported)

        response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

        assert len(response.series_list) == 2
        assert response.series_list[0].is_image_series is True
        assert response.series_list[1].standard_object_type == "DICOM_SR"
        assert response.series_id == response.series_list[0].series_id
    finally:
        series_registry.clear()
        dicom_cache.clear()
