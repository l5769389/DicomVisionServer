import hashlib
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.services.dicom_cache import DicomCache, dicom_cache
from app.services.series_registry import series_registry
from app.services.view_registry import view_registry


def _write_export_source_dicom(path: Path) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientName = "Export^Roundtrip"
    dataset.PatientID = "EXPORT-ROUNDTRIP"
    dataset.StudyID = "STUDY-1"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Export roundtrip source"
    dataset.InstanceNumber = 1
    dataset.Rows = 8
    dataset.Columns = 12
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 1
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [1.5, 0.75]
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1024.0
    dataset.WindowWidth = 800.0
    dataset.WindowCenter = 0.0
    pixels = np.arange(dataset.Rows * dataset.Columns, dtype=np.int16).reshape(dataset.Rows, dataset.Columns)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dicom_secondary_capture_export_roundtrips_rendered_pixels_without_mutating_source(
    tmp_path: Path,
) -> None:
    series_registry.clear()
    dicom_cache.clear()
    source_path = tmp_path / "source.dcm"
    source_dataset = _write_export_source_dicom(source_path)
    source_digest_before = _sha256(source_path)

    try:
        loaded = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
        series_id = loaded.series_list[0].series_id
        client = TestClient(fastapi_app)
        create_response = client.post(
            "/api/v1/view/create",
            json={"seriesId": series_id, "viewType": "Stack"},
        )
        assert create_response.status_code == 200
        view_id = create_response.json()["viewId"]
        view = view_registry.get(view_id)
        view.width = 96
        view.height = 64

        png_response = client.post(
            "/api/v1/view/export",
            json={"viewId": view_id, "exportFormat": "png"},
        )
        dicom_response = client.post(
            "/api/v1/view/export",
            json={"viewId": view_id, "exportFormat": "dicom"},
        )

        assert png_response.status_code == 200
        assert png_response.headers["content-type"].startswith("image/png")
        assert dicom_response.status_code == 200
        assert dicom_response.headers["content-type"].startswith("application/dicom")
        assert ".dcm" in dicom_response.headers["content-disposition"]

        with Image.open(io.BytesIO(png_response.content)) as png_image:
            expected_rgb = np.asarray(png_image.convert("RGB")).copy()

        exported = dcmread(io.BytesIO(dicom_response.content))
        assert exported.file_meta.MediaStorageSOPClassUID == exported.SOPClassUID
        assert exported.file_meta.MediaStorageSOPInstanceUID == exported.SOPInstanceUID
        assert exported.PatientID == source_dataset.PatientID
        assert exported.StudyInstanceUID == source_dataset.StudyInstanceUID
        assert exported.SOPInstanceUID != source_dataset.SOPInstanceUID
        assert exported.SeriesInstanceUID != source_dataset.SeriesInstanceUID
        assert exported.ImageType == ["DERIVED", "SECONDARY", "OTHER"]
        assert exported.BurnedInAnnotation == "YES"
        assert exported.PhotometricInterpretation == "RGB"
        assert (exported.Rows, exported.Columns) == expected_rgb.shape[:2]
        np.testing.assert_array_equal(exported.pixel_array, expected_rgb)

        exported_path = tmp_path / "exported-secondary-capture.dcm"
        exported_path.write_bytes(dicom_response.content)
        cached_export = DicomCache().get(str(exported.SOPInstanceUID), exported_path)
        np.testing.assert_array_equal(cached_export.source_pixels, expected_rgb)

        assert _sha256(source_path) == source_digest_before
        source_after = dcmread(source_path)
        assert source_after.SOPInstanceUID == source_dataset.SOPInstanceUID
        assert source_after.SeriesInstanceUID == source_dataset.SeriesInstanceUID
        np.testing.assert_array_equal(source_after.pixel_array, source_dataset.pixel_array)
    finally:
        view_registry._view_by_id.clear()
        series_registry.clear()
        dicom_cache.clear()
