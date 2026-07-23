from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import py7zr
import pytest
from fastapi import HTTPException
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.services.dicom_cache import dicom_cache
from app.services.dicom_upload_service import DicomUploadService
from app.services.series_registry import series_registry


def _create_dicom(path: Path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientName = "Zip^Tester"
    dataset.PatientID = "zip-patient"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "OT"
    dataset.SeriesDescription = "ZIP Import"
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


def test_zip_import_extracts_dicom_and_ignores_metadata(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "IM0001"
    _create_dicom(dicom_path)
    archive_path = tmp_path / "study.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(dicom_path, "study/IM0001")
        archive.writestr("study/README.txt", "not a DICOM file")
        archive.writestr("__MACOSX/._IM0001", "metadata")

    service = DicomUploadService(upload_root=tmp_path / "uploads")
    response = service.load_archive_path(str(archive_path), workspace_id="zip-test")

    assert len(response.series_list) == 1
    loaded_path = Path(response.series_list[0].folder_path)
    assert loaded_path.is_dir()
    assert not list(loaded_path.rglob("*.zip"))
    series_registry.clear()
    dicom_cache.clear()


def test_zip_import_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("../escape.dcm", b"not a DICOM file")

    service = DicomUploadService(upload_root=tmp_path / "uploads")
    with pytest.raises(HTTPException, match="unsafe file path"):
        service.load_archive_path(str(archive_path))


def test_zip_import_rejects_excessive_compression_ratio(tmp_path: Path) -> None:
    archive_path = tmp_path / "high-ratio.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("compressed.dcm", b"0" * (1024 * 1024))

    service = DicomUploadService(upload_root=tmp_path / "uploads")
    with pytest.raises(HTTPException, match="compression ratio"):
        service.load_archive_path(str(archive_path))


def test_7z_import_extracts_dicom_and_ignores_metadata(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "IM0001"
    _create_dicom(dicom_path)
    archive_path = tmp_path / "study.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(dicom_path, arcname="study/IM0001")
        archive.writestr(b"not a DICOM file", "study/README.txt")

    service = DicomUploadService(upload_root=tmp_path / "uploads")
    response = service.load_archive_path(str(archive_path), workspace_id="seven-zip-test")

    assert len(response.series_list) == 1
    series_registry.clear()
    dicom_cache.clear()


def test_7z_import_accepts_solid_archive_members_without_compressed_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """py7zr may report None for per-file compressed sizes in solid archives."""

    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "IM0001"
    _create_dicom(dicom_path)
    archive_path = tmp_path / "solid-study.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(dicom_path, arcname="study/IM0001")

    original_list = py7zr.SevenZipFile.list

    def list_without_member_compressed_size(archive: py7zr.SevenZipFile):
        entries = original_list(archive)
        for entry in entries:
            entry.compressed = None
        return entries

    monkeypatch.setattr(py7zr.SevenZipFile, "list", list_without_member_compressed_size)
    service = DicomUploadService(upload_root=tmp_path / "uploads")
    response = service.load_archive_path(str(archive_path), workspace_id="solid-seven-zip-test")

    assert len(response.series_list) == 1
    series_registry.clear()
    dicom_cache.clear()


def test_supported_archive_suffixes_are_upload_candidates(tmp_path: Path) -> None:
    service = DicomUploadService(upload_root=tmp_path / "uploads")

    assert service._is_upload_candidate(Path("study.zip"))
    assert service._is_upload_candidate(Path("study.7z"))
    assert service._is_upload_candidate(Path("study.rar"))
