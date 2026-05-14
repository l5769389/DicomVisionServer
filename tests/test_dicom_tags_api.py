from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pydicom
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry


def _create_test_dicom(path: Path, *, series_instance_uid: str | None = None, instance_number: int = 1) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.PatientName = "Tag^Tester"
    dataset.PatientID = "patient-001"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_instance_uid or generate_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesDescription = "Tag API Series"
    dataset.Modality = "OT"
    dataset.InstanceNumber = instance_number
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelData = np.array([[1, 2], [3, 4]], dtype=np.uint16).tobytes()

    nested = Dataset()
    nested.CodeValue = "ABC"
    nested.CodeMeaning = "Nested value"
    dataset.ConceptCodeSequence = [nested]
    dataset.save_as(str(path), write_like_original=False)


def test_dicom_tags_api_returns_instance_tags(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "tag-test.dcm"
    _create_test_dicom(dicom_path)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id

    client = TestClient(fastapi_app)
    response = client.post(
        "/api/v1/dicom/tags",
        json={
            "seriesId": series_id,
            "index": 0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["seriesId"] == series_id
    assert data["index"] == 0
    assert data["total"] == 1
    assert data["instanceNumber"] == 1
    assert data["filePath"].endswith("tag-test.dcm")

    names = {item["name"] for item in data["items"]}
    assert "Patient's Name" in names
    assert "Pixel Data" in names
    assert "Concept Code Sequence" in names

    pixel_data_row = next(item for item in data["items"] if item["name"] == "Pixel Data")
    assert pixel_data_row["value"] == "<Pixel Data omitted>"

    nested_row = next(item for item in data["items"] if item["name"] == "Code Meaning")
    assert nested_row["value"] == "Nested value"
    assert nested_row["depth"] >= 2
    assert nested_row["tagPath"]


def test_load_folder_response_includes_series_thumbnail(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "thumbnail-test.dcm"
    _create_test_dicom(dicom_path)

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/dicom/loadFolder", json={"folderPath": str(tmp_path)})

    assert response.status_code == 200
    data = response.json()
    series = data["seriesList"][0]
    assert series["thumbnailSrc"] == ""
    assert series["thumbnailUrl"].startswith("/api/v1/dicom/thumbnail?seriesId=")

    thumbnail_response = client.get(series["thumbnailUrl"])

    assert thumbnail_response.status_code == 200
    assert thumbnail_response.headers["content-type"] == "image/png"
    assert thumbnail_response.content.startswith(b"\x89PNG")


def test_load_folder_accepts_single_dicom_file_path(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "single-file-test.dcm"
    other_dicom_path = tmp_path / "other-series.dcm"
    _create_test_dicom(dicom_path)
    _create_test_dicom(other_dicom_path)

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/dicom/loadFolder", json={"folderPath": str(dicom_path)})

    assert response.status_code == 200
    data = response.json()
    assert len(data["seriesList"]) == 1
    series = data["seriesList"][0]
    assert series["folderPath"] == str(tmp_path)
    assert series["instanceCount"] == 1


def test_modify_dicom_tag_current_instance_returns_dicom_artifact(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "tag-edit-current.dcm"
    _create_test_dicom(dicom_path)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)
    tags_response = client.post("/api/v1/dicom/tags", json={"seriesId": series_id, "index": 0})
    patient_id_row = next(item for item in tags_response.json()["items"] if item["keyword"] == "PatientID")

    response = client.post(
        "/api/v1/dicom/modifyTag",
        json={
            "seriesId": series_id,
            "index": 0,
            "tagPath": patient_id_row["tagPath"],
            "value": "patient-002",
            "scope": "current",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/dicom"
    assert response.headers["x-dicomvision-artifact-kind"] == "dicom"
    assert response.headers["x-dicomvision-modified-count"] == "1"
    assert response.headers["x-dicomvision-file-name"].endswith(".dcm")

    modified_dataset = pydicom.dcmread(BytesIO(response.content), force=True)
    original_dataset = pydicom.dcmread(str(dicom_path), force=True)
    assert modified_dataset.PatientID == "patient-002"
    assert original_dataset.PatientID == "patient-001"


def test_modify_nested_dicom_tag_writes_copy(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "tag-edit-nested.dcm"
    _create_test_dicom(dicom_path)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)
    tags_response = client.post("/api/v1/dicom/tags", json={"seriesId": series_id, "index": 0})
    code_meaning_row = next(item for item in tags_response.json()["items"] if item["keyword"] == "CodeMeaning")

    response = client.post(
        "/api/v1/dicom/modifyTag",
        json={
            "seriesId": series_id,
            "index": 0,
            "tagPath": code_meaning_row["tagPath"],
            "value": "Edited nested value",
            "scope": "current",
        },
    )

    assert response.status_code == 200
    modified_dataset = pydicom.dcmread(BytesIO(response.content), force=True)
    assert modified_dataset.ConceptCodeSequence[0].CodeMeaning == "Edited nested value"


def test_modify_dicom_tag_series_scope_writes_all_instances(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    series_instance_uid = generate_uid()
    _create_test_dicom(tmp_path / "tag-edit-series-1.dcm", series_instance_uid=series_instance_uid, instance_number=1)
    _create_test_dicom(tmp_path / "tag-edit-series-2.dcm", series_instance_uid=series_instance_uid, instance_number=2)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)
    tags_response = client.post("/api/v1/dicom/tags", json={"seriesId": series_id, "index": 0})
    patient_name_row = next(item for item in tags_response.json()["items"] if item["keyword"] == "PatientName")

    response = client.post(
        "/api/v1/dicom/modifyTag",
        json={
            "seriesId": series_id,
            "index": 0,
            "tagPath": patient_name_row["tagPath"],
            "value": "Edited^Series",
            "scope": "series",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-dicomvision-artifact-kind"] == "zip"
    assert response.headers["x-dicomvision-modified-count"] == "2"
    assert response.headers["x-dicomvision-file-name"].endswith(".zip")

    with ZipFile(BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert len(names) == 2
        assert all(name.endswith(".dcm") for name in names)
        assert len({Path(name).parent for name in names}) == 1
        datasets = [pydicom.dcmread(BytesIO(archive.read(name)), force=True) for name in names]

    for modified_dataset in datasets:
        assert str(modified_dataset.PatientName) == "Edited^Series"
