from io import BytesIO
import os
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pydicom
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.services.dicom_cache import dicom_cache
from app.services.dicom_deidentify_job_service import dicom_deidentify_job_service
from app.services.dicom_tag_job_service import DicomTagModifyJobService, dicom_tag_job_service
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
    dataset.PatientBirthDate = "19700101"
    dataset.PatientSex = "O"
    dataset.StudyDate = "20260514"
    dataset.StudyTime = "101112"
    dataset.AccessionNumber = "ACC-001"
    dataset.InstitutionName = "DicomVision Hospital"
    dataset.ReferringPhysicianName = "Doctor^Demo"
    dataset.StationName = "CT-STATION-1"
    dataset.DeviceSerialNumber = "DEVICE-SERIAL-1"
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
    dataset.add_new((0x0011, 0x0010), "LO", "DICOMVISION_PRIVATE")
    dataset.add_new((0x0011, 0x1001), "LO", "private-patient-note")
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
    assert series["patientName"] == "Tag^Tester"
    assert series["patientId"] == "patient-001"
    assert series["studyDate"] == "20260514"
    assert series["accessionNumber"] == "ACC-001"
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


def test_upload_dicom_files_registers_series(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "upload-test.dcm"
    _create_test_dicom(dicom_path)

    client = TestClient(fastapi_app)
    with dicom_path.open("rb") as handle:
        response = client.post(
            "/api/v1/dicom/upload",
            files=[("files", ("study/series/upload-test.dcm", handle.read(), "application/dicom"))],
            data={"relativePaths": "study/series/upload-test.dcm"},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["seriesList"]) == 1
    series = data["seriesList"][0]
    assert series["patientName"] == "Tag^Tester"
    assert series["instanceCount"] == 1
    assert series["folderPath"]


def test_modify_dicom_tag_current_instance_returns_dicom_artifact(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "tag-edit-current-中文.dcm"
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
    assert "filename*=" in response.headers["content-disposition"]

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


def test_modify_dicom_tag_rejects_invalid_vr_value(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "tag-edit-invalid-vr.dcm"
    _create_test_dicom(dicom_path)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)
    tags_response = client.post("/api/v1/dicom/tags", json={"seriesId": series_id, "index": 0})
    birth_date_row = next(item for item in tags_response.json()["items"] if item["keyword"] == "PatientBirthDate")

    response = client.post(
        "/api/v1/dicom/modifyTag",
        json={
            "seriesId": series_id,
            "index": 0,
            "tagPath": birth_date_row["tagPath"],
            "value": "20260230",
            "scope": "current",
        },
    )

    assert response.status_code == 400
    assert "DA value" in response.json()["detail"]


def test_modify_dicom_tag_normalizes_code_string_value(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_path = tmp_path / "tag-edit-cs-vr.dcm"
    _create_test_dicom(dicom_path)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)
    tags_response = client.post("/api/v1/dicom/tags", json={"seriesId": series_id, "index": 0})
    patient_sex_row = next(item for item in tags_response.json()["items"] if item["keyword"] == "PatientSex")

    response = client.post(
        "/api/v1/dicom/modifyTag",
        json={
            "seriesId": series_id,
            "index": 0,
            "tagPath": patient_sex_row["tagPath"],
            "value": "m",
            "scope": "current",
        },
    )

    assert response.status_code == 200
    modified_dataset = pydicom.dcmread(BytesIO(response.content), force=True)
    assert modified_dataset.PatientSex == "M"


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


def test_deidentify_dicom_series_returns_zip_artifact(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    series_instance_uid = generate_uid()
    first_path = tmp_path / "deid-series-1.dcm"
    second_path = tmp_path / "deid-series-2.dcm"
    _create_test_dicom(first_path, series_instance_uid=series_instance_uid, instance_number=1)
    _create_test_dicom(second_path, series_instance_uid=series_instance_uid, instance_number=2)

    original_dataset = pydicom.dcmread(str(first_path), force=True)
    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)

    response = client.post(
        "/api/v1/dicom/deidentify",
        json={
            "seriesId": series_id,
            "fieldKeys": [
                "patientIdentity",
                "patientDemographics",
                "datesAndTimes",
                "accessionInstitution",
                "physiciansOperators",
                "privateTags",
                "uids",
            ],
            "replacementPrefix": "dv",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-dicomvision-artifact-kind"] == "zip"
    assert response.headers["x-dicomvision-artifact-purpose"] == "deidentify"
    assert response.headers["x-dicomvision-modified-count"] == "2"
    assert response.headers["x-dicomvision-file-name"].endswith(".zip")

    with ZipFile(BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert len(names) == 2
        assert all(name.endswith(".dcm") for name in names)
        assert len({Path(name).parent for name in names}) == 1
        datasets = [pydicom.dcmread(BytesIO(archive.read(name)), force=True) for name in names]

    assert {str(dataset.PatientName) for dataset in datasets} == {"ANONYMIZED"}
    assert all(str(dataset.PatientID).startswith("DV-") for dataset in datasets)
    assert all(dataset.PatientBirthDate == "" for dataset in datasets)
    assert all(dataset.StudyDate == "" for dataset in datasets)
    assert all(dataset.AccessionNumber == "" for dataset in datasets)
    assert all(dataset.InstitutionName == "" for dataset in datasets)
    assert all(dataset.ReferringPhysicianName == "" for dataset in datasets)
    assert all(dataset.PatientIdentityRemoved == "YES" for dataset in datasets)
    assert all(not any(Tag(tag).is_private for tag in dataset.keys()) for dataset in datasets)

    assert datasets[0].StudyInstanceUID != original_dataset.StudyInstanceUID
    assert datasets[0].SeriesInstanceUID != original_dataset.SeriesInstanceUID
    assert datasets[0].SeriesInstanceUID == datasets[1].SeriesInstanceUID
    assert len({str(dataset.SOPInstanceUID) for dataset in datasets}) == 2
    assert all(dataset.file_meta.MediaStorageSOPInstanceUID == dataset.SOPInstanceUID for dataset in datasets)

    unchanged_original = pydicom.dcmread(str(first_path), force=True)
    assert str(unchanged_original.PatientName) == "Tag^Tester"
    assert unchanged_original.PatientID == "patient-001"


def _wait_for_dicom_job(client: TestClient, status_url: str) -> dict:
    deadline = time.monotonic() + 5
    while True:
        response = client.get(status_url)
        assert response.status_code == 200
        data = response.json()
        if data["status"] in {"succeeded", "failed"}:
            return data
        assert time.monotonic() < deadline
        time.sleep(0.05)


def _wait_for_tag_edit_job(client: TestClient, job_id: str) -> dict:
    return _wait_for_dicom_job(client, f"/api/v1/dicom/modifyTag/jobs/{job_id}")


def test_deidentify_dicom_series_async_job_downloads_artifact(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_deidentify_job_service.clear()
    series_instance_uid = generate_uid()
    _create_test_dicom(tmp_path / "deid-async-1.dcm", series_instance_uid=series_instance_uid, instance_number=1)
    _create_test_dicom(tmp_path / "deid-async-2.dcm", series_instance_uid=series_instance_uid, instance_number=2)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)

    create_response = client.post(
        "/api/v1/dicom/deidentify/jobs",
        json={
            "seriesId": series_id,
            "fieldKeys": ["patientIdentity", "privateTags", "uids"],
            "replacementPrefix": "job",
        },
    )

    assert create_response.status_code == 202
    created = create_response.json()
    assert created["status"] in {"pending", "running", "succeeded"}
    assert created["statusUrl"].endswith(created["jobId"])

    completed = _wait_for_dicom_job(client, created["statusUrl"])
    assert completed["status"] == "succeeded"
    assert completed["artifactKind"] == "zip"
    assert completed["modifiedCount"] == 2
    assert completed["processedCount"] == 2
    assert completed["totalCount"] == 2
    assert completed["progressPercent"] == 100
    assert completed["artifactUrl"].endswith(f"/{created['jobId']}/artifact")

    artifact_response = client.get(completed["artifactUrl"])
    assert artifact_response.status_code == 200
    assert artifact_response.headers["content-type"] == "application/zip"
    assert artifact_response.headers["x-dicomvision-artifact-kind"] == "zip"
    assert artifact_response.headers["x-dicomvision-artifact-purpose"] == "deidentify"
    assert artifact_response.headers["x-dicomvision-modified-count"] == "2"

    with ZipFile(BytesIO(artifact_response.content)) as archive:
        datasets = [pydicom.dcmread(BytesIO(archive.read(name)), force=True) for name in archive.namelist()]

    assert len(datasets) == 2
    assert {str(dataset.PatientName) for dataset in datasets} == {"ANONYMIZED"}
    assert all(str(dataset.PatientID).startswith("JOB-") for dataset in datasets)
    assert all(dataset.PatientIdentityRemoved == "YES" for dataset in datasets)
    assert all(not any(Tag(tag).is_private for tag in dataset.keys()) for dataset in datasets)

    dicom_deidentify_job_service.clear()


def test_tag_edit_job_service_removes_stale_temp_artifacts(tmp_path: Path) -> None:
    stale_artifact = tmp_path / f"{'a' * 32}.zip"
    fresh_artifact = tmp_path / f"{'b' * 32}.zip"
    unrelated_file = tmp_path / "notes.zip"
    stale_artifact.write_bytes(b"old")
    fresh_artifact.write_bytes(b"new")
    unrelated_file.write_bytes(b"notes")
    os.utime(stale_artifact, (1, 1))

    service = DicomTagModifyJobService(temp_root=tmp_path)

    assert not stale_artifact.exists()
    assert fresh_artifact.exists()
    assert unrelated_file.exists()
    service.clear()


def test_modify_dicom_tag_series_scope_async_job_downloads_artifact(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    dicom_tag_job_service.clear()
    series_instance_uid = generate_uid()
    _create_test_dicom(tmp_path / "tag-edit-async-1.dcm", series_instance_uid=series_instance_uid, instance_number=1)
    _create_test_dicom(tmp_path / "tag-edit-async-2.dcm", series_instance_uid=series_instance_uid, instance_number=2)

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = load_response.series_list[0].series_id
    client = TestClient(fastapi_app)
    tags_response = client.post("/api/v1/dicom/tags", json={"seriesId": series_id, "index": 0})
    patient_id_row = next(item for item in tags_response.json()["items"] if item["keyword"] == "PatientID")

    create_response = client.post(
        "/api/v1/dicom/modifyTag/jobs",
        json={
            "seriesId": series_id,
            "index": 0,
            "tagPath": patient_id_row["tagPath"],
            "value": "patient-async",
            "scope": "series",
        },
    )

    assert create_response.status_code == 202
    created = create_response.json()
    assert created["status"] in {"pending", "running", "succeeded"}
    assert created["statusUrl"].endswith(created["jobId"])

    completed = _wait_for_tag_edit_job(client, created["jobId"])
    assert completed["status"] == "succeeded"
    assert completed["artifactKind"] == "zip"
    assert completed["modifiedCount"] == 2
    assert completed["processedCount"] == 2
    assert completed["totalCount"] == 2
    assert completed["progressPercent"] == 100
    assert completed["artifactUrl"].endswith(f"/{created['jobId']}/artifact")

    artifact_response = client.get(completed["artifactUrl"])
    assert artifact_response.status_code == 200
    assert artifact_response.headers["content-type"] == "application/zip"
    assert artifact_response.headers["x-dicomvision-artifact-kind"] == "zip"
    assert artifact_response.headers["x-dicomvision-modified-count"] == "2"

    with ZipFile(BytesIO(artifact_response.content)) as archive:
        datasets = [pydicom.dcmread(BytesIO(archive.read(name)), force=True) for name in archive.namelist()]

    assert len(datasets) == 2
    for modified_dataset in datasets:
        assert modified_dataset.PatientID == "patient-async"

    dicom_tag_job_service.clear()
