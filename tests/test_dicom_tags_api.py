from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.services.series_registry import series_registry


def _create_test_dicom(path: Path) -> None:
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
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesDescription = "Tag API Series"
    dataset.Modality = "OT"
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

    nested = Dataset()
    nested.CodeValue = "ABC"
    nested.CodeMeaning = "Nested value"
    dataset.ConceptCodeSequence = [nested]
    dataset.save_as(str(path), write_like_original=False)


def test_dicom_tags_api_returns_instance_tags(tmp_path: Path) -> None:
    series_registry.clear()
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
