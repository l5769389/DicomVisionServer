from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pydicom
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry


def _write_phi_ct_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    z_position: float,
) -> None:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientName = "Sensitive^Patient"
    dataset.PatientID = "SECRET-123"
    dataset.PatientBirthDate = "19601231"
    dataset.StudyDate = "20260719"
    dataset.StudyTime = "123456"
    dataset.AccessionNumber = "PRIV-ACC"
    dataset.InstitutionName = "Private Hospital"
    dataset.ReferringPhysicianName = "Private^Doctor"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Diagnostic source"
    dataset.InstanceNumber = instance_number
    dataset.Rows = 8
    dataset.Columns = 10
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 1
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [0.8, 0.4]
    dataset.SliceThickness = 1.5
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [5.0, 6.0, z_position]
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1024.0
    dataset.WindowCenter = 40.0
    dataset.WindowWidth = 400.0
    pixels = np.arange(80, dtype=np.int16).reshape(8, 10) + instance_number * 100
    dataset.PixelData = pixels.tobytes()

    referenced = Dataset()
    referenced.ReferencedSOPClassUID = CTImageStorage
    referenced.ReferencedSOPInstanceUID = sop_instance_uid
    referenced.PatientName = "Nested^Sensitive"
    referenced.StudyDate = "20260719"
    dataset.ReferencedImageSequence = [referenced]
    dataset.add_new((0x0011, 0x0010), "LO", "PRIVATE_CREATOR")
    dataset.add_new((0x0011, 0x1001), "LO", "private diagnostic note")
    dataset.save_as(path, enforce_file_format=True)


def test_deidentify_preserves_pixels_and_diagnostic_geometry_while_remapping_all_uid_references(
    tmp_path: Path,
) -> None:
    series_registry.clear()
    dicom_cache.clear()
    study_uid = generate_uid()
    series_uid = generate_uid()
    source_paths = [tmp_path / "source-1.dcm", tmp_path / "source-2.dcm"]
    for index, path in enumerate(source_paths, start=1):
        _write_phi_ct_slice(
            path,
            study_uid=study_uid,
            series_uid=series_uid,
            instance_number=index,
            z_position=float(index - 1) * 1.5,
        )

    originals = [pydicom.dcmread(path) for path in source_paths]
    loaded = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    series_id = loaded.series_list[0].series_id

    response = TestClient(fastapi_app).post(
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
            "replacementPrefix": "safe",
        },
    )

    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        exported = [
            pydicom.dcmread(BytesIO(archive.read(name)))
            for name in sorted(archive.namelist())
        ]

    assert len(exported) == len(originals)
    original_by_instance = {int(dataset.InstanceNumber): dataset for dataset in originals}
    exported_by_instance = {int(dataset.InstanceNumber): dataset for dataset in exported}
    assert set(exported_by_instance) == set(original_by_instance)

    for instance_number, dataset in exported_by_instance.items():
        original = original_by_instance[instance_number]
        np.testing.assert_array_equal(dataset.pixel_array, original.pixel_array)
        assert dataset.PixelData == original.PixelData
        assert dataset.Rows == original.Rows
        assert dataset.Columns == original.Columns
        assert dataset.PixelSpacing == original.PixelSpacing
        assert dataset.SliceThickness == original.SliceThickness
        assert dataset.ImageOrientationPatient == original.ImageOrientationPatient
        assert dataset.ImagePositionPatient == original.ImagePositionPatient
        assert dataset.RescaleSlope == original.RescaleSlope
        assert dataset.RescaleIntercept == original.RescaleIntercept
        assert dataset.WindowCenter == original.WindowCenter
        assert dataset.WindowWidth == original.WindowWidth
        assert dataset.Modality == original.Modality
        assert dataset.SOPClassUID == original.SOPClassUID
        assert dataset.file_meta.TransferSyntaxUID == original.file_meta.TransferSyntaxUID

        assert str(dataset.PatientName) == "ANONYMIZED"
        assert str(dataset.PatientID).startswith("SAFE-")
        assert dataset.PatientBirthDate == ""
        assert dataset.StudyDate == ""
        assert dataset.StudyTime == ""
        assert dataset.AccessionNumber == ""
        assert dataset.InstitutionName == ""
        assert dataset.ReferringPhysicianName == ""
        assert str(dataset.ReferencedImageSequence[0].PatientName) == "ANONYMIZED"
        assert dataset.ReferencedImageSequence[0].StudyDate == ""
        assert not any(element.tag.is_private for element in dataset.iterall())

        assert dataset.SOPInstanceUID != original.SOPInstanceUID
        assert dataset.ReferencedImageSequence[0].ReferencedSOPInstanceUID == dataset.SOPInstanceUID
        assert dataset.file_meta.MediaStorageSOPInstanceUID == dataset.SOPInstanceUID

    assert len({str(dataset.StudyInstanceUID) for dataset in exported}) == 1
    assert len({str(dataset.SeriesInstanceUID) for dataset in exported}) == 1
    assert str(exported[0].StudyInstanceUID) != study_uid
    assert str(exported[0].SeriesInstanceUID) != series_uid

    # De-identification must be copy-on-write: source files remain byte-for-byte
    # readable with their original identity, pixels, and UIDs.
    for path, original in zip(source_paths, originals):
        unchanged = pydicom.dcmread(path)
        assert str(unchanged.PatientName) == "Sensitive^Patient"
        assert unchanged.PatientID == "SECRET-123"
        assert unchanged.SOPInstanceUID == original.SOPInstanceUID
        assert unchanged.PixelData == original.PixelData

    series_registry.clear()
    dicom_cache.clear()
