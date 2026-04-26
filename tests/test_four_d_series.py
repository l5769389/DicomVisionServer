from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry


@pytest.fixture(autouse=True)
def isolated_dicom_state() -> Iterator[None]:
    series_registry.clear()
    dicom_cache.clear()
    try:
        yield
    finally:
        series_registry.clear()
        dicom_cache.clear()


def _create_test_dicom(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    series_description: str,
    instance_number: int,
    slice_index: int,
    pixel_value: int,
    temporal_position_identifier: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.PatientName = "FourD^Tester"
    dataset.PatientID = "patient-4d"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesDescription = series_description
    dataset.Modality = "CT"
    dataset.InstanceNumber = instance_number
    dataset.Rows = 6
    dataset.Columns = 6
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
    if temporal_position_identifier is not None:
        dataset.TemporalPositionIdentifier = temporal_position_identifier
        dataset.NumberOfTemporalPositions = 2

    pixels = np.full((6, 6), pixel_value + slice_index, dtype=np.uint16)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(str(path), write_like_original=False)


def test_load_folder_marks_related_phase_series_as_four_d(tmp_path: Path) -> None:
    study_uid = generate_uid()

    for phase_percent, base_value in ((0, 100), (50, 500)):
        series_uid = generate_uid()
        for slice_index in range(3):
            _create_test_dicom(
                tmp_path / f"phase-{phase_percent}" / f"slice-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=f"4D Lung {phase_percent}%",
                instance_number=slice_index + 1,
                slice_index=slice_index,
                pixel_value=base_value,
            )

    response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

    assert len(response.series_list) == 2
    for summary in response.series_list:
        assert summary.is_four_d_series is True
        assert summary.four_d_phase_count == 2
        assert summary.four_d_phases is not None
        assert [phase.phase_index for phase in summary.four_d_phases] == [0, 1]
        assert {phase.label for phase in summary.four_d_phases} == {"Phase 0%", "Phase 50%"}
        assert all(phase.series_id for phase in summary.four_d_phases)
        assert all(phase.status == "ready" for phase in summary.four_d_phases)

        for phase in summary.four_d_phases:
            assert phase.image_src.startswith("data:image/png;base64,")
            assert set(phase.viewport_images) == {"mpr-ax", "mpr-cor", "mpr-sag"}
            assert all(value.startswith("data:image/png;base64,") for value in phase.viewport_images.values())


def test_load_folder_builds_single_series_temporal_phase_items(tmp_path: Path) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()

    instance_number = 1
    for phase_identifier, base_value in ((1, 100), (2, 300)):
        for slice_index in range(2):
            _create_test_dicom(
                tmp_path / f"temporal-{phase_identifier}-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description="Temporal 4D Lung",
                instance_number=instance_number,
                slice_index=slice_index,
                pixel_value=base_value,
                temporal_position_identifier=phase_identifier,
            )
            instance_number += 1

    response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

    assert len(response.series_list) == 1
    summary = response.series_list[0]
    assert summary.is_four_d_series is True
    assert summary.four_d_phase_count == 2
    assert summary.four_d_phases is not None
    assert [phase.label for phase in summary.four_d_phases] == ["Phase 01", "Phase 02"]
    assert {phase.series_id for phase in summary.four_d_phases} == {summary.series_id}
    assert all(phase.status == "ready" for phase in summary.four_d_phases)


def test_four_d_phases_api_returns_phase_manifest_for_selected_series(tmp_path: Path) -> None:
    study_uid = generate_uid()

    for phase_percent, base_value in ((0, 100), (50, 500)):
        series_uid = generate_uid()
        for slice_index in range(2):
            _create_test_dicom(
                tmp_path / f"phase-api-{phase_percent}" / f"slice-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=f"4D Lung {phase_percent}%",
                instance_number=slice_index + 1,
                slice_index=slice_index,
                pixel_value=base_value,
            )

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    selected_series_id = load_response.series_list[0].series_id

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/dicom/fourD/phases", json={"seriesId": selected_series_id})

    assert response.status_code == 200
    data = response.json()
    assert data["seriesId"] == selected_series_id
    assert data["isFourDSeries"] is True
    assert data["fourDPhaseCount"] == 2
    assert [phase["label"] for phase in data["fourDPhases"]] == ["Phase 0%", "Phase 50%"]
    assert all(phase["status"] == "ready" for phase in data["fourDPhases"])
    assert all(phase["viewportImages"]["mpr-ax"].startswith("data:image/png;base64,") for phase in data["fourDPhases"])
