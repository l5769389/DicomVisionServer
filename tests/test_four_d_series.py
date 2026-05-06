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
        assert all(phase.status == "pending" for phase in summary.four_d_phases)
        assert all(phase.image_src == "" for phase in summary.four_d_phases)
        assert all(phase.viewport_images == {} for phase in summary.four_d_phases)


def test_load_folder_recognizes_percent_only_series_description_as_phase(tmp_path: Path) -> None:
    study_uid = generate_uid()

    for series_description, base_value in (("5%", 100), ("50\uFF05", 500)):
        series_uid = generate_uid()
        for slice_index in range(3):
            _create_test_dicom(
                tmp_path / f"series-description-{series_description}" / f"slice-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=series_description,
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
        assert [phase.label for phase in summary.four_d_phases] == ["Phase 5%", "Phase 50%"]


def test_load_folder_groups_percent_only_phase_series_with_different_slice_counts(tmp_path: Path) -> None:
    study_uid = generate_uid()
    phase_percents = list(range(5, 100, 10))

    for phase_index, phase_percent in enumerate(phase_percents):
        series_uid = generate_uid()
        slice_count = 2 + (phase_index % 3)
        for slice_index in range(slice_count):
            _create_test_dicom(
                tmp_path / f"phase-{phase_percent}" / f"slice-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=f"{phase_percent}%",
                instance_number=slice_index + 1,
                slice_index=slice_index,
                pixel_value=100 + phase_index * 100,
            )

    response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

    assert len(response.series_list) == len(phase_percents)
    expected_labels = [f"Phase {phase_percent}%" for phase_percent in phase_percents]
    for summary in response.series_list:
        assert summary.is_four_d_series is True
        assert summary.four_d_phase_count == len(phase_percents)
        assert summary.four_d_phases is not None
        assert [phase.label for phase in summary.four_d_phases] == expected_labels


def test_load_folder_splits_same_series_uid_across_phase_folders_into_distinct_four_d_series(tmp_path: Path) -> None:
    study_uid = generate_uid()
    shared_series_uid = generate_uid()

    for phase_percent, base_value in ((0, 100), (50, 500)):
        for slice_index in range(3):
            _create_test_dicom(
                tmp_path / f"phase-{phase_percent}" / f"slice-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=shared_series_uid,
                series_description="4D Lung",
                instance_number=slice_index + 1,
                slice_index=slice_index,
                pixel_value=base_value,
            )

    response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

    assert len(response.series_list) == 2
    series_ids = {summary.series_id for summary in response.series_list}
    for summary in response.series_list:
        assert summary.is_four_d_series is True
        assert summary.four_d_phases is not None
        assert {phase.series_id for phase in summary.four_d_phases} == series_ids


def test_load_folder_groups_phase_folders_when_series_descriptions_differ(tmp_path: Path) -> None:
    study_uid = generate_uid()

    for phase_percent, description, base_value in (
        (0, "Respiration position 1", 100),
        (50, "Respiration position 2", 500),
    ):
        series_uid = generate_uid()
        for slice_index in range(3):
            _create_test_dicom(
                tmp_path / f"phase-{phase_percent}" / f"slice-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=description,
                instance_number=slice_index + 1,
                slice_index=slice_index,
                pixel_value=base_value,
            )

    response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))

    assert len(response.series_list) == 2
    series_ids = {summary.series_id for summary in response.series_list}
    for summary in response.series_list:
        assert summary.is_four_d_series is True
        assert summary.four_d_phases is not None
        assert [phase.label for phase in summary.four_d_phases] == ["Phase 00", "Phase 50"]
        assert {phase.series_id for phase in summary.four_d_phases} == series_ids


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
    phase_series_ids = [phase.series_id for phase in summary.four_d_phases]
    assert all(phase_series_ids)
    assert len(set(phase_series_ids)) == 2
    assert summary.series_id not in set(phase_series_ids)
    assert all(phase.status == "pending" for phase in summary.four_d_phases)

    for phase_series_id in phase_series_ids:
        phase_series = series_registry.get(str(phase_series_id))
        assert phase_series.is_virtual is True
        assert phase_series.source_series_id == summary.series_id
        assert len(phase_series.instances) == 2


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
    assert all(phase["status"] == "pending" for phase in data["fourDPhases"])
    assert all(phase["imageSrc"] == "" for phase in data["fourDPhases"])
    assert all(phase["viewportImages"] == {} for phase in data["fourDPhases"])

    preview_response = client.post(
        "/api/v1/dicom/fourD/phases",
        json={"seriesId": selected_series_id, "includePreviewImages": True, "previewPhaseIndex": 0},
    )

    assert preview_response.status_code == 200
    preview_data = preview_response.json()
    assert preview_data["fourDPhases"][0]["status"] == "ready"
    preview_url = preview_data["fourDPhases"][0]["viewportImages"]["mpr-ax"]
    assert preview_url.startswith("/api/v1/dicom/fourD/preview?")
    assert preview_data["fourDPhases"][1]["status"] == "pending"
    assert preview_data["fourDPhases"][1]["viewportImages"] == {}

    image_response = client.get(preview_url)
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"
    assert image_response.content.startswith(b"\x89PNG")


def test_four_d_phases_api_returns_virtual_series_ids_for_single_series_phases(tmp_path: Path) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()

    instance_number = 1
    for phase_identifier, base_value in ((1, 100), (2, 300)):
        for slice_index in range(2):
            _create_test_dicom(
                tmp_path / f"single-phase-api-{phase_identifier}-{slice_index}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description="Temporal 4D Lung",
                instance_number=instance_number,
                slice_index=slice_index,
                pixel_value=base_value,
                temporal_position_identifier=phase_identifier,
            )
            instance_number += 1

    load_response = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
    selected_series_id = load_response.series_list[0].series_id

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/dicom/fourD/phases", json={"seriesId": selected_series_id})

    assert response.status_code == 200
    data = response.json()
    assert data["seriesId"] == selected_series_id
    assert data["isFourDSeries"] is True
    assert data["fourDPhaseCount"] == 2
    phase_series_ids = [phase["seriesId"] for phase in data["fourDPhases"]]
    assert len(set(phase_series_ids)) == 2
    assert selected_series_id not in set(phase_series_ids)

    for phase_series_id in phase_series_ids:
        phase_series = series_registry.get(str(phase_series_id))
        assert phase_series.is_virtual is True
        assert phase_series.source_series_id == selected_series_id
