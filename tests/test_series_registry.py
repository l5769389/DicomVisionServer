from pathlib import Path
from types import SimpleNamespace

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, UID, generate_uid

from app.services.series_registry import SeriesRegistry


def _build_dataset(instance_number: object) -> SimpleNamespace:
    return SimpleNamespace(
        SOPInstanceUID="1.2.3",
        InstanceNumber=instance_number,
        Rows=512,
        Columns=256,
    )


def test_build_instance_record_uses_valid_instance_number() -> None:
    record = SeriesRegistry._build_instance_record(Path("slice.dcm"), _build_dataset("12"), 3)

    assert record.instance_number == 12
    assert record.rows == 512
    assert record.columns == 256


def test_build_instance_record_falls_back_for_invalid_instance_number() -> None:
    record = SeriesRegistry._build_instance_record(Path("slice.dcm"), _build_dataset("bad-value"), 3)

    assert record.instance_number == 3


def test_build_instance_record_collects_compatibility_metadata() -> None:
    dataset = SimpleNamespace(
        SOPInstanceUID="1.2.3",
        InstanceNumber="7",
        Rows=512,
        Columns=256,
        file_meta=SimpleNamespace(TransferSyntaxUID=UID("1.2.840.10008.1.2.4.50")),
        PhotometricInterpretation="RGB",
        SamplesPerPixel="3",
        PixelSpacing=["0.7", "0.8"],
        ImageOrientationPatient=["1", "0", "0", "0", "1", "0"],
        ImagePositionPatient=["0", "0", "0"],
        NumberOfFrames="12",
    )

    record = SeriesRegistry._build_instance_record(Path("slice.dcm"), dataset, 3)

    assert record.instance_number == 7
    assert record.transfer_syntax_is_compressed is True
    assert record.photometric_interpretation == "RGB"
    assert record.samples_per_pixel == 3
    assert record.pixel_spacing == (0.7, 0.8)
    assert record.has_image_orientation_patient is True
    assert record.number_of_frames == 12


def test_resolve_scan_target_skips_archives_and_metadata_but_keeps_extensionless_dicom_candidates(tmp_path: Path) -> None:
    dicom_file = tmp_path / "slice.dcm"
    extensionless_dicom = tmp_path / "IM0001"
    archive = tmp_path / "study.zip"
    metadata = tmp_path / "notes.txt"
    macosx_file = tmp_path / "__MACOSX" / "._slice.dcm"
    for path in (dicom_file, extensionless_dicom, archive, metadata, macosx_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"candidate")

    root, scan_paths = SeriesRegistry._resolve_scan_target(str(tmp_path))

    assert root == tmp_path.resolve()
    assert scan_paths == [extensionless_dicom, dicom_file]


def test_resolve_scan_target_skips_hidden_and_symlinked_directories(tmp_path: Path) -> None:
    visible = tmp_path / "study" / "slice.dcm"
    hidden = tmp_path / ".cache" / "hidden.dcm"
    visible.parent.mkdir(parents=True)
    hidden.parent.mkdir(parents=True)
    visible.write_bytes(b"candidate")
    hidden.write_bytes(b"candidate")
    linked_dir = tmp_path / "linked-study"
    linked_dir.symlink_to(visible.parent, target_is_directory=True)

    _, scan_paths = SeriesRegistry._resolve_scan_target(str(tmp_path))

    assert scan_paths == [visible]


def test_scan_header_reads_only_fields_needed_for_series_grouping(tmp_path: Path) -> None:
    path = tmp_path / "slice.dcm"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = "1.2.3.4"
    dataset.SeriesInstanceUID = "1.2.3"
    dataset.Modality = "CT"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.InstitutionName = "Not needed while listing"
    dataset.save_as(path, enforce_file_format=True)

    header = SeriesRegistry._read_dataset_header(path)

    assert header is not None
    assert header.SeriesInstanceUID == "1.2.3"
    assert header.Modality == "CT"
    assert "InstitutionName" not in header
