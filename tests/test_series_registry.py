from pathlib import Path
from types import SimpleNamespace

from pydicom.uid import UID

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
