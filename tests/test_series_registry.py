from pathlib import Path
from types import SimpleNamespace

from pydicom.uid import UID

from app.models.viewer import InstanceRecord, SeriesRecord
from app.services.series_registry import SeriesRegistry


def _build_dataset(instance_number: object) -> SimpleNamespace:
    return SimpleNamespace(
        SOPInstanceUID="1.2.3",
        InstanceNumber=instance_number,
        Rows=512,
        Columns=256,
    )


def _build_complete_instance(**overrides: object) -> InstanceRecord:
    values = {
        "path": Path("slice.dcm"),
        "sop_instance_uid": "1.2.3",
        "instance_number": 1,
        "rows": 512,
        "columns": 256,
        "transfer_syntax_uid": "1.2.840.10008.1.2.1",
        "transfer_syntax_name": "Explicit VR Little Endian",
        "transfer_syntax_is_compressed": False,
        "photometric_interpretation": "MONOCHROME2",
        "samples_per_pixel": 1,
        "pixel_spacing": (0.8, 0.8),
        "imager_pixel_spacing": None,
        "has_image_orientation_patient": True,
        "has_image_position_patient": True,
        "has_rescale_slope": True,
        "has_rescale_intercept": True,
        "has_window_width": True,
        "has_window_center": True,
        "number_of_frames": 1,
    }
    values.update(overrides)
    return InstanceRecord(**values)


def _build_series(instances: list[InstanceRecord], modality: str | None = "CT") -> SeriesRecord:
    return SeriesRecord(
        series_id="series",
        folder_path=".",
        series_instance_uid="1.2.3.series",
        study_instance_uid=None,
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality=modality,
        series_description=None,
        instances=instances,
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


def test_build_compatibility_issues_flags_decoding_and_geometry_risks() -> None:
    first = _build_complete_instance(
        transfer_syntax_is_compressed=True,
        transfer_syntax_name="JPEG Baseline",
        photometric_interpretation="RGB",
        samples_per_pixel=3,
        pixel_spacing=None,
        has_image_orientation_patient=False,
        number_of_frames=4,
    )
    second = _build_complete_instance(
        instance_number=2,
        sop_instance_uid="1.2.4",
        pixel_spacing=None,
        has_image_position_patient=False,
        has_rescale_intercept=False,
    )

    issues = SeriesRegistry._build_compatibility_issues(_build_series([first, second]))
    issue_by_code = {issue.code: issue for issue in issues}

    assert issue_by_code["compressed-transfer-syntax"].affected_instances == 1
    assert issue_by_code["unsupported-photometric"].affected_instances == 1
    assert issue_by_code["multiframe-first-frame"].affected_instances == 1
    assert issue_by_code["missing-pixel-spacing"].affected_instances == 2
    assert issue_by_code["missing-spatial-geometry"].affected_instances == 2
    assert issue_by_code["missing-rescale"].affected_instances == 1
