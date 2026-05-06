from pathlib import Path
from types import SimpleNamespace

from app.services.series_registry import SeriesRegistry


def _build_dataset(instance_number: object) -> SimpleNamespace:
    return SimpleNamespace(SOPInstanceUID="1.2.3", InstanceNumber=instance_number, Rows=512, Columns=256)


def test_build_instance_record_uses_valid_instance_number() -> None:
    record = SeriesRegistry._build_instance_record(Path("slice.dcm"), _build_dataset("12"), 3)

    assert record.instance_number == 12
    assert record.rows == 512
    assert record.columns == 256


def test_build_instance_record_falls_back_for_invalid_instance_number() -> None:
    record = SeriesRegistry._build_instance_record(Path("slice.dcm"), _build_dataset("bad-value"), 3)

    assert record.instance_number == 3
