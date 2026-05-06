import math

from pydicom.multival import MultiValue

from app.services.dicom_cache import DicomCache


def test_get_first_number_accepts_scalar_and_multivalue_inputs() -> None:
    assert DicomCache._get_first_number("40.5") == 40.5
    assert DicomCache._get_first_number(MultiValue(float, ["80", "120"])) == 80.0


def test_get_first_number_ignores_empty_invalid_and_non_finite_values() -> None:
    assert DicomCache._get_first_number(None) is None
    assert DicomCache._get_first_number(MultiValue(float, [])) is None
    assert DicomCache._get_first_number("not-a-number") is None
    assert DicomCache._get_first_number(math.nan) is None
