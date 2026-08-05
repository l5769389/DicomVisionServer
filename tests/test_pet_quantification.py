from __future__ import annotations

from math import exp, log

import numpy as np
import pytest
from pydicom import dcmread
from pydicom.dataset import Dataset

from app.services.pet_quantification import (
    PET_UNIT_KBQML,
    PET_UNIT_PERCENT_ID_G,
    PET_UNIT_SOURCE,
    PET_UNIT_SUL,
    PET_UNIT_SUV_BSA,
    PET_UNIT_SUV_BW,
    apply_pet_mapping,
    build_pet_quantification_context,
    derive_pet_auto_range,
)
from tests.support.dicom_phantoms import write_pet_bqml_series


def _quantitative_bqml_dataset(*, tracer: str = "F-18 FDG") -> Dataset:
    dataset = Dataset()
    dataset.Units = "BQML"
    dataset.PatientWeight = 70.0
    dataset.PatientSize = 1.75
    dataset.PatientSex = "M"
    dataset.CorrectedImage = ["DECY", "ATTN"]
    dataset.DecayCorrection = "START"
    dataset.AcquisitionDateTime = "20260102001000"
    item = Dataset()
    item.Radiopharmaceutical = tracer
    item.RadionuclideTotalDose = 350_000_000.0
    item.RadionuclideHalfLife = 6586.2
    item.RadiopharmaceuticalStartDateTime = "20260101235000"
    dataset.RadiopharmaceuticalInformationSequence = [item]
    return dataset


def test_bqml_quantification_known_scale_and_cross_midnight_decay() -> None:
    dataset = _quantitative_bqml_dataset()
    context = build_pet_quantification_context(dataset)
    corrected_dose = 350_000_000.0 * exp(-log(2.0) * 1200.0 / 6586.2)

    assert context.is_fdg is True
    assert context.mapping_for(PET_UNIT_KBQML).scale == pytest.approx(0.001)
    assert context.mapping_for(PET_UNIT_SUV_BW).scale == pytest.approx(70_000.0 / corrected_dose)

    height_cm = 175.0
    bsa_cm2 = 0.007184 * height_cm**0.725 * 70.0**0.425 * 10_000.0
    assert context.mapping_for(PET_UNIT_SUV_BSA).scale == pytest.approx(bsa_cm2 / corrected_dose)

    bmi = 70.0 / 1.75**2
    lbm_kg = 9270.0 * 70.0 / (6680.0 + 216.0 * bmi)
    assert context.mapping_for(PET_UNIT_SUL).scale == pytest.approx(lbm_kg * 1000.0 / corrected_dose)

    source = np.array([0.0, 5_000.0, 10_000.0], dtype=np.float32)
    suv = apply_pet_mapping(source, context.mapping_for(PET_UNIT_SUV_BW))
    assert suv[1] == pytest.approx(5_000.0 * 70_000.0 / corrected_dose, rel=1e-6)


def test_admin_decay_correction_uses_administered_dose_without_second_decay() -> None:
    dataset = _quantitative_bqml_dataset()
    dataset.DecayCorrection = "ADMIN"
    context = build_pet_quantification_context(dataset)

    assert context.mapping_for(PET_UNIT_SUV_BW).scale == pytest.approx(70_000.0 / 350_000_000.0)


def test_ambiguous_decay_reference_disables_derived_suv_units() -> None:
    dataset = _quantitative_bqml_dataset()
    dataset.DecayCorrection = "NONE"
    context = build_pet_quantification_context(dataset)

    option = context.option_for(PET_UNIT_SUV_BW)
    assert option.available is False
    assert "START or ADMIN" in str(option.reason)


def test_missing_pet_metadata_disables_units_without_silent_fallback() -> None:
    dataset = Dataset()
    dataset.Units = "BQML"
    context = build_pet_quantification_context(dataset)

    assert context.mapping_for(PET_UNIT_SOURCE) is not None
    assert context.mapping_for(PET_UNIT_KBQML) is not None
    assert context.mapping_for(PET_UNIT_SUV_BW) is None
    assert context.mapping_for(PET_UNIT_SUV_BSA) is None
    assert context.mapping_for(PET_UNIT_SUL) is None
    assert context.option_for(PET_UNIT_SUV_BW).available is False
    assert "Radiopharmaceutical" in str(context.option_for(PET_UNIT_SUV_BW).reason)


def test_invalid_animal_height_disables_height_dependent_units_only() -> None:
    dataset = _quantitative_bqml_dataset()
    dataset.PatientWeight = 0.018
    dataset.PatientSize = 0.001
    context = build_pet_quantification_context(dataset)

    # The source and activity concentration remain usable.  Anthropometric
    # formulae must not turn an obviously invalid animal height into a
    # plausible-looking human SUVbsa/SUL result.
    assert context.mapping_for(PET_UNIT_KBQML) is not None
    assert context.mapping_for(PET_UNIT_SUV_BSA) is None
    assert context.mapping_for(PET_UNIT_SUL) is None
    assert "adult-range height" in str(context.option_for(PET_UNIT_SUV_BSA).reason)


@pytest.mark.parametrize(
    ("attribute", "value", "expected_reason"),
    [
        ("NumberOfFrames", 2, "multi-frame"),
        ("NumberOfTimeSlices", 2, "Dynamic PET"),
        ("NumberOfTimeSlots", 2, "Gated PET"),
    ],
)
def test_non_static_pet_is_explicitly_unsupported(
    attribute: str,
    value: int,
    expected_reason: str,
) -> None:
    dataset = _quantitative_bqml_dataset()
    setattr(dataset, attribute, value)

    context = build_pet_quantification_context(dataset)

    assert context.support_status == "unsupported"
    assert expected_reason in str(context.support_reason)


@pytest.mark.parametrize(
    ("suv_type", "expected_unit"),
    [
        (None, PET_UNIT_SUV_BW),
        ("BW", PET_UNIT_SUV_BW),
        ("BSA", PET_UNIT_SUV_BSA),
        ("LBMJANMA", PET_UNIT_SUL),
    ],
)
def test_native_gml_uses_suv_type(suv_type: str | None, expected_unit: str) -> None:
    dataset = Dataset()
    dataset.Units = "GML"
    if suv_type is not None:
        dataset.SUVType = suv_type
    context = build_pet_quantification_context(dataset)

    assert context.mapping_for(expected_unit).scale == pytest.approx(1.0)
    if suv_type is None:
        assert any("SUV Type is absent" in warning for warning in context.warnings)


def test_real_world_value_mapping_controls_explicit_percent_id_per_gram() -> None:
    dataset = Dataset()
    dataset.Units = "CNTS"
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = 10.0
    mapping = Dataset()
    mapping.RealWorldValueSlope = 0.5
    mapping.RealWorldValueIntercept = 3.0
    unit = Dataset()
    unit.CodeValue = "%ID/g"
    unit.CodeMeaning = "Percent Injected Dose per Gram"
    mapping.MeasurementUnitsCodeSequence = [unit]
    dataset.RealWorldValueMappingSequence = [mapping]

    context = build_pet_quantification_context(dataset)
    percent_mapping = context.mapping_for(PET_UNIT_PERCENT_ID_G)
    assert percent_mapping is not None
    assert percent_mapping.scale == pytest.approx(0.25)
    assert percent_mapping.offset == pytest.approx(0.5)


def test_pet_auto_range_uses_robust_distribution_for_all_units() -> None:
    volume = np.arange(1, 1001, dtype=np.float32)
    expected_high = float(np.percentile(volume, 99.5))
    assert derive_pet_auto_range(volume, PET_UNIT_SUV_BW) == pytest.approx((0.0, expected_high))
    low, high = derive_pet_auto_range(volume, PET_UNIT_SOURCE)
    assert low == 0.0
    assert high == pytest.approx(expected_high)


def test_synthetic_pet_dicom_preserves_physical_and_quantitative_truth(tmp_path) -> None:
    volume = np.zeros((5, 7, 9), dtype=np.float32)
    volume[1:4, 2:6, 3:7] = 5_000.0
    phantom = write_pet_bqml_series(
        tmp_path / "pet",
        volume,
        spacing_zyx_mm=(3.0, 2.0, 1.5),
    )
    dataset = dcmread(phantom.paths[2])
    context = build_pet_quantification_context(dataset)

    assert dataset.Modality == "PT"
    assert tuple(float(value) for value in dataset.PixelSpacing) == pytest.approx((2.0, 1.5))
    assert float(dataset.SliceThickness) == pytest.approx(3.0)
    assert np.asarray(dataset.pixel_array)[3, 4] == 5_000
    assert context.mapping_for(PET_UNIT_SUV_BW) is not None
    assert context.is_fdg is True
