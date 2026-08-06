from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import exp, floor, isfinite, log, log10
import re
from typing import Iterable

import numpy as np
from pydicom.dataset import Dataset


PET_UNIT_SOURCE = "source"
PET_UNIT_KBQML = "kBqml"
PET_UNIT_SUV_BW = "SUVbw"
PET_UNIT_SUV_BSA = "SUVbsa"
PET_UNIT_SUL = "SUL"
PET_UNIT_PERCENT_ID_G = "percentIDg"

PET_UNIT_ORDER = (
    PET_UNIT_SOURCE,
    PET_UNIT_KBQML,
    PET_UNIT_SUV_BW,
    PET_UNIT_SUV_BSA,
    PET_UNIT_SUL,
    PET_UNIT_PERCENT_ID_G,
)

PET_UNIT_LABELS: dict[str, str] = {
    PET_UNIT_SOURCE: "Source",
    PET_UNIT_KBQML: "kBq/ml",
    PET_UNIT_SUV_BW: "g/ml (SUVbw)",
    PET_UNIT_SUV_BSA: "cm²/ml (SUVbsa)",
    PET_UNIT_SUL: "g/ml (SUL, Janmahasatian)",
    PET_UNIT_PERCENT_ID_G: "%ID/g",
}


@dataclass(frozen=True)
class PetLinearMapping:
    scale: float
    offset: float
    label: str
    provenance: str


@dataclass(frozen=True)
class PetUnitOption:
    unit: str
    label: str
    available: bool
    reason: str | None = None
    provenance: str | None = None


@dataclass(frozen=True)
class PetQuantificationContext:
    source_units: str
    source_unit_label: str
    suv_type: str | None
    mappings: dict[str, PetLinearMapping]
    unit_options: tuple[PetUnitOption, ...]
    warnings: tuple[str, ...] = ()
    quantitative: bool = False
    tracer_name: str | None = None
    is_fdg: bool = False
    support_status: str = "static-supported"
    support_reason: str | None = None
    mapping_provenance: str | None = None
    photometric_interpretation: str | None = None

    def mapping_for(self, unit: str) -> PetLinearMapping | None:
        return self.mappings.get(normalize_pet_unit(unit))

    def option_for(self, unit: str) -> PetUnitOption:
        normalized = normalize_pet_unit(unit)
        return next(
            (item for item in self.unit_options if item.unit == normalized),
            PetUnitOption(normalized, PET_UNIT_LABELS.get(normalized, normalized), False, "Unsupported PET unit"),
        )

    def preferred_unit(self) -> str:
        for unit in (PET_UNIT_SUV_BW, PET_UNIT_SUL, PET_UNIT_SUV_BSA, PET_UNIT_KBQML, PET_UNIT_SOURCE):
            if self.option_for(unit).available:
                return unit
        return PET_UNIT_SOURCE


def normalize_pet_unit(value: object | None) -> str:
    # Absence of a requested unit must never imply SUVbw.  The caller may
    # explicitly choose context.preferred_unit() after capability validation.
    text = str(value or PET_UNIT_SOURCE).strip()
    aliases = {
        "RAW": PET_UNIT_SOURCE,
        "SOURCE": PET_UNIT_SOURCE,
        "BQML": PET_UNIT_SOURCE,
        "KBQ/ML": PET_UNIT_KBQML,
        "KBQML": PET_UNIT_KBQML,
        "UPTAKE": PET_UNIT_KBQML,
        "SUV": PET_UNIT_SUV_BW,
        "SUVBW": PET_UNIT_SUV_BW,
        "GML": PET_UNIT_SUV_BW,
        "SUVBSA": PET_UNIT_SUV_BSA,
        "SUL": PET_UNIT_SUL,
        "%ID/G": PET_UNIT_PERCENT_ID_G,
        "PERCENTIDG": PET_UNIT_PERCENT_ID_G,
    }
    return aliases.get(text.upper(), PET_UNIT_SOURCE)


def safe_float(value: object | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _first_sequence_item(dataset: Dataset | None, name: str) -> Dataset | None:
    if dataset is None:
        return None
    sequence = getattr(dataset, name, None)
    try:
        return sequence[0] if sequence else None
    except Exception:
        return None


def _sequence_items(dataset: Dataset | None, name: str) -> tuple[Dataset, ...]:
    if dataset is None:
        return ()
    sequence = getattr(dataset, name, None)
    if not sequence:
        return ()
    try:
        return tuple(item for item in sequence if item is not None)
    except Exception:
        return ()


def _parse_timezone_offset(value: object | None) -> timezone | None:
    text = str(value or "").strip()
    if len(text) != 5 or text[0] not in "+-" or not text[1:].isdigit():
        return None
    hours = int(text[1:3])
    minutes = int(text[3:5])
    if hours > 14 or minutes > 59:
        return None
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(delta if text[0] == "+" else -delta)


def _parse_dicom_datetime(
    date_value: object | None,
    time_value: object | None = None,
    *,
    default_timezone: timezone | None = None,
) -> datetime | None:
    original_date = str(date_value or "").strip()
    date_text = "".join(char for char in original_date if char.isdigit())
    raw_time = str(time_value or "").strip()
    if time_value is None and len(date_text) > 8:
        # DICOM DT is YYYYMMDDHHMMSS.FFFFFF&ZZXX.  Parse the fractional
        # seconds and UTC suffix independently: collecting every digit would
        # otherwise accidentally append timezone digits to the fraction.
        match = re.fullmatch(
            r"(?P<body>\d{4,14})(?:\.(?P<fraction>\d{1,6}))?(?P<offset>[+-]\d{4})?",
            original_date,
        )
        if match is None:
            return None
        datetime_text = match.group("body")[:14].ljust(14, "0")
        fraction = match.group("fraction") or ""
        try:
            parsed = datetime.strptime(datetime_text, "%Y%m%d%H%M%S")
            parsed_timezone = _parse_timezone_offset(match.group("offset"))
            return parsed.replace(
                microsecond=int(fraction.ljust(6, "0") or "0"),
                tzinfo=parsed_timezone or default_timezone,
            )
        except ValueError:
            return None
    if len(date_text) < 8:
        return None
    time_head, _, time_fraction = raw_time.partition(".")
    time_text = "".join(char for char in time_head if char.isdigit())[:6].ljust(6, "0")
    try:
        parsed = datetime.strptime(f"{date_text[:8]}{time_text}", "%Y%m%d%H%M%S")
        microsecond = int("".join(char for char in time_fraction if char.isdigit())[:6].ljust(6, "0") or "0")
        return parsed.replace(microsecond=microsecond, tzinfo=default_timezone)
    except ValueError:
        return None


def resolve_decay_corrected_dose_bq(dataset: Dataset | None) -> tuple[float | None, str | None]:
    radiopharmaceutical = _first_sequence_item(dataset, "RadiopharmaceuticalInformationSequence")
    if dataset is None or radiopharmaceutical is None:
        return None, "Missing Radiopharmaceutical Information Sequence"
    dose = safe_float(getattr(radiopharmaceutical, "RadionuclideTotalDose", None))
    if dose is None or dose <= 0.0:
        return None, "Missing or invalid injected dose"

    corrected_values = getattr(dataset, "CorrectedImage", []) or []
    corrected_image = (
        str(corrected_values).upper()
        if isinstance(corrected_values, str)
        else " ".join(str(value).upper() for value in corrected_values)
    )
    if "DECY" not in corrected_image:
        return None, "PET pixels are not marked as decay corrected"
    decay_correction = str(getattr(dataset, "DecayCorrection", "") or "").strip().upper()
    if decay_correction == "ADMIN":
        return float(dose), None
    if decay_correction != "START":
        return None, "Decay Correction must identify START or ADMIN"

    half_life = safe_float(getattr(radiopharmaceutical, "RadionuclideHalfLife", None))
    if half_life is None or half_life <= 0.0:
        return None, "Missing or invalid radionuclide half-life"

    dataset_timezone = _parse_timezone_offset(getattr(dataset, "TimezoneOffsetFromUTC", None))
    injection_datetime = _parse_dicom_datetime(
        getattr(radiopharmaceutical, "RadiopharmaceuticalStartDateTime", None),
        default_timezone=dataset_timezone,
    )
    scan_datetime = _parse_dicom_datetime(
        getattr(dataset, "AcquisitionDateTime", None),
        default_timezone=dataset_timezone,
    )
    scan_date = (
        getattr(dataset, "AcquisitionDate", None)
        or getattr(dataset, "SeriesDate", None)
        or getattr(dataset, "StudyDate", None)
    )
    if injection_datetime is None:
        injection_datetime = _parse_dicom_datetime(
            scan_date,
            getattr(radiopharmaceutical, "RadiopharmaceuticalStartTime", None),
            default_timezone=dataset_timezone,
        )
    if scan_datetime is None:
        scan_datetime = _parse_dicom_datetime(
            scan_date,
            getattr(dataset, "AcquisitionTime", None)
            or getattr(dataset, "SeriesTime", None)
            or getattr(dataset, "StudyTime", None),
            default_timezone=dataset_timezone,
        )
    if injection_datetime is None or scan_datetime is None:
        return None, "Missing injection or acquisition time"
    if injection_datetime > scan_datetime:
        injection_datetime -= timedelta(days=1)

    elapsed_seconds = (scan_datetime - injection_datetime).total_seconds()
    if elapsed_seconds < 0.0 or elapsed_seconds > 14 * 24 * 3600:
        return None, "Injection and acquisition times are inconsistent"
    return float(dose) * exp(-log(2.0) * elapsed_seconds / float(half_life)), None


def _measurement_unit_from_rwvm(dataset: Dataset | None) -> tuple[str | None, float, float, str | None]:
    """Return the first recognized linear PET mapping, not blindly item zero."""

    for item in _sequence_items(dataset, "RealWorldValueMappingSequence"):
        units_item = _first_sequence_item(item, "MeasurementUnitsCodeSequence")
        code_value = str(getattr(units_item, "CodeValue", "") or "").strip()
        code_meaning = str(getattr(units_item, "CodeMeaning", "") or "").strip()
        code = f"{code_value} {code_meaning}".upper()
        mapped_unit: str | None = None
        if "%ID/G" in code or "PERCENT INJECTED DOSE PER GRAM" in code:
            mapped_unit = PET_UNIT_PERCENT_ID_G
        elif "KBQ/ML" in code or "KBQ.ML-1" in code:
            mapped_unit = PET_UNIT_KBQML
        elif "BQ/ML" in code or "BQ.ML-1" in code:
            mapped_unit = "BQML"
        elif "SUVBSA" in code or "BODY SURFACE" in code:
            mapped_unit = PET_UNIT_SUV_BSA
        elif "SUL" in code or "LEAN BODY" in code:
            mapped_unit = PET_UNIT_SUL
        elif "SUVBW" in code or ("G/ML" in code and "LEAN" not in code and "BODY SURFACE" not in code):
            mapped_unit = PET_UNIT_SUV_BW
        if mapped_unit is None:
            continue
        slope = safe_float(getattr(item, "RealWorldValueSlope", None))
        intercept = safe_float(getattr(item, "RealWorldValueIntercept", None))
        return (
            mapped_unit,
            slope if slope is not None else 1.0,
            intercept if intercept is not None else 0.0,
            code_meaning or code_value or None,
        )
    return None, 1.0, 0.0, None


def _mapping_from_stored_to_cached(dataset: Dataset | None, rwv_slope: float, rwv_intercept: float) -> tuple[float, float]:
    rescale_slope = safe_float(getattr(dataset, "RescaleSlope", None)) if dataset is not None else None
    rescale_intercept = safe_float(getattr(dataset, "RescaleIntercept", None)) if dataset is not None else None
    slope = rescale_slope if rescale_slope not in (None, 0.0) else 1.0
    intercept = rescale_intercept or 0.0
    return rwv_slope / slope, rwv_intercept - intercept * rwv_slope / slope


def _valid_positive(value: float | None) -> bool:
    return value is not None and value > 0.0 and isfinite(value)


def _valid_adult_anthropometrics(weight_kg: float | None, height_m: float | None) -> bool:
    return (
        weight_kg is not None
        and height_m is not None
        and 20.0 <= weight_kg <= 350.0
        and 1.2 <= height_m <= 2.3
    )


def _janmahasatian_lbm_kg(weight_kg: float, height_m: float, sex: str) -> float | None:
    bmi = weight_kg / (height_m * height_m)
    if sex == "M":
        result = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    elif sex == "F":
        result = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)
    else:
        return None
    return result if 0.0 < result <= weight_kg * 1.25 else None


def _radiopharmaceutical_name(dataset: Dataset | None) -> str | None:
    item = _first_sequence_item(dataset, "RadiopharmaceuticalInformationSequence")
    if item is None:
        return None
    for attribute in ("Radiopharmaceutical", "RadiopharmaceuticalCodeMeaning"):
        value = str(getattr(item, attribute, "") or "").strip()
        if value:
            return value
    code_item = _first_sequence_item(item, "RadiopharmaceuticalCodeSequence")
    value = str(getattr(code_item, "CodeMeaning", "") or "").strip()
    return value or None


def _is_fdg_name(value: str | None) -> bool:
    normalized = "".join(char for char in str(value or "").upper() if char.isalnum())
    return "FDG" in normalized or "FLUORODEOXYGLUCOSE" in normalized or "18FFDG" in normalized


def build_pet_quantification_context(dataset: Dataset | None) -> PetQuantificationContext:
    source_units = str(getattr(dataset, "Units", "") or "").strip().upper() if dataset is not None else ""
    suv_type = str(getattr(dataset, "SUVType", "") or "").strip().upper() or None if dataset is not None else None
    source_label = source_units or "stored value"
    warnings: list[str] = []
    photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "").strip().upper() if dataset is not None else ""
    if photometric and photometric != "MONOCHROME2":
        warnings.append(
            f"PET Photometric Interpretation is {photometric}; quantitative values are preserved and inversion is display-only"
        )
    number_of_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1) if dataset is not None else 1
    number_of_time_slices = int(getattr(dataset, "NumberOfTimeSlices", 1) or 1) if dataset is not None else 1
    number_of_time_slots = int(getattr(dataset, "NumberOfTimeSlots", 1) or 1) if dataset is not None else 1
    series_type_value = getattr(dataset, "SeriesType", []) if dataset is not None else []
    series_type = (
        str(series_type_value)
        if isinstance(series_type_value, str)
        else "\\".join(str(value) for value in (series_type_value or []))
    )
    unsupported_reason: str | None = None
    if number_of_frames > 1:
        unsupported_reason = "Enhanced or multi-frame PET is not supported"
    elif number_of_time_slices > 1 or "DYNAMIC" in series_type.upper():
        unsupported_reason = "Dynamic PET is not supported"
    elif number_of_time_slots > 1 or hasattr(dataset, "GatedInformationSequence"):
        unsupported_reason = "Gated PET is not supported"
    if unsupported_reason:
        warnings.append(unsupported_reason)
    mappings: dict[str, PetLinearMapping] = {
        PET_UNIT_SOURCE: PetLinearMapping(1.0, 0.0, source_label, "DICOM source values")
    }

    rwvm_unit, rwvm_slope, rwvm_intercept, rwvm_label = _measurement_unit_from_rwvm(dataset)
    if rwvm_unit is not None:
        rwvm_scale, rwvm_offset = _mapping_from_stored_to_cached(dataset, rwvm_slope, rwvm_intercept)
        label = rwvm_label or PET_UNIT_LABELS.get(rwvm_unit, rwvm_unit)
        mappings[rwvm_unit] = PetLinearMapping(rwvm_scale, rwvm_offset, label, "Real World Value Mapping")
        if rwvm_unit == "BQML":
            mappings[PET_UNIT_KBQML] = PetLinearMapping(
                rwvm_scale * 0.001,
                rwvm_offset * 0.001,
                PET_UNIT_LABELS[PET_UNIT_KBQML],
                "Real World Value Mapping",
            )

    native_suv_unit: str | None = None
    if source_units in {"GML", "SUV", "SUVBW"}:
        if suv_type in {None, "", "BW"}:
            native_suv_unit = PET_UNIT_SUV_BW
            if not suv_type:
                warnings.append("SUV Type is absent; DICOM GML is interpreted as SUVbw")
        elif suv_type == "BSA":
            native_suv_unit = PET_UNIT_SUV_BSA
        elif suv_type == "LBMJANMA":
            native_suv_unit = PET_UNIT_SUL
        elif suv_type in {"LBM", "LBMJAMES", "LBMJAMES128"}:
            warnings.append(
                f"Native SUV Type {suv_type} is not relabelled as Janmahasatian SUL"
            )
        else:
            warnings.append(f"Unsupported native SUV Type: {suv_type}")
    if native_suv_unit is not None and native_suv_unit not in mappings:
        mappings[native_suv_unit] = PetLinearMapping(
            1.0,
            0.0,
            PET_UNIT_LABELS[native_suv_unit],
            f"Native DICOM {source_units}{f' / {suv_type}' if suv_type else ''}",
        )

    bqml_mapping: PetLinearMapping | None = None
    if source_units == "BQML":
        bqml_mapping = PetLinearMapping(1.0, 0.0, "Bq/ml", "Native DICOM BQML")
        mappings.setdefault(
            PET_UNIT_KBQML,
            PetLinearMapping(0.001, 0.0, PET_UNIT_LABELS[PET_UNIT_KBQML], "Native DICOM BQML"),
        )
    elif rwvm_unit == "BQML":
        bqml_mapping = PetLinearMapping(rwvm_scale, rwvm_offset, "Bq/ml", "Real World Value Mapping")
    elif rwvm_unit == PET_UNIT_KBQML:
        bqml_mapping = PetLinearMapping(rwvm_scale * 1000.0, rwvm_offset * 1000.0, "Bq/ml", "Real World Value Mapping")

    dose_bq, dose_reason = resolve_decay_corrected_dose_bq(dataset)
    weight_kg = safe_float(getattr(dataset, "PatientWeight", None)) if dataset is not None else None
    height_m = safe_float(getattr(dataset, "PatientSize", None)) if dataset is not None else None
    sex = str(getattr(dataset, "PatientSex", "") or "").strip().upper() if dataset is not None else ""

    if bqml_mapping is not None and _valid_positive(dose_bq):
        if _valid_positive(weight_kg):
            factor = weight_kg * 1000.0 / float(dose_bq)
            mappings.setdefault(
                PET_UNIT_SUV_BW,
                PetLinearMapping(
                    bqml_mapping.scale * factor,
                    bqml_mapping.offset * factor,
                    PET_UNIT_LABELS[PET_UNIT_SUV_BW],
                    "BQML + decay-corrected dose + body weight",
                ),
            )
        if _valid_adult_anthropometrics(weight_kg, height_m):
            height_cm = float(height_m) * 100.0
            bsa_cm2 = 0.007184 * height_cm**0.725 * float(weight_kg) ** 0.425 * 10000.0
            factor = bsa_cm2 / float(dose_bq)
            mappings.setdefault(
                PET_UNIT_SUV_BSA,
                PetLinearMapping(
                    bqml_mapping.scale * factor,
                    bqml_mapping.offset * factor,
                    PET_UNIT_LABELS[PET_UNIT_SUV_BSA],
                    "BQML + Du Bois body-surface area",
                ),
            )
            lbm_kg = _janmahasatian_lbm_kg(float(weight_kg), float(height_m), sex)
            if lbm_kg is not None:
                factor = lbm_kg * 1000.0 / float(dose_bq)
                mappings.setdefault(
                    PET_UNIT_SUL,
                    PetLinearMapping(
                        bqml_mapping.scale * factor,
                        bqml_mapping.offset * factor,
                        PET_UNIT_LABELS[PET_UNIT_SUL],
                        "BQML + Janmahasatian lean body mass",
                    ),
                )

    if rwvm_unit == PET_UNIT_PERCENT_ID_G:
        mappings[PET_UNIT_PERCENT_ID_G] = PetLinearMapping(
            rwvm_scale,
            rwvm_offset,
            PET_UNIT_LABELS[PET_UNIT_PERCENT_ID_G],
            "Real World Value Mapping",
        )

    reasons: dict[str, str] = {}
    if PET_UNIT_KBQML not in mappings:
        reasons[PET_UNIT_KBQML] = "Requires native BQML or a reliable Real World Value Mapping"
    if PET_UNIT_SUV_BW not in mappings:
        reasons[PET_UNIT_SUV_BW] = dose_reason or "Requires BQML, valid body weight, dose and decay timing"
    if PET_UNIT_SUV_BSA not in mappings:
        reasons[PET_UNIT_SUV_BSA] = (
            "Requires adult-range height and weight plus reliable BQML dose/timing metadata"
        )
    if PET_UNIT_SUL not in mappings:
        reasons[PET_UNIT_SUL] = (
            "Requires sex, adult-range height and weight plus reliable BQML dose/timing metadata"
        )
    if PET_UNIT_PERCENT_ID_G not in mappings:
        reasons[PET_UNIT_PERCENT_ID_G] = "Requires an explicit reliable %ID/g Real World Value Mapping"

    options = tuple(
        PetUnitOption(
            unit=unit,
            label=(
                f"Source ({source_label})"
                if unit == PET_UNIT_SOURCE
                else mappings[unit].label if unit in mappings else PET_UNIT_LABELS[unit]
            ),
            available=unit in mappings,
            reason=None if unit in mappings else reasons.get(unit),
            provenance=mappings[unit].provenance if unit in mappings else None,
        )
        for unit in PET_UNIT_ORDER
    )
    tracer_name = _radiopharmaceutical_name(dataset)
    return PetQuantificationContext(
        source_units=source_units or "UNKNOWN",
        source_unit_label=source_label,
        suv_type=suv_type,
        mappings=mappings,
        unit_options=options,
        warnings=tuple(warnings),
        quantitative=any(unit in mappings for unit in PET_UNIT_ORDER[1:]),
        tracer_name=tracer_name,
        is_fdg=_is_fdg_name(tracer_name),
        support_status="unsupported" if unsupported_reason else "static-supported",
        support_reason=unsupported_reason,
        mapping_provenance=(
            mappings[next(iter(unit for unit in PET_UNIT_ORDER if unit in mappings and unit != PET_UNIT_SOURCE))].provenance
            if any(unit in mappings for unit in PET_UNIT_ORDER if unit != PET_UNIT_SOURCE)
            else mappings[PET_UNIT_SOURCE].provenance
        ),
        photometric_interpretation=photometric or None,
    )


def apply_pet_mapping(volume: np.ndarray, mapping: PetLinearMapping) -> np.ndarray:
    source = np.asarray(volume, dtype=np.float32)
    if abs(mapping.scale - 1.0) <= 1e-12 and abs(mapping.offset) <= 1e-12:
        return source
    return source * np.float32(mapping.scale) + np.float32(mapping.offset)


def derive_pet_auto_range(volume: np.ndarray, unit: str) -> tuple[float, float]:
    finite = np.asarray(volume, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    positive = finite[finite > 0.0]
    values = positive if positive.size else finite
    # A robust distribution-derived suggestion avoids a product-specific fixed
    # SUV ceiling while preventing isolated hot/noisy voxels from dominating.
    high = float(np.nanpercentile(values, 99.5))
    low = 0.0 if float(np.nanmin(finite)) >= 0.0 else float(np.nanpercentile(finite, 1.0))
    if not isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def derive_pet_control_range_max(auto_high: float, unit: str) -> float:
    """Return a stable UI control ceiling without changing the display ceiling.

    SUV/SUL workflows conventionally need room for common presets up to 30.
    Activity/source units can span many orders of magnitude, so use a
    human-readable 1/2/5 × 10ⁿ ceiling derived from the actual volume.
    """

    high = float(auto_high) if isfinite(float(auto_high)) and float(auto_high) > 0 else 1.0
    normalized = normalize_pet_unit(unit)
    if normalized in {PET_UNIT_SUV_BW, PET_UNIT_SUV_BSA, PET_UNIT_SUL}:
        return max(30.0, high)

    exponent = floor(log10(high))
    magnitude = 10.0**exponent
    normalized_high = high / magnitude
    step = next((candidate for candidate in (1.0, 2.0, 5.0, 10.0) if normalized_high <= candidate), 10.0)
    ceiling = step * magnitude
    return max(high, ceiling, 1e-6)


def serialize_pet_unit_options(options: Iterable[PetUnitOption]) -> list[dict[str, object]]:
    return [
        {
            "unit": option.unit,
            "label": option.label,
            "available": option.available,
            "reason": option.reason,
            "provenance": option.provenance,
        }
        for option in options
    ]
