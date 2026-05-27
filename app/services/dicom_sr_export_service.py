import io
import re
import unicodedata
from copy import deepcopy
from datetime import datetime

import numpy as np
from fastapi import HTTPException
from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    EnhancedSRStorage,
    ExplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID,
    generate_uid,
)

from app.models.viewer import ViewRecord
from app.schemas.view import (
    ViewExportMeasurementOverlayPayload,
    ViewExportOverlaysPayload,
    ViewExportPointPayload,
)


DICOMVISION_CODE_SCHEME = "99DICOMVISION"
MEASUREMENT_LABEL_VALUE_PATTERN = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([^\s,;]*)"
)


def build_measurement_sr_dicom_bytes(
    view: ViewRecord,
    overlays: ViewExportOverlaysPayload | None,
    reference_dataset: Dataset | None,
) -> bytes:
    measurements = list(overlays.measurements if overlays else [])
    if not measurements:
        raise HTTPException(status_code=400, detail="No measurements available for DICOM SR export")

    now = datetime.now()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = EnhancedSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    dataset = Dataset()
    dataset.file_meta = file_meta

    if reference_dataset is not None:
        for attribute in (
            "PatientName",
            "PatientID",
            "PatientBirthDate",
            "PatientSex",
            "StudyInstanceUID",
            "StudyID",
            "AccessionNumber",
            "StudyDate",
            "StudyTime",
            "ReferringPhysicianName",
            "InstitutionName",
        ):
            value = getattr(reference_dataset, attribute, None)
            if value not in (None, ""):
                setattr(dataset, attribute, value)

    content_date = now.strftime("%Y%m%d")
    content_time = now.strftime("%H%M%S")
    dataset.SpecificCharacterSet = "ISO_IR 192"
    dataset.SOPClassUID = EnhancedSRStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = getattr(dataset, "StudyInstanceUID", None) or generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "SR"
    dataset.SeriesNumber = 1001
    dataset.InstanceNumber = 1
    dataset.SeriesDescription = "DicomVision Measurement Report"
    dataset.Manufacturer = "DicomVision"
    dataset.ContentDate = content_date
    dataset.ContentTime = content_time
    dataset.InstanceCreationDate = content_date
    dataset.InstanceCreationTime = content_time
    if not getattr(dataset, "StudyDate", None):
        dataset.StudyDate = content_date
    if not getattr(dataset, "StudyTime", None):
        dataset.StudyTime = content_time
    dataset.CompletionFlag = "COMPLETE"
    dataset.VerificationFlag = "UNVERIFIED"
    dataset.ValueType = "CONTAINER"
    dataset.ContinuityOfContent = "SEPARATE"
    dataset.ConceptNameCodeSequence = _code_sequence("126000", "DCM", "Imaging Measurement Report")

    referenced_sop = _build_referenced_sop_item(reference_dataset)
    evidence_sequence = _build_current_requested_evidence_sequence(reference_dataset, referenced_sop)
    if evidence_sequence is not None:
        dataset.CurrentRequestedProcedureEvidenceSequence = evidence_sequence

    reference_size = _resolve_sr_graphic_reference_size(view, reference_dataset)
    dataset.ContentSequence = Sequence(
        [
            _build_sr_measurement_group_item(
                measurement,
                index=index,
                reference_size=reference_size,
                referenced_sop=referenced_sop,
            )
            for index, measurement in enumerate(measurements, start=1)
        ]
    )

    output = io.BytesIO()
    dcmwrite(output, dataset, enforce_file_format=True)
    return output.getvalue()


def _build_sr_measurement_group_item(
    measurement: ViewExportMeasurementOverlayPayload,
    *,
    index: int,
    reference_size: tuple[int, int],
    referenced_sop: Dataset | None,
) -> Dataset:
    group = Dataset()
    group.RelationshipType = "CONTAINS"
    group.ValueType = "CONTAINER"
    group.ContinuityOfContent = "SEPARATE"
    group.ConceptNameCodeSequence = _code_sequence("125007", "DCM", "Measurement Group")

    content_items = [
        _build_text_content_item("MEAS_INDEX", "Measurement Index", str(index)),
        _build_text_content_item("MEAS_ID", "Measurement ID", measurement.measurement_id),
        _build_text_content_item("MEAS_TOOL", "Measurement Tool", measurement.tool_type),
    ]

    graphic_item = _build_spatial_coordinate_content_item(measurement, reference_size, referenced_sop)
    if graphic_item is not None:
        content_items.append(graphic_item)

    for label in measurement.label_lines:
        clean_label = str(label).strip()
        if clean_label:
            content_items.append(_build_text_content_item("MEAS_LABEL", "Measurement Label", clean_label))
            numeric_item = _build_numeric_content_item(clean_label)
            if numeric_item is not None:
                content_items.append(numeric_item)

    group.ContentSequence = Sequence(content_items)
    return group


def _build_spatial_coordinate_content_item(
    measurement: ViewExportMeasurementOverlayPayload,
    reference_size: tuple[int, int],
    referenced_sop: Dataset | None,
) -> Dataset | None:
    graphic_type, graphic_data = _build_sr_graphic_data(measurement, reference_size)
    if graphic_type is None or not graphic_data:
        return None

    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "SCOORD"
    item.ConceptNameCodeSequence = _code_sequence("MEAS_GEOM", DICOMVISION_CODE_SCHEME, "Measurement Geometry")
    item.GraphicType = graphic_type
    item.GraphicData = graphic_data
    if referenced_sop is not None:
        item.ReferencedSOPSequence = Sequence([deepcopy(referenced_sop)])
    return item


def _build_sr_graphic_data(
    measurement: ViewExportMeasurementOverlayPayload,
    reference_size: tuple[int, int],
) -> tuple[str | None, list[float]]:
    columns, rows = reference_size
    points = [
        _normalized_point_to_pixel(point, columns, rows)
        for point in measurement.points
    ]
    points = [point for point in points if point is not None]
    if not points:
        return None, []

    if len(points) == 1:
        return "POINT", [points[0][0], points[0][1]]

    if measurement.tool_type == "rect" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[0], points[1]
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    elif measurement.tool_type == "ellipse" and len(points) >= 2:
        points = _ellipse_outline_points(points[0], points[1])
    elif measurement.tool_type == "freeform" and len(points) > 2 and points[0] != points[-1]:
        points = [*points, points[0]]

    return "POLYLINE", [coordinate for point in points for coordinate in point]


def _normalized_point_to_pixel(point: ViewExportPointPayload, columns: int, rows: int) -> tuple[float, float] | None:
    try:
        x = float(point.x)
        y = float(point.y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    x = min(max(x, 0.0), 1.0) * max(float(columns - 1), 1.0)
    y = min(max(y, 0.0), 1.0) * max(float(rows - 1), 1.0)
    return x, y


def _ellipse_outline_points(
    first_corner: tuple[float, float],
    opposite_corner: tuple[float, float],
    *,
    segments: int = 32,
) -> list[tuple[float, float]]:
    x1, y1 = first_corner
    x2, y2 = opposite_corner
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    radius_x = abs(x2 - x1) / 2.0
    radius_y = abs(y2 - y1) / 2.0
    points = [
        (
            center_x + radius_x * float(np.cos((2.0 * np.pi * index) / segments)),
            center_y + radius_y * float(np.sin((2.0 * np.pi * index) / segments)),
        )
        for index in range(segments)
    ]
    return [*points, points[0]]


def _build_numeric_content_item(label: str) -> Dataset | None:
    match = MEASUREMENT_LABEL_VALUE_PATTERN.search(label)
    if match is None:
        return None

    value_text, unit_text = match.groups()
    try:
        numeric_value = float(value_text)
    except ValueError:
        return None

    concept_name = label[: match.start()].strip(" :-") or "Measurement Value"
    measured_value = Dataset()
    measured_value.NumericValue = f"{numeric_value:.12g}"
    measured_value.MeasurementUnitsCodeSequence = _unit_code_sequence(unit_text)

    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "NUM"
    item.ConceptNameCodeSequence = _private_code_sequence(concept_name)
    item.MeasuredValueSequence = Sequence([measured_value])
    return item


def _build_text_content_item(code_value: str, code_meaning: str, text: str) -> Dataset:
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = _code_sequence(code_value, DICOMVISION_CODE_SCHEME, code_meaning)
    item.TextValue = str(text)
    return item


def _private_code_sequence(code_meaning: str) -> Sequence:
    normalized = unicodedata.normalize("NFKD", code_meaning).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[A-Za-z0-9]+", normalized.upper())
    code_value = "_".join(words)[:16].strip("_") or "MEAS_VALUE"
    return _code_sequence(code_value, DICOMVISION_CODE_SCHEME, code_meaning[:64] or "Measurement Value")


def _unit_code_sequence(unit_text: str | None) -> Sequence:
    normalized = _normalize_unit_text(unit_text)
    units = {
        "": ("1", "UCUM", "no units"),
        "mm": ("mm", "UCUM", "millimeter"),
        "cm": ("cm", "UCUM", "centimeter"),
        "m": ("m", "UCUM", "meter"),
        "mm2": ("mm2", "UCUM", "square millimeter"),
        "cm2": ("cm2", "UCUM", "square centimeter"),
        "deg": ("deg", "UCUM", "degree"),
        "degree": ("deg", "UCUM", "degree"),
        "degrees": ("deg", "UCUM", "degree"),
        "px": ("{pixel}", "UCUM", "pixel"),
        "pixel": ("{pixel}", "UCUM", "pixel"),
        "pixels": ("{pixel}", "UCUM", "pixel"),
        "%": ("%", "UCUM", "percent"),
        "hu": ("[hnsf'U]", "UCUM", "Hounsfield unit"),
    }
    code_value, coding_scheme, code_meaning = units.get(normalized, (normalized or "1", "UCUM", normalized or "no units"))
    return _code_sequence(code_value[:16], coding_scheme, code_meaning[:64])


def _normalize_unit_text(unit_text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", unit_text or "").encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().strip(".").lower().replace("^2", "2")
    if normalized.startswith("sq") and len(normalized) > 2:
        return f"{normalized[2:]}2"
    return normalized


def _code_sequence(code_value: str, coding_scheme: str, code_meaning: str) -> Sequence:
    code_item = Dataset()
    code_item.CodeValue = code_value
    code_item.CodingSchemeDesignator = coding_scheme
    code_item.CodeMeaning = code_meaning
    return Sequence([code_item])


def _build_referenced_sop_item(reference_dataset: Dataset | None) -> Dataset | None:
    if reference_dataset is None:
        return None
    sop_class_uid = getattr(reference_dataset, "SOPClassUID", None)
    sop_instance_uid = getattr(reference_dataset, "SOPInstanceUID", None)
    if not sop_class_uid or not sop_instance_uid:
        return None

    item = Dataset()
    item.ReferencedSOPClassUID = sop_class_uid
    item.ReferencedSOPInstanceUID = sop_instance_uid
    return item


def _build_current_requested_evidence_sequence(
    reference_dataset: Dataset | None,
    referenced_sop: Dataset | None,
) -> Sequence | None:
    if reference_dataset is None or referenced_sop is None:
        return None

    study_uid = getattr(reference_dataset, "StudyInstanceUID", None)
    series_uid = getattr(reference_dataset, "SeriesInstanceUID", None)
    if not study_uid or not series_uid:
        return None

    series_item = Dataset()
    series_item.SeriesInstanceUID = series_uid
    series_item.ReferencedSOPSequence = Sequence([deepcopy(referenced_sop)])

    study_item = Dataset()
    study_item.StudyInstanceUID = study_uid
    study_item.ReferencedSeriesSequence = Sequence([series_item])
    return Sequence([study_item])


def _resolve_sr_graphic_reference_size(view: ViewRecord, reference_dataset: Dataset | None) -> tuple[int, int]:
    columns = _positive_int(getattr(reference_dataset, "Columns", None) if reference_dataset else None)
    rows = _positive_int(getattr(reference_dataset, "Rows", None) if reference_dataset else None)
    columns = columns or _positive_int(view.width) or 1
    rows = rows or _positive_int(view.height) or 1
    return columns, rows


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
