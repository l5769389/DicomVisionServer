from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydicom.uid import GrayscaleSoftcopyPresentationStateStorage

from app.models.measurement import MeasurementPoint
from app.models.viewer import (
    PresentationAnnotationRecord,
    PresentationMeasurementRecord,
    PresentationStateRecord,
)


GSPS_SOP_CLASS_UIDS = {str(GrayscaleSoftcopyPresentationStateStorage)}
MEASUREMENT_LAYER_NAMES = {"MEASURE", "MEASUREMENT", "MEASUREMENTS"}
ANNOTATION_LAYER_NAMES = {"ANNOT", "ANNOTATION", "ANNOTATIONS"}


def is_gsps_dataset(dataset: Any) -> bool:
    sop_class_uid = str(getattr(dataset, "SOPClassUID", "") or "").strip()
    modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
    return sop_class_uid in GSPS_SOP_CLASS_UIDS or modality == "PR"


def parse_gsps_dataset(dataset: Any, path: Path) -> list[PresentationStateRecord]:
    records_by_sop_uid: dict[str, dict[str, object]] = {}
    default_referenced_sop_uids = _collect_default_referenced_sop_uids(dataset)
    gsps_sop_uid = _safe_text(getattr(dataset, "SOPInstanceUID", None))

    for annotation_index, annotation_item in enumerate(_sequence_items(getattr(dataset, "GraphicAnnotationSequence", None))):
        referenced_sop_uid = _resolve_annotation_referenced_sop_uid(annotation_item, default_referenced_sop_uids)
        if not referenced_sop_uid:
            continue

        target = records_by_sop_uid.setdefault(
            referenced_sop_uid,
            {
                "measurements": [],
                "annotations": [],
            },
        )
        layer_name = _safe_text(getattr(annotation_item, "GraphicLayer", None)).upper()
        text_records = _collect_text_records(annotation_item)
        graphic_records = _collect_graphic_records(annotation_item)

        for graphic_index, graphic in enumerate(graphic_records):
            text = text_records[graphic_index].text if graphic_index < len(text_records) else ""
            overlay_id = _build_overlay_id(gsps_sop_uid, annotation_index, graphic_index)
            if layer_name in ANNOTATION_LAYER_NAMES:
                annotation = _build_annotation_record(overlay_id, graphic.points, text)
                if annotation is not None:
                    target["annotations"].append(annotation)
                continue

            measurement = _build_measurement_record(overlay_id, graphic, text)
            if measurement is not None:
                target["measurements"].append(measurement)

        for text_index, text_record in enumerate(text_records[len(graphic_records):], start=len(graphic_records)):
            annotation = _build_text_only_annotation_record(
                _build_overlay_id(gsps_sop_uid, annotation_index, text_index),
                text_record,
            )
            if annotation is not None:
                target["annotations"].append(annotation)

    return [
        PresentationStateRecord(
            path=path,
            sop_instance_uid=gsps_sop_uid,
            referenced_sop_instance_uid=sop_uid,
            measurements=tuple(value["measurements"]),
            annotations=tuple(value["annotations"]),
        )
        for sop_uid, value in records_by_sop_uid.items()
        if value["measurements"] or value["annotations"]
    ]


class _TextRecord:
    def __init__(self, text: str, anchor: MeasurementPoint | None) -> None:
        self.text = text
        self.anchor = anchor


class _GraphicRecord:
    def __init__(self, graphic_type: str, points: tuple[MeasurementPoint, ...]) -> None:
        self.graphic_type = graphic_type
        self.points = points


def _build_overlay_id(gsps_sop_uid: str | None, annotation_index: int, item_index: int) -> str:
    prefix = gsps_sop_uid or "gsps"
    return f"{prefix}:{annotation_index}:{item_index}"


def _build_measurement_record(
    overlay_id: str,
    graphic: _GraphicRecord,
    text: str,
) -> PresentationMeasurementRecord | None:
    tool_type, points = _resolve_measurement_tool_and_points(graphic)
    if tool_type is None or len(points) < 2:
        return None

    label_lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    return PresentationMeasurementRecord(
        measurement_id=f"gsps-measure-{overlay_id}",
        tool_type=tool_type,
        points=points,
        label_lines=label_lines,
    )


def _build_annotation_record(
    overlay_id: str,
    points: tuple[MeasurementPoint, ...],
    text: str,
) -> PresentationAnnotationRecord | None:
    if len(points) < 2:
        return None

    return PresentationAnnotationRecord(
        annotation_id=f"gsps-annotation-{overlay_id}",
        tool_type="arrow",
        points=(points[0], points[-1]),
        text=text,
    )


def _build_text_only_annotation_record(
    overlay_id: str,
    text_record: _TextRecord,
) -> PresentationAnnotationRecord | None:
    if not text_record.text or text_record.anchor is None:
        return None

    anchor = text_record.anchor
    end = MeasurementPoint(x=anchor.x + 1.0, y=anchor.y + 1.0)
    return PresentationAnnotationRecord(
        annotation_id=f"gsps-text-{overlay_id}",
        tool_type="arrow",
        points=(anchor, end),
        text=text_record.text,
    )


def _resolve_measurement_tool_and_points(graphic: _GraphicRecord) -> tuple[str | None, tuple[MeasurementPoint, ...]]:
    points = graphic.points
    if len(points) < 2:
        return None, ()

    if graphic.graphic_type == "ELLIPSE":
        return "ellipse", _bbox_diagonal_points(points)
    if graphic.graphic_type == "CIRCLE":
        return "ellipse", _bbox_diagonal_points(points)
    if graphic.graphic_type != "POLYLINE":
        return None, ()
    if len(points) == 2:
        return "line", points
    if _is_axis_aligned_closed_rectangle(points):
        return "rect", (points[0], points[2])
    return "freeform", points


def _bbox_diagonal_points(points: tuple[MeasurementPoint, ...]) -> tuple[MeasurementPoint, ...]:
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    return (
        MeasurementPoint(x=min_x, y=min_y),
        MeasurementPoint(x=max_x, y=max_y),
    )


def _is_axis_aligned_closed_rectangle(points: tuple[MeasurementPoint, ...]) -> bool:
    if len(points) != 5:
        return False
    if not _points_close(points[0], points[-1]):
        return False
    unique_x = {round(point.x, 3) for point in points[:-1]}
    unique_y = {round(point.y, 3) for point in points[:-1]}
    return len(unique_x) == 2 and len(unique_y) == 2


def _points_close(first: MeasurementPoint, second: MeasurementPoint) -> bool:
    return abs(first.x - second.x) <= 1e-3 and abs(first.y - second.y) <= 1e-3


def _collect_graphic_records(annotation_item: Any) -> list[_GraphicRecord]:
    records: list[_GraphicRecord] = []
    for graphic_item in _sequence_items(getattr(annotation_item, "GraphicObjectSequence", None)):
        units = _safe_text(getattr(graphic_item, "GraphicAnnotationUnits", None)).upper()
        if units and units != "PIXEL":
            continue

        graphic_type = _safe_text(getattr(graphic_item, "GraphicType", None)).upper()
        graphic_data = _numeric_list(getattr(graphic_item, "GraphicData", None))
        if len(graphic_data) < 2:
            continue

        points = tuple(
            MeasurementPoint(x=max(float(graphic_data[index]) - 1.0, 0.0), y=max(float(graphic_data[index + 1]) - 1.0, 0.0))
            for index in range(0, len(graphic_data) - 1, 2)
        )
        if points:
            records.append(_GraphicRecord(graphic_type=graphic_type, points=points))
    return records


def _collect_text_records(annotation_item: Any) -> list[_TextRecord]:
    records: list[_TextRecord] = []
    for text_item in _sequence_items(getattr(annotation_item, "TextObjectSequence", None)):
        text = _safe_text(getattr(text_item, "UnformattedTextValue", None))
        if not text:
            continue

        anchor = None
        units = _safe_text(getattr(text_item, "AnchorPointAnnotationUnits", None)).upper()
        anchor_data = _numeric_list(getattr(text_item, "AnchorPoint", None))
        if (not units or units == "PIXEL") and len(anchor_data) >= 2:
            anchor = MeasurementPoint(x=max(float(anchor_data[0]) - 1.0, 0.0), y=max(float(anchor_data[1]) - 1.0, 0.0))
        records.append(_TextRecord(text=text, anchor=anchor))
    return records


def _collect_default_referenced_sop_uids(dataset: Any) -> list[str]:
    sop_uids: list[str] = []
    for series_item in _sequence_items(getattr(dataset, "ReferencedSeriesSequence", None)):
        for image_item in _sequence_items(getattr(series_item, "ReferencedImageSequence", None)):
            sop_uid = _safe_text(getattr(image_item, "ReferencedSOPInstanceUID", None))
            if sop_uid:
                sop_uids.append(sop_uid)
    return sop_uids


def _resolve_annotation_referenced_sop_uid(annotation_item: Any, default_referenced_sop_uids: list[str]) -> str | None:
    for image_item in _sequence_items(getattr(annotation_item, "ReferencedImageSequence", None)):
        sop_uid = _safe_text(getattr(image_item, "ReferencedSOPInstanceUID", None))
        if sop_uid:
            return sop_uid
    return default_referenced_sop_uids[0] if default_referenced_sop_uids else None


def _sequence_items(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _numeric_list(value: Any) -> list[float]:
    try:
        values = list(value)
    except TypeError:
        return []

    numbers: list[float] = []
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numbers.append(number)
    return numbers


def _safe_text(value: Any) -> str:
    return str(value or "").strip()
