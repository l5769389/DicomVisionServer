import io
from copy import deepcopy
from datetime import datetime

import numpy as np
from fastapi import HTTPException
from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    GrayscaleSoftcopyPresentationStateStorage,
    PYDICOM_IMPLEMENTATION_UID,
    generate_uid,
)

from app.models.viewer import ViewRecord
from app.schemas.view import (
    ViewExportAnnotationOverlayPayload,
    ViewExportMeasurementOverlayPayload,
    ViewExportOverlaysPayload,
    ViewExportPointPayload,
)


GSPS_MEASUREMENT_LAYER = "MEASURE"
GSPS_ANNOTATION_LAYER = "ANNOT"


def build_gsps_dicom_bytes(
    view: ViewRecord,
    overlays: ViewExportOverlaysPayload | None,
    reference_dataset: Dataset | None,
) -> bytes:
    if view.view_type != "Stack":
        raise HTTPException(status_code=400, detail="DICOM GSPS export is only supported for Stack views")

    measurements = list(overlays.measurements if overlays else [])
    annotations = list(overlays.annotations if overlays else [])
    if not measurements and not annotations:
        raise HTTPException(status_code=400, detail="No annotations or measurements available for DICOM GSPS export")

    referenced_sop = _build_referenced_sop_item(reference_dataset)
    if reference_dataset is None or referenced_sop is None:
        raise HTTPException(status_code=400, detail="DICOM GSPS export requires a source DICOM image reference")

    study_uid = getattr(reference_dataset, "StudyInstanceUID", None)
    series_uid = getattr(reference_dataset, "SeriesInstanceUID", None)
    if not study_uid or not series_uid:
        raise HTTPException(status_code=400, detail="Source DICOM image is missing StudyInstanceUID or SeriesInstanceUID")

    reference_size = _resolve_graphic_reference_size(view, reference_dataset)
    now = datetime.now()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = GrayscaleSoftcopyPresentationStateStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    dataset = Dataset()
    dataset.file_meta = file_meta
    _copy_reference_patient_and_study(dataset, reference_dataset)

    creation_date = now.strftime("%Y%m%d")
    creation_time = now.strftime("%H%M%S")
    dataset.SpecificCharacterSet = "ISO_IR 192"
    dataset.SOPClassUID = GrayscaleSoftcopyPresentationStateStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "PR"
    dataset.SeriesNumber = 1002
    dataset.InstanceNumber = 1
    dataset.SeriesDescription = "DicomVision Presentation State"
    dataset.Manufacturer = "DicomVision"
    dataset.PresentationCreationDate = creation_date
    dataset.PresentationCreationTime = creation_time
    dataset.ContentLabel = "DICOMVISION"
    dataset.ContentDescription = "DicomVision annotations and measurements"
    dataset.ContentCreatorName = "DicomVision"
    dataset.PresentationLUTShape = "IDENTITY"

    dataset.ReferencedSeriesSequence = _build_referenced_series_sequence(series_uid, referenced_sop)
    dataset.DisplayedAreaSelectionSequence = Sequence([
        _build_displayed_area_item(reference_size, referenced_sop),
    ])
    dataset.GraphicLayerSequence = Sequence([
        _build_graphic_layer_item(GSPS_MEASUREMENT_LAYER, order=1),
        _build_graphic_layer_item(GSPS_ANNOTATION_LAYER, order=2),
    ])
    dataset.GraphicAnnotationSequence = _build_graphic_annotation_sequence(
        measurements,
        annotations,
        reference_size=reference_size,
        referenced_sop=referenced_sop,
    )

    output = io.BytesIO()
    dcmwrite(output, dataset, enforce_file_format=True)
    return output.getvalue()


def _copy_reference_patient_and_study(dataset: Dataset, reference_dataset: Dataset) -> None:
    for attribute in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
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


def _build_graphic_annotation_sequence(
    measurements: list[ViewExportMeasurementOverlayPayload],
    annotations: list[ViewExportAnnotationOverlayPayload],
    *,
    reference_size: tuple[int, int],
    referenced_sop: Dataset,
) -> Sequence:
    items: list[Dataset] = []

    measurement_item = _build_measurement_annotation_item(measurements, reference_size, referenced_sop)
    if measurement_item is not None:
        items.append(measurement_item)

    annotation_item = _build_free_annotation_item(annotations, reference_size, referenced_sop)
    if annotation_item is not None:
        items.append(annotation_item)

    return Sequence(items)


def _build_measurement_annotation_item(
    measurements: list[ViewExportMeasurementOverlayPayload],
    reference_size: tuple[int, int],
    referenced_sop: Dataset,
) -> Dataset | None:
    graphic_objects = []
    text_objects = []
    for measurement in measurements:
        graphic_object = _build_graphic_object(measurement.tool_type, measurement.points, reference_size)
        if graphic_object is not None:
            graphic_objects.append(graphic_object)

        label = "\n".join(line for line in measurement.label_lines if line.strip()).strip()
        anchor = _resolve_text_anchor(measurement.points, reference_size)
        if label and anchor is not None:
            text_objects.append(_build_text_object(label, anchor))

    return _build_graphic_annotation_item(
        GSPS_MEASUREMENT_LAYER,
        referenced_sop,
        graphic_objects=graphic_objects,
        text_objects=text_objects,
    )


def _build_free_annotation_item(
    annotations: list[ViewExportAnnotationOverlayPayload],
    reference_size: tuple[int, int],
    referenced_sop: Dataset,
) -> Dataset | None:
    graphic_objects = []
    text_objects = []
    for annotation in annotations:
        graphic_object = _build_graphic_object(annotation.tool_type, annotation.points, reference_size)
        if graphic_object is not None:
            graphic_objects.append(graphic_object)

        anchor = _resolve_text_anchor(annotation.points, reference_size)
        text = annotation.text.strip()
        if text and anchor is not None:
            text_objects.append(_build_text_object(text, anchor))

    return _build_graphic_annotation_item(
        GSPS_ANNOTATION_LAYER,
        referenced_sop,
        graphic_objects=graphic_objects,
        text_objects=text_objects,
    )


def _build_graphic_annotation_item(
    layer_name: str,
    referenced_sop: Dataset,
    *,
    graphic_objects: list[Dataset],
    text_objects: list[Dataset],
) -> Dataset | None:
    if not graphic_objects and not text_objects:
        return None

    item = Dataset()
    item.GraphicLayer = layer_name
    item.ReferencedImageSequence = Sequence([deepcopy(referenced_sop)])
    if graphic_objects:
        item.GraphicObjectSequence = Sequence(graphic_objects)
    if text_objects:
        item.TextObjectSequence = Sequence(text_objects)
    return item


def _build_graphic_object(
    tool_type: str,
    points: list[ViewExportPointPayload],
    reference_size: tuple[int, int],
) -> Dataset | None:
    graphic_type, graphic_data = _build_graphic_data(tool_type, points, reference_size)
    if graphic_type is None or not graphic_data:
        return None

    item = Dataset()
    item.GraphicAnnotationUnits = "PIXEL"
    item.GraphicDimensions = 2
    item.NumberOfGraphicPoints = len(graphic_data) // 2
    item.GraphicData = graphic_data
    item.GraphicType = graphic_type
    item.GraphicFilled = "N"
    return item


def _build_graphic_data(
    tool_type: str,
    points: list[ViewExportPointPayload],
    reference_size: tuple[int, int],
) -> tuple[str | None, list[float]]:
    columns, rows = reference_size
    pixel_points = [
        _normalized_point_to_pixel(point, columns, rows)
        for point in points
    ]
    pixel_points = [point for point in pixel_points if point is not None]
    if not pixel_points:
        return None, []

    if len(pixel_points) == 1:
        return "POINT", [pixel_points[0][0], pixel_points[0][1]]

    if tool_type == "rect" and len(pixel_points) >= 2:
        (x1, y1), (x2, y2) = pixel_points[0], pixel_points[1]
        pixel_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    elif tool_type == "ellipse" and len(pixel_points) >= 2:
        pixel_points = _ellipse_outline_points(pixel_points[0], pixel_points[1])
    elif tool_type == "freeform" and len(pixel_points) > 2 and pixel_points[0] != pixel_points[-1]:
        pixel_points = [*pixel_points, pixel_points[0]]

    return "POLYLINE", [coordinate for point in pixel_points for coordinate in point]


def _build_text_object(text: str, anchor: tuple[float, float]) -> Dataset:
    item = Dataset()
    item.UnformattedTextValue = text
    item.AnchorPointAnnotationUnits = "PIXEL"
    item.AnchorPoint = [anchor[0], anchor[1]]
    item.AnchorPointVisibility = "Y"
    return item


def _resolve_text_anchor(
    points: list[ViewExportPointPayload],
    reference_size: tuple[int, int],
) -> tuple[float, float] | None:
    if not points:
        return None
    columns, rows = reference_size
    return _normalized_point_to_pixel(points[-1], columns, rows)


def _normalized_point_to_pixel(point: ViewExportPointPayload, columns: int, rows: int) -> tuple[float, float] | None:
    try:
        x = float(point.x)
        y = float(point.y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None

    x = 1.0 + min(max(x, 0.0), 1.0) * max(float(columns - 1), 0.0)
    y = 1.0 + min(max(y, 0.0), 1.0) * max(float(rows - 1), 0.0)
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


def _build_referenced_series_sequence(series_uid: str, referenced_sop: Dataset) -> Sequence:
    series_item = Dataset()
    series_item.SeriesInstanceUID = series_uid
    series_item.ReferencedImageSequence = Sequence([deepcopy(referenced_sop)])
    return Sequence([series_item])


def _build_displayed_area_item(reference_size: tuple[int, int], referenced_sop: Dataset) -> Dataset:
    columns, rows = reference_size
    item = Dataset()
    item.ReferencedImageSequence = Sequence([deepcopy(referenced_sop)])
    item.DisplayedAreaTopLeftHandCorner = [1, 1]
    item.DisplayedAreaBottomRightHandCorner = [columns, rows]
    item.PresentationSizeMode = "SCALE TO FIT"
    return item


def _build_graphic_layer_item(layer_name: str, *, order: int) -> Dataset:
    item = Dataset()
    item.GraphicLayer = layer_name
    item.GraphicLayerOrder = order
    return item


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


def _resolve_graphic_reference_size(view: ViewRecord, reference_dataset: Dataset) -> tuple[int, int]:
    columns = _positive_int(getattr(reference_dataset, "Columns", None))
    rows = _positive_int(getattr(reference_dataset, "Rows", None))
    columns = columns or _positive_int(view.width) or 1
    rows = rows or _positive_int(view.height) or 1
    return columns, rows


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
