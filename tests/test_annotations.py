from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.measurement import MeasurementPoint, MeasurementSliceContext
from app.models.viewer import AnnotationRecord, ViewRecord
from app.services.viewer_service import ViewerService


@dataclass(frozen=True)
class _Transform:
    matrix: np.ndarray


def test_annotation_serialization_uses_backend_image_transform() -> None:
    annotation = AnnotationRecord(
        annotation_id="annotation-1",
        tool_type="arrow",
        points=(MeasurementPoint(10.0, 20.0), MeasurementPoint(30.0, 40.0)),
        slice_context=MeasurementSliceContext(kind="stack", slice_index=0, sop_instance_uid="sop-1"),
        text="A",
        color="#ffd166",
        size="md",
    )
    transform = _Transform(
        matrix=np.asarray(
            [
                [2.0, 0.0, 5.0],
                [0.0, 3.0, 7.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    [payload] = ViewerService._serialize_annotations(
        (annotation,),
        image_transform=transform,
        canvas_width=100,
        canvas_height=200,
    )

    assert payload.annotation_id == "annotation-1"
    assert payload.points[0].x == 0.25
    assert payload.points[0].y == 0.335
    assert payload.points[1].x == 0.65
    assert payload.points[1].y == 0.635


def test_visible_annotations_follow_current_slice() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", current_index=3)
    view.annotations = [
        AnnotationRecord(
            annotation_id="visible",
            tool_type="arrow",
            points=(MeasurementPoint(1.0, 1.0), MeasurementPoint(2.0, 2.0)),
            slice_context=MeasurementSliceContext(kind="stack", slice_index=3, sop_instance_uid="sop-3"),
        ),
        AnnotationRecord(
            annotation_id="hidden",
            tool_type="arrow",
            points=(MeasurementPoint(1.0, 1.0), MeasurementPoint(2.0, 2.0)),
            slice_context=MeasurementSliceContext(kind="stack", slice_index=4, sop_instance_uid="sop-4"),
        ),
    ]

    visible = service._build_visible_annotations(view)

    assert [annotation.annotation_id for annotation in visible] == ["visible"]
