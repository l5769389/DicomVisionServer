from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.models.measurement import MeasurementMetrics, MeasurementPoint, MeasurementRecord, MeasurementSliceContext
from app.models.viewer import ViewRecord
from app.schemas.view import OverlayPointPayload, ViewHoverRequest, ViewOperationRequest
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import viewport_transformer


def _identity_hover_context(view: ViewRecord, width: int, height: int):
    transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=width,
        image_height=height,
        canvas_width=width,
        canvas_height=height,
        view=view,
    )
    return width, height, transform, width, height


def test_ct_hover_returns_one_based_coordinates_and_hu(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=2, height=2)
    pixels = np.asarray([[-1024.0, -824.0], [-524.0, -24.0]], dtype=np.float32)
    dataset = SimpleNamespace(PhotometricInterpretation="MONOCHROME2")
    monkeypatch.setattr(
        service,
        "_get_hover_source_context",
        lambda *_args, **_kwargs: (pixels, dataset, "CT", "CT", "HU"),
    )
    monkeypatch.setattr(
        service,
        "_build_hover_mapping_context",
        lambda *_args, **_kwargs: _identity_hover_context(view, 2, 2),
    )

    result = service._resolve_hover_sample_for_workspace(view, 0.75, 0.75)

    assert result == (2, 2, -24.0, "CT", "HU")


def test_mpr_hover_returns_interpolated_hu_value(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-mpr", series_id="series-1", view_type="AX", width=2, height=2)
    plane = np.asarray([[10.25, 20.5], [30.75, 40.125]], dtype=np.float32)
    dataset = SimpleNamespace(PhotometricInterpretation="MONOCHROME2")
    monkeypatch.setattr(
        service,
        "_get_hover_source_context",
        lambda *_args, **_kwargs: (plane, dataset, "CT", "CT", "HU"),
    )
    monkeypatch.setattr(
        service,
        "_build_hover_mapping_context",
        lambda *_args, **_kwargs: _identity_hover_context(view, 2, 2),
    )

    assert service._resolve_hover_sample_for_workspace(view, 0.75, 0.75) == (2, 2, 40.125, "CT", "HU")


def test_backend_formats_authoritative_hover_text_without_ct_prefix() -> None:
    service = ViewerService()

    assert service._format_hover_display_text(99, 210, 115.4, "HU") == "X: 210 Y:  99    115 HU"
    assert service._format_hover_display_text(8, 12, 2.375, "SUVbw") == "X:  12 Y:   8  2.375 SUVbw"
    assert service._format_hover_display_text(0, 0, None, "HU") is None


def test_hover_response_pushes_authoritative_display_text(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=512, height=512)
    monkeypatch.setattr("app.services.viewer.interaction.compat.view_registry.get", lambda *_args, **_kwargs: view)
    monkeypatch.setattr(
        service,
        "_resolve_hover_sample_for_workspace",
        lambda *_args, **_kwargs: (99, 210, 115.4, "CT", "HU"),
    )

    result = service.handle_view_hover(ViewHoverRequest(viewId="view-1", x=0.4, y=0.2))

    assert result.model_dump(by_alias=True)["displayText"] == "X: 210 Y:  99    115 HU"


def test_hover_outside_image_does_not_return_a_pixel_value(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=4, height=2)
    pixels = np.ones((2, 2), dtype=np.float32)
    dataset = SimpleNamespace(PhotometricInterpretation="MONOCHROME2")
    monkeypatch.setattr(
        service,
        "_get_hover_source_context",
        lambda *_args, **_kwargs: (pixels, dataset, "CT", "CT", "HU"),
    )
    monkeypatch.setattr(
        service,
        "_build_hover_mapping_context",
        lambda *_args, **_kwargs: (
            2,
            2,
            viewport_transformer.build_image_to_canvas_transform(
                image_width=2,
                image_height=2,
                canvas_width=4,
                canvas_height=2,
                view=view,
            ),
            4,
            2,
        ),
    )

    assert service._resolve_hover_sample_for_workspace(view, 0.0, 0.0) == (0, 0, None, "CT", "HU")


def test_measurement_serialization_preserves_offscreen_geometry() -> None:
    measurement = MeasurementRecord(
        measurement_id="measurement-1",
        tool_type="line",
        points=(MeasurementPoint(-25.0, 20.0), MeasurementPoint(125.0, 80.0)),
        slice_context=MeasurementSliceContext(kind="stack", slice_index=0, sop_instance_uid="sop-1"),
        metrics=MeasurementMetrics(unit="px", area_unit="px2"),
        label_anchor=MeasurementPoint(0.0, 0.0),
    )
    transform = SimpleNamespace(matrix=np.eye(3, dtype=np.float64))

    [payload] = ViewerService._serialize_measurements(
        (measurement,),
        image_transform=transform,
        canvas_width=100,
        canvas_height=100,
    )

    assert [(point.x, point.y) for point in payload.points] == [(-0.25, 0.2), (1.25, 0.8)]


def test_overlay_points_allow_offscreen_values_but_operation_inputs_remain_bounded() -> None:
    assert OverlayPointPayload(x=-0.25, y=1.25).model_dump() == {"x": -0.25, "y": 1.25}

    with pytest.raises(ValidationError):
        ViewOperationRequest(
            viewId="view-1",
            opType="measurement",
            points=[{"x": -0.25, "y": 0.5}],
        )
