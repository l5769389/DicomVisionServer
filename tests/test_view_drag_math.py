import math

import numpy as np
import pytest

from app.models.viewer import ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import viewport_transformer


def _drag_request(
    *,
    op_type: str,
    action_type: str,
    x: float,
    y: float,
    interaction_id: str = "drag-1",
    anchor_x: float | None = None,
    anchor_y: float | None = None,
) -> ViewOperationRequest:
    return ViewOperationRequest.model_validate(
        {
            "viewId": "view-1",
            "opType": op_type,
            "actionType": action_type,
            "x": x,
            "y": y,
            "canvasWidth": 500,
            "canvasHeight": 250,
            "interactionId": interaction_id,
            "anchorX": anchor_x,
            "anchorY": anchor_y,
        }
    )


@pytest.mark.parametrize("zoom", [0.25, 1.0, 4.0, 12.0])
def test_pan_tracks_css_pixels_independently_of_zoom(zoom: float) -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    view.width = 1000
    view.height = 500
    view.zoom = zoom

    service._handle_drag_pan(view, _drag_request(op_type="pan", action_type="start", x=0, y=0))
    service._handle_drag_pan(view, _drag_request(op_type="pan", action_type="move", x=100, y=-50))

    # The render canvas is 2x the CSS viewport, so 100 CSS pixels become 200
    # render pixels regardless of image zoom.
    assert view.offset_x == pytest.approx(200.0)
    assert view.offset_y == pytest.approx(-100.0)


def test_pan_end_applies_release_position_and_clears_drag_origin() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    view.width = 1000
    view.height = 500
    view.offset_x = 12.0
    view.offset_y = -8.0

    service._handle_drag_pan(view, _drag_request(op_type="pan", action_type="start", x=0, y=0))
    service._handle_drag_pan(view, _drag_request(op_type="pan", action_type="end", x=75, y=25))

    assert view.offset_x == pytest.approx(162.0)
    assert view.offset_y == pytest.approx(42.0)
    assert view.drag_origin_offset_x is None
    assert view.drag_origin_offset_y is None


def test_exponential_zoom_is_symmetric_for_equal_up_and_down_drags() -> None:
    service = ViewerService()

    def zoom_after(delta_y: float) -> float:
        view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
        view.zoom = 2.0
        service._handle_drag_zoom(view, _drag_request(op_type="zoom", action_type="start", x=0, y=0))
        service._handle_drag_zoom(view, _drag_request(op_type="zoom", action_type="move", x=0, y=delta_y))
        return view.zoom

    zoom_up = zoom_after(-50.0)
    zoom_down = zoom_after(50.0)

    assert zoom_up == pytest.approx(2.0 * math.exp(0.64))
    assert zoom_down == pytest.approx(2.0 * math.exp(-0.64))
    assert (zoom_up / 2.0) * (zoom_down / 2.0) == pytest.approx(1.0)


def test_zoom_end_uses_final_coalesced_delta() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    view.zoom = 1.5

    service._handle_drag_zoom(view, _drag_request(op_type="zoom", action_type="start", x=0, y=0))
    service._handle_drag_zoom(view, _drag_request(op_type="zoom", action_type="end", x=0, y=-25))

    assert view.zoom == pytest.approx(1.5 * math.exp(0.32))
    assert view.drag_origin_zoom is None


def test_window_end_uses_final_coalesced_delta_without_an_extra_move() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    view.window_width = 400.0
    view.window_center = 40.0

    service._handle_drag_window(view, _drag_request(op_type="window", action_type="start", x=0, y=0))
    service._handle_drag_window(view, _drag_request(op_type="window", action_type="end", x=30, y=-20))

    assert view.window_width == pytest.approx(430.0)
    assert view.window_center == pytest.approx(60.0)
    assert view.drag_origin_window_width is None
    assert view.drag_origin_window_center is None


def test_zoom_anchor_keeps_the_image_point_under_the_cursor_stationary() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    view.width = 1000
    view.height = 500
    view.zoom = 1.5
    view.offset_x = 40.0
    view.offset_y = -20.0
    anchor_css_x = 100.0
    anchor_css_y = 75.0
    anchor_canvas = (anchor_css_x * 2.0, anchor_css_y * 2.0)

    old_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=512,
        image_height=512,
        canvas_width=view.width,
        canvas_height=view.height,
        view=view,
    ).matrix
    source_point = np.linalg.inv(old_transform) @ np.asarray([anchor_canvas[0], anchor_canvas[1], 1.0])

    start = _drag_request(
        op_type="zoom",
        action_type="start",
        x=0,
        y=0,
        anchor_x=anchor_css_x,
        anchor_y=anchor_css_y,
    )
    move = _drag_request(
        op_type="zoom",
        action_type="move",
        x=0,
        y=-25,
        anchor_x=anchor_css_x,
        anchor_y=anchor_css_y,
    )
    service._handle_drag_zoom(view, start)
    service._handle_drag_zoom(view, move)

    new_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=512,
        image_height=512,
        canvas_width=view.width,
        canvas_height=view.height,
        view=view,
    ).matrix
    projected = new_transform @ source_point

    assert projected[0] == pytest.approx(anchor_canvas[0])
    assert projected[1] == pytest.approx(anchor_canvas[1])
