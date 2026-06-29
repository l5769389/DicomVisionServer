from app.models.viewer import ViewRecord
from app.services.hover_mapping import map_normalized_canvas_to_image_row_col
from app.services.viewport_transformer import viewport_transformer


def test_canvas_normalized_hover_coordinates_map_to_changing_one_based_pixels() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=300, height=200)
    transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=200,
        image_height=100,
        canvas_width=300,
        canvas_height=200,
        view=view,
    )

    assert map_normalized_canvas_to_image_row_col(
        50 / 300,
        50 / 200,
        image_width=200,
        image_height=100,
        canvas_width=300,
        canvas_height=200,
        image_transform=transform,
    ) == (1, 1)
    assert map_normalized_canvas_to_image_row_col(
        249 / 300,
        149 / 200,
        image_width=200,
        image_height=100,
        canvas_width=300,
        canvas_height=200,
        image_transform=transform,
    ) == (100, 200)


def test_canvas_normalized_hover_coordinates_outside_rendered_image_return_zero() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=300, height=200)
    transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=200,
        image_height=100,
        canvas_width=300,
        canvas_height=200,
        view=view,
    )

    assert map_normalized_canvas_to_image_row_col(
        0,
        0,
        image_width=200,
        image_height=100,
        canvas_width=300,
        canvas_height=200,
        image_transform=transform,
    ) == (0, 0)
