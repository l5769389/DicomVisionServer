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


def test_zoomed_hover_coordinates_are_mapped_through_transform_without_exceeding_image_bounds() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack", width=512, height=512)
    view.zoom = 2.0
    view.offset_x = 16.0
    view.offset_y = -8.0
    transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=512,
        image_height=512,
        canvas_width=512,
        canvas_height=512,
        view=view,
    )

    first = map_normalized_canvas_to_image_row_col(
        256 / 512,
        256 / 512,
        image_width=512,
        image_height=512,
        canvas_width=512,
        canvas_height=512,
        image_transform=transform,
    )
    second = map_normalized_canvas_to_image_row_col(
        257 / 512,
        256 / 512,
        image_width=512,
        image_height=512,
        canvas_width=512,
        canvas_height=512,
        image_transform=transform,
    )

    assert first == (261, 249)
    assert second == (261, 249)
    assert 1 <= first[0] <= 512
    assert 1 <= first[1] <= 512
    assert 1 <= second[0] <= 512
    assert 1 <= second[1] <= 512
