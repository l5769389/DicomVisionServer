import numpy as np

from app.models.viewer import ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.pseudocolor import (
    DEFAULT_PSEUDOCOLOR_PRESET,
    apply_pseudocolor,
    normalize_pseudocolor_preset,
)
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import AffineTransform


def test_normalize_pseudocolor_preset_accepts_prefixed_values() -> None:
    assert normalize_pseudocolor_preset("pseudocolor:pet") == "pet"
    assert normalize_pseudocolor_preset("HOTIRON") == "hotiron"
    assert normalize_pseudocolor_preset("invalid") == DEFAULT_PSEUDOCOLOR_PRESET


def test_apply_pseudocolor_returns_rgb_pixels() -> None:
    grayscale = np.asarray([[0, 127, 255]], dtype=np.uint8)
    colored = apply_pseudocolor(grayscale, "pet")

    assert colored.shape == (1, 3, 3)
    assert colored.dtype == np.uint8
    assert tuple(colored[0, 0]) == (20, 0, 61)
    assert tuple(colored[0, -1]) == (255, 77, 90)


def test_render_fast_base_image_switches_to_rgb_for_pseudocolor() -> None:
    source_pixels = np.asarray([[0.0, 50.0], [100.0, 150.0]], dtype=np.float32)
    transform = AffineTransform(matrix=np.eye(3, dtype=np.float64))
    grayscale_view = ViewRecord(
        view_id="view-gray",
        series_id="series-1",
        view_type="Stack",
        pseudocolor_preset="bw",
        width=2,
        height=2,
    )
    color_view = ViewRecord(
        view_id="view-color",
        series_id="series-1",
        view_type="Stack",
        pseudocolor_preset="pet",
        width=2,
        height=2,
    )

    grayscale_image = ViewerService._render_fast_base_image(
        source_pixels=source_pixels,
        pixel_min=0.0,
        pixel_max=150.0,
        render_view=grayscale_view,
        image_transform=transform,
    )
    color_image = ViewerService._render_fast_base_image(
        source_pixels=source_pixels,
        pixel_min=0.0,
        pixel_max=150.0,
        render_view=color_view,
        image_transform=transform,
    )

    assert grayscale_image.mode == "L"
    assert color_image.mode == "RGB"


def test_handle_pseudocolor_normalizes_payload_alias() -> None:
    view = ViewRecord(view_id="view-1", series_id="series-1", view_type="Stack")
    payload = ViewOperationRequest.model_validate(
        {
            "viewId": "view-1",
            "opType": "pseudocolor",
            "pseudocolorPreset": "pseudocolor:rainbow",
        }
    )

    changed = ViewerService._handle_pseudocolor(view, payload)

    assert changed is True
    assert view.pseudocolor_preset == "rainbow"


