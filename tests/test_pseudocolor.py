from hashlib import sha256

import numpy as np

from app.models.viewer import ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services.pseudocolor import (
    DEFAULT_PSEUDOCOLOR_PRESET,
    apply_pseudocolor,
    normalize_pseudocolor_preset,
    pseudocolor_background_color,
    pseudocolor_definition,
)
from app.services.render_layers.corner_info_layer import _overlay_text_colors
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import AffineTransform


def test_normalize_pseudocolor_preset_accepts_prefixed_values() -> None:
    assert normalize_pseudocolor_preset("pseudocolor:pet") == "pet"
    assert normalize_pseudocolor_preset("HOTIRON") == "hotiron"
    assert normalize_pseudocolor_preset("HotMetal") == "hotmetal"
    assert normalize_pseudocolor_preset("invalid") == DEFAULT_PSEUDOCOLOR_PRESET


def test_apply_pseudocolor_returns_rgb_pixels() -> None:
    grayscale = np.asarray([[0, 127, 255]], dtype=np.uint8)
    colored = apply_pseudocolor(grayscale, "pet")

    assert colored.shape == (1, 3, 3)
    assert colored.dtype == np.uint8
    assert tuple(colored[0, 0]) == (0, 0, 0)
    assert tuple(colored[0, -1]) == (255, 255, 255)


def test_versioned_lut_registry_matches_complete_256_entry_golden_tables() -> None:
    expected_hashes = {
        "bw": "72432263dbfe17abc40ed269f24c7a344e077e3671007dfc8a2f3851f8193dc2",
        "bwinverse": "346f6a25ad11ec5b45a83392366f269058ba209d2877491634a2405f86beb3db",
        "blackbody": "c50f9920dc226cbf2db7c62a94a0dd9d9a5a81e0973e73ca67c36cc66a2583d2",
        "hotiron": "03db970356806b8fa1dfddb6022c69d5e6cc8fd62dfc67dbce8a90724506356c",
        "hotmetal": "d0e644c3e683ed0cb0983315e7c1af18180027d77de9d3f09f2f76a05bdbd2b8",
        "pet": "e8aaab29be18ff65e063f3d691b9fa09db0fc85c9a28007e16fc872220411dd2",
        "rainbow": "4317d05e950432d6dafc6363cff8ed2e30f6a5ab1fd958207c24721f6270f585",
    }

    for preset, expected_hash in expected_hashes.items():
        definition = pseudocolor_definition(preset)
        assert definition.lut.shape == (256, 3)
        assert definition.lut.dtype == np.uint8
        assert definition.version
        assert definition.provenance
        assert definition.license
        assert definition.sha256 == expected_hash
        assert sha256(definition.lut.tobytes()).hexdigest() == expected_hash


def test_petct_rainbow_uses_reference_black_red_yellow_white_ramp() -> None:
    grayscale = np.asarray([[0, 80, 180, 255]], dtype=np.uint8)
    colored = apply_pseudocolor(grayscale, "petct-rainbow")

    assert tuple(colored[0, 0]) == (0, 0, 0)
    assert colored[0, 1, 0] > colored[0, 1, 1]
    assert colored[0, 2, 0] >= colored[0, 2, 1] > colored[0, 2, 2]
    assert tuple(colored[0, -1]) == (255, 255, 255)


def test_hotmetal_uses_a_distinct_high_contrast_pet_ramp() -> None:
    grayscale = np.asarray([[0, 64, 128, 192, 255]], dtype=np.uint8)
    colored = apply_pseudocolor(grayscale, "hotmetal")

    assert tuple(colored[0, 0]) == (0, 0, 0)
    assert colored[0, 2, 0] > colored[0, 2, 1] > colored[0, 2, 2]
    assert tuple(colored[0, -1]) == (255, 255, 255)


def test_lut_background_and_corner_labels_use_matching_high_contrast_treatments() -> None:
    assert pseudocolor_background_color("bw") == (0, 0, 0)
    assert pseudocolor_background_color("bwinverse") == (255, 255, 255)
    assert pseudocolor_background_color("pet") == (0, 0, 0)

    dark_foreground, dark_outline = _overlay_text_colors("hotiron")
    light_foreground, light_outline = _overlay_text_colors("bwinverse")

    assert dark_foreground[:3] == (248, 250, 252)
    assert dark_outline[:3] == (0, 0, 0)
    assert light_foreground[:3] == (24, 35, 52)
    assert light_outline[:3] == (255, 255, 255)


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
