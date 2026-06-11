from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL
from app.models.viewer import MprSegmentationState, MprSegmentationVoiBoxState, ViewGroupRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
from app.services import viewer_service as viewer_service_module
from app.services.view_registry import view_registry
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import viewport_transformer


def test_mpr_segmentation_threshold_mask_includes_hu_range() -> None:
    source_pixels = np.asarray(
        [
            [-1000.0, 299.0, 300.0],
            [512.0, 3071.0, 3072.0],
        ],
        dtype=np.float32,
    )
    state = MprSegmentationState(enabled=True, lower_hu=300.0, upper_hu=3071.0, opacity=0.45, voi_box=None)

    mask = ViewerService._build_mpr_segmentation_plane_mask(source_pixels, state, MPR_VIEWPORT_AXIAL)

    assert mask is not None
    assert mask.tolist() == [
        [False, False, True],
        [True, True, False],
    ]


def test_mpr_segmentation_voi_limits_mask_to_projected_plane_box() -> None:
    source_pixels = np.full((4, 4), 500.0, dtype=np.float32)
    state = MprSegmentationState(
        enabled=True,
        lower_hu=300.0,
        upper_hu=700.0,
        opacity=0.45,
        voi_box=MprSegmentationVoiBoxState(
            x_min=0.25,
            x_max=0.75,
            y_min=0.0,
            y_max=1.0,
            z_min=0.5,
            z_max=1.0,
        ),
    )

    mask = ViewerService._build_mpr_segmentation_plane_mask(source_pixels, state, MPR_VIEWPORT_CORONAL)

    assert mask is not None
    assert mask.tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [False, True, True, False],
        [False, True, True, False],
    ]


def test_mpr_segmentation_disabled_does_not_blend_overlay() -> None:
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR")
    view.width = 2
    view.height = 2
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=2,
        image_height=2,
        canvas_width=2,
        canvas_height=2,
        view=view,
    )
    base_image = Image.new("RGB", (2, 2), (10, 20, 30))
    state = MprSegmentationState(enabled=False, lower_hu=300.0, upper_hu=700.0, opacity=1.0, color="#ff0000")

    rendered = ViewerService._apply_mpr_segmentation_overlay(
        base_image,
        state,
        np.full((2, 2), 500.0, dtype=np.float32),
        MPR_VIEWPORT_AXIAL,
        image_transform,
        2,
        2,
    )

    assert np.asarray(rendered).tolist() == np.asarray(base_image).tolist()


def test_mpr_segmentation_overlay_blends_selected_pixels() -> None:
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR")
    view.width = 2
    view.height = 2
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=2,
        image_height=2,
        canvas_width=2,
        canvas_height=2,
        view=view,
    )
    source_pixels = np.asarray([[0.0, 500.0], [800.0, 900.0]], dtype=np.float32)
    state = MprSegmentationState(enabled=True, lower_hu=300.0, upper_hu=800.0, opacity=1.0, color="#ff0000", voi_box=None)

    rendered = ViewerService._apply_mpr_segmentation_overlay(
        Image.new("RGB", (2, 2), (10, 20, 30)),
        state,
        source_pixels,
        MPR_VIEWPORT_AXIAL,
        image_transform,
        2,
        2,
    )

    pixels = np.asarray(rendered)
    assert pixels[0, 0, :3].tolist() == [10, 20, 30]
    assert pixels[0, 1, :3].tolist() == [255, 0, 0]
    assert pixels[1, 0, :3].tolist() == [255, 0, 0]
    assert pixels[1, 1, :3].tolist() == [10, 20, 30]


def test_mpr_segmentation_operation_updates_shared_group_and_broadcasts(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id=series.series_id)
    axial_view = ViewRecord(view_id="v-ax", series_id=series.series_id, view_type="MPR", view_group=group)
    coronal_view = ViewRecord(view_id="v-cor", series_id=series.series_id, view_type="COR", view_group=group)
    sagittal_view = ViewRecord(view_id="v-sag", series_id=series.series_id, view_type="SAG", view_group=group)
    for candidate_view in (axial_view, coronal_view, sagittal_view):
        candidate_view.width = 240
        candidate_view.height = 240
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update({
            axial_view.view_id: axial_view,
            coronal_view.view_id: coronal_view,
            sagittal_view.view_id: sagittal_view,
        })

        outcome = service.handle_view_operation(
            ViewOperationRequest(
                viewId=axial_view.view_id,
                opType="mprSegmentation",
                actionType="end",
                mprSegmentationConfig={
                    "enabled": True,
                    "lowerHu": 700,
                    "upperHu": 300,
                    "opacity": 0.8,
                    "color": "#ABCDEF",
                    "voiBox": {
                        "xMin": 0.2,
                        "xMax": 0.8,
                        "yMin": 0.1,
                        "yMax": 0.9,
                        "zMin": 0.0,
                        "zMax": 1.0,
                    },
                },
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert group.mpr_segmentation.enabled is True
    assert group.mpr_segmentation.lower_hu == 300
    assert group.mpr_segmentation.upper_hu == 700
    assert group.mpr_segmentation.opacity == 0.8
    assert group.mpr_segmentation.color == "#abcdef"
    assert group.mpr_segmentation.voi_box is not None
    assert group.mpr_segmentation.voi_box.x_min == 0.2
    assert outcome.broadcast_view_ids == ("v-ax", "v-cor", "v-sag")
    assert outcome.broadcast_fast_preview is False
    assert outcome.mpr_revision == 1


def test_mpr_segmentation_operation_is_noop_for_non_mpr_view() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="v", series_id="s", view_type="STACK")
    payload = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprSegmentation",
        mprSegmentationConfig={"enabled": True, "lowerHu": 300, "upperHu": 700},
    )

    assert service._handle_mpr_segmentation_config(view, payload) is False
