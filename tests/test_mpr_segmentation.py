from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.models.viewer import (
    MprSegmentationState,
    MprSegmentationVoiBoxState,
    MprThresholdRegionBoxState,
    MprThresholdRegionState,
    MprVoiSphereState,
    ViewGroupRecord,
    ViewRecord,
)
from app.schemas.view import MprSegmentationConfig, ViewOperationRequest
from app.services import viewer_service as viewer_service_module
from app.services.mpr import PlanePose, build_identity_geometry
from app.services.view_registry import view_registry
from app.services.viewer_operation_handlers import _handle_mpr_segmentation_operation
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import viewport_transformer


def _plane_pose(viewport: str, center: tuple[float, float, float] = (0.0, 2.0, 2.0)) -> PlanePose:
    if viewport == MPR_VIEWPORT_CORONAL:
        row = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        col = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        normal = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    elif viewport == MPR_VIEWPORT_SAGITTAL:
        row = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        col = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    else:
        row = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        col = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        normal = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    return PlanePose(
        viewport=viewport,
        center_world=np.asarray(center, dtype=np.float64),
        cursor_center_world=np.asarray(center, dtype=np.float64),
        row_world=row,
        col_world=col,
        normal_world=normal,
        pixel_spacing_row_mm=1.0,
        pixel_spacing_col_mm=1.0,
        output_shape=(5, 5),
        is_oblique=False,
    )


def _region(
    region_id: str,
    *,
    center: tuple[float, float, float] = (0.0, 2.0, 2.0),
    threshold_hu: float = 300.0,
    threshold_mode: str = "hu",
    threshold_percentile: float = 80.0,
    enabled: bool = True,
    width_mm: float = 5.0,
    height_mm: float = 5.0,
    depth_mm: float = 1.0,
) -> MprThresholdRegionState:
    return MprThresholdRegionState(
        id=region_id,
        enabled=enabled,
        threshold_hu=threshold_hu,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
        color="#ff4df8",
        box=MprThresholdRegionBoxState(
            center_world=center,
            row_world=(0.0, 1.0, 0.0),
            col_world=(0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
            width_mm=width_mm,
            height_mm=height_mm,
            depth_mm=depth_mm,
            source_viewport=MPR_VIEWPORT_AXIAL,
        ),
    )


def test_mpr_plane_payload_includes_image_to_canvas_matrix() -> None:
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR", view_group=group)
    transform = SimpleNamespace(
        matrix=np.asarray(
            [
                [2.0, 0.0, 10.0],
                [0.0, 3.0, 20.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    plane = ViewerService()._build_mpr_plane_payload(
        view,
        MPR_VIEWPORT_AXIAL,
        plane_pose=_plane_pose(MPR_VIEWPORT_AXIAL),
        geometry=build_identity_geometry((5, 5, 5)),
        image_transform=transform,
    )

    assert plane is not None
    assert plane.image_to_canvas_matrix == (
        (2.0, 0.0, 10.0),
        (0.0, 3.0, 20.0),
        (0.0, 0.0, 1.0),
    )


def test_mpr_segmentation_selected_voi_is_mutually_exclusive_with_region_selection() -> None:
    config = MprSegmentationConfig.model_validate({
        "enabled": True,
        "selectedRegionId": "r1",
        "selectedVoi": True,
        "thresholdRegions": [
            {
                "id": "r1",
                "enabled": True,
                "label": "1",
                "thresholdHu": 300,
                "box": {
                    "centerWorld": [0, 0, 0],
                    "rowWorld": [0, 1, 0],
                    "colWorld": [0, 0, 1],
                    "normalWorld": [1, 0, 0],
                    "widthMm": 10,
                    "heightMm": 10,
                    "depthMm": 1,
                    "sourceViewport": MPR_VIEWPORT_AXIAL,
                },
            }
        ],
        "voiSphere": {
            "enabled": True,
            "centerWorld": [0, 0, 0],
            "radiusMm": 8,
            "color": "#22d3ee",
        },
    })

    state = ViewerService._normalize_mpr_segmentation_state(config)
    serialized = ViewerService._serialize_mpr_segmentation_config(state)

    assert state.selected_voi is True
    assert state.selected_region_id is None
    assert serialized.selected_voi is True
    assert serialized.selected_region_id is None


def test_mpr_segmentation_supports_multiple_selected_voi_spheres() -> None:
    config = MprSegmentationConfig.model_validate({
        "enabled": True,
        "selectedRegionId": "r1",
        "selectedVoiId": "v2",
        "thresholdRegions": [
            {
                "id": "r1",
                "enabled": True,
                "label": "1",
                "thresholdHu": 300,
                "box": {
                    "centerWorld": [0, 0, 0],
                    "rowWorld": [0, 1, 0],
                    "colWorld": [0, 0, 1],
                    "normalWorld": [1, 0, 0],
                    "widthMm": 10,
                    "heightMm": 10,
                    "depthMm": 1,
                    "sourceViewport": MPR_VIEWPORT_AXIAL,
                },
            }
        ],
        "voiSpheres": [
            {
                "id": "v1",
                "label": "A",
                "enabled": True,
                "centerWorld": [0, 0, 0],
                "radiusMm": 8,
                "color": "#22d3ee",
            },
            {
                "id": "v2",
                "label": "B",
                "enabled": True,
                "centerWorld": [1, 1, 1],
                "radiusMm": 6,
                "color": "#22d3ee",
            },
        ],
    })

    state = ViewerService._normalize_mpr_segmentation_state(config)
    serialized = ViewerService._serialize_mpr_segmentation_config(state)

    assert len(state.voi_spheres) == 2
    assert state.selected_voi is True
    assert state.selected_voi_id == "v2"
    assert state.selected_region_id is None
    assert state.voi_sphere is state.voi_spheres[1]
    assert len(serialized.voi_spheres) == 2
    assert serialized.selected_voi_id == "v2"
    assert serialized.voi_sphere is not None
    assert serialized.voi_sphere.id == "v2"


def test_mpr_segmentation_threshold_region_uses_hu_greater_than_threshold() -> None:
    source_pixels = np.asarray(
        [
            [-1000.0, 299.0, 300.0, 301.0, 900.0],
            [512.0, 3071.0, 3072.0, 250.0, 400.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [100.0, 200.0, 300.0, 301.0, 302.0],
            [600.0, 700.0, 800.0, 900.0, 1000.0],
        ],
        dtype=np.float32,
    )
    state = MprSegmentationState(
        enabled=True,
        selected_region_id="r1",
        threshold_regions=[_region("r1", threshold_hu=300.0)],
    )

    mask = ViewerService._build_mpr_segmentation_plane_mask(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
    )

    assert mask is not None
    assert mask.tolist() == (source_pixels > 300.0).tolist()


def test_mpr_segmentation_percentile_region_uses_effective_threshold() -> None:
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    geometry = build_identity_geometry(volume.shape)
    region = _region(
        "r1",
        center=(1.0, 1.0, 1.0),
        threshold_hu=-100.0,
        threshold_mode="percentile",
        threshold_percentile=80.0,
        width_mm=3.0,
        height_mm=3.0,
        depth_mm=3.0,
    )

    stats = ViewerService._compute_mpr_threshold_region_stats(volume, geometry, region)
    region.stats = stats
    mask = ViewerService._build_mpr_segmentation_plane_mask(
        np.asarray(
            [
                [18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0],
                [24.0, 25.0, 26.0],
            ],
            dtype=np.float32,
        ),
        MprSegmentationState(enabled=True, threshold_regions=[region]),
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL, center=(1.0, 1.0, 1.0)),
    )

    assert np.isclose(stats.effective_threshold_hu, 20.8)
    assert stats.sample_count == 6
    assert stats.hu_min == 21.0
    assert mask is not None
    assert mask.tolist() == [
        [False, False, False],
        [True, True, True],
        [True, True, True],
    ]


def test_mpr_segmentation_merges_multiple_regions_and_skips_disabled_regions() -> None:
    source_pixels = np.full((5, 5), 500.0, dtype=np.float32)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[
            _region("r1", center=(0.0, 1.0, 1.0), width_mm=1.0, height_mm=1.0),
            _region("r2", center=(0.0, 3.0, 3.0), enabled=False, width_mm=1.0, height_mm=1.0),
            _region("r3", center=(0.0, 1.0, 3.0), width_mm=1.0, height_mm=1.0),
        ],
    )

    mask = ViewerService._build_mpr_segmentation_plane_mask(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
    )

    expected = np.zeros((5, 5), dtype=bool)
    expected[1, 1] = True
    expected[1, 3] = True
    assert mask is not None
    assert mask.tolist() == expected.tolist()


def test_mpr_segmentation_box_projects_consistently_to_three_mpr_planes() -> None:
    source_pixels = np.full((5, 5), 500.0, dtype=np.float32)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[
            _region("r1", center=(2.0, 2.0, 2.0), width_mm=3.0, height_mm=3.0, depth_mm=3.0),
        ],
    )

    for viewport, center in (
        (MPR_VIEWPORT_AXIAL, (2.0, 2.0, 2.0)),
        (MPR_VIEWPORT_CORONAL, (2.0, 2.0, 2.0)),
        (MPR_VIEWPORT_SAGITTAL, (2.0, 2.0, 2.0)),
    ):
        mask = ViewerService._build_mpr_segmentation_plane_mask(
            source_pixels,
            state,
            viewport,
            _plane_pose(viewport, center=center),
        )

        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        assert mask is not None
        assert mask.tolist() == expected.tolist()


def test_mpr_segmentation_region_stats_use_thresholded_box_voxels() -> None:
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    geometry = build_identity_geometry(volume.shape)
    region = _region(
        "r1",
        center=(1.0, 1.0, 1.0),
        threshold_hu=23.0,
        width_mm=3.0,
        height_mm=3.0,
        depth_mm=3.0,
    )

    stats = ViewerService._compute_mpr_threshold_region_stats(volume, geometry, region)

    assert stats.sample_count == 3
    assert stats.hu_min == 24.0
    assert stats.hu_max == 26.0
    assert stats.hu_mean == 25.0
    assert np.isclose(stats.hu_std_dev, np.sqrt(2.0 / 3.0))
    assert np.isclose(stats.volume_cm3, 0.003)
    assert stats.effective_threshold_hu == 23.0


def test_mpr_voi_sphere_stats_use_all_sphere_voxels() -> None:
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    geometry = build_identity_geometry(volume.shape)
    sphere = MprVoiSphereState(center_world=(1.0, 1.0, 1.0), radius_mm=1.01)

    stats = ViewerService._compute_mpr_voi_sphere_stats(volume, geometry, sphere)

    assert stats.sample_count == 7
    assert stats.hu_min == 4.0
    assert stats.hu_max == 22.0
    assert stats.hu_mean == 13.0
    assert np.isclose(stats.hu_std_dev, np.sqrt(26.0))
    assert np.isclose(stats.volume_cm3, 0.007)


def test_mpr_segmentation_refresh_updates_voi_sphere_stats() -> None:
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    geometry = build_identity_geometry(volume.shape)
    state = MprSegmentationState(
        enabled=True,
        voi_sphere=MprVoiSphereState(center_world=(1.0, 1.0, 1.0), radius_mm=1.01),
    )

    ViewerService._refresh_mpr_segmentation_stats(state, volume, geometry)

    assert state.voi_sphere is not None
    assert state.voi_sphere.stats is not None
    assert state.voi_sphere.stats.sample_count == 7
    assert state.voi_sphere.stats.hu_mean == 13.0


def test_mpr_segmentation_refresh_clears_disabled_region_stats() -> None:
    volume = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    geometry = build_identity_geometry(volume.shape)
    region = _region(
        "r1",
        center=(1.0, 1.0, 1.0),
        enabled=False,
        threshold_hu=0.0,
        width_mm=3.0,
        height_mm=3.0,
        depth_mm=3.0,
    )
    region.stats = ViewerService._compute_mpr_threshold_region_stats(volume, geometry, region)
    assert region.stats.sample_count > 0

    ViewerService._refresh_mpr_segmentation_region_stats(
        MprSegmentationState(enabled=True, threshold_regions=[region]),
        volume,
        geometry,
    )

    assert region.stats is not None
    assert region.stats.sample_count == 0
    assert region.stats.hu_mean is None
    assert region.stats.volume_cm3 == 0.0
    assert region.stats.effective_threshold_hu is None


def test_mpr_segmentation_overlay_rect_uses_actual_threshold_mask_bbox() -> None:
    source_pixels = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 500.0, 500.0, 0.0, 0.0],
            [0.0, 500.0, 500.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", threshold_hu=300.0)],
    )

    overlay = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
    )

    assert overlay is not None
    assert len(overlay.regions) == 1
    assert overlay.regions[0].visible is True
    assert overlay.regions[0].rect is not None
    assert overlay.regions[0].rect.x_min == 0.2
    assert overlay.regions[0].rect.y_min == 0.2
    assert overlay.regions[0].rect.x_max == 0.6
    assert overlay.regions[0].rect.y_max == 0.6


def test_mpr_segmentation_overlay_samples_use_geometry_mask_before_threshold() -> None:
    source_pixels = np.arange(25, dtype=np.float32).reshape(5, 5)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", threshold_hu=10_000.0)],
    )

    overlay = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
    )

    assert overlay is not None
    region = overlay.regions[0]
    assert region.visible is False
    assert region.rect is None
    assert region.sample_revision > 0
    assert region.samples is not None
    assert region.samples.total_count == 25
    assert region.samples.sampled_count == 25
    assert region.samples.points[:6] == [0.5, 0.5, 0.0, 1.5, 0.5, 1.0]


def test_mpr_segmentation_fast_preview_overlay_omits_samples() -> None:
    source_pixels = np.arange(25, dtype=np.float32).reshape(5, 5)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", threshold_hu=-1_000.0)],
    )

    overlay = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
        include_samples=False,
    )

    assert overlay is not None
    region = overlay.regions[0]
    assert region.visible is True
    assert region.rect is not None
    assert region.sample_revision > 0
    assert region.samples is None


def test_mpr_segmentation_overlay_sample_revision_ignores_threshold_but_tracks_geometry() -> None:
    source_pixels = np.arange(25, dtype=np.float32).reshape(5, 5)
    plane = _plane_pose(MPR_VIEWPORT_AXIAL)
    first = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        MprSegmentationState(enabled=True, threshold_regions=[_region("r1", threshold_hu=100.0, depth_mm=1.0)]),
        MPR_VIEWPORT_AXIAL,
        plane,
    )
    threshold_changed = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        MprSegmentationState(enabled=True, threshold_regions=[_region("r1", threshold_hu=900.0, depth_mm=1.0)]),
        MPR_VIEWPORT_AXIAL,
        plane,
    )
    depth_changed = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        MprSegmentationState(enabled=True, threshold_regions=[_region("r1", threshold_hu=100.0, depth_mm=2.0)]),
        MPR_VIEWPORT_AXIAL,
        plane,
    )

    assert first is not None
    assert threshold_changed is not None
    assert depth_changed is not None
    assert first.regions[0].sample_revision == threshold_changed.regions[0].sample_revision
    assert first.regions[0].sample_revision != depth_changed.regions[0].sample_revision


def test_mpr_segmentation_overlay_samples_are_capped_with_stable_sampling() -> None:
    source_pixels = np.arange(25, dtype=np.float32).reshape(5, 5)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", threshold_hu=-1000.0)],
    )

    first = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
        sample_limit=5,
    )
    second = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
        sample_limit=5,
    )

    assert first is not None
    assert second is not None
    assert first.regions[0].samples is not None
    assert first.regions[0].samples.total_count == 25
    assert first.regions[0].samples.sampled_count == 5
    assert first.regions[0].samples.points == second.regions[0].samples.points


def test_mpr_segmentation_preview_overlay_includes_limited_samples() -> None:
    source_pixels = np.arange(100, dtype=np.float32).reshape(10, 10)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", threshold_hu=-1000.0, width_mm=10.0, height_mm=10.0)],
    )

    overlay = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        PlanePose(
            viewport=MPR_VIEWPORT_AXIAL,
            center_world=np.asarray((0.0, 4.5, 4.5), dtype=np.float64),
            cursor_center_world=np.asarray((0.0, 4.5, 4.5), dtype=np.float64),
            row_world=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
            col_world=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            normal_world=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            pixel_spacing_row_mm=1.0,
            pixel_spacing_col_mm=1.0,
            output_shape=(10, 10),
            is_oblique=False,
        ),
        include_samples=True,
        sample_limit=7,
    )

    assert overlay is not None
    samples = overlay.regions[0].samples
    assert samples is not None
    assert samples.total_count == 64
    assert samples.sampled_count == 7


def test_mpr_voi_sphere_projection_reports_intersection_and_dashed_projection_state() -> None:
    plane = _plane_pose(MPR_VIEWPORT_AXIAL, center=(0.0, 0.0, 0.0))
    sphere = MprVoiSphereState(center_world=(3.0, 4.0, 0.0), radius_mm=5.0)

    projection = ViewerService._project_mpr_voi_sphere_to_plane(sphere, plane)

    assert projection["intersects"] is True
    assert projection["centerMm"] == (4.0, 0.0)
    assert projection["distanceToPlaneMm"] == 3.0
    assert projection["radiusMm"] == 4.0

    outside_projection = ViewerService._project_mpr_voi_sphere_to_plane(
        MprVoiSphereState(center_world=(6.0, 4.0, 0.0), radius_mm=5.0),
        plane,
    )
    assert outside_projection["intersects"] is False
    assert outside_projection["radiusMm"] == 5.0


def test_mpr_segmentation_overlay_uses_bright_dotted_threshold_points() -> None:
    view = ViewRecord(view_id="v", series_id="s", view_type="MPR")
    view.width = 5
    view.height = 5
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=5,
        image_height=5,
        canvas_width=5,
        canvas_height=5,
        view=view,
    )
    source_pixels = np.full((5, 5), 500.0, dtype=np.float32)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", threshold_hu=300.0)],
    )

    rendered = ViewerService._apply_mpr_segmentation_overlay(
        Image.new("RGB", (5, 5), (10, 20, 30)),
        state,
        source_pixels,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL),
        image_transform,
        5,
        5,
    )

    pixels = np.asarray(rendered)
    highlighted = np.any(pixels[:, :, :3] != np.asarray([10, 20, 30], dtype=np.uint8), axis=2)
    highlighted_count = int(np.count_nonzero(highlighted))
    assert 8 <= highlighted_count <= 17
    assert highlighted_count < source_pixels.size
    assert pixels[highlighted][0, :3].tolist() == [225, 70, 221]


def test_mpr_segmentation_dot_pattern_is_canvas_space_stipple_not_diagonal_stripes() -> None:
    mask = np.ones((64, 64), dtype=bool)

    dotted = ViewerService._apply_segmentation_dot_pattern(mask)

    row_index, col_index = np.indices(mask.shape, dtype=np.int32)
    old_diagonal_pattern = ((row_index + col_index) % 3) == 0
    coverage = float(np.count_nonzero(dotted)) / float(mask.size)
    assert 0.45 <= coverage <= 0.55
    assert not np.array_equal(dotted, old_diagonal_pattern)


def test_mpr_segmentation_overlay_bbox_uses_full_mask_not_dotted_mask() -> None:
    source_pixels = np.full((8, 8), 500.0, dtype=np.float32)
    state = MprSegmentationState(
        enabled=True,
        threshold_regions=[_region("r1", center=(0.0, 3.5, 3.5), threshold_hu=300.0, width_mm=8.0, height_mm=8.0)],
    )

    overlay = ViewerService._build_mpr_segmentation_overlay_payload(
        source_pixels,
        state,
        MPR_VIEWPORT_AXIAL,
        _plane_pose(MPR_VIEWPORT_AXIAL, center=(0.0, 3.5, 3.5)),
    )

    assert overlay is not None
    assert overlay.regions[0].rect is not None
    assert overlay.regions[0].rect.x_min == 0.0
    assert overlay.regions[0].rect.y_min == 0.0
    assert overlay.regions[0].rect.x_max == 1.0
    assert overlay.regions[0].rect.y_max == 1.0


def test_legacy_mpr_segmentation_fields_remain_supported_for_input() -> None:
    source_pixels = np.asarray(
        [
            [-1000.0, 299.0, 300.0],
            [512.0, 3071.0, 3072.0],
        ],
        dtype=np.float32,
    )
    state = MprSegmentationState(
        enabled=True,
        lower_hu=300.0,
        upper_hu=3071.0,
        opacity=0.45,
        voi_box=None,
        legacy_enabled=True,
    )

    mask = ViewerService._build_mpr_segmentation_plane_mask(source_pixels, state, MPR_VIEWPORT_AXIAL)

    assert mask is not None
    assert mask.tolist() == [
        [False, False, True],
        [True, True, False],
    ]


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
                    "clientRevision": 7,
                    "selectedRegionId": "r1",
                    "thresholdRegions": [
                        {
                            "id": "r1",
                            "enabled": True,
                            "label": "1",
                            "thresholdHu": 350,
                            "thresholdMode": "percentile",
                            "thresholdPercentile": 75,
                            "color": "#FF4DF8",
                            "box": {
                                "centerWorld": [0, 1, 1],
                                "rowWorld": [0, 1, 0],
                                "colWorld": [0, 0, 1],
                                "normalWorld": [1, 0, 0],
                                "widthMm": 12,
                                "heightMm": 8,
                                "depthMm": 2,
                                "sourceViewport": MPR_VIEWPORT_AXIAL,
                            },
                        }
                    ],
                    "voiSphere": {
                        "enabled": True,
                        "centerWorld": [0, 1, 1],
                        "radiusMm": 10,
                        "color": "#22D3EE",
                    },
                },
            )
        )
    finally:
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)

    assert group.mpr_segmentation.enabled is True
    assert group.mpr_segmentation.client_revision == 7
    assert group.mpr_segmentation.selected_region_id == "r1"
    assert len(group.mpr_segmentation.threshold_regions) == 1
    assert group.mpr_segmentation.threshold_regions[0].threshold_hu == 350
    assert group.mpr_segmentation.threshold_regions[0].threshold_mode == "percentile"
    assert group.mpr_segmentation.threshold_regions[0].threshold_percentile == 75
    assert group.mpr_segmentation.threshold_regions[0].color == "#ff4df8"
    assert group.mpr_segmentation.voi_sphere is not None
    assert group.mpr_segmentation.voi_sphere.color == "#22d3ee"
    assert outcome.broadcast_view_ids == ("v-ax", "v-cor", "v-sag")
    assert outcome.broadcast_fast_preview is False
    assert outcome.mpr_revision == 1


def test_mpr_segmentation_move_does_not_refresh_stats(monkeypatch) -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id="s")
    view = ViewRecord(view_id="v-ax", series_id="s", view_type="MPR", view_group=group)
    payload = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprSegmentation",
        actionType="move",
        mprSegmentationConfig={
            "enabled": True,
            "selectedRegionId": "r1",
            "thresholdRegions": [
                {
                    "id": "r1",
                    "enabled": True,
                    "label": "1",
                    "thresholdHu": 350,
                    "color": "#ff4df8",
                    "box": {
                        "centerWorld": [0, 1, 1],
                        "rowWorld": [0, 1, 0],
                        "colWorld": [0, 0, 1],
                        "normalWorld": [1, 0, 0],
                        "widthMm": 12,
                        "heightMm": 8,
                        "depthMm": 2,
                        "sourceViewport": MPR_VIEWPORT_AXIAL,
                    },
                }
            ],
        },
    )
    calls = {"count": 0}

    def count_refresh(*args, **kwargs):
        calls["count"] += 1

    monkeypatch.setattr(service, "_refresh_mpr_segmentation_stats_for_view", count_refresh)

    assert service._handle_mpr_segmentation_config(view, payload, refresh_stats=False) is True

    assert calls["count"] == 0


def test_mpr_segmentation_move_schedules_current_view_preview_with_samples_metadata() -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id=series.series_id)
    view = ViewRecord(view_id="v-ax", series_id=series.series_id, view_type="MPR", view_group=group)
    payload = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprSegmentation",
        actionType="move",
        mprSegmentationConfig={
            "enabled": True,
            "selectedRegionId": "r1",
            "thresholdRegions": [
                {
                    "id": "r1",
                    "enabled": True,
                    "label": "1",
                    "thresholdHu": 350,
                    "color": "#ff4df8",
                    "box": {
                        "centerWorld": [0, 1, 1],
                        "rowWorld": [0, 1, 0],
                        "colWorld": [0, 0, 1],
                        "normalWorld": [1, 0, 0],
                        "widthMm": 12,
                        "heightMm": 8,
                        "depthMm": 2,
                        "sourceViewport": MPR_VIEWPORT_AXIAL,
                    },
                }
            ],
        },
    )

    outcome = _handle_mpr_segmentation_operation(service, view, series, payload, True)

    assert outcome.mode == "single"
    assert outcome.fast_preview is True
    assert outcome.fast_preview_full_resolution is True
    assert outcome.defer_single is True
    assert outcome.metadata_mode == "mpr-segmentation-preview"


def test_legacy_mpr_segmentation_operation_input_sets_legacy_state(monkeypatch) -> None:
    service = ViewerService()
    series = SimpleNamespace(series_id="s", instances=[])
    group = ViewGroupRecord(group_id="g", group_type="MPR", series_id=series.series_id)
    axial_view = ViewRecord(view_id="v-ax", series_id=series.series_id, view_type="MPR", view_group=group)
    monkeypatch.setattr(viewer_service_module.series_registry, "get", lambda series_id: series)

    previous_views = dict(view_registry._view_by_id)
    try:
        view_registry._view_by_id.clear()
        view_registry._view_by_id[axial_view.view_id] = axial_view
        service.handle_view_operation(
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
    assert group.mpr_segmentation.legacy_enabled is True
    assert group.mpr_segmentation.lower_hu == 300
    assert group.mpr_segmentation.upper_hu == 700
    assert group.mpr_segmentation.opacity == 0.8
    assert group.mpr_segmentation.color == "#abcdef"
    assert group.mpr_segmentation.voi_box is not None
    assert group.mpr_segmentation.voi_box.x_min == 0.2


def test_mpr_segmentation_operation_is_noop_for_non_mpr_view() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="v", series_id="s", view_type="STACK")
    payload = ViewOperationRequest(
        viewId=view.view_id,
        opType="mprSegmentation",
        mprSegmentationConfig={"enabled": True, "lowerHu": 300, "upperHu": 700},
    )

    assert service._handle_mpr_segmentation_config(view, payload) is False
