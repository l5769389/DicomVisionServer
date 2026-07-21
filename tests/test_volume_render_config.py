from types import SimpleNamespace

import numpy as np

from app.models.viewer import SeriesRecord, ViewRecord
from app.services.surface_render_config import create_default_surface_render_config
from app.services.viewer_service import ViewerService
from app.services.volume_render_config import (
    build_volume_intensity_stats,
    create_adaptive_volume_render_config,
    create_default_volume_render_config,
    normalize_volume_preset_name,
    select_default_volume_preset,
)
from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer


def _layer_by_key(config: dict[str, object], key: str) -> dict[str, object]:
    return next(layer for layer in config["layers"] if isinstance(layer, dict) and layer["key"] == key)


def _series(modality: str = "CT") -> SeriesRecord:
    return SeriesRecord(
        series_id="series-adaptive",
        folder_path=".",
        series_instance_uid="1.2.3",
        study_instance_uid=None,
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality=modality,
        series_description="Adaptive preset test",
    )


class _FakeVolumeMapper:
    def __init__(self) -> None:
        self.image_sample_distances: list[float] = []
        self.sample_distances: list[float] = []
        self.jittering: list[int] = []

    def SetImageSampleDistance(self, value: float) -> None:
        self.image_sample_distances.append(float(value))

    def SetSampleDistance(self, value: float) -> None:
        self.sample_distances.append(float(value))

    def SetUseJittering(self, value: int) -> None:
        self.jittering.append(int(value))


class _FakeRenderWindow:
    def __init__(self) -> None:
        self.multi_samples: list[int] = []

    def SetMultiSamples(self, value: int) -> None:
        self.multi_samples.append(int(value))


class _FakeImageData:
    def __init__(self, spacing: tuple[float, float, float]) -> None:
        self._spacing = spacing

    def GetSpacing(self) -> tuple[float, float, float]:
        return self._spacing


def test_default_volume_config_is_bone_focused_not_red_angiography() -> None:
    config = create_default_volume_render_config("unknown")

    assert config["preset"] == "bone"
    assert config["blendMode"] == "composite"

    bone = _layer_by_key(config, "bone")
    soft_tissue = _layer_by_key(config, "softTissue")
    blood = _layer_by_key(config, "blood")
    lighting = config["lighting"]

    assert bone["enabled"] is True
    assert bone["wl"] == 360.0
    assert bone["opacity"] == 0.96
    assert soft_tissue["enabled"] is True
    assert soft_tissue["opacity"] <= 0.08
    assert blood["enabled"] is False
    assert lighting["ambient"] == 0.08
    assert lighting["specular"] == 0.32


def test_volume_preset_normalization_keeps_aaa_explicit_but_defaults_to_bone() -> None:
    assert normalize_volume_preset_name("") == "bone"
    assert normalize_volume_preset_name("volumePreset:bone") == "bone"
    assert normalize_volume_preset_name("ct bone") == "bone"
    assert normalize_volume_preset_name("volumePreset:aaa") == "aaa"
    assert normalize_volume_preset_name("volumePreset:bonePlusPlate") == "bonePlusPlate"
    assert normalize_volume_preset_name("volumePreset:coronaryCta") == "coronaryCta"
    assert normalize_volume_preset_name("MR-Angio") == "mrAngio"
    assert normalize_volume_preset_name("CBCT-Bone2") == "cbctBone2"


def test_volume_renderer_uses_higher_quality_sampling_for_final_frame() -> None:
    preview_mapper = _FakeVolumeMapper()
    final_mapper = _FakeVolumeMapper()
    preview_window = _FakeRenderWindow()
    final_window = _FakeRenderWindow()
    preview_session = SimpleNamespace(
        mapper=preview_mapper,
        render_window=preview_window,
        image_data=_FakeImageData((1.0, 1.0, 1.5)),
    )
    final_session = SimpleNamespace(
        mapper=final_mapper,
        render_window=final_window,
        image_data=_FakeImageData((1.0, 1.0, 1.5)),
    )

    VtkVolumeRenderer._update_sampling(preview_session, SimpleNamespace(fast_preview=True))
    VtkVolumeRenderer._update_sampling(final_session, SimpleNamespace(fast_preview=False))
    VtkVolumeRenderer._update_render_quality(preview_session, SimpleNamespace(fast_preview=True))
    VtkVolumeRenderer._update_render_quality(final_session, SimpleNamespace(fast_preview=False))

    assert preview_mapper.image_sample_distances[-1] == 1.15
    assert final_mapper.image_sample_distances[-1] == 1.0
    assert preview_mapper.sample_distances[-1] == 0.9
    assert final_mapper.sample_distances[-1] == 0.45
    assert preview_window.multi_samples[-1] == 0
    assert final_window.multi_samples[-1] == 8
    assert preview_mapper.jittering[-1] == 0
    assert final_mapper.jittering[-1] == 1


def test_volume_final_render_supersamples_without_changing_preview_size() -> None:
    assert VtkVolumeRenderer._resolve_render_size(
        SimpleNamespace(canvas_width=800, canvas_height=600, fast_preview=True)
    ) == (800, 600)
    assert VtkVolumeRenderer._resolve_render_size(
        SimpleNamespace(canvas_width=800, canvas_height=600, fast_preview=False)
    ) == (1000, 750)
    assert VtkVolumeRenderer._resolve_render_size(
        SimpleNamespace(canvas_width=1500, canvas_height=900, fast_preview=False)
    ) == (1600, 960)


def test_volume_renderer_quality_feature_detection_tolerates_missing_vtk_methods() -> None:
    VtkVolumeRenderer._update_render_quality(
        SimpleNamespace(mapper=object(), render_window=object()),
        SimpleNamespace(fast_preview=False),
    )


def _synthetic_py_test_path2_like_volume() -> np.ndarray:
    values = np.concatenate(
        [
            np.full(90_000, -1000.0, dtype=np.float32),
            np.full(1_000, -80.0, dtype=np.float32),
            np.full(2_500, 70.0, dtype=np.float32),
            np.full(3_500, 115.0, dtype=np.float32),
            np.full(2_500, 150.0, dtype=np.float32),
            np.full(350, 185.0, dtype=np.float32),
            np.full(120, 270.0, dtype=np.float32),
            np.full(30, 1250.0, dtype=np.float32),
        ]
    )
    return values.reshape(100, 100, 10)


def _layer_opacity_at_hu(config: dict[str, object], key: str, hu: float) -> float:
    renderer = VtkVolumeRenderer()
    _, points = renderer._build_layer_transfer_points(_layer_by_key(config, key))
    points = sorted((float(position), float(opacity)) for position, opacity in points)
    if not points:
        return 0.0
    if hu <= points[0][0]:
        return points[0][1]
    for (left_x, left_opacity), (right_x, right_opacity) in zip(points, points[1:], strict=False):
        if left_x <= hu <= right_x:
            if right_x == left_x:
                return max(left_opacity, right_opacity)
            ratio = (hu - left_x) / (right_x - left_x)
            return left_opacity + (right_opacity - left_opacity) * ratio
    return points[-1][1]


def test_aaa_adaptive_config_covers_low_ct_body_range() -> None:
    volume = _synthetic_py_test_path2_like_volume()
    stats = build_volume_intensity_stats(volume, modality="CT")
    config = create_adaptive_volume_render_config("aaa", volume, modality="CT")

    blood = _layer_by_key(config, "blood")
    muscle = _layer_by_key(config, "muscle")
    soft_tissue = _layer_by_key(config, "softTissue")
    bone = _layer_by_key(config, "bone")
    lighting = config["lighting"]

    assert stats.is_ct_hu is True
    assert 80.0 <= stats.p50 <= 150.0
    assert 90.0 <= stats.p95 <= 240.0
    assert stats.p95 - stats.p50 <= 100.0
    assert muscle["enabled"] is True
    assert float(muscle["opacity"]) == 0.34
    assert blood["enabled"] is True
    assert 145.0 <= float(blood["wl"]) <= 205.0
    assert float(blood["ww"]) >= 160.0
    assert float(blood["opacity"]) == 0.55
    assert soft_tissue["enabled"] is True
    assert float(soft_tissue["opacity"]) == 0.39
    assert float(soft_tissue["wl"]) - float(soft_tissue["ww"]) / 2.0 <= 80.0
    assert float(soft_tissue["wl"]) + float(soft_tissue["ww"]) / 2.0 >= 180.0
    assert bone["enabled"] is True
    assert 260.0 <= float(bone["wl"]) <= 620.0
    assert float(bone["ww"]) >= 480.0
    assert float(bone["opacity"]) >= 0.55
    assert int(str(bone["colorEnd"]).replace("#", ""), 16) == 0xFFFFFF
    assert _layer_opacity_at_hu(config, "muscle", 80.0) >= 0.10
    assert _layer_opacity_at_hu(config, "softTissue", 115.0) >= 0.18
    assert _layer_opacity_at_hu(config, "softTissue", 160.0) >= 0.25
    assert _layer_opacity_at_hu(config, "blood", 160.0) > _layer_opacity_at_hu(config, "blood", 115.0)
    assert lighting["ambient"] == 0.42
    assert lighting["diffuse"] == 0.72


def test_aaa_soft_detail_keeps_low_contrast_body_detail_under_stronger_red_layer() -> None:
    volume = _synthetic_py_test_path2_like_volume()
    config = create_adaptive_volume_render_config("aaa", volume, modality="CT")

    soft_115 = _layer_opacity_at_hu(config, "softTissue", 115.0)
    soft_160 = _layer_opacity_at_hu(config, "softTissue", 160.0)
    blood_115 = _layer_opacity_at_hu(config, "blood", 115.0)
    blood_160 = _layer_opacity_at_hu(config, "blood", 160.0)
    blood_190 = _layer_opacity_at_hu(config, "blood", 190.0)

    assert soft_115 >= 0.20
    assert soft_160 >= soft_115
    assert 0.12 <= blood_115 < blood_160
    assert blood_160 >= 0.85
    assert blood_190 >= 0.65
    assert _layer_opacity_at_hu(config, "bone", 160.0) == 0.0


def test_aaa_adaptive_config_keeps_bone_highlights_for_standard_ct() -> None:
    volume = np.concatenate(
        [
            np.full(40_000, -1000.0, dtype=np.float32),
            np.linspace(-120.0, 110.0, 30_000, dtype=np.float32),
            np.linspace(450.0, 1150.0, 2_000, dtype=np.float32),
        ]
    ).reshape(120, 120, 5)

    config = create_adaptive_volume_render_config("aaa", volume, modality="CT")
    blood = _layer_by_key(config, "blood")
    muscle = _layer_by_key(config, "muscle")
    soft_tissue = _layer_by_key(config, "softTissue")
    bone = _layer_by_key(config, "bone")

    assert muscle["enabled"] is False
    assert 120.0 <= float(blood["wl"]) <= 285.0
    assert float(soft_tissue["wl"]) < float(bone["wl"])
    assert float(bone["wl"]) >= 420.0
    assert config["lighting"]["ambient"] == 0.28
    assert int(str(bone["colorEnd"]).replace("#", ""), 16) == 0xFFFFFF


def test_aaa_adaptive_config_falls_back_to_percentiles_for_non_hu_data() -> None:
    volume = np.linspace(0.0, 1.0, 10_000, dtype=np.float32).reshape(100, 100, 1)

    stats = build_volume_intensity_stats(volume, modality="MR")
    config = create_adaptive_volume_render_config("aaa", volume, modality="MR")
    blood = _layer_by_key(config, "blood")
    muscle = _layer_by_key(config, "muscle")
    bone = _layer_by_key(config, "bone")

    assert stats.is_ct_hu is False
    assert muscle["enabled"] is False
    assert 0.7 <= float(blood["wl"]) <= 0.95
    assert 0.9 <= float(bone["wl"]) <= 1.0


def test_default_volume_preset_selects_aaa_for_py_test_path2_like_ct() -> None:
    volume = _synthetic_py_test_path2_like_volume()
    series = _series("CT")

    preset = select_default_volume_preset(series, volume)

    assert preset == "aaa"


def test_remove_bed_filters_outer_low_density_slab_but_preserves_high_density() -> None:
    service = ViewerService()
    volume = np.full((24, 24, 24), -1000.0, dtype=np.float32)
    volume[8:16, 8:16, 8:16] = 140.0
    volume[:, 2:4, :] = 70.0
    volume[12, 3, 12] = 1200.0

    filtered = service._remove_bed_from_render_volume(volume)

    assert float(filtered[12, 2, 12]) <= -900.0
    assert float(filtered[12, 12, 12]) == 140.0
    assert float(filtered[12, 3, 12]) == 1200.0


def test_remove_bed_filters_curved_and_slanted_supports_but_preserves_body() -> None:
    service = ViewerService()
    volume = np.full((40, 44, 48), -1000.0, dtype=np.float32)
    volume[12:28, 14:30, 16:32] = 145.0
    volume[20, 22, 24] = 1250.0
    for z_index in range(volume.shape[0]):
        for x_index in range(volume.shape[2]):
            y_index = int(round(33 + 4 * ((x_index - 24) / 24) ** 2))
            volume[z_index, max(0, y_index - 1):min(volume.shape[1], y_index + 1), x_index] = 70.0
    for z_index in range(volume.shape[0]):
        x_index = int(3 + z_index * 0.05)
        volume[z_index, :, x_index:x_index + 2] = 85.0

    filtered = service._remove_bed_from_render_volume(volume)

    assert float(filtered[20, 33, 24]) <= -900.0
    assert float(filtered[20, 36, 0]) <= -900.0
    assert float(filtered[20, 22, 24]) == 1250.0
    assert float(filtered[18, 22, 24]) == 145.0
    assert int(np.count_nonzero(filtered[12:28, 14:30, 16:32] == 145.0)) >= 4095


def test_remove_bed_protects_central_square_phantom_from_candidate_mask() -> None:
    service = ViewerService()
    volume = np.full((40, 44, 48), -1000.0, dtype=np.float32)
    volume[12:28, 14:30, 16:32] = 145.0
    volume[:, 2:4, :] = 72.0
    volume[12:28, 28:31, 16:32] = 118.0
    volume[20, 22, 24] = 1250.0

    filtered = service._remove_bed_from_render_volume(volume)
    body_core = filtered[12:28, 14:28, 16:32]
    low_contrast_bottom = filtered[12:28, 28:31, 16:32]

    assert float(filtered[20, 2, 24]) <= -900.0
    assert float(filtered[20, 22, 24]) == 1250.0
    assert int(np.count_nonzero(body_core <= -900.0)) == 0
    assert int(np.count_nonzero(body_core == 145.0)) >= body_core.size - 1
    assert int(np.count_nonzero(low_contrast_bottom == 118.0)) >= int(low_contrast_bottom.size * 0.98)


def test_volume_clip_inside_outside_uses_current_view_projection() -> None:
    service = ViewerService()
    volume = np.full((6, 8, 10), 100.0, dtype=np.float32)
    polygon = ((0.30, 0.30), (0.70, 0.30), (0.70, 0.70), (0.30, 0.70))
    identity_quaternion = (0.0, 0.0, 0.0, 1.0)

    inside = service._apply_3d_volume_clip(
        volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        mode="inside",
        points=polygon,
        rotation_quaternion=identity_quaternion,
    )
    outside = service._apply_3d_volume_clip(
        volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        mode="outside",
        points=polygon,
        rotation_quaternion=identity_quaternion,
    )

    assert float(inside[3, 4, 5]) == 100.0
    assert float(inside[0, 4, 0]) < 100.0
    assert float(outside[3, 4, 5]) < 100.0
    assert float(outside[0, 4, 0]) == 100.0


def test_volume_clip_prefilters_candidates_with_polygon_bbox(monkeypatch) -> None:
    service = ViewerService()
    volume = np.full((12, 40, 50), 100.0, dtype=np.float32)
    polygon = ((0.08, 0.08), (0.20, 0.08), (0.20, 0.20), (0.08, 0.20))
    original = ViewerService._points_inside_polygon
    candidate_counts: list[int] = []

    def wrapped_points_inside_polygon(x: np.ndarray, y: np.ndarray, polygon_array: np.ndarray) -> np.ndarray:
        candidate_counts.append(int(np.asarray(x).size))
        return original(x, y, polygon_array)

    monkeypatch.setattr(ViewerService, "_points_inside_polygon", staticmethod(wrapped_points_inside_polygon))

    clipped = service._apply_3d_volume_clip(
        volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        mode="outside",
        points=polygon,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
    )

    assert int(np.count_nonzero(clipped < 100.0)) > 0
    assert 0 < sum(candidate_counts) < int(volume.size * 0.2)


def test_default_volume_preset_selects_lung_ct_from_low_hu_fraction() -> None:
    values = np.concatenate(
        [
            np.full(50_000, -1000.0, dtype=np.float32),
            np.full(35_000, -720.0, dtype=np.float32),
            np.full(10_000, -80.0, dtype=np.float32),
            np.full(5_000, 620.0, dtype=np.float32),
        ]
    )
    volume = values.reshape(100, 100, 10)

    assert select_default_volume_preset(_series("CT"), volume) == "lung"


def test_default_volume_preset_uses_series_keywords_for_cta_mr_and_cbct() -> None:
    volume = _synthetic_py_test_path2_like_volume()
    coronary = _series("CT")
    coronary.series_description = "Coronary CTA"
    mr = _series("MR")
    mr.series_description = "Brain"
    cbct = _series("CBCT")
    cbct.series_description = "Dental scan"

    assert select_default_volume_preset(coronary, volume) == "coronaryCta"
    assert select_default_volume_preset(mr, np.linspace(0.0, 1.0, 1000, dtype=np.float32)) == "mrDefault"
    assert select_default_volume_preset(cbct, volume) == "cbctRealist"


def test_adaptive_non_aaa_presets_follow_data_distribution() -> None:
    volume = np.concatenate(
        [
            np.full(35_000, -1000.0, dtype=np.float32),
            np.linspace(-60.0, 120.0, 40_000, dtype=np.float32),
            np.linspace(180.0, 420.0, 8_000, dtype=np.float32),
            np.linspace(500.0, 1150.0, 5_000, dtype=np.float32),
        ]
    ).reshape(110, 80, 10)

    cta = create_adaptive_volume_render_config("coronaryCta", volume, modality="CT")
    bones = create_adaptive_volume_render_config("bones", volume, modality="CT")
    lung = create_adaptive_volume_render_config("lung", volume, modality="CT")

    assert cta["preset"] == "coronaryCta"
    assert _layer_by_key(cta, "blood")["enabled"] is True
    assert 135.0 <= float(_layer_by_key(cta, "blood")["wl"]) <= 430.0
    assert _layer_by_key(bones, "bone")["enabled"] is True
    assert float(_layer_by_key(bones, "bone")["wl"]) >= 320.0
    assert _layer_by_key(lung, "lung")["enabled"] is True
    assert float(_layer_by_key(lung, "lung")["wl"]) < -400.0


def test_adaptive_mr_preset_uses_percentile_fallback() -> None:
    volume = np.linspace(0.0, 1.0, 10_000, dtype=np.float32).reshape(100, 100, 1)

    config = create_adaptive_volume_render_config("mrAngio", volume, modality="MR")
    custom = _layer_by_key(config, "custom")

    assert config["preset"] == "mrAngio"
    assert config["blendMode"] == "mip"
    assert custom["enabled"] is True
    assert 0.8 <= float(custom["wl"]) <= 1.0


def test_surface_and_volume_presets_share_names_not_config_shape() -> None:
    volume_config = create_default_volume_render_config("bone")
    surface_config = create_default_surface_render_config("bone")

    assert volume_config["preset"] == surface_config["preset"] == "bone"
    assert "layers" in volume_config
    assert "isoValue" not in volume_config
    assert "isoValue" in surface_config
    assert "layers" not in surface_config


def test_viewer_service_resolves_aaa_preset_to_cached_adaptive_config() -> None:
    service = ViewerService()
    volume = _synthetic_py_test_path2_like_volume()
    view = ViewRecord(view_id="adaptive-view", series_id="series-adaptive", view_type="3D")
    view.volume_preset = "aaa"
    view.volume_render_config = create_default_volume_render_config("aaa")
    view.volume_render_config_source = "preset"

    config = service._resolve_volume_render_config_for_render(
        view,
        series=_series("CT"),
        volume=volume,
        volume_token="volume-a",
    )
    second_config = service._resolve_volume_render_config_for_render(
        view,
        series=_series("CT"),
        volume=volume,
        volume_token="volume-a",
    )

    assert config is second_config
    assert view.volume_render_config_token is not None
    assert _layer_by_key(config, "muscle")["enabled"] is True
    assert 130.0 <= float(_layer_by_key(config, "blood")["wl"]) <= 210.0


def test_viewer_service_resolves_non_aaa_preset_to_cached_adaptive_config() -> None:
    service = ViewerService()
    volume = _synthetic_py_test_path2_like_volume()
    view = ViewRecord(view_id="adaptive-cta-view", series_id="series-adaptive", view_type="3D")
    view.volume_preset = "coronaryCta"
    view.volume_render_config = create_default_volume_render_config("coronaryCta")
    view.volume_render_config_source = "preset"

    config = service._resolve_volume_render_config_for_render(
        view,
        series=_series("CT"),
        volume=volume,
        volume_token="volume-cta",
    )
    second_config = service._resolve_volume_render_config_for_render(
        view,
        series=_series("CT"),
        volume=volume,
        volume_token="volume-cta",
    )

    assert config is second_config
    assert view.volume_render_config_token is not None
    assert config["preset"] == "coronaryCta"
    assert _layer_by_key(config, "blood")["enabled"] is True


def test_viewer_service_keeps_manual_aaa_config_stable() -> None:
    service = ViewerService()
    volume = _synthetic_py_test_path2_like_volume()
    manual_config = create_default_volume_render_config("aaa")
    _layer_by_key(manual_config, "blood")["wl"] = 222.0
    view = ViewRecord(view_id="manual-view", series_id="series-adaptive", view_type="3D")
    view.volume_preset = "aaa"
    view.volume_render_config = manual_config
    view.volume_render_config_source = "manual"

    config = service._resolve_volume_render_config_for_render(
        view,
        series=_series("CT"),
        volume=volume,
        volume_token="volume-a",
    )

    assert float(_layer_by_key(config, "blood")["wl"]) == 222.0
    assert view.volume_render_config_source == "manual"
    assert view.volume_render_config_token is None


def test_aaa_gradient_opacity_keeps_low_gradient_body_visible() -> None:
    points = VtkVolumeRenderer._build_gradient_opacity_points("aaa", "composite", True)

    assert points[0] == (0.0, 0.24)
    assert points[1] == (16.0, 0.36)
    assert points[2][1] >= 0.62
    assert points[3][1] >= 0.86
    assert points[-1][1] >= 0.95
