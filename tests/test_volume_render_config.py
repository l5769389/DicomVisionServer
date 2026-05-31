from app.services.volume_render_config import create_default_volume_render_config, normalize_volume_preset_name


def _layer_by_key(config: dict[str, object], key: str) -> dict[str, object]:
    return next(layer for layer in config["layers"] if isinstance(layer, dict) and layer["key"] == key)


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
