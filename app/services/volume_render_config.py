from __future__ import annotations

from typing import Any

from app.schemas.view import VolumeRenderConfig


def create_default_volume_render_config(preset_value: str) -> dict[str, object]:
    """Build a backend-owned baseline config for a named 3D preset.

    The preset definitions live here so ViewerService can stay focused on orchestration
    instead of carrying a large block of static configuration data.
    """

    preset = normalize_volume_preset_name(preset_value)

    layers = {
        "bone": {
            "key": "bone",
            "label": "\u9aa8\u9abc",
            "enabled": False,
            "ww": 500.0,
            "wl": 400.0,
            "opacity": 1.0,
            "colorStart": "#ffffff",
            "colorEnd": "#ffffff",
        },
        "blood": {
            "key": "blood",
            "label": "\u8840\u6db2",
            "enabled": False,
            "ww": 200.0,
            "wl": 220.0,
            "opacity": 0.2,
            "colorStart": "#d31b1b",
            "colorEnd": "#ffd54a",
        },
        "muscle": {
            "key": "muscle",
            "label": "\u808c\u8089",
            "enabled": False,
            "ww": 320.0,
            "wl": 45.0,
            "opacity": 0.55,
            "colorStart": "#f2c7b8",
            "colorEnd": "#8a3426",
        },
        "softTissue": {
            "key": "softTissue",
            "label": "\u8f6f\u7ec4\u7ec7",
            "enabled": False,
            "ww": 380.0,
            "wl": 55.0,
            "opacity": 0.32,
            "colorStart": "#f1d8c8",
            "colorEnd": "#b06b56",
        },
        "lung": {
            "key": "lung",
            "label": "\u80ba",
            "enabled": False,
            "ww": 1500.0,
            "wl": -550.0,
            "opacity": 0.22,
            "colorStart": "#9fd8ff",
            "colorEnd": "#e5f6ff",
        },
        "custom": {
            "key": "custom",
            "label": "\u81ea\u5b9a\u4e49",
            "enabled": False,
            "ww": 400.0,
            "wl": 40.0,
            "opacity": 0.3,
            "colorStart": "#7dd3fc",
            "colorEnd": "#f8fafc",
        },
    }
    blend_mode = "composite"
    lighting = {
        "shading": True,
        "interpolation": "linear",
        "ambient": 0.16,
        "diffuse": 0.86,
        "specular": 0.18,
        "roughness": 0.78,
    }

    if preset == "aaa":
        layers["bone"].update({"enabled": True, "ww": 500.0, "wl": 400.0, "opacity": 1.0, "colorStart": "#ffffff", "colorEnd": "#ffffff"})
        layers["blood"].update({"enabled": True, "ww": 200.0, "wl": 220.0, "opacity": 0.2, "colorStart": "#d31b1b", "colorEnd": "#ffd54a"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.12, "diffuse": 0.9, "specular": 0.2, "roughness": 0.74})
    elif preset == "red":
        layers["bone"].update({"enabled": True, "ww": 442.0, "wl": 115.0, "opacity": 1.0, "colorStart": "#c31616", "colorEnd": "#ff6666"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.14, "diffuse": 0.88, "specular": 0.16, "roughness": 0.8})
    elif preset == "cardiac":
        layers["bone"].update({"enabled": True, "ww": 170.0, "wl": 176.0, "opacity": 0.9, "colorStart": "#fff9f2", "colorEnd": "#7f1720"})
        layers["blood"].update({"enabled": True, "ww": 170.0, "wl": 7.0, "opacity": 0.3, "colorStart": "#ffe082", "colorEnd": "#ffb300"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.1, "diffuse": 0.88, "specular": 0.22, "roughness": 0.72})
    elif preset == "muscle":
        layers["muscle"].update({"enabled": True, "ww": 280.0, "wl": 40.0, "opacity": 0.58, "colorStart": "#f4cfbf", "colorEnd": "#8c3d2e"})
        layers["softTissue"].update({"enabled": True, "ww": 360.0, "wl": 50.0, "opacity": 0.28, "colorStart": "#f3ddd1", "colorEnd": "#9e6a5a"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.18, "diffuse": 0.82, "specular": 0.08, "roughness": 0.9})
    elif preset == "mip":
        blend_mode = "mip"
        layers["bone"].update({"enabled": True, "ww": 900.0, "wl": 350.0, "opacity": 0.35, "colorStart": "#9a9a9a", "colorEnd": "#ffffff"})
        layers["blood"].update({"enabled": True, "ww": 260.0, "wl": 200.0, "opacity": 0.85, "colorStart": "#f7f1b6", "colorEnd": "#ffffff"})
        lighting.update({"shading": False, "interpolation": "linear", "ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0})

    return {
        "preset": preset,
        "blendMode": blend_mode,
        "layers": list(layers.values()),
        "lighting": lighting,
    }


def normalize_volume_render_config(
    value: VolumeRenderConfig | dict[str, object] | None,
    fallback_preset: str,
) -> dict[str, object]:
    fallback = create_default_volume_render_config(fallback_preset)
    if value is None:
        return fallback

    if isinstance(value, VolumeRenderConfig):
        payload: dict[str, Any] = value.model_dump(by_alias=True)
    else:
        payload = dict(value)

    preset = str(payload.get("preset") or fallback["preset"]).strip().lower()
    normalized = create_default_volume_render_config(preset)
    normalized["blendMode"] = "mip" if payload.get("blendMode") == "mip" else "composite"

    incoming_layers = payload.get("layers") if isinstance(payload.get("layers"), list) else []
    layer_map = {str(layer["key"]): layer for layer in normalized["layers"] if isinstance(layer, dict)}

    for entry in incoming_layers:
        if not isinstance(entry, dict):
            continue
        layer = layer_map.get(str(entry.get("key") or ""))
        if layer is None:
            continue
        layer["label"] = str(entry.get("label") or layer["label"])
        layer["enabled"] = bool(entry.get("enabled", layer["enabled"]))
        layer["ww"] = max(1.0, float(entry.get("ww", layer["ww"])))
        layer["wl"] = float(entry.get("wl", layer["wl"]))
        layer["opacity"] = _normalize_unit_interval(entry.get("opacity"), float(layer["opacity"]))
        layer["colorStart"] = _normalize_hex_color(str(entry.get("colorStart") or layer["colorStart"]), str(layer["colorStart"]))
        layer["colorEnd"] = _normalize_hex_color(str(entry.get("colorEnd") or layer["colorEnd"]), str(layer["colorEnd"]))

    lighting_payload = payload.get("lighting") if isinstance(payload.get("lighting"), dict) else {}
    lighting = normalized.get("lighting") if isinstance(normalized.get("lighting"), dict) else {}
    lighting["shading"] = bool(lighting_payload.get("shading", lighting.get("shading", True)))
    lighting["interpolation"] = _normalize_volume_interpolation(
        str(lighting_payload.get("interpolation") or lighting.get("interpolation") or "linear")
    )
    lighting["ambient"] = _normalize_unit_interval(lighting_payload.get("ambient"), float(lighting.get("ambient", 0.18)))
    lighting["diffuse"] = _normalize_unit_interval(lighting_payload.get("diffuse"), float(lighting.get("diffuse", 0.82)))
    lighting["specular"] = _normalize_unit_interval(lighting_payload.get("specular"), float(lighting.get("specular", 0.12)))
    lighting["roughness"] = _normalize_unit_interval(lighting_payload.get("roughness"), float(lighting.get("roughness", 0.85)))
    normalized["lighting"] = lighting

    return normalized


def normalize_volume_preset_name(value: str) -> str:
    preset_value = str(value or "aaa").strip().lower()
    if ":" in preset_value:
        preset_value = preset_value.split(":", 1)[1]

    preset_aliases = {
        "aaa": "aaa",
        "red": "red",
        "cardiac": "cardiac",
        "cardiac-muscle": "cardiac",
        "cardiac_muscle": "cardiac",
        "cardiac muscle": "cardiac",
        "muscle": "muscle",
        "mip": "mip",
    }
    return preset_aliases.get(preset_value, "aaa")


def _normalize_hex_color(value: str, fallback: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 7 and text.startswith("#") and all(ch in "0123456789abcdef" for ch in text[1:]):
        return text
    return fallback


def _normalize_unit_interval(value: object, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, numeric))


def _normalize_volume_interpolation(value: str) -> str:
    normalized = str(value or "linear").strip().lower()
    if normalized in {"nearest", "linear", "cubic"}:
        return normalized
    return "linear"
