from __future__ import annotations

from typing import Any

from app.schemas.view import SurfaceRenderConfig


def create_default_surface_render_config(preset_value: str = "bone") -> dict[str, object]:
    preset = normalize_surface_preset_name(preset_value)
    if preset == "bone":
        return {
            "preset": "bone",
            "isoValue": 240.0,
            "smoothing": 0.28,
            "decimation": 0.18,
            "color": "#f3eadb",
            "ambient": 0.2,
            "diffuse": 0.76,
            "specular": 0.36,
            "roughness": 0.34,
        }
    return create_default_surface_render_config("bone")


def normalize_surface_render_config(
    value: SurfaceRenderConfig | dict[str, object] | None,
    fallback_preset: str = "bone",
) -> dict[str, object]:
    fallback = create_default_surface_render_config(fallback_preset)
    if value is None:
        return fallback

    if isinstance(value, SurfaceRenderConfig):
        payload: dict[str, Any] = value.model_dump(by_alias=True)
    else:
        payload = dict(value)

    preset = normalize_surface_preset_name(str(payload.get("preset") or fallback["preset"]))
    normalized = create_default_surface_render_config(preset)
    normalized["isoValue"] = _normalize_numeric(payload.get("isoValue"), float(normalized["isoValue"]), -2000.0, 4000.0)
    normalized["smoothing"] = _normalize_numeric(payload.get("smoothing"), float(normalized["smoothing"]), 0.0, 1.0)
    normalized["decimation"] = _normalize_numeric(payload.get("decimation"), float(normalized["decimation"]), 0.0, 0.9)
    normalized["color"] = _normalize_hex_color(str(payload.get("color") or normalized["color"]), str(normalized["color"]))
    normalized["ambient"] = _normalize_numeric(payload.get("ambient"), float(normalized["ambient"]), 0.0, 1.0)
    normalized["diffuse"] = _normalize_numeric(payload.get("diffuse"), float(normalized["diffuse"]), 0.0, 1.0)
    normalized["specular"] = _normalize_numeric(payload.get("specular"), float(normalized["specular"]), 0.0, 1.0)
    normalized["roughness"] = _normalize_numeric(payload.get("roughness"), float(normalized["roughness"]), 0.0, 1.0)
    return normalized


def normalize_surface_preset_name(value: str) -> str:
    preset = str(value or "bone").strip().lower()
    if ":" in preset:
        preset = preset.split(":", 1)[1]
    return "bone" if preset in {"bone", "bones", "skull", "surface"} else "bone"


def _normalize_numeric(value: object, fallback: float, lower: float, upper: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(lower, min(upper, numeric))


def _normalize_hex_color(value: str, fallback: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 7 and text.startswith("#") and all(ch in "0123456789abcdef" for ch in text[1:]):
        return text
    return fallback
