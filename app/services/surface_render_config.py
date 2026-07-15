from __future__ import annotations

from typing import Any

import numpy as np

from app.schemas.view import SurfaceRenderConfig
from app.services.volume_render_config import VolumeIntensityStats, build_volume_intensity_stats


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
    if preset == "softTissue":
        return {
            "preset": "softTissue",
            "isoValue": 85.0,
            "smoothing": 0.18,
            "decimation": 0.08,
            "color": "#b86642",
            "ambient": 0.28,
            "diffuse": 0.72,
            "specular": 0.08,
            "roughness": 0.86,
        }
    if preset == "highDensity":
        return {
            "preset": "highDensity",
            "isoValue": 420.0,
            "smoothing": 0.22,
            "decimation": 0.12,
            "color": "#f8fafc",
            "ambient": 0.16,
            "diffuse": 0.82,
            "specular": 0.46,
            "roughness": 0.26,
        }
    return create_default_surface_render_config("bone")


def create_adaptive_surface_render_config(
    preset_value: str = "bone",
    volume: np.ndarray | None = None,
    *,
    modality: str | None = None,
    stats: VolumeIntensityStats | None = None,
) -> dict[str, object]:
    """Create a surface config whose iso threshold follows the current dataset."""

    preset = normalize_surface_preset_name(preset_value)
    config = create_default_surface_render_config(preset)
    if stats is None and volume is not None:
        stats = build_volume_intensity_stats(volume, modality=modality)
    if stats is None or stats.foreground_count <= 0:
        return config

    modality_value = str(modality or "").strip().upper()
    hu_range_is_plausible = stats.source_min <= -500.0 or stats.source_max >= 300.0
    use_hu_anchors = stats.is_ct_hu and hu_range_is_plausible and modality_value not in {"MR", "CBCT"}
    if use_hu_anchors:
        if preset == "softTissue":
            low_anchor = (stats.p10 + stats.p25) / 2.0
            if stats.p10 <= -120.0:
                iso_value = _clamp(low_anchor, -350.0, -80.0)
            else:
                iso_value = _clamp((stats.p10 + stats.p50) / 2.0, -80.0, 160.0)
        elif preset == "highDensity":
            dense_anchor = max(stats.p95, stats.p99, stats.p995 * 0.78)
            minimum_dense_hu = 700.0 if stats.very_dense_foreground_fraction >= 0.001 else 450.0
            iso_value = _clamp(max(minimum_dense_hu, dense_anchor), 350.0, 1800.0)
        else:
            iso_value = _clamp(max(220.0, min(520.0, max(stats.p90, stats.p75 + 90.0))), 160.0, 650.0)
    else:
        span = max(stats.p99 - stats.p10, 1.0)
        if preset == "softTissue":
            iso_value = _clamp(stats.p50, stats.source_min, stats.source_max)
        elif preset == "highDensity":
            iso_value = _clamp(stats.p90 + span * 0.08, stats.source_min, stats.source_max)
        else:
            iso_value = _clamp(stats.p75 + span * 0.05, stats.source_min, stats.source_max)

    config["isoValue"] = round(float(iso_value), 3)
    return config


def normalize_surface_render_config(
    value: SurfaceRenderConfig | dict[str, object] | None,
    fallback_preset: str = "bone",
) -> dict[str, object]:
    fallback = create_default_surface_render_config(fallback_preset)
    if value is None:
        return fallback

    if isinstance(value, SurfaceRenderConfig):
        payload: dict[str, Any] = value.model_dump(by_alias=True, exclude_unset=True)
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
    preset_aliases = {
        "bone": "bone",
        "bones": "bone",
        "skull": "bone",
        "surface": "bone",
        "softtissue": "softTissue",
        "soft-tissue": "softTissue",
        "soft_tissue": "softTissue",
        "soft tissue": "softTissue",
        "tissue": "softTissue",
        "highdensity": "highDensity",
        "high-density": "highDensity",
        "high_density": "highDensity",
        "high density": "highDensity",
        "dense": "highDensity",
        "metal": "highDensity",
    }
    return preset_aliases.get(preset, "bone")


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
