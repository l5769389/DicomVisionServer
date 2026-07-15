from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.schemas.view import VolumeRenderConfig

logger = get_logger(__name__)

MAX_VOLUME_STATS_SAMPLES = 2_000_000
AAA_FOREGROUND_AIR_THRESHOLD_HU = -700.0


@dataclass(frozen=True)
class VolumeIntensityStats:
    """Robust intensity summary used to adapt 3D transfer functions."""

    source_min: float
    source_max: float
    foreground_count: int
    sample_count: int
    foreground_fraction: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    p995: float
    low_hu_fraction: float
    dense_foreground_fraction: float
    very_dense_foreground_fraction: float
    is_ct_hu: bool

    def as_log_dict(self) -> dict[str, object]:
        return {
            "sourceRange": [round(self.source_min, 3), round(self.source_max, 3)],
            "foregroundCount": self.foreground_count,
            "sampleCount": self.sample_count,
            "foregroundFraction": round(self.foreground_fraction, 6),
            "lowHuFraction": round(self.low_hu_fraction, 6),
            "denseForegroundFraction": round(self.dense_foreground_fraction, 6),
            "veryDenseForegroundFraction": round(self.very_dense_foreground_fraction, 6),
            "isCtHu": self.is_ct_hu,
            "percentiles": {
                "p10": round(self.p10, 3),
                "p25": round(self.p25, 3),
                "p50": round(self.p50, 3),
                "p75": round(self.p75, 3),
                "p90": round(self.p90, 3),
                "p95": round(self.p95, 3),
                "p99": round(self.p99, 3),
                "p995": round(self.p995, 3),
            },
        }


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

    if preset == "bone":
        layers["bone"].update({"enabled": True, "ww": 820.0, "wl": 360.0, "opacity": 0.96, "colorStart": "#e8dcc8", "colorEnd": "#ffffff"})
        layers["softTissue"].update({"enabled": True, "ww": 430.0, "wl": 55.0, "opacity": 0.075, "colorStart": "#d6a18a", "colorEnd": "#f0d8c9"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.08, "diffuse": 0.9, "specular": 0.32, "roughness": 0.52})
    elif preset == "aaa":
        layers["bone"].update({"enabled": True, "ww": 560.0, "wl": 330.0, "opacity": 0.58, "colorStart": "#f2f2f2", "colorEnd": "#ffffff"})
        layers["blood"].update({"enabled": True, "ww": 150.0, "wl": 160.0, "opacity": 0.52, "colorStart": "#9d0b0b", "colorEnd": "#ff3b1f"})
        layers["muscle"].update({"enabled": True, "ww": 260.0, "wl": 125.0, "opacity": 0.38, "colorStart": "#d7784a", "colorEnd": "#ffc090"})
        layers["softTissue"].update({"enabled": True, "ww": 300.0, "wl": 130.0, "opacity": 0.44, "colorStart": "#8f2d18", "colorEnd": "#e89b64"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.38, "diffuse": 0.62, "specular": 0.08, "roughness": 0.92})
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
    elif preset == "xray":
        blend_mode = "mip"
        layers["bone"].update({"enabled": True, "ww": 1200.0, "wl": 460.0, "opacity": 0.78, "colorStart": "#5b6470", "colorEnd": "#ffffff"})
        layers["softTissue"].update({"enabled": True, "ww": 650.0, "wl": 80.0, "opacity": 0.1, "colorStart": "#2d333a", "colorEnd": "#b8c0cc"})
        lighting.update({"shading": False, "interpolation": "linear", "ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0})
    elif preset in {"carotid", "coronaryCta", "bodyCta", "neckCta"}:
        layers["blood"].update({"enabled": True, "ww": 260.0, "wl": 240.0, "opacity": 0.72, "colorStart": "#f6c45b", "colorEnd": "#fff6d2"})
        layers["bone"].update({"enabled": True, "ww": 900.0, "wl": 500.0, "opacity": 0.42, "colorStart": "#e8e8e8", "colorEnd": "#ffffff"})
        layers["softTissue"].update({"enabled": True, "ww": 420.0, "wl": 80.0, "opacity": 0.12, "colorStart": "#4a1c14", "colorEnd": "#a86745"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.18, "diffuse": 0.78, "specular": 0.28, "roughness": 0.64})
    elif preset in {"bonePlusPlate", "fracture", "lumbar", "hardware", "bones", "cbctBone", "cbctBone2"}:
        bone_opacity = 0.92
        if preset == "hardware":
            bone_opacity = 0.78
        elif preset == "cbctBone2":
            bone_opacity = 0.98
        layers["bone"].update({"enabled": True, "ww": 980.0, "wl": 470.0, "opacity": bone_opacity, "colorStart": "#d6c4a6", "colorEnd": "#ffffff"})
        layers["softTissue"].update({"enabled": True, "ww": 460.0, "wl": 70.0, "opacity": 0.06, "colorStart": "#b78872", "colorEnd": "#efd6c5"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.08, "diffuse": 0.9, "specular": 0.34 if preset != "hardware" else 0.46, "roughness": 0.5 if preset != "hardware" else 0.38})
    elif preset in {"lung", "lung2", "lung3"}:
        layers["lung"].update({"enabled": True, "ww": 1550.0, "wl": -610.0, "opacity": 0.32, "colorStart": "#28435a", "colorEnd": "#d9f1ff"})
        layers["bone"].update({"enabled": preset != "lung3", "ww": 1000.0, "wl": 430.0, "opacity": 0.28, "colorStart": "#d9d9d9", "colorEnd": "#ffffff"})
        layers["softTissue"].update({"enabled": preset == "lung2", "ww": 420.0, "wl": 45.0, "opacity": 0.08, "colorStart": "#5d3024", "colorEnd": "#d5a58b"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.26, "diffuse": 0.68, "specular": 0.08, "roughness": 0.9})
    elif preset in {"renalsStomach", "mrDefault"}:
        layers["muscle"].update({"enabled": True, "ww": 320.0, "wl": 55.0, "opacity": 0.46, "colorStart": "#d7b09e", "colorEnd": "#8d4b3b"})
        layers["softTissue"].update({"enabled": True, "ww": 420.0, "wl": 70.0, "opacity": 0.26, "colorStart": "#f0d0bd", "colorEnd": "#aa6a55"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.18, "diffuse": 0.78, "specular": 0.1, "roughness": 0.88})
    elif preset == "vesselOutline":
        layers["blood"].update({"enabled": True, "ww": 220.0, "wl": 210.0, "opacity": 0.86, "colorStart": "#ffe9a8", "colorEnd": "#ffffff"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.22, "diffuse": 0.62, "specular": 0.36, "roughness": 0.58})
    elif preset in {"mrMip", "mrAngio"}:
        blend_mode = "mip"
        layers["custom"].update({"enabled": True, "ww": 1.0, "wl": 0.82, "opacity": 0.72, "colorStart": "#b9c8ff", "colorEnd": "#ffffff"})
        layers["blood"].update({"enabled": preset == "mrAngio", "ww": 1.0, "wl": 0.72, "opacity": 0.76, "colorStart": "#dce9ff", "colorEnd": "#ffffff"})
        lighting.update({"shading": False, "interpolation": "linear", "ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "roughness": 1.0})
    elif preset == "cbctRealist":
        layers["bone"].update({"enabled": True, "ww": 850.0, "wl": 360.0, "opacity": 0.86, "colorStart": "#d9c2a4", "colorEnd": "#fff8ec"})
        layers["softTissue"].update({"enabled": True, "ww": 380.0, "wl": 70.0, "opacity": 0.12, "colorStart": "#b18472", "colorEnd": "#e5c7b3"})
        lighting.update({"shading": True, "interpolation": "linear", "ambient": 0.16, "diffuse": 0.82, "specular": 0.24, "roughness": 0.64})

    return {
        "preset": preset,
        "blendMode": blend_mode,
        "layers": list(layers.values()),
        "lighting": lighting,
    }


def create_adaptive_volume_render_config(
    preset_value: str,
    volume: np.ndarray,
    *,
    modality: str | None = None,
    stats: VolumeIntensityStats | None = None,
) -> dict[str, object]:
    """Build a data-aware preset while preserving the named preset's visual intent."""

    preset = normalize_volume_preset_name(preset_value)
    stats = stats or build_volume_intensity_stats(volume, modality=modality)
    config = create_default_volume_render_config(preset)
    if stats.is_ct_hu:
        _apply_ct_adaptive_layers(config, preset, stats)
    else:
        _apply_percentile_adaptive_layers(config, preset, stats)
    logger.info(
        "adaptive volume config preset=%s stats=%s layers=%s",
        preset,
        stats.as_log_dict(),
        _summarize_volume_layers(config),
    )
    return config


def select_default_volume_preset(
    series: object,
    volume: np.ndarray,
    *,
    stats: VolumeIntensityStats | None = None,
) -> str:
    stats = stats or build_volume_intensity_stats(volume, modality=getattr(series, "modality", None))
    modality = str(getattr(series, "modality", "") or "").strip().upper()
    text = " ".join(
        str(getattr(series, attr, "") or "")
        for attr in (
            "modality",
            "series_description",
            "study_description",
            "standard_object_type",
            "preferred_view_type",
        )
    ).lower()
    compact_text = text.replace("_", " ").replace("-", " ")

    if modality in {"MR", "MRI"}:
        if any(keyword in compact_text for keyword in ("angio", "mra", "vessel")):
            return "mrAngio"
        if "mip" in compact_text:
            return "mrMip"
        return "mrDefault"

    if modality == "CBCT" or "cbct" in compact_text or "cone beam" in compact_text:
        if "bone2" in compact_text or "bone 2" in compact_text:
            return "cbctBone2"
        if "bone" in compact_text:
            return "cbctBone"
        return "cbctRealist"

    is_ct_like = modality in {"CT", "CTA"} or stats.is_ct_hu
    if is_ct_like:
        if "coronary" in compact_text:
            return "coronaryCta"
        if "carotid" in compact_text:
            return "carotid"
        if "neck" in compact_text and any(keyword in compact_text for keyword in ("cta", "angio", "vessel")):
            return "neckCta"
        if any(keyword in compact_text for keyword in ("cta", "angiogram", "angio", "vessel")):
            return "bodyCta"
        if any(keyword in compact_text for keyword in ("hardware", "metal", "implant")):
            return "hardware"
        if any(keyword in compact_text for keyword in ("plate", "screw")):
            return "bonePlusPlate"
        if "fracture" in compact_text:
            return "fracture"
        if any(keyword in compact_text for keyword in ("lumbar", "spine", "vertebra")):
            return "lumbar"
        if any(keyword in compact_text for keyword in ("lung", "chest")) or stats.low_hu_fraction >= 0.24:
            return "lung"
        if stats.dense_foreground_fraction >= 0.18 and stats.very_dense_foreground_fraction >= 0.04:
            return "hardware"
        if stats.dense_foreground_fraction >= 0.22:
            return "bones"
        return "aaa"

    return "bone"


def build_volume_intensity_stats(volume: np.ndarray, *, modality: str | None = None) -> VolumeIntensityStats:
    values = _sample_finite_volume_values(volume)
    if values.size == 0:
        return VolumeIntensityStats(
            source_min=0.0,
            source_max=1.0,
            foreground_count=0,
            sample_count=0,
            foreground_fraction=0.0,
            p10=0.0,
            p25=0.0,
            p50=0.0,
            p75=0.25,
            p90=0.5,
            p95=0.75,
            p99=1.0,
            p995=1.0,
            low_hu_fraction=0.0,
            dense_foreground_fraction=0.0,
            very_dense_foreground_fraction=0.0,
            is_ct_hu=False,
        )

    source_min = float(np.min(values))
    source_max = float(np.max(values))
    modality_text = str(modality or "").strip().upper()
    modality_is_ct = modality_text in {"CT", "CTA", "CBCT"}
    hu_like_range = source_min <= -850.0 and source_max >= 80.0
    prefer_ct_hu = modality_is_ct or (not modality_text and hu_like_range)

    foreground = _ct_foreground_values(values) if prefer_ct_hu else np.asarray([], dtype=np.float32)
    is_ct_hu = prefer_ct_hu and foreground.size >= max(64, int(values.size * 0.0005))
    if not is_ct_hu:
        foreground = _percentile_foreground_values(values)

    if foreground.size == 0:
        foreground = values

    percentiles = _percentile_map(foreground, (10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5))
    low_hu_fraction = float(np.count_nonzero((values >= -950.0) & (values <= -300.0)) / max(1, values.size))
    dense_foreground_fraction = float(np.count_nonzero(foreground >= 350.0) / max(1, foreground.size))
    very_dense_foreground_fraction = float(np.count_nonzero(foreground >= 1000.0) / max(1, foreground.size))
    return VolumeIntensityStats(
        source_min=source_min,
        source_max=source_max,
        foreground_count=int(foreground.size),
        sample_count=int(values.size),
        foreground_fraction=float(foreground.size / max(1, values.size)),
        p10=percentiles[10.0],
        p25=percentiles[25.0],
        p50=percentiles[50.0],
        p75=percentiles[75.0],
        p90=percentiles[90.0],
        p95=percentiles[95.0],
        p99=percentiles[99.0],
        p995=percentiles[99.5],
        low_hu_fraction=low_hu_fraction,
        dense_foreground_fraction=dense_foreground_fraction,
        very_dense_foreground_fraction=very_dense_foreground_fraction,
        is_ct_hu=is_ct_hu,
    )


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
        layer["ww"] = float(entry.get("ww", layer["ww"]))
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
    preset_value = str(value or "bone").strip().lower()
    if ":" in preset_value:
        preset_value = preset_value.split(":", 1)[1]
    preset_value = preset_value.replace("_", "-").replace(" ", "-")

    preset_aliases = {
        "volume": "bone",
        "bone": "bone",
        "ct-bone": "bone",
        "aaa": "aaa",
        "red": "red",
        "cardiac": "cardiac",
        "cardiac-muscle": "cardiac",
        "muscle": "muscle",
        "mip": "mip",
        "xray": "xray",
        "x-ray": "xray",
        "carotid": "carotid",
        "boneplusplate": "bonePlusPlate",
        "bone-plus-plate": "bonePlusPlate",
        "boneplate": "bonePlusPlate",
        "fracture": "fracture",
        "lumbar": "lumbar",
        "spine": "lumbar",
        "hardware": "hardware",
        "metal": "hardware",
        "implant": "hardware",
        "lung": "lung",
        "lung2": "lung2",
        "lung-2": "lung2",
        "lung3": "lung3",
        "lung-3": "lung3",
        "renals-stomach": "renalsStomach",
        "renal-stomach": "renalsStomach",
        "renalsstomach": "renalsStomach",
        "vessel-outline": "vesselOutline",
        "vesseloutline": "vesselOutline",
        "bones": "bones",
        "coronarycta": "coronaryCta",
        "coronary-cta": "coronaryCta",
        "bodycta": "bodyCta",
        "body-cta": "bodyCta",
        "neckcta": "neckCta",
        "neck-cta": "neckCta",
        "mrdefault": "mrDefault",
        "mr-default": "mrDefault",
        "mrmip": "mrMip",
        "mr-mip": "mrMip",
        "mrangio": "mrAngio",
        "mr-angio": "mrAngio",
        "mra": "mrAngio",
        "cbctrealist": "cbctRealist",
        "cbct-realist": "cbctRealist",
        "cbctbone": "cbctBone",
        "cbct-bone": "cbctBone",
        "cbctbone2": "cbctBone2",
        "cbct-bone2": "cbctBone2",
        "cbct-bone-2": "cbctBone2",
    }
    return preset_aliases.get(preset_value, "bone")


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


def _sample_finite_volume_values(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.size == 0:
        return np.asarray([], dtype=np.float32)
    flat = array.ravel()
    if flat.size > MAX_VOLUME_STATS_SAMPLES:
        step = max(1, int(ceil(flat.size / MAX_VOLUME_STATS_SAMPLES)))
        flat = flat[::step]
    finite = flat[np.isfinite(flat)]
    return np.asarray(finite, dtype=np.float32)


def _ct_foreground_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(values[values > AAA_FOREGROUND_AIR_THRESHOLD_HU], dtype=np.float32)


def _percentile_foreground_values(values: np.ndarray) -> np.ndarray:
    if values.size < 16:
        return values
    low, high = np.percentile(values, [2.0, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return values
    foreground = values[(values >= low) & (values <= high)]
    if foreground.size < 16:
        return values
    return np.asarray(foreground, dtype=np.float32)


def _percentile_map(values: np.ndarray, percentiles: tuple[float, ...]) -> dict[float, float]:
    if values.size == 0:
        return {percentile: 0.0 for percentile in percentiles}
    results = np.percentile(values, percentiles)
    return {percentile: float(result) for percentile, result in zip(percentiles, results, strict=True)}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _layers_by_key(config: dict[str, object]) -> dict[str, dict[str, object]]:
    layers = config.get("layers")
    if not isinstance(layers, list):
        return {}
    return {
        str(layer.get("key")): layer
        for layer in layers
        if isinstance(layer, dict) and isinstance(layer.get("key"), str)
    }


def _update_layer(config: dict[str, object], key: str, **updates: object) -> None:
    layer = _layers_by_key(config).get(key)
    if layer is None:
        return
    layer.update(updates)


def _update_lighting(config: dict[str, object], **updates: object) -> None:
    lighting = config.get("lighting")
    if not isinstance(lighting, dict):
        lighting = {}
        config["lighting"] = lighting
    lighting.update(updates)


def _apply_ct_aaa_layers(config: dict[str, object], stats: VolumeIntensityStats) -> None:
    if _is_soft_detail_ct_distribution(stats):
        _apply_ct_aaa_soft_detail_layers(config, stats)
        return

    _update_layer(config, "muscle", enabled=False)
    soft_low = _clamp(stats.p50, 25.0, 125.0)
    soft_high = _clamp(max(stats.p90, soft_low + 70.0), 95.0, 220.0)
    soft_width = _clamp(max((soft_high - soft_low) * 2.35, 220.0), 220.0, 420.0)
    soft_center = _clamp((soft_low + soft_high) / 2.0, 60.0, 150.0)

    blood_center_seed = max(stats.p75, stats.p90, 120.0)
    if stats.p95 >= 260.0:
        blood_center_seed = max(blood_center_seed, (stats.p90 + stats.p95) / 2.0)
    blood_center = _clamp(blood_center_seed, 120.0, 285.0)
    blood_width = _clamp(max((stats.p95 - stats.p75) * 3.0, 185.0), 185.0, 340.0)

    bone_center = _clamp(max(stats.p995, stats.p99, 420.0), 380.0, 760.0)
    bone_width = _clamp(max((stats.p995 - stats.p95) * 4.0, 720.0), 680.0, 1220.0)
    dense_span = max(0.0, stats.p995 - max(stats.p95, 1.0))
    bone_opacity = _clamp(0.58 + min(0.16, dense_span / 2400.0), 0.58, 0.74)
    blood_opacity = _clamp(0.56 + min(0.09, max(0.0, stats.p95 - stats.p75) / 360.0), 0.54, 0.67)
    soft_opacity = _clamp(0.22 + min(0.08, max(0.0, stats.p90 - stats.p50) / 900.0), 0.22, 0.3)

    _update_layer(
        config,
        "softTissue",
        enabled=True,
        ww=_rounded(soft_width),
        wl=_rounded(soft_center),
        opacity=_rounded(soft_opacity),
        colorStart="#3b130d",
        colorEnd="#a55a32",
    )
    _update_layer(
        config,
        "blood",
        enabled=True,
        ww=_rounded(blood_width),
        wl=_rounded(blood_center),
        opacity=_rounded(blood_opacity),
        colorStart="#9d0b0b",
        colorEnd="#ff3b1f",
    )
    _update_layer(
        config,
        "bone",
        enabled=True,
        ww=_rounded(bone_width),
        wl=_rounded(bone_center),
        opacity=_rounded(bone_opacity),
        colorStart="#f2f2f2",
        colorEnd="#ffffff",
    )
    _update_lighting(config, ambient=0.28, diffuse=0.75, specular=0.12, roughness=0.85)


def _is_soft_detail_ct_distribution(stats: VolumeIntensityStats) -> bool:
    return (
        stats.is_ct_hu
        and 40.0 <= stats.p50 <= 150.0
        and 90.0 <= stats.p95 <= 240.0
        and stats.p95 - stats.p50 <= 100.0
        and stats.dense_foreground_fraction < 0.02
        and stats.very_dense_foreground_fraction < 0.005
    )


def _apply_ct_aaa_soft_detail_layers(config: dict[str, object], stats: VolumeIntensityStats) -> None:
    muscle_center = (stats.p10 + stats.p75) / 2.0
    muscle_width = max((stats.p90 - stats.p10) * 2.2, 280.0)
    soft_center = (stats.p10 + stats.p95) / 2.0
    soft_width = max((stats.p95 - stats.p10) * 2.2, 360.0)
    blood_center = _clamp((stats.p75 + stats.p95) / 2.0, 145.0, 205.0)
    blood_width = max((stats.p95 - stats.p75) * 5.5, 160.0)
    bone_center = _clamp(max(stats.p99, stats.p995, 300.0), 260.0, 620.0)
    bone_width = _clamp(max((stats.p995 - stats.p95) * 3.0, 520.0), 480.0, 920.0)
    bone_opacity = 0.58 if stats.dense_foreground_fraction < 0.01 else 0.64

    _update_layer(
        config,
        "muscle",
        enabled=True,
        ww=_rounded(muscle_width),
        wl=_rounded(muscle_center),
        opacity=0.34,
        colorStart="#b65a38",
        colorEnd="#ffd0a6",
    )
    _update_layer(
        config,
        "softTissue",
        enabled=True,
        ww=_rounded(soft_width),
        wl=_rounded(soft_center),
        opacity=0.39,
        colorStart="#b3492a",
        colorEnd="#ffc18a",
    )
    _update_layer(
        config,
        "blood",
        enabled=True,
        ww=_rounded(blood_width),
        wl=_rounded(blood_center),
        opacity=0.55,
        colorStart="#a60008",
        colorEnd="#ff2418",
    )
    _update_layer(
        config,
        "bone",
        enabled=True,
        ww=_rounded(bone_width),
        wl=_rounded(bone_center),
        opacity=bone_opacity,
        colorStart="#f2f2f2",
        colorEnd="#ffffff",
    )
    _update_lighting(config, ambient=0.42, diffuse=0.72, specular=0.07, roughness=0.9)


def _apply_percentile_aaa_layers(config: dict[str, object], stats: VolumeIntensityStats) -> None:
    foreground_span = max(stats.p99 - stats.p50, 1.0)
    soft_width = _clamp(max((stats.p90 - stats.p50) * 2.4, foreground_span * 0.38, 1.0), 1.0, max(1.0, foreground_span * 1.4))
    blood_width = _clamp(max((stats.p95 - stats.p75) * 2.6, foreground_span * 0.28, 1.0), 1.0, max(1.0, foreground_span * 1.2))
    bone_width = _clamp(max((stats.p995 - stats.p90) * 2.8, foreground_span * 0.18, 1.0), 1.0, max(1.0, foreground_span * 1.4))

    _update_layer(
        config,
        "muscle",
        enabled=False,
    )
    _update_layer(
        config,
        "softTissue",
        enabled=True,
        ww=_rounded(soft_width),
        wl=_rounded((stats.p50 + stats.p90) / 2.0),
        opacity=0.26,
        colorStart="#3b130d",
        colorEnd="#a55a32",
    )
    _update_layer(
        config,
        "blood",
        enabled=True,
        ww=_rounded(blood_width),
        wl=_rounded((stats.p75 + stats.p95) / 2.0),
        opacity=0.58,
        colorStart="#9d0b0b",
        colorEnd="#ff3b1f",
    )
    _update_layer(
        config,
        "bone",
        enabled=True,
        ww=_rounded(bone_width),
        wl=_rounded(max(stats.p99, stats.p95)),
        opacity=0.62,
        colorStart="#f2f2f2",
        colorEnd="#ffffff",
    )
    _update_lighting(config, ambient=0.28, diffuse=0.75, specular=0.12, roughness=0.85)


def _apply_ct_adaptive_layers(config: dict[str, object], preset: str, stats: VolumeIntensityStats) -> None:
    if preset == "aaa":
        _apply_ct_aaa_layers(config, stats)
        return
    if preset in {"carotid", "coronaryCta", "bodyCta", "neckCta", "cardiac", "vesselOutline"}:
        _apply_ct_vascular_layers(config, preset, stats)
        return
    if preset in {"lung", "lung2", "lung3"}:
        _apply_ct_lung_layers(config, preset, stats)
        return
    if preset in {"bone", "bones", "bonePlusPlate", "fracture", "lumbar", "hardware", "xray", "cbctRealist", "cbctBone", "cbctBone2"}:
        _apply_ct_bone_layers(config, preset, stats)
        return
    if preset in {"mip"}:
        _apply_ct_mip_layers(config, stats)
        return
    if preset in {"red", "muscle", "renalsStomach"}:
        _apply_ct_soft_tissue_layers(config, preset, stats)
        return
    _apply_percentile_adaptive_layers(config, preset, stats)


def _apply_ct_vascular_layers(config: dict[str, object], preset: str, stats: VolumeIntensityStats) -> None:
    vessel_center_seed = max(stats.p75, stats.p90, 145.0)
    if stats.p95 >= 240.0:
        vessel_center_seed = max(vessel_center_seed, (stats.p90 + stats.p95) / 2.0)
    upper_bound = 430.0 if preset in {"coronaryCta", "bodyCta", "neckCta"} else 340.0
    vessel_center = _clamp(vessel_center_seed, 135.0, upper_bound)
    vessel_width = _clamp(max((stats.p95 - stats.p75) * 2.8, 180.0), 160.0, 420.0)
    soft_center = _clamp((stats.p50 + stats.p90) / 2.0, 45.0, 145.0)
    soft_width = _clamp(max((stats.p90 - stats.p50) * 2.4, 240.0), 220.0, 520.0)
    bone_center = _clamp(max(stats.p995, 430.0), 380.0, 850.0)
    bone_width = _clamp(max((stats.p995 - stats.p95) * 4.0, 760.0), 680.0, 1400.0)
    vessel_color = ("#ffe9a8", "#ffffff") if preset == "vesselOutline" else ("#f6c45b", "#fff6d2")

    _update_layer(config, "blood", enabled=True, ww=_rounded(vessel_width), wl=_rounded(vessel_center), opacity=0.82 if preset == "vesselOutline" else 0.68, colorStart=vessel_color[0], colorEnd=vessel_color[1])
    _update_layer(config, "softTissue", enabled=preset != "vesselOutline", ww=_rounded(soft_width), wl=_rounded(soft_center), opacity=0.1 if preset != "cardiac" else 0.16, colorStart="#4a1c14", colorEnd="#a86745")
    _update_layer(config, "bone", enabled=preset != "vesselOutline", ww=_rounded(bone_width), wl=_rounded(bone_center), opacity=0.34 if preset != "cardiac" else 0.46, colorStart="#e8e8e8", colorEnd="#ffffff")
    _update_lighting(config, ambient=0.18, diffuse=0.78, specular=0.3, roughness=0.62)


def _apply_ct_bone_layers(config: dict[str, object], preset: str, stats: VolumeIntensityStats) -> None:
    high_anchor = max(stats.p95, stats.p99, 360.0)
    if preset in {"hardware", "bonePlusPlate"}:
        high_anchor = max(high_anchor, stats.p995, 620.0)
    bone_center = _clamp(high_anchor, 320.0, 1050.0)
    bone_width = _clamp(max((stats.p995 - stats.p75) * 1.6, 650.0), 560.0, 1800.0)
    soft_center = _clamp((stats.p50 + stats.p90) / 2.0, 35.0, 135.0)
    soft_width = _clamp(max((stats.p90 - stats.p50) * 2.2, 260.0), 220.0, 560.0)
    bone_opacity = 0.9
    if preset in {"bones", "cbctBone2"}:
        bone_opacity = 0.98
    elif preset in {"hardware", "xray"}:
        bone_opacity = 0.76
    elif preset == "fracture":
        bone_opacity = 0.94

    _update_layer(config, "bone", enabled=True, ww=_rounded(bone_width), wl=_rounded(bone_center), opacity=bone_opacity, colorStart="#d6c4a6", colorEnd="#ffffff")
    _update_layer(config, "softTissue", enabled=preset not in {"xray", "hardware"}, ww=_rounded(soft_width), wl=_rounded(soft_center), opacity=0.05 if preset != "cbctRealist" else 0.12, colorStart="#b78872", colorEnd="#efd6c5")
    if preset == "xray":
        config["blendMode"] = "mip"
        _update_lighting(config, shading=False, ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0)
    else:
        _update_lighting(config, ambient=0.08 if preset != "cbctRealist" else 0.16, diffuse=0.88, specular=0.44 if preset == "hardware" else 0.32, roughness=0.42 if preset == "hardware" else 0.56)


def _apply_ct_lung_layers(config: dict[str, object], preset: str, stats: VolumeIntensityStats) -> None:
    lung_center = _clamp(stats.source_min * 0.35 + (-520.0 * 0.65), -760.0, -420.0)
    lung_width = _clamp(max(1200.0, abs(stats.source_min - min(stats.p50, -120.0)) * 1.8), 1100.0, 1900.0)
    _update_layer(config, "lung", enabled=True, ww=_rounded(lung_width), wl=_rounded(lung_center), opacity=0.26 if preset == "lung" else 0.34, colorStart="#28435a", colorEnd="#d9f1ff")
    _update_layer(config, "bone", enabled=preset != "lung3", ww=950.0, wl=_rounded(_clamp(max(stats.p99, 420.0), 360.0, 760.0)), opacity=0.22, colorStart="#d9d9d9", colorEnd="#ffffff")
    _update_layer(config, "softTissue", enabled=preset == "lung2", ww=420.0, wl=_rounded(_clamp((stats.p50 + stats.p90) / 2.0, 35.0, 120.0)), opacity=0.08, colorStart="#5d3024", colorEnd="#d5a58b")
    _update_lighting(config, ambient=0.26, diffuse=0.68, specular=0.08, roughness=0.9)


def _apply_ct_soft_tissue_layers(config: dict[str, object], preset: str, stats: VolumeIntensityStats) -> None:
    soft_low = _clamp(stats.p50, -40.0, 90.0)
    soft_high = _clamp(max(stats.p90, soft_low + 60.0), 70.0, 190.0)
    soft_width = _clamp(max((soft_high - soft_low) * 2.3, 240.0), 220.0, 520.0)
    soft_center = _clamp((soft_low + soft_high) / 2.0, 35.0, 130.0)
    target_key = "muscle" if preset in {"muscle", "renalsStomach"} else "softTissue"
    _update_layer(config, target_key, enabled=True, ww=_rounded(soft_width), wl=_rounded(soft_center), opacity=0.54 if target_key == "muscle" else 0.42)
    _update_layer(config, "softTissue", enabled=True, ww=_rounded(soft_width * 1.18), wl=_rounded(soft_center), opacity=0.24 if preset != "red" else 0.2)
    if preset == "red":
        _update_layer(config, "bone", enabled=True, ww=_rounded(max(320.0, (stats.p95 - stats.p50) * 2.0)), wl=_rounded(_clamp(max(stats.p90, 115.0), 95.0, 260.0)), opacity=0.86, colorStart="#b30f0f", colorEnd="#ff5b5b")
    _update_lighting(config, ambient=0.16, diffuse=0.82, specular=0.1, roughness=0.88)


def _apply_ct_mip_layers(config: dict[str, object], stats: VolumeIntensityStats) -> None:
    config["blendMode"] = "mip"
    high_center = _clamp(max(stats.p95, stats.p99, 220.0), 160.0, 820.0)
    high_width = _clamp(max((stats.p995 - stats.p75) * 1.7, 480.0), 360.0, 1500.0)
    _update_layer(config, "bone", enabled=True, ww=_rounded(high_width), wl=_rounded(high_center), opacity=0.42, colorStart="#9a9a9a", colorEnd="#ffffff")
    _update_layer(config, "blood", enabled=True, ww=_rounded(max((stats.p95 - stats.p75) * 2.4, 220.0)), wl=_rounded(_clamp(max(stats.p90, 160.0), 120.0, 420.0)), opacity=0.82, colorStart="#f7f1b6", colorEnd="#ffffff")
    _update_lighting(config, shading=False, ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0)


def _apply_percentile_adaptive_layers(config: dict[str, object], preset: str, stats: VolumeIntensityStats) -> None:
    if preset == "aaa":
        _apply_percentile_aaa_layers(config, stats)
        return
    foreground_span = max(stats.p995 - stats.p50, 1.0)
    soft_center = _rounded((stats.p50 + stats.p90) / 2.0)
    soft_width = _rounded(_clamp(max((stats.p90 - stats.p50) * 2.4, foreground_span * 0.35, 1.0), 1.0, max(1.0, foreground_span * 1.5)))
    high_center = _rounded(max(stats.p95, stats.p99))
    high_width = _rounded(_clamp(max((stats.p995 - stats.p75) * 1.6, foreground_span * 0.25, 1.0), 1.0, max(1.0, foreground_span * 1.8)))

    for key, layer in _layers_by_key(config).items():
        if not bool(layer.get("enabled")):
            continue
        if key in {"bone", "blood", "custom"} and preset in {"mip", "mrMip", "mrAngio", "xray"}:
            _update_layer(config, key, ww=high_width, wl=high_center, opacity=_rounded(max(0.42, float(layer.get("opacity", 0.4)))))
        elif key in {"bone", "blood"}:
            _update_layer(config, key, ww=high_width, wl=high_center)
        else:
            _update_layer(config, key, ww=soft_width, wl=soft_center)

    if preset in {"mip", "mrMip", "mrAngio", "xray"}:
        config["blendMode"] = "mip"
        _update_lighting(config, shading=False, ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0)


def _summarize_volume_layers(config: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for key, layer in _layers_by_key(config).items():
        if not bool(layer.get("enabled")):
            continue
        summary[key] = {
            "ww": layer.get("ww"),
            "wl": layer.get("wl"),
            "opacity": layer.get("opacity"),
        }
    return summary
