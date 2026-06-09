from __future__ import annotations

from functools import lru_cache

import numpy as np


DEFAULT_PSEUDOCOLOR_PRESET = "bw"

_PRESET_STOPS: dict[str, tuple[tuple[float, str], ...]] = {
    "blackbody": (
        (0.00, "#000000"),
        (0.22, "#5b0f00"),
        (0.46, "#c73a00"),
        (0.70, "#ffb000"),
        (1.00, "#fff6d5"),
    ),
    "bw": (
        (0.00, "#000000"),
        (0.36, "#3f3f3f"),
        (0.70, "#9f9f9f"),
        (1.00, "#ffffff"),
    ),
    "bwinverse": (
        (0.00, "#ffffff"),
        (0.34, "#b8b8b8"),
        (0.68, "#626262"),
        (1.00, "#000000"),
    ),
    "cardiac": (
        (0.00, "#09111c"),
        (0.24, "#124d79"),
        (0.46, "#1da6a6"),
        (0.72, "#f4d35e"),
        (1.00, "#f1635e"),
    ),
    "hotiron": (
        (0.00, "#000000"),
        (0.24, "#520000"),
        (0.48, "#b10d0d"),
        (0.72, "#ff7a00"),
        (1.00, "#fff2bf"),
    ),
    "pet": (
        (0.00, "#14003d"),
        (0.18, "#1b4cff"),
        (0.38, "#00b7ff"),
        (0.56, "#1de5a3"),
        (0.78, "#ffe14a"),
        (1.00, "#ff4d5a"),
    ),
    "petct-rainbow": (
        (0.00, "#000000"),
        (0.12, "#3a0000"),
        (0.30, "#8d0000"),
        (0.52, "#e21b00"),
        (0.72, "#ff8a00"),
        (0.88, "#ffe100"),
        (1.00, "#fffef0"),
    ),
    "rainbow": (
        (0.00, "#5b00d6"),
        (0.18, "#1558ff"),
        (0.34, "#00b0ff"),
        (0.52, "#00cf7c"),
        (0.74, "#ffd000"),
        (1.00, "#ff5d39"),
    ),
}


def normalize_pseudocolor_preset(value: str | None) -> str:
    normalized = str(value or "").strip().lower().removeprefix("pseudocolor:")
    return normalized if normalized in _PRESET_STOPS else DEFAULT_PSEUDOCOLOR_PRESET


def apply_pseudocolor(grayscale_pixels: np.ndarray, preset: str | None) -> np.ndarray:
    normalized_preset = normalize_pseudocolor_preset(preset)
    lut = build_lut(normalized_preset)
    return lut[np.asarray(grayscale_pixels, dtype=np.uint8)]


@lru_cache(maxsize=None)
def build_lut(preset: str) -> np.ndarray:
    normalized_preset = normalize_pseudocolor_preset(preset)
    stops = _PRESET_STOPS[normalized_preset]
    indices = np.arange(256, dtype=np.float32)
    anchors = np.asarray([position * 255.0 for position, _ in stops], dtype=np.float32)
    colors = np.asarray([_hex_to_rgb(color) for _, color in stops], dtype=np.float32)
    channels = [
        np.interp(indices, anchors, colors[:, channel_index])
        for channel_index in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    stripped = value.strip().removeprefix("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))
