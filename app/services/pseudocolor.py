from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import numpy as np


DEFAULT_PSEUDOCOLOR_PRESET = "bw"
PSEUDOCOLOR_REGISTRY_VERSION = "dicomvision-2026.2"


@dataclass(frozen=True)
class PseudocolorDefinition:
    key: str
    label: str
    version: str
    provenance: str
    license: str
    lut: np.ndarray

    @property
    def sha256(self) -> str:
        return sha256(np.ascontiguousarray(self.lut, dtype=np.uint8).tobytes()).hexdigest()


def _channel(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _linear_gray(*, inverse: bool = False) -> np.ndarray:
    values = np.arange(256, dtype=np.float64)
    if inverse:
        values = 255.0 - values
    channel = _channel(values)
    return np.stack((channel, channel, channel), axis=-1)


def _normalized_axis() -> np.ndarray:
    return np.arange(256, dtype=np.float64) / 255.0


def _hot_iron() -> np.ndarray:
    """Continuous black/red/orange/white ramp generated at all 256 entries."""
    x = _normalized_axis()
    return _channel(
        np.stack(
            (
                np.clip(2.0 * x, 0.0, 1.0),
                np.clip(2.0 * x - 1.0, 0.0, 1.0),
                np.clip(4.0 * x - 3.0, 0.0, 1.0),
            ),
            axis=-1,
        )
        * 255.0
    )


def _hot_metal() -> np.ndarray:
    """Metal-style hot ramp with longer red and yellow transition regions."""
    x = _normalized_axis()
    return _channel(
        np.stack(
            (
                np.clip(1.4 * x, 0.0, 1.0),
                np.clip(2.8 * x - 1.4, 0.0, 1.0),
                np.clip(4.0 * x - 3.0, 0.0, 1.0),
            ),
            axis=-1,
        )
        * 255.0
    )


def _black_body() -> np.ndarray:
    x = _normalized_axis()
    return _channel(
        np.stack(
            (
                np.clip(3.0 * x, 0.0, 1.0),
                np.clip(3.0 * x - 1.0, 0.0, 1.0),
                np.clip(3.0 * x - 2.0, 0.0, 1.0),
            ),
            axis=-1,
        )
        * 255.0
    )


def _hsv_ramp(
    *,
    hue_start: float,
    hue_end: float,
    saturation_start: float = 1.0,
    saturation_end: float = 0.0,
    value_start: float = 0.20,
    value_end: float = 1.0,
) -> np.ndarray:
    count = 256
    hue = np.linspace(hue_start, hue_end, count, dtype=np.float64)
    saturation = np.linspace(saturation_start, saturation_end, count, dtype=np.float64)
    value = np.linspace(value_start, value_end, count, dtype=np.float64)
    chroma = value * saturation
    hue_sector = (hue % 1.0) * 6.0
    x = chroma * (1.0 - np.abs(hue_sector % 2.0 - 1.0))
    zeros = np.zeros_like(chroma)
    rgb_prime = np.empty((count, 3), dtype=np.float64)
    sectors = np.floor(hue_sector).astype(np.int32) % 6
    candidates = (
        (chroma, x, zeros),
        (x, chroma, zeros),
        (zeros, chroma, x),
        (zeros, x, chroma),
        (x, zeros, chroma),
        (chroma, zeros, x),
    )
    for sector, candidate in enumerate(candidates):
        mask = sectors == sector
        rgb_prime[mask] = np.stack(candidate, axis=-1)[mask]
    rgb = rgb_prime + (value - chroma)[:, None]
    return _channel(rgb * 255.0)


def _pet_ramp() -> np.ndarray:
    # A continuous PET spectrum with a true black zero. Keeping LUT(0) black is
    # important because padding outside the acquired field must remain background.
    x = _normalized_axis()
    red = np.where(
        x < 0.25,
        0.0,
        np.where(x < 0.75, 2.0 * x - 0.5, 1.0),
    )
    green = np.where(
        x < 0.25,
        2.0 * x,
        np.where(x < 0.5, 1.0 - 2.0 * x, 2.0 * x - 1.0),
    )
    blue = np.where(
        x < 0.5,
        2.0 * x,
        np.where(x < 0.75, 3.0 - 4.0 * x, 4.0 * x - 3.0),
    )
    return _channel(np.stack((red, green, blue), axis=-1) * 255.0)


def _rainbow() -> np.ndarray:
    return _hsv_ramp(
        hue_start=0.75,
        hue_end=0.0,
        saturation_start=1.0,
        saturation_end=0.82,
        value_start=0.40,
        value_end=1.0,
    )


def _definition(key: str, label: str, lut: np.ndarray) -> PseudocolorDefinition:
    frozen = np.ascontiguousarray(lut, dtype=np.uint8)
    frozen.setflags(write=False)
    return PseudocolorDefinition(
        key=key,
        label=label,
        version=PSEUDOCOLOR_REGISTRY_VERSION,
        provenance="DicomVision analytic 256-entry palette",
        license="DicomVision project license",
        lut=frozen,
    )


_DEFINITIONS: dict[str, PseudocolorDefinition] = {
    "bw": _definition("bw", "BW", _linear_gray()),
    "bwinverse": _definition("bwinverse", "BWInverse", _linear_gray(inverse=True)),
    "blackbody": _definition("blackbody", "BlackBody", _black_body()),
    "hotiron": _definition("hotiron", "HotIron", _hot_iron()),
    "hotmetal": _definition("hotmetal", "HotMetal", _hot_metal()),
    "pet": _definition("pet", "PET", _pet_ramp()),
    "rainbow": _definition("rainbow", "Rainbow", _rainbow()),
}

# Historical fusion palette names remain readable but resolve to a versioned
# palette instead of maintaining a second, visually different approximation.
_ALIASES = {
    "petct-rainbow": "hotmetal",
}


def normalize_pseudocolor_preset(value: str | None) -> str:
    normalized = str(value or "").strip().lower().removeprefix("pseudocolor:")
    normalized = _ALIASES.get(normalized, normalized)
    return normalized if normalized in _DEFINITIONS else DEFAULT_PSEUDOCOLOR_PRESET


def pseudocolor_definition(preset: str | None) -> PseudocolorDefinition:
    return _DEFINITIONS[normalize_pseudocolor_preset(preset)]


def apply_pseudocolor(grayscale_pixels: np.ndarray, preset: str | None) -> np.ndarray:
    return build_lut(normalize_pseudocolor_preset(preset))[
        np.asarray(grayscale_pixels, dtype=np.uint8)
    ]


def pseudocolor_background_color(preset: str | None) -> tuple[int, int, int]:
    """Return the active LUT colour for pixels outside the acquired FOV."""
    return tuple(int(component) for component in build_lut(normalize_pseudocolor_preset(preset))[0])


@lru_cache(maxsize=None)
def build_lut(preset: str) -> np.ndarray:
    return pseudocolor_definition(preset).lut
