from __future__ import annotations

import io
from threading import Lock
from time import perf_counter

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

AUTO_WEBP_METHODS = (0, 1, 2)
AUTO_WEBP_SAMPLE_MAX_SIDE = 256

_selection_lock = Lock()
_selected_auto_method: int | None = None


def _encode_lossless_webp(image: Image.Image, *, method: int) -> bytes:
    output = io.BytesIO()
    image.save(output, format="WEBP", lossless=True, method=method)
    return output.getvalue()


def _make_calibration_sample(image: Image.Image) -> Image.Image:
    sample = image.convert("RGB") if image.mode != "RGB" else image.copy()
    sample.thumbnail(
        (AUTO_WEBP_SAMPLE_MAX_SIDE, AUTO_WEBP_SAMPLE_MAX_SIDE),
        Image.Resampling.BILINEAR,
    )
    return sample


def _select_fastest_method(image: Image.Image) -> int:
    sample = _make_calibration_sample(image)
    measurements: list[tuple[float, int, int]] = []
    for method in AUTO_WEBP_METHODS:
        started_at = perf_counter()
        payload = _encode_lossless_webp(sample, method=method)
        measurements.append(((perf_counter() - started_at) * 1000.0, len(payload), method))

    fastest_ms = min(item[0] for item in measurements)
    # Methods within 10% of the fastest are effectively tied. Prefer the smaller
    # payload among them so cloud deployments do not trade a negligible CPU win
    # for a much larger final still.
    candidates = [item for item in measurements if item[0] <= fastest_ms * 1.10 + 0.05]
    selected = min(candidates, key=lambda item: (item[1], item[0], item[2]))[2]
    logger.info(
        "3d final webp calibration selected_method=%s sample=%sx%s candidates=%s",
        selected,
        sample.width,
        sample.height,
        ",".join(f"m{method}:{elapsed_ms:.1f}ms/{size}b" for elapsed_ms, size, method in measurements),
    )
    return selected


def resolve_3d_final_webp_method(image: Image.Image) -> int:
    configured = get_settings().normalized_three_d_final_webp_method
    if isinstance(configured, int):
        return configured

    global _selected_auto_method
    if _selected_auto_method is not None:
        return _selected_auto_method
    with _selection_lock:
        if _selected_auto_method is None:
            _selected_auto_method = _select_fastest_method(image)
        return _selected_auto_method


def encode_lossless_3d_webp(image: Image.Image) -> bytes:
    return _encode_lossless_webp(image, method=resolve_3d_final_webp_method(image))


def get_calibrated_3d_final_webp_method() -> int | None:
    return _selected_auto_method


def reset_3d_final_webp_method_selection() -> None:
    """Reset process-local calibration state for tests and repeatable benchmarks."""
    global _selected_auto_method
    with _selection_lock:
        _selected_auto_method = None
