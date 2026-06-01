from __future__ import annotations

import numpy as np

REPRESENTATIVE_SLICE_SAMPLE_LIMIT = 48


def build_representative_sample_indexes(count: int, *, sample_limit: int = REPRESENTATIVE_SLICE_SAMPLE_LIMIT) -> list[int]:
    if count <= sample_limit:
        return list(range(count))
    indexes = np.linspace(0, count - 1, sample_limit)
    return sorted({int(round(float(index))) for index in indexes})


def to_content_luminance(pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels)
    if array.ndim >= 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3].astype(np.float32, copy=False)
        return (
            rgb[..., 0] * np.float32(0.299)
            + rgb[..., 1] * np.float32(0.587)
            + rgb[..., 2] * np.float32(0.114)
        )
    if array.ndim > 2:
        array = np.squeeze(array)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
    return array.astype(np.float32, copy=False)


def estimate_background_value(image: np.ndarray) -> float:
    if image.ndim < 2 or image.size == 0:
        finite = image[np.isfinite(image)]
        return float(np.median(finite)) if finite.size else 0.0

    border = np.concatenate(
        [
            image[0, :].ravel(),
            image[-1, :].ravel(),
            image[:, 0].ravel(),
            image[:, -1].ravel(),
        ]
    )
    finite_border = border[np.isfinite(border)]
    if finite_border.size:
        return float(np.median(finite_border))
    finite = image[np.isfinite(image)]
    return float(np.median(finite)) if finite.size else 0.0


def build_content_threshold(values: np.ndarray, robust_span: float) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0 or not np.isfinite(robust_span) or robust_span <= 1e-6:
        return float("inf")
    low = float(np.min(finite_values))
    high = float(np.max(finite_values))
    intensity_floor = 1.0 if 0.0 <= low and high <= 255.0 else 8.0
    return max(float(robust_span) * 0.06, intensity_floor)


def score_representative_pixels(pixels: np.ndarray) -> float:
    image = to_content_luminance(pixels)
    if image.size == 0:
        return 0.0

    finite = image[np.isfinite(image)]
    if finite.size < 16:
        return 0.0

    low, high = np.percentile(finite, [1.0, 99.0])
    robust_span = float(high - low)
    threshold = build_content_threshold(finite, robust_span)
    if not np.isfinite(threshold):
        return 0.0

    background = estimate_background_value(image)
    foreground = np.abs(image - background) > threshold
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count <= 0:
        return 0.0

    foreground_ratio = foreground_count / float(image.size)
    foreground_values = image[foreground]
    foreground_low, foreground_high = np.percentile(foreground_values, [5.0, 95.0])
    foreground_span = float(max(0.0, foreground_high - foreground_low))
    return float(min(foreground_ratio, 0.72) * max(robust_span, foreground_span))
