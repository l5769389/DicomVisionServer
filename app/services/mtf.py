from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MtfAnalysisResult:
    frequencies: np.ndarray
    values: np.ndarray
    mtf50: float
    mtf10: float
    peak_value: float
    center_x: float
    center_y: float
    fwhm_w: float
    fwhm_h: float


class MtfAnalyzer:
    @staticmethod
    def analyze_roi(
        roi: np.ndarray,
        *,
        spacing_xy: tuple[float, float] | None = None,
    ) -> MtfAnalysisResult:
        roi_array = np.asarray(roi, dtype=np.float64)
        if roi_array.ndim != 2:
            raise ValueError("MTF analysis requires a 2D ROI")
        if roi_array.shape[0] < 5 or roi_array.shape[1] < 5:
            raise ValueError("MTF ROI is too small")

        prepared_roi, polarity = MtfAnalyzer._prepare_roi(roi_array)
        center_y, center_x = MtfAnalyzer._find_bead_center(prepared_roi)
        centered_psf = MtfAnalyzer._extract_centered_psf(prepared_roi, center_x, center_y)
        frequencies, values = MtfAnalyzer._psf_to_mtf(centered_psf, spacing_xy=spacing_xy)

        center_row = MtfAnalyzer._sample_profile(prepared_roi, center_x=center_x, center_y=center_y, axis="horizontal")
        center_col = MtfAnalyzer._sample_profile(prepared_roi, center_x=center_x, center_y=center_y, axis="vertical")
        fwhm_w = MtfAnalyzer._calculate_fwhm(center_row)
        fwhm_h = MtfAnalyzer._calculate_fwhm(center_col)

        if spacing_xy is not None:
            fwhm_w *= max(abs(float(spacing_xy[0])), 1e-6)
            fwhm_h *= max(abs(float(spacing_xy[1])), 1e-6)

        if polarity < 0:
            peak_value = float(np.min(roi_array))
        else:
            peak_value = float(np.max(roi_array))

        return MtfAnalysisResult(
            frequencies=frequencies,
            values=values,
            mtf50=MtfAnalyzer._find_frequency_at(frequencies, values, 0.5),
            mtf10=MtfAnalyzer._find_frequency_at(frequencies, values, 0.1),
            peak_value=peak_value,
            center_x=center_x,
            center_y=center_y,
            fwhm_w=fwhm_w,
            fwhm_h=fwhm_h,
        )

    @staticmethod
    def _prepare_roi(roi: np.ndarray) -> tuple[np.ndarray, int]:
        median = float(np.median(roi))
        bright_contrast = float(np.max(roi) - median)
        dark_contrast = float(median - np.min(roi))
        polarity = -1 if dark_contrast > bright_contrast else 1
        working = -roi if polarity < 0 else roi.copy()
        background = float(np.percentile(working, 15))
        prepared = np.clip(working - background, 0.0, None)
        return prepared, polarity

    @staticmethod
    def _find_bead_center(roi: np.ndarray) -> tuple[float, float]:
        smoothed = ndimage.gaussian_filter(roi, sigma=1.0)
        peak_y, peak_x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)

        win = 5
        y0 = max(0, peak_y - win)
        y1 = min(roi.shape[0], peak_y + win + 1)
        x0 = max(0, peak_x - win)
        x1 = min(roi.shape[1], peak_x + win + 1)
        patch = roi[y0:y1, x0:x1]
        total = float(np.sum(patch))

        if total <= 0.0:
            return float(peak_y), float(peak_x)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        center_y = float(np.sum(yy * patch) / total)
        center_x = float(np.sum(xx * patch) / total)
        return center_y, center_x

    @staticmethod
    def _extract_centered_psf(roi: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
        size = min(roi.shape)
        if size % 2 == 0:
            size -= 1
        if size < 5:
            raise ValueError("MTF ROI is too small")

        shifted = ndimage.shift(
            roi,
            shift=((size // 2) - center_y, (size // 2) - center_x),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        start_y = max((shifted.shape[0] - size) // 2, 0)
        start_x = max((shifted.shape[1] - size) // 2, 0)
        psf = shifted[start_y : start_y + size, start_x : start_x + size].copy()

        border_pixels = np.concatenate((psf[0, :], psf[-1, :], psf[:, 0], psf[:, -1]))
        background = float(np.median(border_pixels))
        psf = np.clip(psf - background, 0.0, None)
        total = float(np.sum(psf))
        if total <= 0.0:
            raise ValueError("MTF ROI does not contain a detectable bead signal")

        return psf / total

    @staticmethod
    def _sample_profile(
        roi: np.ndarray,
        *,
        center_x: float,
        center_y: float,
        axis: str,
    ) -> np.ndarray:
        if axis == "horizontal":
            coords = np.vstack((np.full(roi.shape[1], center_y, dtype=np.float64), np.arange(roi.shape[1], dtype=np.float64)))
        else:
            coords = np.vstack((np.arange(roi.shape[0], dtype=np.float64), np.full(roi.shape[0], center_x, dtype=np.float64)))
        return ndimage.map_coordinates(roi, coords, order=1, mode="nearest")

    @staticmethod
    def _calculate_fwhm(profile: np.ndarray) -> float:
        if profile.size < 3:
            return 0.0

        working = np.asarray(profile, dtype=np.float64)
        baseline = float(np.min(working))
        signal = np.clip(working - baseline, 0.0, None)
        peak_value = float(np.max(signal))
        if peak_value <= 0.0:
            return 0.0

        half_max = peak_value / 2.0
        peak_index = int(np.argmax(signal))
        left = MtfAnalyzer._interp_crossing(signal, half_max, peak_index, go_left=True)
        right = MtfAnalyzer._interp_crossing(signal, half_max, peak_index, go_left=False)
        if left is None or right is None or right <= left:
            return float(np.count_nonzero(signal >= half_max))
        return float(right - left)

    @staticmethod
    def _interp_crossing(profile: np.ndarray, threshold: float, start: int, *, go_left: bool) -> float | None:
        step = -1 if go_left else 1
        previous_index = start
        current_index = start + step

        while 0 <= current_index < profile.size:
            if profile[current_index] < threshold <= profile[previous_index]:
                delta = float(profile[previous_index] - profile[current_index])
                if delta <= 1e-12:
                    return float(current_index)
                fraction = float((profile[previous_index] - threshold) / delta)
                return float(previous_index + fraction * (current_index - previous_index))
            previous_index = current_index
            current_index += step
        return None

    @staticmethod
    def _psf_to_mtf(
        psf: np.ndarray,
        *,
        spacing_xy: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = psf.shape
        dx = float(spacing_xy[0]) if spacing_xy is not None else 1.0
        dy = float(spacing_xy[1]) if spacing_xy is not None else 1.0
        dx = max(abs(dx), 1e-6)
        dy = max(abs(dy), 1e-6)

        wy = np.hanning(height)
        wx = np.hanning(width)
        windowed = psf * np.outer(wy, wx)
        total = float(np.sum(windowed))
        if total > 0.0:
            windowed /= total

        otf = np.fft.fftshift(np.fft.fft2(windowed))
        mtf_2d = np.abs(otf)
        dc = float(mtf_2d[height // 2, width // 2])
        if dc <= 0.0:
            raise ValueError("MTF ROI produced an invalid zero-frequency response")
        mtf_2d /= dc

        fx = np.fft.fftshift(np.fft.fftfreq(width, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(height, d=dy))
        freq_x, freq_y = np.meshgrid(fx, fy)
        radial_freq = np.sqrt(freq_x**2 + freq_y**2)

        radial_positive = radial_freq.ravel()
        mtf_positive = mtf_2d.ravel()
        max_frequency = min(0.5 / dx, 0.5 / dy)

        bin_count = max(min(min(height, width) // 2, 128), 8)
        edges = np.linspace(0.0, max_frequency, bin_count + 1)
        values = np.zeros(bin_count, dtype=np.float64)

        for index in range(bin_count):
            if index == bin_count - 1:
                mask = (radial_positive >= edges[index]) & (radial_positive <= edges[index + 1])
            else:
                mask = (radial_positive >= edges[index]) & (radial_positive < edges[index + 1])
            if np.any(mask):
                values[index] = float(np.mean(mtf_positive[mask]))
            elif index > 0:
                values[index] = values[index - 1]

        frequencies = (edges[:-1] + edges[1:]) / 2.0
        values = ndimage.uniform_filter1d(values, size=3, mode="nearest")
        values = np.clip(values, 0.0, 1.0)
        values[0] = 1.0
        values = np.minimum.accumulate(values)
        frequencies = np.concatenate(([0.0], frequencies))
        values = np.concatenate(([1.0], values))
        return frequencies, values

    @staticmethod
    def _find_frequency_at(freqs: np.ndarray, mtf: np.ndarray, target: float) -> float:
        if freqs.size == 0 or mtf.size == 0:
            return 0.0

        if float(mtf[0]) <= target:
            return float(freqs[0])

        for index in range(1, len(mtf)):
            if mtf[index] <= target <= mtf[index - 1]:
                delta = float(mtf[index - 1] - mtf[index])
                if delta <= 1e-12:
                    return float(freqs[index])
                fraction = float((mtf[index - 1] - target) / delta)
                return float(freqs[index - 1] + fraction * (freqs[index] - freqs[index - 1]))
        return float(freqs[-1])
