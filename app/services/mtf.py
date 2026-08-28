from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MtfQualityWarning:
    code: str
    message: str


@dataclass(frozen=True)
class MtfAnalysisResult:
    frequencies: np.ndarray
    values: np.ndarray
    mtf50: float | None
    mtf10: float | None
    mtf50_w: float | None
    mtf10_w: float | None
    mtf50_h: float | None
    mtf10_h: float | None
    nyquist_w: float
    nyquist_h: float
    radial_nyquist: float
    peak_value: float
    center_x: float
    center_y: float
    fwhm_w: float | None
    fwhm_h: float | None
    quality_warnings: tuple[MtfQualityWarning, ...]


@dataclass(frozen=True)
class _PreparedRoi:
    psf: np.ndarray
    detection_signal: np.ndarray
    polarity: int
    signal_peak: float
    background_plane_range: float
    background_residual_std: float
    used_positive_component_fallback: bool


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
        if roi_array.shape[0] < 9 or roi_array.shape[1] < 9:
            raise ValueError("MTF ROI must be at least 9 x 9 pixels")
        if not np.all(np.isfinite(roi_array)):
            raise ValueError("MTF ROI contains non-finite pixel values")

        dx, dy = MtfAnalyzer._normalize_spacing(spacing_xy)
        prepared = MtfAnalyzer._prepare_roi(roi_array)
        center_y, center_x = MtfAnalyzer._find_bead_center(prepared.detection_signal)

        lsf_w = np.sum(prepared.psf, axis=0)
        lsf_h = np.sum(prepared.psf, axis=1)
        fwhm_w_pixels = MtfAnalyzer._calculate_fwhm(lsf_w)
        fwhm_h_pixels = MtfAnalyzer._calculate_fwhm(lsf_h)

        frequencies, values = MtfAnalyzer._psf_to_radial_mtf(prepared.psf, dx=dx, dy=dy)
        frequencies_w, values_w = MtfAnalyzer._lsf_to_mtf(lsf_w, spacing=dx)
        frequencies_h, values_h = MtfAnalyzer._lsf_to_mtf(lsf_h, spacing=dy)

        mtf50 = MtfAnalyzer._find_frequency_at(frequencies, values, 0.5)
        mtf10 = MtfAnalyzer._find_frequency_at(frequencies, values, 0.1)
        mtf50_w = MtfAnalyzer._find_frequency_at(frequencies_w, values_w, 0.5)
        mtf10_w = MtfAnalyzer._find_frequency_at(frequencies_w, values_w, 0.1)
        mtf50_h = MtfAnalyzer._find_frequency_at(frequencies_h, values_h, 0.5)
        mtf10_h = MtfAnalyzer._find_frequency_at(frequencies_h, values_h, 0.1)

        warnings: list[MtfQualityWarning] = [
            MtfQualityWarning(
                code="source-size-uncorrected",
                message="Finite point-source size correction was not applied.",
            )
        ]
        if prepared.used_positive_component_fallback:
            warnings.append(
                MtfQualityWarning(
                    code="unstable-dc-fallback",
                    message=(
                        "The signed point-spread response had an unstable zero-frequency value; "
                        "MTF was estimated from the non-negative component containing the detected source."
                    ),
                )
            )
        if roi_array.shape[0] < 21 or roi_array.shape[1] < 21:
            warnings.append(
                MtfQualityWarning(
                    code="roi-small",
                    message="The ROI is smaller than 21 x 21 pixels; frequency estimates may be unstable.",
                )
            )

        if fwhm_w_pixels is None:
            warnings.append(
                MtfQualityWarning(
                    code="fwhm-w-incomplete",
                    message="FWHM-W could not be measured because both half-height crossings are not inside the ROI.",
                )
            )
        if fwhm_h_pixels is None:
            warnings.append(
                MtfQualityWarning(
                    code="fwhm-h-incomplete",
                    message="FWHM-H could not be measured because both half-height crossings are not inside the ROI.",
                )
            )

        measurable_fwhm = [value for value in (fwhm_w_pixels, fwhm_h_pixels) if value is not None]
        required_margin = max(4.0, 1.5 * max(measurable_fwhm, default=0.0))
        edge_distance = min(
            center_x,
            center_y,
            roi_array.shape[1] - 1.0 - center_x,
            roi_array.shape[0] - 1.0 - center_y,
        )
        if edge_distance < required_margin:
            warnings.append(
                MtfQualityWarning(
                    code="point-near-roi-edge",
                    message="The point source is too close to an ROI edge for a reliable full PSF measurement.",
                )
            )

        if prepared.background_residual_std > 0.0:
            snr = prepared.signal_peak / prepared.background_residual_std
            if snr < 10.0:
                warnings.append(
                    MtfQualityWarning(
                        code="low-snr",
                        message=f"Estimated point-source SNR is low ({snr:.1f}); use a larger or cleaner ROI.",
                    )
                )
        if prepared.background_plane_range > prepared.signal_peak * 0.05:
            warnings.append(
                MtfQualityWarning(
                    code="nonuniform-background",
                    message="The fitted background varies by more than 5% of the point-source peak.",
                )
            )
        elif prepared.background_residual_std > prepared.signal_peak * 0.10:
            warnings.append(
                MtfQualityWarning(
                    code="nonuniform-background",
                    message="Background residual variation exceeds 10% of the point-source peak.",
                )
            )

        thresholds = (
            ("mtf50-beyond-nyquist", "Radial MTF50", mtf50),
            ("mtf10-beyond-nyquist", "Radial MTF10", mtf10),
            ("mtf50-w-beyond-nyquist", "MTF50-W", mtf50_w),
            ("mtf10-w-beyond-nyquist", "MTF10-W", mtf10_w),
            ("mtf50-h-beyond-nyquist", "MTF50-H", mtf50_h),
            ("mtf10-h-beyond-nyquist", "MTF10-H", mtf10_h),
        )
        for code, label, value in thresholds:
            if value is None:
                warnings.append(
                    MtfQualityWarning(
                        code=code,
                        message=f"{label} remains above its threshold at Nyquist.",
                    )
                )

        peak_value = float(np.min(roi_array) if prepared.polarity < 0 else np.max(roi_array))
        return MtfAnalysisResult(
            frequencies=frequencies,
            values=values,
            mtf50=mtf50,
            mtf10=mtf10,
            mtf50_w=mtf50_w,
            mtf10_w=mtf10_w,
            mtf50_h=mtf50_h,
            mtf10_h=mtf10_h,
            nyquist_w=0.5 / dx,
            nyquist_h=0.5 / dy,
            radial_nyquist=min(0.5 / dx, 0.5 / dy),
            peak_value=peak_value,
            center_x=center_x,
            center_y=center_y,
            fwhm_w=None if fwhm_w_pixels is None else fwhm_w_pixels * dx,
            fwhm_h=None if fwhm_h_pixels is None else fwhm_h_pixels * dy,
            quality_warnings=tuple(warnings),
        )

    @staticmethod
    def _normalize_spacing(spacing_xy: tuple[float, float] | None) -> tuple[float, float]:
        if spacing_xy is None:
            return 1.0, 1.0
        return (
            max(abs(float(spacing_xy[0])), 1e-6),
            max(abs(float(spacing_xy[1])), 1e-6),
        )

    @staticmethod
    def _prepare_roi(roi: np.ndarray) -> _PreparedRoi:
        height, width = roi.shape
        yy, xx = np.mgrid[:height, :width]
        border_mask = np.zeros_like(roi, dtype=bool)
        border_mask[[0, -1], :] = True
        border_mask[:, [0, -1]] = True
        border_x = xx[border_mask].astype(np.float64)
        border_y = yy[border_mask].astype(np.float64)
        border_values = roi[border_mask]
        design = np.column_stack((border_x, border_y, np.ones(border_values.size)))
        keep = np.ones(border_values.size, dtype=bool)
        coefficients = np.zeros(3, dtype=np.float64)

        for _ in range(4):
            coefficients, *_ = np.linalg.lstsq(design[keep], border_values[keep], rcond=None)
            residuals = border_values - design @ coefficients
            center = float(np.median(residuals[keep]))
            mad = float(np.median(np.abs(residuals[keep] - center)))
            robust_sigma = 1.4826 * mad
            if robust_sigma <= 1e-12:
                break
            next_keep = np.abs(residuals - center) <= 3.5 * robust_sigma
            if np.count_nonzero(next_keep) < 6 or np.array_equal(next_keep, keep):
                break
            keep = next_keep

        background_plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
        residual = roi - background_plane
        smoothed = ndimage.gaussian_filter(residual, sigma=1.0)
        bright_contrast = float(np.max(smoothed))
        dark_contrast = float(-np.min(smoothed))
        polarity = -1 if dark_contrast > bright_contrast else 1
        psf = residual * polarity
        detection_signal = np.clip(psf, 0.0, None)
        signal_peak = float(np.max(detection_signal))
        if signal_peak <= max(float(np.ptp(roi)), 1.0) * 1e-9:
            raise ValueError("MTF ROI does not contain a detectable point-source signal")

        border_residuals = residual[border_mask]
        border_center = float(np.median(border_residuals))
        border_mad = float(np.median(np.abs(border_residuals - border_center)))
        background_residual_std = 1.4826 * border_mad
        if background_residual_std <= 1e-12:
            background_residual_std = float(np.std(border_residuals))

        if background_residual_std > 1e-12 and signal_peak / background_residual_std < 3.0:
            raise ValueError("MTF ROI point-source SNR is below the minimum reliable value of 3")

        signed_total = float(np.sum(psf))
        positive_total = float(np.sum(detection_signal))
        minimum_stable_dc = max(signal_peak * 1e-9, positive_total * 0.05)
        used_positive_component_fallback = signed_total <= minimum_stable_dc
        if used_positive_component_fallback:
            psf = MtfAnalyzer._extract_peak_component(
                detection_signal,
                noise_sigma=background_residual_std,
            )
            total = float(np.sum(psf))
        else:
            total = signed_total
        if total <= signal_peak * 1e-9:
            raise ValueError("MTF ROI does not contain a stable point-source component")
        psf = psf / total

        return _PreparedRoi(
            psf=psf,
            detection_signal=detection_signal,
            polarity=polarity,
            signal_peak=signal_peak,
            background_plane_range=float(np.ptp(background_plane)),
            background_residual_std=background_residual_std,
            used_positive_component_fallback=used_positive_component_fallback,
        )

    @staticmethod
    def _extract_peak_component(signal: np.ndarray, *, noise_sigma: float) -> np.ndarray:
        peak_y, peak_x = np.unravel_index(int(np.argmax(signal)), signal.shape)
        signal_peak = float(signal[peak_y, peak_x])
        threshold = max(signal_peak * 0.05, noise_sigma * 1.5)
        labels, _ = ndimage.label(signal >= threshold)
        peak_label = int(labels[peak_y, peak_x])
        if peak_label <= 0:
            return np.zeros_like(signal)

        support = labels == peak_label
        support = ndimage.binary_dilation(support, iterations=2)
        return np.where(support, signal, 0.0)

    @staticmethod
    def _find_bead_center(signal: np.ndarray) -> tuple[float, float]:
        smoothed = ndimage.gaussian_filter(signal, sigma=1.0)
        peak_y, peak_x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
        radius = 5
        y0 = max(0, peak_y - radius)
        y1 = min(signal.shape[0], peak_y + radius + 1)
        x0 = max(0, peak_x - radius)
        x1 = min(signal.shape[1], peak_x + radius + 1)
        patch = signal[y0:y1, x0:x1]
        total = float(np.sum(patch))
        if total <= 0.0:
            return float(peak_y), float(peak_x)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        return float(np.sum(yy * patch) / total), float(np.sum(xx * patch) / total)

    @staticmethod
    def _calculate_fwhm(profile: np.ndarray) -> float | None:
        working = np.asarray(profile, dtype=np.float64)
        if working.size < 3 or not np.all(np.isfinite(working)):
            return None
        edge_count = max(1, min(working.size // 10, 5))
        baseline = float(np.median(np.concatenate((working[:edge_count], working[-edge_count:]))))
        signal = working - baseline
        peak_value = float(np.max(signal))
        if peak_value <= 0.0:
            return None
        half_max = peak_value / 2.0
        peak_index = int(np.argmax(signal))
        left = MtfAnalyzer._interp_crossing(signal, half_max, peak_index, go_left=True)
        right = MtfAnalyzer._interp_crossing(signal, half_max, peak_index, go_left=False)
        if left is None or right is None or right <= left:
            return None
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
    def _next_fft_size(length: int) -> int:
        target = max(length, length * 8)
        return 1 << (target - 1).bit_length()

    @staticmethod
    def _lsf_to_mtf(lsf: np.ndarray, *, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        n_fft = MtfAnalyzer._next_fft_size(int(lsf.size))
        values = np.abs(np.fft.rfft(np.asarray(lsf, dtype=np.float64), n=n_fft))
        dc = float(values[0])
        if dc <= 1e-12:
            raise ValueError("MTF ROI produced an invalid zero-frequency response")
        values /= dc
        return np.fft.rfftfreq(n_fft, d=spacing), values

    @staticmethod
    def _psf_to_radial_mtf(psf: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
        height, width = psf.shape
        n_fft_h = MtfAnalyzer._next_fft_size(height)
        n_fft_w = MtfAnalyzer._next_fft_size(width)
        mtf_2d = np.abs(np.fft.fftshift(np.fft.fft2(psf, s=(n_fft_h, n_fft_w))))
        dc = float(mtf_2d[n_fft_h // 2, n_fft_w // 2])
        if dc <= 1e-12:
            raise ValueError("MTF ROI produced an invalid zero-frequency response")
        mtf_2d /= dc

        frequency_step_x = 1.0 / (n_fft_w * dx)
        frequency_step_y = 1.0 / (n_fft_h * dy)
        frequency_step = max(frequency_step_x, frequency_step_y)
        radial_nyquist = min(0.5 / dx, 0.5 / dy)
        frequencies = np.arange(0.0, radial_nyquist + frequency_step * 0.25, frequency_step)
        frequencies = frequencies[frequencies <= radial_nyquist + 1e-12]

        angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        sample_x = n_fft_w // 2 + frequencies[:, None] * np.cos(angles)[None, :] / frequency_step_x
        sample_y = n_fft_h // 2 + frequencies[:, None] * np.sin(angles)[None, :] / frequency_step_y
        samples = ndimage.map_coordinates(
            mtf_2d,
            np.vstack((sample_y.ravel(), sample_x.ravel())),
            order=1,
            mode="wrap",
            prefilter=False,
        ).reshape(frequencies.size, angles.size)
        values = np.mean(samples, axis=1)
        values[0] = 1.0
        return frequencies, values

    @staticmethod
    def _find_frequency_at(freqs: np.ndarray, mtf: np.ndarray, target: float) -> float | None:
        if freqs.size == 0 or mtf.size == 0:
            return None
        if float(mtf[0]) <= target:
            return float(freqs[0])
        for index in range(1, len(mtf)):
            if mtf[index] <= target < mtf[index - 1]:
                delta = float(mtf[index - 1] - mtf[index])
                if delta <= 1e-12:
                    return float(freqs[index])
                fraction = float((mtf[index - 1] - target) / delta)
                return float(freqs[index - 1] + fraction * (freqs[index] - freqs[index - 1]))
        return None
