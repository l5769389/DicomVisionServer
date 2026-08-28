import math

import numpy as np
import pytest

from app.models.viewer import ViewRecord
from app.schemas.view import ViewQaWaterAnalyzeRequest
from app.services.mtf import MtfAnalyzer
from app.services.view_registry import view_registry
from app.services.viewport_transformer import AffineTransform
from app.services.water_phantom_qa_service import WaterPhantomQaService


def test_gaussian_point_source_mtf_matches_physical_frequency_and_fwhm_truth() -> None:
    size = 65
    sigma_pixels = 2.0
    spacing_mm = 0.5
    y_grid, x_grid = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    roi = 1000.0 * np.exp(
        -((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2.0 * sigma_pixels**2)
    )

    result = MtfAnalyzer.analyze_roi(roi, spacing_xy=(spacing_mm, spacing_mm))

    expected_mtf50_lp_per_mm = math.sqrt(math.log(2.0)) / (
        math.sqrt(2.0) * math.pi * sigma_pixels * spacing_mm
    )
    expected_mtf10_lp_per_mm = math.sqrt(math.log(10.0)) / (
        math.sqrt(2.0) * math.pi * sigma_pixels * spacing_mm
    )
    expected_fwhm_mm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_pixels * spacing_mm

    assert result.mtf50 == pytest.approx(expected_mtf50_lp_per_mm, rel=0.01)
    assert result.mtf10 == pytest.approx(expected_mtf10_lp_per_mm, rel=0.01)
    assert result.mtf50_w == pytest.approx(expected_mtf50_lp_per_mm, rel=0.01)
    assert result.mtf10_w == pytest.approx(expected_mtf10_lp_per_mm, rel=0.01)
    assert result.mtf50_h == pytest.approx(expected_mtf50_lp_per_mm, rel=0.01)
    assert result.mtf10_h == pytest.approx(expected_mtf10_lp_per_mm, rel=0.01)
    assert result.fwhm_w == pytest.approx(expected_fwhm_mm, rel=0.02)
    assert result.fwhm_h == pytest.approx(expected_fwhm_mm, rel=0.02)
    assert result.peak_value == pytest.approx(1000.0)


@pytest.mark.parametrize("shape", [(65, 65), (41, 81), (81, 41)])
def test_gaussian_truth_is_stable_for_square_and_rectangular_rois(shape: tuple[int, int]) -> None:
    height, width = shape
    sigma_pixels = 2.0
    y_grid, x_grid = np.mgrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    roi = 750.0 * np.exp(
        -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma_pixels**2)
    )

    result = MtfAnalyzer.analyze_roi(roi)

    expected_mtf50 = math.sqrt(math.log(2.0)) / (math.sqrt(2.0) * math.pi * sigma_pixels)
    expected_mtf10 = math.sqrt(math.log(10.0)) / (math.sqrt(2.0) * math.pi * sigma_pixels)
    expected_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_pixels
    assert result.mtf50 == pytest.approx(expected_mtf50, rel=0.01)
    assert result.mtf10 == pytest.approx(expected_mtf10, rel=0.01)
    assert result.fwhm_w == pytest.approx(expected_fwhm, rel=0.02)
    assert result.fwhm_h == pytest.approx(expected_fwhm, rel=0.02)


def test_subpixel_dark_point_with_linear_background_matches_gaussian_truth() -> None:
    height, width = 57, 73
    sigma_pixels = 2.4
    center_x, center_y = 35.35, 27.65
    y_grid, x_grid = np.mgrid[:height, :width]
    background = 120.0 + 0.45 * x_grid - 0.3 * y_grid
    roi = background - 900.0 * np.exp(
        -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma_pixels**2)
    )

    result = MtfAnalyzer.analyze_roi(roi)

    expected_mtf50 = math.sqrt(math.log(2.0)) / (math.sqrt(2.0) * math.pi * sigma_pixels)
    expected_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_pixels
    assert result.center_x == pytest.approx(center_x, abs=0.08)
    assert result.center_y == pytest.approx(center_y, abs=0.08)
    assert result.mtf50 == pytest.approx(expected_mtf50, rel=0.015)
    assert result.fwhm_w == pytest.approx(expected_fwhm, rel=0.02)
    assert result.fwhm_h == pytest.approx(expected_fwhm, rel=0.02)
    assert result.peak_value == pytest.approx(float(np.min(roi)))
    assert "nonuniform-background" in {warning.code for warning in result.quality_warnings}


def test_directional_mtf_and_fwhm_use_dicom_x_y_spacing() -> None:
    size = 81
    spacing_x, spacing_y = 0.4, 0.8
    physical_sigma = 1.6
    sigma_x = physical_sigma / spacing_x
    sigma_y = physical_sigma / spacing_y
    y_grid, x_grid = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    roi = 1000.0 * np.exp(
        -(
            (x_grid - center) ** 2 / (2.0 * sigma_x**2)
            + (y_grid - center) ** 2 / (2.0 * sigma_y**2)
        )
    )

    result = MtfAnalyzer.analyze_roi(roi, spacing_xy=(spacing_x, spacing_y))

    expected_mtf50 = math.sqrt(math.log(2.0)) / (math.sqrt(2.0) * math.pi * physical_sigma)
    expected_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * physical_sigma
    assert result.mtf50_w == pytest.approx(expected_mtf50, rel=0.01)
    assert result.mtf50_h == pytest.approx(expected_mtf50, rel=0.01)
    assert result.fwhm_w == pytest.approx(expected_fwhm, rel=0.02)
    assert result.fwhm_h == pytest.approx(expected_fwhm, rel=0.02)
    assert result.nyquist_w == pytest.approx(1.25)
    assert result.nyquist_h == pytest.approx(0.625)
    assert result.radial_nyquist == pytest.approx(0.625)


def test_edge_clipped_point_returns_null_fwhm_instead_of_pixel_count() -> None:
    size = 65
    sigma_pixels = 2.0
    y_grid, x_grid = np.mgrid[:size, :size]
    roi = 1000.0 * np.exp(-((x_grid - 0.0) ** 2 + (y_grid - 32.0) ** 2) / (2.0 * sigma_pixels**2))

    result = MtfAnalyzer.analyze_roi(roi)
    warning_codes = {warning.code for warning in result.quality_warnings}

    assert result.fwhm_w is None
    assert result.fwhm_h is not None
    assert "fwhm-w-incomplete" in warning_codes
    assert "point-near-roi-edge" in warning_codes


def test_delta_point_reports_thresholds_above_nyquist() -> None:
    roi = np.zeros((65, 65), dtype=np.float64)
    roi[32, 32] = 1000.0

    result = MtfAnalyzer.analyze_roi(roi)
    warning_codes = {warning.code for warning in result.quality_warnings}

    assert result.mtf50 is None
    assert result.mtf10 is None
    assert result.mtf50_w is None
    assert result.mtf10_w is None
    assert "mtf50-beyond-nyquist" in warning_codes
    assert "mtf10-w-beyond-nyquist" in warning_codes


def test_raw_mtf_curve_preserves_reconstruction_overshoot() -> None:
    roi = np.zeros((65, 65), dtype=np.float64)
    roi[32, 32] = 100.0
    roi[32, 31] = -20.0
    roi[32, 33] = -20.0

    result = MtfAnalyzer.analyze_roi(roi)

    assert float(np.max(result.values)) > 1.1
    assert np.any(np.diff(result.values) > 0.0)


def test_unstable_signed_dc_uses_detected_positive_component_with_warning() -> None:
    roi = np.zeros((65, 65), dtype=np.float64)
    roi[32, 32] = 100.0
    roi[31, 32] = -30.0
    roi[33, 32] = -30.0
    roi[32, 31] = -30.0
    roi[32, 33] = -30.0

    result = MtfAnalyzer.analyze_roi(roi)
    warning_codes = {warning.code for warning in result.quality_warnings}

    assert "unstable-dc-fallback" in warning_codes
    assert result.values[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(result.values))


def test_small_but_analyzable_roi_returns_quality_warning() -> None:
    size = 17
    y_grid, x_grid = np.mgrid[:size, :size]
    roi = 1000.0 * np.exp(-((x_grid - 8.0) ** 2 + (y_grid - 8.0) ** 2) / 8.0)

    result = MtfAnalyzer.analyze_roi(roi)

    assert result.mtf50 is not None
    assert "roi-small" in {warning.code for warning in result.quality_warnings}


def test_low_snr_point_returns_quality_warning_without_discarding_all_metrics() -> None:
    size = 65
    rng = np.random.default_rng(3)
    y_grid, x_grid = np.mgrid[:size, :size]
    roi = (
        100.0
        + rng.normal(0.0, 5.0, (size, size))
        + 30.0 * np.exp(-((x_grid - 32.0) ** 2 + (y_grid - 32.0) ** 2) / 8.0)
    )

    result = MtfAnalyzer.analyze_roi(roi)

    assert result.mtf50 is not None
    assert "low-snr" in {warning.code for warning in result.quality_warnings}


def test_roi_smaller_than_nine_pixels_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 9 x 9"):
        MtfAnalyzer.analyze_roi(np.ones((8, 9), dtype=np.float64))


class _WaterPhantomTruthContext:
    def __init__(self, pixels: np.ndarray, spacing_xy: tuple[float, float]) -> None:
        self._pixels = pixels
        self._spacing_xy = spacing_xy

    def _resolve_measurement_source_context(self, view: ViewRecord):
        return self._pixels, self._spacing_xy, object()

    def _build_hover_mapping_context(self, view: ViewRecord):
        height, width = self._pixels.shape
        return width, height, AffineTransform(np.eye(3, dtype=np.float64)), width, height


def test_uniform_water_phantom_reports_known_hu_and_physical_roi_dimensions() -> None:
    size = 256
    center = 128
    radius = 80
    pixels = np.full((size, size), -1000.0, dtype=np.float32)
    y_grid, x_grid = np.ogrid[:size, :size]
    water_mask = (x_grid - center) ** 2 + (y_grid - center) ** 2 <= radius**2
    pixels[water_mask] = 7.0

    view = ViewRecord(
        view_id="water-phantom-truth",
        series_id="water-phantom-series",
        view_type="Stack",
        width=size,
        height=size,
    )
    view_registry._view_by_id[view.view_id] = view
    service = WaterPhantomQaService(
        _WaterPhantomTruthContext(pixels, spacing_xy=(0.5, 1.0))
    )

    try:
        result = service.analyze(
            ViewQaWaterAnalyzeRequest(
                viewId=view.view_id,
                viewportKey="single",
                metrics=["accuracy", "uniformity", "noise"],
            )
        )
    finally:
        view_registry._view_by_id.pop(view.view_id, None)

    assert result.status == "ready"
    assert result.metrics.accuracy is not None
    assert result.metrics.uniformity is not None
    assert result.metrics.noise is not None
    assert result.metrics.accuracy.center_mean == pytest.approx(7.0)
    assert result.metrics.accuracy.deviation_hu == pytest.approx(7.0)
    assert result.metrics.uniformity.peripheral_means == pytest.approx([7.0, 7.0, 7.0, 7.0])
    assert result.metrics.uniformity.max_deviation == pytest.approx(0.0)
    assert result.metrics.noise.std_dev == pytest.approx(0.0)

    center_roi = next(item for item in result.metrics.uniformity.roi_stats if item.id == "center")
    assert center_roi.unit == "HU"
    assert center_roi.size_unit == "mm"
    assert center_roi.area_unit == "mm2"
    assert center_roi.height == pytest.approx(center_roi.width * 2.0, abs=0.02)
    assert center_roi.area == pytest.approx(
        math.pi * (center_roi.width / 2.0) * (center_roi.height / 2.0),
        abs=0.15,
    )
