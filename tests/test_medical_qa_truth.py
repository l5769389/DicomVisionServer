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

    # The implementation includes Hann windowing and radial binning, so the
    # analytical Gaussian truth is compared with an explicit numerical tolerance.
    assert result.mtf50 == pytest.approx(expected_mtf50_lp_per_mm, rel=0.03)
    assert result.mtf10 == pytest.approx(expected_mtf10_lp_per_mm, rel=0.05)
    assert result.fwhm_w == pytest.approx(expected_fwhm_mm, rel=0.03)
    assert result.fwhm_h == pytest.approx(expected_fwhm_mm, rel=0.03)
    assert result.peak_value == pytest.approx(1000.0)


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
