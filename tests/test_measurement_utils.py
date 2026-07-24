import math

import numpy as np
import pytest

from app.models.measurement import MeasurementPoint, MeasurementToolType
from app.services.measurement_utils import build_measurement_metrics


def test_rect_measurement_metrics_tolerate_points_outside_image() -> None:
    source_pixels = np.arange(25, dtype=np.float32).reshape(5, 5)

    metrics, label_lines = build_measurement_metrics(
        "rect",
        (
            MeasurementPoint(x=-2.0, y=-1.0),
            MeasurementPoint(x=2.0, y=2.0),
        ),
        source_pixels,
        None,
    )

    assert metrics.width == 4.0
    assert metrics.height == 3.0
    assert metrics.mean == float(np.mean(source_pixels[0:3, 0:3]))
    assert label_lines[0] == "Size 4.0 * 3.0 px"


def test_freeform_measurement_metrics_tolerate_polygon_outside_image() -> None:
    source_pixels = np.ones((5, 5), dtype=np.float32)

    metrics, _ = build_measurement_metrics(
        "freeform",
        (
            MeasurementPoint(x=-2.0, y=-2.0),
            MeasurementPoint(x=3.0, y=0.0),
            MeasurementPoint(x=2.0, y=3.0),
        ),
        source_pixels,
        None,
    )

    assert metrics.width == 5.0
    assert metrics.height == 5.0
    assert metrics.area is not None
    assert metrics.area > 0


def test_line_measurement_uses_anisotropic_dicom_pixel_spacing() -> None:
    metrics, label_lines = build_measurement_metrics(
        "line",
        (
            MeasurementPoint(x=2.0, y=3.0),
            MeasurementPoint(x=10.0, y=7.0),
        ),
        np.zeros((16, 16), dtype=np.float32),
        (0.5, 2.0),
    )

    assert metrics.unit == "mm"
    assert metrics.length == pytest.approx(math.sqrt((8 * 0.5) ** 2 + (4 * 2.0) ** 2))
    assert label_lines == ("8.9 mm",)


@pytest.mark.parametrize(
    ("tool_type", "expected_angle", "expected_label"),
    [
        ("alignment-horizontal", math.degrees(math.atan2(2.0, 4.0)), "ΔH 26.6°"),
        ("alignment-vertical", math.degrees(math.atan2(4.0, 2.0)), "ΔV 63.4°"),
    ],
)
def test_alignment_angle_uses_dicom_physical_spacing_for_horizontal_and_vertical_references(
    tool_type: MeasurementToolType,
    expected_angle: float,
    expected_label: str,
) -> None:
    metrics, label_lines = build_measurement_metrics(
        tool_type,
        (
            MeasurementPoint(x=0.0, y=0.0),
            MeasurementPoint(x=40.0, y=5.0),
        ),
        np.zeros((64, 64), dtype=np.float32),
        (0.5, 2.0),
    )

    assert metrics.unit == "mm"
    assert metrics.length == pytest.approx(math.hypot(20.0, 10.0))
    assert metrics.angle_degrees == pytest.approx(expected_angle)
    assert label_lines == (expected_label, "22.4 mm")


def test_alignment_angle_is_unchanged_when_the_reference_line_endpoints_are_reversed() -> None:
    forward_metrics, _ = build_measurement_metrics(
        "alignment-horizontal",
        (MeasurementPoint(x=0.0, y=0.0), MeasurementPoint(x=40.0, y=5.0)),
        np.zeros((64, 64), dtype=np.float32),
        (0.5, 2.0),
    )
    reversed_metrics, _ = build_measurement_metrics(
        "alignment-horizontal",
        (MeasurementPoint(x=40.0, y=5.0), MeasurementPoint(x=0.0, y=0.0)),
        np.zeros((64, 64), dtype=np.float32),
        (0.5, 2.0),
    )

    assert reversed_metrics.angle_degrees == pytest.approx(forward_metrics.angle_degrees)


def test_alignment_angle_rejects_missing_or_invalid_physical_spacing() -> None:
    points = (MeasurementPoint(x=0.0, y=0.0), MeasurementPoint(x=8.0, y=1.0))
    pixels = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="valid physical pixel spacing"):
        build_measurement_metrics("alignment-horizontal", points, pixels, None)
    with pytest.raises(ValueError, match="valid physical pixel spacing"):
        build_measurement_metrics("alignment-horizontal", points, pixels, (float("nan"), 0.5))


def test_alignment_angle_does_not_report_an_unstable_result_for_a_short_reference_line() -> None:
    metrics, label_lines = build_measurement_metrics(
        "alignment-horizontal",
        (MeasurementPoint(x=0.0, y=0.0), MeasurementPoint(x=8.0, y=1.0)),
        np.zeros((16, 16), dtype=np.float32),
        (0.5, 2.0),
    )

    assert metrics.length == pytest.approx(math.hypot(4.0, 2.0))
    assert metrics.angle_degrees is None
    assert label_lines == ("Reference line < 20 mm", "4.5 mm")


def test_rect_measurement_reports_physical_size_area_and_source_value_statistics() -> None:
    source_pixels = np.arange(100, dtype=np.float32).reshape(10, 10)

    metrics, label_lines = build_measurement_metrics(
        "rect",
        (
            MeasurementPoint(x=1.0, y=2.0),
            MeasurementPoint(x=6.0, y=6.0),
        ),
        source_pixels,
        (0.8, 1.5),
    )

    expected_roi = source_pixels[2:7, 1:7]
    assert metrics.unit == "mm"
    assert metrics.area_unit == "mm2"
    assert metrics.width == pytest.approx(4.0)
    assert metrics.height == pytest.approx(6.0)
    assert metrics.area == pytest.approx(24.0)
    assert metrics.mean == pytest.approx(float(np.mean(expected_roi)))
    assert metrics.standard_deviation == pytest.approx(float(np.std(expected_roi)))
    assert metrics.minimum == pytest.approx(float(np.min(expected_roi)))
    assert metrics.maximum == pytest.approx(float(np.max(expected_roi)))
    assert label_lines[:2] == ("Size 4.0 * 6.0 mm", "Area 24.0 mm2")


def test_ellipse_measurement_uses_physical_axes_for_area() -> None:
    metrics, _ = build_measurement_metrics(
        "ellipse",
        (
            MeasurementPoint(x=2.0, y=3.0),
            MeasurementPoint(x=10.0, y=9.0),
        ),
        np.ones((16, 16), dtype=np.float32),
        (0.5, 2.0),
    )

    assert metrics.width == pytest.approx(4.0)
    assert metrics.height == pytest.approx(12.0)
    assert metrics.area == pytest.approx(math.pi * 2.0 * 6.0)
    assert metrics.mean == pytest.approx(1.0)


def test_angle_and_curve_measurements_have_known_geometric_truth() -> None:
    angle_metrics, _ = build_measurement_metrics(
        "angle",
        (
            MeasurementPoint(x=1.0, y=0.0),
            MeasurementPoint(x=0.0, y=0.0),
            MeasurementPoint(x=0.0, y=1.0),
        ),
        np.zeros((4, 4), dtype=np.float32),
        (0.5, 2.0),
    )
    curve_metrics, _ = build_measurement_metrics(
        "curve",
        (
            MeasurementPoint(x=0.0, y=0.0),
            MeasurementPoint(x=6.0, y=0.0),
            MeasurementPoint(x=6.0, y=4.0),
        ),
        np.zeros((8, 8), dtype=np.float32),
        (0.5, 2.0),
    )

    assert angle_metrics.angle_degrees == pytest.approx(90.0)
    assert curve_metrics.unit == "mm"
    assert curve_metrics.length == pytest.approx(3.0 + 8.0)


def test_angle_measurement_uses_anisotropic_dicom_pixel_spacing() -> None:
    metrics, label_lines = build_measurement_metrics(
        "angle",
        (
            MeasurementPoint(x=2.0, y=0.0),
            MeasurementPoint(x=0.0, y=0.0),
            MeasurementPoint(x=2.0, y=2.0),
        ),
        np.zeros((4, 4), dtype=np.float32),
        (0.5, 2.0),
    )

    expected_angle = math.degrees(math.acos(1.0 / math.sqrt(17.0)))
    assert metrics.angle_degrees == pytest.approx(expected_angle)
    assert label_lines == (f"{expected_angle:.1f}\N{DEGREE SIGN}",)


def test_freeform_area_counts_enclosed_pixels_in_physical_square_millimetres() -> None:
    metrics, _ = build_measurement_metrics(
        "freeform",
        (
            MeasurementPoint(x=1.0, y=1.0),
            MeasurementPoint(x=5.0, y=1.0),
            MeasurementPoint(x=5.0, y=5.0),
            MeasurementPoint(x=1.0, y=5.0),
        ),
        np.full((8, 8), 42.0, dtype=np.float32),
        (0.5, 2.0),
    )

    assert metrics.width == pytest.approx(2.0)
    assert metrics.height == pytest.approx(8.0)
    assert metrics.area == pytest.approx(16.0)
    assert metrics.mean == pytest.approx(42.0)
