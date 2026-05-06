import numpy as np

from app.models.measurement import MeasurementPoint
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
