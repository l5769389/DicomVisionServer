import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset

from app.models.viewer import InstanceRecord, SeriesRecord, ViewRecord
from app.services.dicom_cache import CachedDicom
from app.services.render_layers.base_image_layer import BaseImageLayer
from app.services.render_layers.render_context import RenderContext
from app.services.series_registry import SeriesRegistry
from app.services.viewport_transformer import AffineTransform


def _build_rgb_cached(pixels: np.ndarray) -> CachedDicom:
    return CachedDicom(
        dataset=Dataset(),
        source_pixels=pixels,
        window_width=None,
        window_center=None,
        pixel_min=float(np.min(pixels)),
        pixel_max=float(np.max(pixels)),
        byte_size=int(pixels.nbytes),
    )


def _build_series() -> SeriesRecord:
    return SeriesRecord(
        series_id="rgb-series",
        folder_path=".",
        series_instance_uid="1.2.3.rgb",
        study_instance_uid=None,
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="OT",
        series_description="RGB Secondary Capture",
        instances=[
            InstanceRecord(
                path=Path("rgb.dcm"),
                sop_instance_uid="1.2.3.rgb.1",
                instance_number=1,
                rows=2,
                columns=2,
                photometric_interpretation="RGB",
                samples_per_pixel=3,
            )
        ],
    )


def test_rgb_secondary_capture_thumbnail_uses_color_pixels(monkeypatch) -> None:
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    fake_cache = SimpleNamespace(get=lambda instance_uid, path: _build_rgb_cached(pixels))
    monkeypatch.setattr("app.services.series_registry.dicom_cache", fake_cache)

    thumbnail = SeriesRegistry()._build_series_thumbnail_png(_build_series())

    assert thumbnail is not None
    with Image.open(io.BytesIO(thumbnail)) as image:
        assert image.mode == "RGB"
        assert image.size == (96, 96)
        assert image.getbbox() is not None


def test_stack_base_image_layer_renders_rgb_pixels_as_color_image() -> None:
    pixels = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    cached = _build_rgb_cached(pixels)
    view = ViewRecord(view_id="view-rgb", series_id="rgb-series", view_type="Stack")
    context = RenderContext(
        view=view,
        source_pixels=cached.source_pixels,
        pixel_min=cached.pixel_min,
        pixel_max=cached.pixel_max,
        image_transform=AffineTransform(np.eye(3, dtype=np.float64)),
        cached=cached,
    )

    rendered_pixels = BaseImageLayer().render_pixels(context)

    assert rendered_pixels.dtype == np.uint8
    assert rendered_pixels.shape == (2, 2, 3)
    np.testing.assert_array_equal(rendered_pixels, pixels)
