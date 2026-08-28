from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydicom.dataset import Dataset

from app.models.viewer import InstanceRecord, SeriesRecord, ViewRecord
from app.schemas.view import ViewSetSizeRequest
from app.services.viewer_service import ViewerService


def _build_stack_context(monkeypatch, *, width: int = 200, height: int = 100):
    service = ViewerService()
    instance = InstanceRecord(
        path=Path("stack-fit.dcm"),
        sop_instance_uid="1.2.3.4",
        instance_number=1,
        rows=height,
        columns=width,
    )
    series = SeriesRecord(
        series_id="stack-series",
        folder_path=".",
        series_instance_uid="1.2.3",
        study_instance_uid="1.2",
        patient_id="patient",
        patient_name="Patient",
        study_date="20260818",
        study_description="Stack fit",
        accession_number="ACC",
        modality="CT",
        series_description="Stack fit",
        instances=[instance],
    )
    dataset = Dataset()
    # DICOM order is [row, column]. The pixels are 200 x 100, but their
    # physical display extent is square: 200 mm x 200 mm.
    dataset.PixelSpacing = [2.0, 1.0]
    cached = SimpleNamespace(
        dataset=dataset,
        source_pixels=np.zeros((height, width), dtype=np.float32),
        window_width=400.0,
        window_center=40.0,
    )
    view = ViewRecord(
        view_id="stack-view",
        series_id=series.series_id,
        view_type="Stack",
        width=200,
        height=100,
    )

    monkeypatch.setattr("app.services.viewer.state.compat.series_registry.get", lambda *_args, **_kwargs: series)
    monkeypatch.setattr("app.services.viewer.state.compat.dicom_cache.get", lambda *_args, **_kwargs: cached)
    monkeypatch.setattr("app.services.viewer_service.view_registry.get", lambda *_args, **_kwargs: view)
    monkeypatch.setattr(service, "_resolve_representative_stack_index", lambda _series: 0)
    return service, series, view


def test_stack_resize_refits_an_untouched_auto_fit_view(monkeypatch) -> None:
    service, _series, view = _build_stack_context(monkeypatch)
    service._initialize_viewport(view)
    view.is_initialized = True

    assert view.zoom == pytest.approx(0.5)

    service.set_view_size(
        ViewSetSizeRequest(
            viewId=view.view_id,
            opType="setSize",
            size={"width": 400, "height": 200},
        )
    )

    assert view.zoom == pytest.approx(1.0)
    assert view.offset_x == pytest.approx(0.0)
    assert view.offset_y == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("transform_field", "transform_value"),
    (
        ("zoom", 1.5),
        ("offset_x", 12.0),
        ("rotation_degrees", 90),
        ("hor_flip", True),
    ),
)
def test_stack_resize_preserves_a_user_transform(monkeypatch, transform_field: str, transform_value: object) -> None:
    service, _series, view = _build_stack_context(monkeypatch)
    service._initialize_viewport(view)
    view.is_initialized = True
    setattr(view, transform_field, transform_value)
    expected_zoom = view.zoom

    service.set_view_size(
        ViewSetSizeRequest(
            viewId=view.view_id,
            opType="setSize",
            size={"width": 400, "height": 200},
        )
    )

    assert view.zoom == pytest.approx(expected_zoom)
    assert getattr(view, transform_field) == transform_value


def test_stack_zoom_reset_uses_the_same_physical_contain_fit_as_initialization(monkeypatch) -> None:
    service, series, view = _build_stack_context(monkeypatch)
    view.width = 400
    view.height = 200
    view.zoom = 4.0

    assert service._reset_view_zoom_state(view, series) is True

    assert view.zoom == pytest.approx(1.0)

