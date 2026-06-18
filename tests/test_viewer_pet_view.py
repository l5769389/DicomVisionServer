from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from pydicom.dataset import Dataset

from app.models.viewer import InstanceRecord, SeriesRecord, ViewRecord
from app.schemas.view import ViewCreateRequest, ViewOperationRequest
from app.services import view_registry as view_registry_module
from app.services.layered_renderer import RenderContext, layered_renderer
from app.services.view_registry import ViewRegistry
from app.services.viewer_service import ViewerService
from app.services.viewport_transformer import viewport_transformer


def _instance(index: int = 1) -> InstanceRecord:
    return InstanceRecord(
        path=Path(f"IM{index:06d}.dcm"),
        sop_instance_uid=f"1.2.3.{index}",
        instance_number=index,
        rows=4,
        columns=5,
    )


def _series(series_id: str = "pet", modality: str = "PT") -> SeriesRecord:
    return SeriesRecord(
        series_id=series_id,
        folder_path=".",
        series_instance_uid=f"1.2.840.{series_id}",
        study_instance_uid="1.2.840.study",
        patient_id="patient",
        patient_name="Patient",
        study_date="20260101",
        study_description="Study",
        accession_number="ACC",
        modality=modality,
        series_description="PET FDG SUV",
        instances=[_instance(1), _instance(2), _instance(3)],
    )


def _dataset(units: str = "GML") -> Dataset:
    dataset = Dataset()
    dataset.Units = units
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.InstanceNumber = 1
    dataset.SOPInstanceUID = "1.2.3.1"
    return dataset


def _patch_pet_render_dependencies(
    monkeypatch,
    service: ViewerService,
    series: SeriesRecord,
    volume: np.ndarray,
    *,
    stub_renderer: bool = True,
) -> None:
    dataset = _dataset("GML")
    cached = SimpleNamespace(dataset=dataset, source_pixels=volume[0])
    monkeypatch.setattr("app.services.viewer_service.series_registry.get", lambda *_args, **_kwargs: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(service, "_resolve_representative_stack_index", lambda _series: 1)
    monkeypatch.setattr(service, "_get_reference_instance_and_cache", lambda _series: (series.instances[0], cached))
    monkeypatch.setattr(service, "_get_indexed_instance_and_cache", lambda _series, index: (series.instances[index], cached))
    monkeypatch.setattr(service, "_get_stack_spacing_xy", lambda _dataset: (1.0, 1.0))
    monkeypatch.setattr(service, "_build_scale_bar_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_build_visible_measurements", lambda _view: ())
    monkeypatch.setattr(service, "_build_visible_presentation_measurements", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(service, "_build_visible_presentation_annotations", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(service, "_build_stack_orientation_overlay", lambda *_args, **_kwargs: None)
    if stub_renderer:
        monkeypatch.setattr("app.services.viewer_service.layered_renderer.render", lambda _context: Image.new("RGB", (5, 4)))


def test_pet_view_create_accepts_pet_and_rejects_non_pet(monkeypatch) -> None:
    pet_series = _series("pet", "PT")
    ct_series = _series("ct", "CT")

    def get_series(series_id: str, **_kwargs):
        return pet_series if series_id == "pet" else ct_series

    monkeypatch.setattr(view_registry_module.series_registry, "get", get_series)
    registry = ViewRegistry()

    created = registry.create(ViewCreateRequest(seriesId="pet", viewType="PET"))
    assert registry.get(created.view_id).view_type == "PET"

    with pytest.raises(HTTPException) as error:
        registry.create(ViewCreateRequest(seriesId="ct", viewType="PET"))
    assert error.value.status_code == 400
    assert "PET view" in str(error.value.detail)


def test_render_pet_view_returns_pet_info(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)

    view = ViewRecord(view_id="pet-view", series_id=series.series_id, view_type="PET", width=128, height=96)
    result = service._render_pet_view(view)

    assert result.meta.slice_info.current == 1
    assert result.meta.slice_info.total == 3
    assert result.meta.pet_info is not None
    assert result.meta.pet_info.series_id == "pet"
    assert result.meta.pet_info.pet_unit == "SUVbw"
    assert result.meta.pet_info.pet_unit_label == "g/ml (SUVbw)"
    assert result.meta.pet_info.pseudocolor_preset == "bwinverse"
    assert result.meta.pet_info.pet_window_min == pytest.approx(0.0)
    assert result.meta.pet_info.pet_window_max == pytest.approx(4.49)


def test_render_pet_view_passes_white_background_to_renderer(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    volume[:, 1:3, 2:4] = 10.0
    _patch_pet_render_dependencies(monkeypatch, service, series, volume, stub_renderer=False)
    captured: dict[str, float] = {}

    def render_with_capture(context: RenderContext) -> Image.Image:
        captured["background_cval"] = context.background_cval
        return Image.new("RGB", (5, 4))

    monkeypatch.setattr("app.services.viewer_service.layered_renderer.render", render_with_capture)

    view = ViewRecord(view_id="pet-view", series_id=series.series_id, view_type="PET", width=96, height=96)
    result = service._render_pet_view(view)

    assert result.meta.pet_info is not None
    assert captured["background_cval"] == 255.0


def test_pet_render_context_uses_white_background_after_window_render() -> None:
    view = ViewRecord(view_id="pet-view", series_id="pet", view_type="PET", width=24, height=24)
    view.pseudocolor_preset = "bwinverse"
    view.window_width = 10.0
    view.window_center = 5.0
    source_pixels = np.zeros((4, 4), dtype=np.float32)
    source_pixels[1:3, 1:3] = 10.0
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=4,
        image_height=4,
        canvas_width=24,
        canvas_height=24,
        view=view,
    )
    image = layered_renderer.render(
        RenderContext(
            view=view,
            source_pixels=source_pixels,
            pixel_min=0.0,
            pixel_max=10.0,
            image_transform=image_transform,
            background_cval=255.0,
        )
    ).convert("RGB")
    pixels = np.asarray(image)

    assert np.all(pixels[:4, :4] >= 248)
    assert np.all(pixels[:4, -4:] >= 248)
    assert np.all(pixels[-4:, :4] >= 248)
    assert np.all(pixels[-4:, -4:] >= 248)


def test_pet_standalone_edge_background_is_suppressed_to_white() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="pet-view", series_id="pet", view_type="PET", width=24, height=24)
    view.pseudocolor_preset = "bwinverse"
    view.window_width = 4.49
    view.window_center = 2.245
    source_pixels = np.ones((4, 4), dtype=np.float32)
    source_pixels[1:3, 1:3] = 4.49
    prepared_pixels = service._prepare_pet_standalone_source_pixels(
        source_pixels,
        view.window_width,
        view.window_center,
    )
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=4,
        image_height=4,
        canvas_width=24,
        canvas_height=24,
        view=view,
    )
    image = layered_renderer.render(
        RenderContext(
            view=view,
            source_pixels=prepared_pixels,
            pixel_min=float(np.nanmin(prepared_pixels)),
            pixel_max=float(np.nanmax(prepared_pixels)),
            image_transform=image_transform,
            background_cval=255.0,
        )
    ).convert("RGB")
    pixels = np.asarray(image)

    assert np.all(pixels[3:7, 3:7] >= 248)
    assert np.any(pixels[10:14, 10:14] < 80)


def test_pet_config_unit_resets_default_window(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.ones((3, 4, 5), dtype=np.float32)
    monkeypatch.setattr("app.services.viewer_service.series_registry.get", lambda *_args, **_kwargs: series)
    monkeypatch.setattr(service, "_get_series_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(
        service,
        "_build_fusion_pet_display_volume",
        lambda *_args, **_kwargs: SimpleNamespace(
            volume=volume * 2.0,
            unit="kBqml",
            unit_label="kBq/ml (uptake)",
            source_units="BQML",
            scale=0.001,
        ),
    )
    monkeypatch.setattr(service, "_derive_default_pet_window_for_display_volume", lambda _display: (6.0, 3.0))

    view = ViewRecord(
        view_id="pet-view",
        series_id=series.series_id,
        view_type="PET",
        width=128,
        height=96,
        is_initialized=True,
        pet_unit="SUVbw",
    )
    view.window_width = 12.0
    view.window_center = 6.0

    changed = service._handle_pet_config(view, ViewOperationRequest(viewId=view.view_id, opType="petConfig", petUnit="kBqml"))

    assert changed is True
    assert view.pet_unit == "kBqml"
    assert view.pet_unit_label == "kBq/ml (uptake)"
    assert view.window_width == 6.0
    assert view.window_center == 3.0


def test_pet_config_window_updates_range() -> None:
    service = ViewerService()
    view = ViewRecord(
        view_id="pet-view",
        series_id="pet",
        view_type="PET",
        width=128,
        height=96,
        pseudocolor_preset="rainbow",
    )

    changed = service._handle_pet_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="petConfig",
            petWindowMin=1.0,
            petWindowMax=9.0,
            pseudocolorPreset="rainbow",
        ),
    )

    assert changed is True
    assert view.window_width == 8.0
    assert view.window_center == 5.0
    assert view.pseudocolor_preset == "bwinverse"
