from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from pydicom.dataset import Dataset

from app.core import MPR_VIEWPORT_CORONAL
from app.models.viewer import InstanceRecord, SeriesRecord, ViewGroupRecord, ViewRecord
from app.schemas.view import (
    SurfaceRenderConfig,
    ViewCreateRequest,
    ViewHoverRequest,
    ViewOperationRequest,
    ViewSetSizeRequest,
)
from app.services import view_registry as view_registry_module
from app.services.layered_renderer import RenderContext, layered_renderer
from app.services.mpr import build_identity_geometry
from app.services.pseudocolor import pseudocolor_background_color
from app.services.view_registry import ViewRegistry
from app.services.viewer_operation_handlers import _handle_reset_operation
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


def test_pet_3d_rejects_surface_and_remove_bed_operations(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    monkeypatch.setattr("app.services.viewer.operations.compat.series_registry.get", lambda *_args, **_kwargs: series)
    view = ViewRecord(view_id="pet-3d", series_id=series.series_id, view_type="3D")

    service._handle_render_3d_mode(
        view,
        ViewOperationRequest(viewId=view.view_id, opType="render3dMode", render3dMode="surface"),
    )
    assert view.render_3d_mode == "volume"

    service._handle_surface_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="surfaceConfig",
            surfaceConfig=SurfaceRenderConfig(isoValue=500.0),
        ),
    )
    assert view.render_3d_mode == "volume"
    assert view.surface_render_config is None

    service._handle_volume_render_options(
        view,
        ViewOperationRequest(viewId=view.view_id, opType="volumeRenderOptions", removeBed=True),
    )
    assert view.volume_remove_bed is False


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
    assert result.meta.pet_info.pseudocolor_preset == "hotiron"
    assert result.meta.pet_info.pet_window_min == pytest.approx(0.0)
    assert result.meta.pet_info.pet_window_max == pytest.approx(np.percentile(volume, 99.5), abs=0.01)
    suv_option = next(option for option in result.meta.pet_info.unit_options if option.unit == "SUVbw")
    assert suv_option.available is True
    assert suv_option.auto_window_min == pytest.approx(0.0)
    assert suv_option.auto_window_max == pytest.approx(result.meta.pet_info.auto_window_max)
    assert suv_option.control_window_max is not None


def test_pet_view_resizes_from_initial_auto_fit_and_hover_maps_inside_image(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.arange(3 * 70 * 140, dtype=np.float32).reshape((3, 70, 140))
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    hover_cached = SimpleNamespace(dataset=_dataset("GML"), source_pixels=volume[1])
    monkeypatch.setattr("app.services.viewer.interaction.compat.dicom_cache.get", lambda *_args, **_kwargs: hover_cached)

    view = ViewRecord(view_id="pet-view", series_id=series.series_id, view_type="PET", width=140, height=70)
    service._initialize_pet_viewport(view)
    view.is_initialized = True

    assert view.zoom == pytest.approx(0.98)

    monkeypatch.setattr("app.services.viewer_service.view_registry.get", lambda *_args, **_kwargs: view)

    service.set_view_size(
        ViewSetSizeRequest(
            viewId=view.view_id,
            opType="setSize",
            size={"width": 1400, "height": 700},
        )
    )

    # PET fit must be allowed above the former 8x PET cap.  A 140 × 70
    # acquisition in a 1400 × 700 viewport has a 10x contain fit; retain a
    # small edge-safe margin rather than silently under-fitting it.
    assert view.zoom == pytest.approx(9.8)
    assert view.offset_x == pytest.approx(0.0)
    assert view.offset_y == pytest.approx(0.0)

    hover = service.handle_view_hover(ViewHoverRequest(viewId=view.view_id, x=0.5, y=0.5))
    assert 1 <= hover.row <= volume.shape[1]
    assert 1 <= hover.col <= volume.shape[2]
    assert hover.display_text is not None
    assert "SUVbw" in hover.display_text


def test_initial_pet_config_before_size_does_not_leave_standalone_view_at_one_x(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((3, 70, 140), dtype=np.float32)
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    view = ViewRecord(view_id="pet-view", series_id=series.series_id, view_type="PET")

    assert service._handle_pet_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="petConfig",
            pseudocolorPreset="hotiron",
            petWindowMin=0.0,
            petWindowMax=4.49,
        ),
    )
    assert view.is_initialized is False

    monkeypatch.setattr("app.services.viewer_service.view_registry.get", lambda *_args, **_kwargs: view)
    service.set_view_size(
        ViewSetSizeRequest(
            viewId=view.view_id,
            opType="setSize",
            size={"width": 1400, "height": 700},
        )
    )

    assert view.is_initialized is True
    assert view.zoom == pytest.approx(9.8)


def test_pet_auto_fit_keeps_a_safe_margin_for_standalone_and_mpr_views(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((104, 70, 140), dtype=np.float32)
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    monkeypatch.setattr(
        service,
        "_get_series_volume_geometry",
        lambda _series, shape: build_identity_geometry(tuple(int(value) for value in shape)),
    )
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda _series: None)

    standalone = ViewRecord(
        view_id="pet-view",
        series_id=series.series_id,
        view_type="PET",
        width=1400,
        height=700,
    )
    service._initialize_pet_viewport(standalone)
    assert standalone.zoom == pytest.approx(9.8)

    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    mpr = ViewRecord(
        view_id="pet-mpr-ax",
        series_id=series.series_id,
        view_type="MPR",
        view_group=group,
        width=1400,
        height=700,
    )
    mpr_result = service._render_mpr_view(mpr)
    assert mpr.zoom == pytest.approx(9.8)
    assert group.pet_unit == "SUVbw"
    assert mpr_result.meta.pet_info is not None
    assert mpr_result.meta.pet_info.pet_unit == "SUVbw"


def test_pet_auto_fit_can_exceed_legacy_ten_x_limit(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((3, 70, 140), dtype=np.float32)
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)

    view = ViewRecord(
        view_id="pet-view",
        series_id=series.series_id,
        view_type="PET",
        width=2800,
        height=1400,
    )

    service._initialize_pet_viewport(view)

    # A 140 × 70 PET slice needs a 20x physical contain fit here.  The 2%
    # safety margin must not be clipped to the former 8x/10x limits.
    assert view.zoom == pytest.approx(19.6)


def test_pet_render_uses_pixel_spacing_for_the_same_fit_geometry(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((3, 140, 70), dtype=np.float32)
    _patch_pet_render_dependencies(monkeypatch, service, series, volume, stub_renderer=False)

    dataset = _dataset("GML")
    dataset.PixelSpacing = [0.5, 0.5]
    cached = SimpleNamespace(dataset=dataset, source_pixels=volume[0])
    monkeypatch.setattr(service, "_get_indexed_instance_and_cache", lambda *_args, **_kwargs: (series.instances[1], cached))
    monkeypatch.setattr(service, "_get_reference_instance_and_cache", lambda *_args, **_kwargs: (series.instances[0], cached))

    captured_context: RenderContext | None = None

    def capture_context(context: RenderContext) -> Image.Image:
        nonlocal captured_context
        captured_context = context
        return Image.new("RGB", (context.view.width or 1, context.view.height or 1))

    monkeypatch.setattr("app.services.viewer_service.layered_renderer.render", capture_context)
    view = ViewRecord(view_id="pet-view", series_id=series.series_id, view_type="PET", width=1400, height=700)

    service._render_pet_view(view)

    assert captured_context is not None
    matrix = captured_context.image_transform.matrix
    # The 70 × 140 pixels are 35 × 70 mm.  With the physical contain fit,
    # their mapped height stays inside the 700 px canvas instead of being
    # magnified by the missing 0.5 mm pixel spacing factor.
    top = float(matrix[1, 2])
    bottom = top + 140 * float(matrix[1, 1])
    assert top >= 0.0
    assert bottom <= 700.0


def test_each_pet_mpr_plane_uses_its_own_physical_contain_fit(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((104, 140, 70), dtype=np.float32)
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    monkeypatch.setattr(
        service,
        "_get_series_volume_geometry",
        lambda _series, shape: build_identity_geometry(tuple(int(value) for value in shape)),
    )
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda _series: None)

    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    views = [
        ViewRecord(view_id="pet-mpr-ax", series_id=series.series_id, view_type="MPR", view_group=group, width=1400, height=700),
        ViewRecord(view_id="pet-mpr-cor", series_id=series.series_id, view_type="COR", view_group=group, width=1400, height=700),
        ViewRecord(view_id="pet-mpr-sag", series_id=series.series_id, view_type="SAG", view_group=group, width=1400, height=700),
    ]

    for view in views:
        service._render_mpr_view(view)

    # Axial is 140 × 70, coronal is 104 × 70, sagittal is 104 × 140.
    # Each viewport must fit its own reslice geometry rather than inheriting a
    # 1x or another plane's transform.
    assert views[0].zoom == pytest.approx(4.9)
    assert views[1].zoom == pytest.approx(6.596153846)
    assert views[2].zoom == pytest.approx(6.596153846)


def test_pet_config_before_first_mpr_render_does_not_skip_geometry_initialization(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((104, 140, 70), dtype=np.float32)
    volume[52, 70, 35] = 4.0
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    monkeypatch.setattr(
        service,
        "_get_series_volume_geometry",
        lambda _series, shape: build_identity_geometry(tuple(int(value) for value in shape)),
    )
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda _series: None)

    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    view = ViewRecord(
        view_id="pet-mpr-ax",
        series_id=series.series_id,
        view_type="MPR",
        view_group=group,
        width=240,
        height=240,
    )

    changed = service._handle_pet_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="petConfig",
            pseudocolorPreset="hotiron",
            petWindowMin=0.0,
            petWindowMax=3.2,
        ),
    )

    assert changed is True
    assert view.is_initialized is False
    assert group.axial_index == 0

    result = service._render_mpr_view(view)

    assert view.is_initialized is True
    assert group.axial_index == volume.shape[0] // 2
    assert group.coronal_index == volume.shape[1] // 2
    assert group.sagittal_index == volume.shape[2] // 2
    assert result.meta.slice_info.current == volume.shape[0] // 2
    assert result.meta.pet_info is not None
    assert result.meta.pet_info.pseudocolor_preset == "hotiron"
    assert result.meta.pet_info.pet_window_max == pytest.approx(3.2)
    assert result.meta.corner_info is not None
    assert any(line.startswith("PET:") for line in result.meta.corner_info.bottom_left)
    assert all(not line.strip().upper().startswith("W:") for line in result.meta.corner_info.bottom_left)
    assert result.meta.corner_info.tags["windowLevel"][0].startswith("PET:")


def test_pet_mpr_zoom_reset_refits_to_initial_viewport_zoom(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((104, 140, 70), dtype=np.float32)
    volume[52, 70, 35] = 4.0
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    monkeypatch.setattr(
        service,
        "_get_series_volume_geometry",
        lambda _series, shape: build_identity_geometry(tuple(int(value) for value in shape)),
    )
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda _series: None)

    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    view = ViewRecord(
        view_id="pet-mpr-ax",
        series_id=series.series_id,
        view_type="MPR",
        view_group=group,
        width=240,
        height=240,
    )

    service._render_mpr_view(view)
    initial_fit_zoom = float(view.zoom)
    view.zoom = 1.0

    assert service._reset_view_zoom_state(view, series) is True

    assert view.zoom == pytest.approx(initial_fit_zoom)
    assert view.zoom != pytest.approx(1.0)


def test_pet_mpr_view_reset_preserves_active_viewport_and_is_idempotent(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((104, 140, 70), dtype=np.float32)
    volume[52, 70, 35] = 4.0
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    monkeypatch.setattr(
        service,
        "_get_series_volume_geometry",
        lambda _series, shape: build_identity_geometry(tuple(int(value) for value in shape)),
    )
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda _series: None)

    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    view = ViewRecord(
        view_id="pet-mpr-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=group,
        width=240,
        height=240,
    )

    service._render_mpr_view(view)
    group.active_viewport = MPR_VIEWPORT_CORONAL

    service._reset_view(view)
    first_render = service._render_mpr_view(view)
    first_cursor = group.mpr_cursor
    assert first_cursor is not None
    assert first_render.meta.mpr_crosshair is not None
    first_snapshot = (
        group.active_viewport,
        group.axial_index,
        group.coronal_index,
        group.sagittal_index,
        tuple(first_cursor.center_world),
        tuple(first_cursor.reference_center_world),
        tuple(tuple(row) for row in first_cursor.orientation_world),
        first_render.meta.mpr_crosshair.center_x,
        first_render.meta.mpr_crosshair.center_y,
    )

    service._reset_view(view)
    second_render = service._render_mpr_view(view)
    second_cursor = group.mpr_cursor
    assert second_cursor is not None
    assert second_render.meta.mpr_crosshair is not None
    second_snapshot = (
        group.active_viewport,
        group.axial_index,
        group.coronal_index,
        group.sagittal_index,
        tuple(second_cursor.center_world),
        tuple(second_cursor.reference_center_world),
        tuple(tuple(row) for row in second_cursor.orientation_world),
        second_render.meta.mpr_crosshair.center_x,
        second_render.meta.mpr_crosshair.center_y,
    )

    assert first_snapshot == second_snapshot
    assert group.active_viewport == MPR_VIEWPORT_CORONAL


def test_pet_mpr_auto_window_preserves_quantitative_ranges_below_one() -> None:
    service = ViewerService()
    series = _series()
    volume = np.linspace(0.0, 0.08, num=8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    group.pet_unit = "SUVbw"
    view = ViewRecord(
        view_id="pet-mpr-ax",
        series_id=series.series_id,
        view_type="MPR",
        view_group=group,
    )

    service._reset_mpr_view_window(view, series, volume)

    assert 0.0 < view.window_width < 0.1
    assert view.window_center == pytest.approx(view.window_width / 2.0)


def test_pet_mpr_reset_embeds_pet_config_before_first_reset_render(monkeypatch) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((104, 140, 70), dtype=np.float32)
    volume[52, 70, 35] = 4.0
    _patch_pet_render_dependencies(monkeypatch, service, series, volume)
    monkeypatch.setattr(
        service,
        "_get_series_volume_geometry",
        lambda _series, shape: build_identity_geometry(tuple(int(value) for value in shape)),
    )
    monkeypatch.setattr(service, "_get_series_patient_transform", lambda _series: None)

    group = ViewGroupRecord(group_id="pet-mpr", group_type="mpr", series_id=series.series_id)
    view = ViewRecord(
        view_id="pet-mpr-cor",
        series_id=series.series_id,
        view_type="COR",
        view_group=group,
        width=240,
        height=240,
    )
    service._render_mpr_view(view)
    group.active_viewport = MPR_VIEWPORT_CORONAL
    group.pet_pseudocolor_preset = "bwinverse"
    view.pseudocolor_preset = "bwinverse"

    decision = _handle_reset_operation(
        service,
        view,
        series,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="reset",
            subOpType="view",
            pseudocolorPreset="rainbow",
            petWindowMin=0.0,
            petWindowMax=2.5,
        ),
        True,
    )

    assert decision.mode == "broadcast"
    assert group.active_viewport == MPR_VIEWPORT_CORONAL
    assert group.pet_pseudocolor_preset == "rainbow"
    assert view.pseudocolor_preset == "rainbow"
    assert view.window_width == pytest.approx(2.5)
    assert view.window_center == pytest.approx(1.25)

    result = service._render_mpr_view(view)

    assert result.meta.pet_info is not None
    assert result.meta.pet_info.pseudocolor_preset == "rainbow"
    assert result.meta.pet_info.pet_window_min == pytest.approx(0.0)
    assert result.meta.pet_info.pet_window_max == pytest.approx(2.5)


def test_scale_bar_uses_shorter_length_when_ten_cm_exceeds_viewport() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="scale", series_id="pet", view_type="MPR", width=120, height=120)
    view.zoom = 10.0
    image_transform = viewport_transformer.build_image_to_canvas_transform(
        image_width=100,
        image_height=100,
        canvas_width=120,
        canvas_height=120,
        view=view,
    )

    scale_bar = service._build_scale_bar_info(view, image_transform, (1.0, 1.0))

    assert scale_bar is not None
    assert scale_bar.label != "10 cm"
    assert 8.0 <= scale_bar.length_norm * view.width <= view.width - 32.0


@pytest.mark.parametrize(
    ("preset", "expected_background"),
    [
        ("bw", (0, 0, 0)),
        ("bwinverse", (255, 255, 255)),
        ("hotiron", (0, 0, 0)),
        ("pet", (0, 0, 0)),
        ("rainbow", (51, 0, 102)),
    ],
)
def test_render_pet_view_uses_the_active_lut_zero_colour_for_canvas_padding(
    monkeypatch,
    preset: str,
    expected_background: tuple[int, int, int],
) -> None:
    service = ViewerService()
    series = _series()
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    volume[:, 1:3, 2:4] = 10.0
    _patch_pet_render_dependencies(monkeypatch, service, series, volume, stub_renderer=False)
    captured: dict[str, tuple[int, int, int]] = {}

    def render_with_capture(context: RenderContext) -> Image.Image:
        assert isinstance(context.background_cval, tuple)
        captured["background_cval"] = context.background_cval
        return Image.new("RGB", (5, 4))

    monkeypatch.setattr("app.services.viewer_service.layered_renderer.render", render_with_capture)

    view = ViewRecord(view_id="pet-view", series_id=series.series_id, view_type="PET", width=96, height=96)
    view.pseudocolor_preset = preset
    view.is_initialized = True
    result = service._render_pet_view(view)

    assert result.meta.pet_info is not None
    assert captured["background_cval"] == expected_background


@pytest.mark.parametrize(
    ("preset", "expected_background"),
    [
        ("bw", (0, 0, 0)),
        ("bwinverse", (255, 255, 255)),
        ("hotiron", (0, 0, 0)),
        ("pet", (0, 0, 0)),
        ("rainbow", (51, 0, 102)),
    ],
)
def test_pet_canvas_padding_is_rendered_in_the_active_lut_zero_colour(
    preset: str,
    expected_background: tuple[int, int, int],
) -> None:
    view = ViewRecord(view_id="pet-view", series_id="pet", view_type="PET", width=24, height=24)
    view.pseudocolor_preset = preset
    view.window_width = 10.0
    view.window_center = 5.0
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
            source_pixels=np.full((4, 4), 10.0, dtype=np.float32),
            pixel_min=0.0,
            pixel_max=10.0,
            image_transform=image_transform,
            background_cval=pseudocolor_background_color(preset),
        )
    ).convert("RGB")
    pixels = np.asarray(image)

    assert tuple(pixels[0, 0]) == expected_background


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


def test_pet_standalone_keeps_low_uptake_pixels_for_lut_mapping() -> None:
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
    assert np.array_equal(prepared_pixels, source_pixels)


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
    assert view.pseudocolor_preset == "rainbow"


def test_pet_display_range_update_does_not_change_control_ceiling() -> None:
    service = ViewerService()
    view = ViewRecord(
        view_id="pet-view",
        series_id="pet",
        view_type="PET",
        width=128,
        height=96,
        pet_control_window_max=30.0,
    )
    view.window_width = 4.0
    view.window_center = 2.0

    changed = service._handle_pet_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType="petConfig",
            petWindowMin=0.0,
            petWindowMax=8.0,
        ),
    )

    assert changed is True
    assert view.window_width == pytest.approx(8.0)
    assert view.window_center == pytest.approx(4.0)
    assert view.pet_control_window_max == pytest.approx(30.0)
