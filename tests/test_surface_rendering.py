from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core import VIEW_OP_TYPE_RENDER_3D_MODE, VIEW_OP_TYPE_ROTATE_3D, VIEW_OP_TYPE_SURFACE_CONFIG
from app.models.viewer import SeriesRecord, ViewRecord
from app.schemas.view import SurfaceRenderConfig, ViewOperationRequest
from app.services.surface_render_config import (
    create_default_surface_render_config,
    normalize_surface_render_config,
)
from app.services.volume_rendering.vtk_surface_renderer import SurfaceRenderRequest, VtkSurfaceRenderer
from app.services.viewer_service import ViewerService


def _build_series() -> SeriesRecord:
    return SeriesRecord(
        series_id="surface-series",
        folder_path=".",
        series_instance_uid="1.2.3.surface",
        study_instance_uid=None,
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="Surface CT",
    )


def test_surface_render_config_normalizes_out_of_range_values() -> None:
    config = normalize_surface_render_config(
        {
            "preset": "surface",
            "isoValue": 9000,
            "smoothing": -0.5,
            "decimation": 3.0,
            "color": "not-a-color",
            "ambient": "0.35",
            "diffuse": None,
            "specular": 4.0,
            "roughness": -1.0,
        }
    )

    assert config["preset"] == "bone"
    assert config["isoValue"] == 4000.0
    assert config["smoothing"] == 0.0
    assert config["decimation"] == 0.9
    assert config["color"] == create_default_surface_render_config()["color"]
    assert config["ambient"] == 0.35
    assert config["diffuse"] == create_default_surface_render_config()["diffuse"]
    assert config["specular"] == 1.0
    assert config["roughness"] == 0.0


def test_surface_fast_preview_uses_lower_cost_volume_and_render_size() -> None:
    volume = np.zeros((64, 512, 384), dtype=np.float32)

    sampled, spacing = VtkSurfaceRenderer._prepare_surface_volume(volume, (0.5, 0.6, 1.2), fast_preview=True)

    assert sampled.shape == (32, 128, 96)
    assert spacing == pytest.approx((2.0, 2.4, 2.4))

    request = SurfaceRenderRequest(
        view_id="surface-view",
        volume=volume,
        spacing_xyz=(0.5, 0.6, 1.2),
        canvas_width=1000,
        canvas_height=800,
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
        fast_preview=True,
    )

    assert VtkSurfaceRenderer._resolve_render_size(request) == (500, 400)
    assert VtkSurfaceRenderer._resolve_render_size(
        SurfaceRenderRequest(
            **{
                **request.__dict__,
                "fast_preview": False,
            }
        )
    ) == (1000, 800)


def test_3d_mode_and_surface_config_handlers_initialize_surface_state() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="surface-view", series_id="surface-series", view_type="3D")

    service._handle_render_3d_mode(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_RENDER_3D_MODE,
            render3dMode="surface",
        ),
    )

    assert view.render_3d_mode == "surface"
    assert view.surface_render_config == create_default_surface_render_config("bone")
    assert view.is_initialized is True

    view.is_initialized = False
    service._handle_surface_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_SURFACE_CONFIG,
            surfaceConfig=SurfaceRenderConfig(
                isoValue=720.0,
                smoothing=0.7,
                decimation=0.2,
                color="#ABCDEF",
            ),
        ),
    )

    assert view.render_3d_mode == "surface"
    assert view.surface_render_config["isoValue"] == 720.0
    assert view.surface_render_config["smoothing"] == 0.7
    assert view.surface_render_config["decimation"] == 0.2
    assert view.surface_render_config["color"] == "#abcdef"
    assert view.is_initialized is True


def test_surface_mode_3d_rotation_uses_surface_renderer(monkeypatch) -> None:
    service = ViewerService()
    series = _build_series()
    volume = np.zeros((5, 6, 7), dtype=np.float32)
    view = ViewRecord(
        view_id="surface-view",
        series_id=series.series_id,
        view_type="3D",
        width=200,
        height=100,
    )
    view.render_3d_mode = "surface"
    view.surface_render_config = create_default_surface_render_config("bone")

    monkeypatch.setattr("app.services.viewer_service.series_registry", SimpleNamespace(get=lambda series_id: series))
    monkeypatch.setattr(service, "_get_series_volume", lambda active_series: volume)
    monkeypatch.setattr(service, "_get_3d_spacing_xyz", lambda active_series: (0.7, 0.8, 1.2))

    calls: dict[str, object] = {}

    def fake_apply_trackball_camera_delta(request, *, delta_x_pixels: float, delta_y_pixels: float):
        calls["request"] = request
        calls["delta"] = (delta_x_pixels, delta_y_pixels)
        return (0.1, 0.2, 0.3, 0.9)

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_surface_renderer.apply_trackball_camera_delta",
        fake_apply_trackball_camera_delta,
    )

    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="start",
            x=0.5,
            y=0.5,
        ),
    )
    service._handle_drag_rotate_3d(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_ROTATE_3D,
            actionType="move",
            x=0.55,
            y=0.45,
        ),
    )

    request = calls["request"]
    assert request.view_id == view.view_id
    assert request.volume is volume
    assert request.spacing_xyz == (0.7, 0.8, 1.2)
    assert request.fast_preview is True
    assert calls["delta"] == pytest.approx((10.0, -5.0))
    assert view.rotation_quaternion == (0.1, 0.2, 0.3, 0.9)
    assert view.is_initialized is True


def _build_surface_view() -> ViewRecord:
    view = ViewRecord(
        view_id="surface-view",
        series_id="surface-series",
        view_type="3D",
        width=200,
        height=100,
    )
    view.render_3d_mode = "surface"
    view.surface_render_config = create_default_surface_render_config("bone")
    view.is_initialized = True
    return view


def _patch_surface_render_dependencies(monkeypatch, service: ViewerService, series: SeriesRecord, volume: np.ndarray) -> None:
    monkeypatch.setattr("app.services.viewer_service.series_registry", SimpleNamespace(get=lambda series_id: series))
    monkeypatch.setattr(service, "_get_series_volume", lambda active_series, progress_callback=None: volume)
    monkeypatch.setattr(service, "_get_3d_spacing_xyz", lambda active_series: (0.7, 0.8, 1.2))
    monkeypatch.setattr(
        "app.services.viewer_service.vtk_surface_renderer.render",
        lambda request: Image.new("RGB", (2, 2), color=(255, 255, 255)),
    )


def test_full_surface_render_warms_preview_session(monkeypatch) -> None:
    service = ViewerService()
    series = _build_series()
    volume = np.zeros((5, 6, 7), dtype=np.float32)
    view = _build_surface_view()
    _patch_surface_render_dependencies(monkeypatch, service, series, volume)
    warm_requests = []

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_surface_renderer.warm_preview_session",
        lambda request: warm_requests.append(request),
    )

    service._render_3d_view(view, fast_preview=False)

    assert len(warm_requests) == 1
    assert warm_requests[0].view_id == view.view_id
    assert warm_requests[0].fast_preview is False


def test_surface_fast_preview_render_does_not_warm_preview_session(monkeypatch) -> None:
    service = ViewerService()
    series = _build_series()
    volume = np.zeros((5, 6, 7), dtype=np.float32)
    view = _build_surface_view()
    _patch_surface_render_dependencies(monkeypatch, service, series, volume)
    warm_requests = []

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_surface_renderer.warm_preview_session",
        lambda request: warm_requests.append(request),
    )

    service._render_3d_view(view, fast_preview=True)

    assert warm_requests == []


def test_surface_preview_warm_failure_does_not_fail_render(monkeypatch) -> None:
    service = ViewerService()
    series = _build_series()
    volume = np.zeros((5, 6, 7), dtype=np.float32)
    view = _build_surface_view()
    _patch_surface_render_dependencies(monkeypatch, service, series, volume)

    def fail_warm_preview_session(request) -> None:
        raise RuntimeError("warm failed")

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_surface_renderer.warm_preview_session",
        fail_warm_preview_session,
    )

    result = service._render_3d_view(view, fast_preview=False)

    assert result.image_bytes


def _build_surface_request(view_id: str = "surface-view", *, fast_preview: bool = False) -> SurfaceRenderRequest:
    return SurfaceRenderRequest(
        view_id=view_id,
        volume=np.zeros((4, 5, 6), dtype=np.float32),
        spacing_xyz=(0.7, 0.8, 1.2),
        canvas_width=200,
        canvas_height=100,
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
        surface_config=create_default_surface_render_config("bone"),
        fast_preview=fast_preview,
    )


def test_surface_preview_warm_deduplicates_pending_requests() -> None:
    renderer = VtkSurfaceRenderer()
    submitted = []

    class FakeFuture:
        def __init__(self) -> None:
            self.callbacks = []

        def add_done_callback(self, callback) -> None:
            self.callbacks.append(callback)

        def result(self) -> None:
            return None

    def fake_submit(fn, request):
        future = FakeFuture()
        submitted.append((fn, request, future))
        return future

    renderer._executor = SimpleNamespace(submit=fake_submit)
    request = _build_surface_request()

    renderer.warm_preview_session(request)
    renderer.warm_preview_session(request)

    assert len(submitted) == 1
    assert submitted[0][1].fast_preview is True

    submitted[0][2].callbacks[0](submitted[0][2])
    renderer.warm_preview_session(request)
    assert len(submitted) == 2


def test_surface_session_lru_finalizes_oldest_session(monkeypatch) -> None:
    renderer = VtkSurfaceRenderer()
    finalized = []

    def fake_create_session(volume, spacing_xyz, volume_token, config, config_token, fast_preview):
        render_window = SimpleNamespace(Finalize=lambda: finalized.append(len(finalized)))
        return SimpleNamespace(
            volume_token=volume_token,
            config_token=config_token,
            render_window=render_window,
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    for index in range(9):
        request = _build_surface_request(view_id=f"surface-view-{index}")
        renderer._get_or_create_session(request, request.volume)

    assert len(renderer._sessions) == 8
    assert len(finalized) == 1
    assert ("surface-view-0", "final") not in renderer._sessions


def test_surface_drop_session_cleans_final_preview_and_warm_keys() -> None:
    renderer = VtkSurfaceRenderer()
    finalized = []
    renderer._sessions[("view-a", "final")] = SimpleNamespace(render_window=SimpleNamespace(Finalize=lambda: finalized.append("a-final")))
    renderer._sessions[("view-a", "preview")] = SimpleNamespace(render_window=SimpleNamespace(Finalize=lambda: finalized.append("a-preview")))
    renderer._sessions[("view-b", "final")] = SimpleNamespace(render_window=SimpleNamespace(Finalize=lambda: finalized.append("b-final")))
    renderer._warm_preview_keys = {
        ("view-a", "warm"),
        ("view-b", "warm"),
    }

    renderer._drop_session_in_executor("view-a")

    assert sorted(finalized) == ["a-final", "a-preview"]
    assert list(renderer._sessions.keys()) == [("view-b", "final")]
    assert renderer._warm_preview_keys == {("view-b", "warm")}
