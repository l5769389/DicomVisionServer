from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData

from app.core import VIEW_OP_TYPE_RENDER_3D_MODE, VIEW_OP_TYPE_ROTATE_3D, VIEW_OP_TYPE_SURFACE_CONFIG
from app.models.viewer import SeriesRecord, ViewRecord
from app.schemas.view import SurfaceRenderConfig, ViewOperationRequest
from app.services.surface_render_config import (
    create_adaptive_surface_render_config,
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


def test_surface_render_config_supports_surface_presets() -> None:
    soft_tissue = create_default_surface_render_config("surfacePreset:softTissue")
    high_density = create_default_surface_render_config("high-density")

    assert soft_tissue["preset"] == "softTissue"
    assert soft_tissue["isoValue"] == 85.0
    assert soft_tissue["color"] == "#b86642"
    assert high_density["preset"] == "highDensity"
    assert high_density["isoValue"] == 420.0
    assert high_density["specular"] == 0.46


def test_adaptive_surface_config_uses_ct_hu_and_percentile_fallback() -> None:
    ct_volume = np.concatenate(
        [
            np.full(900, -1000, dtype=np.float32),
            np.linspace(40, 220, 900, dtype=np.float32),
            np.linspace(360, 900, 120, dtype=np.float32),
        ]
    ).reshape(12, 16, 10)
    bone_config = create_adaptive_surface_render_config("bone", ct_volume, modality="CT")
    soft_config = create_adaptive_surface_render_config("softTissue", ct_volume, modality="CT")
    high_density_config = create_adaptive_surface_render_config("highDensity", ct_volume, modality="CT")

    assert 180.0 <= bone_config["isoValue"] <= 520.0
    assert -80.0 <= soft_config["isoValue"] <= 180.0
    assert high_density_config["isoValue"] >= 450.0

    mr_volume = np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape(10, 10, 10)
    mr_config = create_adaptive_surface_render_config("bone", mr_volume, modality="MR")
    cbct_config = create_adaptive_surface_render_config("bone", mr_volume, modality="CBCT")

    assert 0.65 <= mr_config["isoValue"] <= 0.9
    assert 0.65 <= cbct_config["isoValue"] <= 0.9


def test_adaptive_surface_presets_separate_skin_bone_and_dense_metal() -> None:
    ct_volume = np.concatenate(
        [
            np.full(1200, -1000, dtype=np.float32),
            np.linspace(-320, -80, 400, dtype=np.float32),
            np.linspace(20, 180, 700, dtype=np.float32),
            np.linspace(250, 850, 180, dtype=np.float32),
            np.linspace(1200, 2600, 40, dtype=np.float32),
        ]
    ).reshape(18, 14, 10)

    skin = create_adaptive_surface_render_config("softTissue", ct_volume, modality="CT")
    bone = create_adaptive_surface_render_config("bone", ct_volume, modality="CT")
    dense = create_adaptive_surface_render_config("highDensity", ct_volume, modality="CT")

    assert -350.0 <= skin["isoValue"] <= -80.0
    assert 160.0 <= bone["isoValue"] <= 650.0
    assert dense["isoValue"] >= 700.0
    assert skin["isoValue"] < bone["isoValue"] < dense["isoValue"]


def test_surface_mesh_cache_reuses_geometry_and_limits_entries(monkeypatch) -> None:
    renderer = VtkSurfaceRenderer()
    image_data = vtkImageData()
    volume_token = ("volume-token", (8, 8, 8), (1.0, 1.0, 1.0))
    builds: list[float] = []

    def fake_build_surface_mesh(_image_data, config):
        builds.append(float(config["isoValue"]))
        return vtkPolyData()

    monkeypatch.setattr(renderer, "_build_surface_mesh", fake_build_surface_mesh)
    config = create_default_surface_render_config("bone")

    first = renderer._get_or_create_mesh(image_data, volume_token, config)
    second = renderer._get_or_create_mesh(image_data, volume_token, config)

    assert first is second
    assert builds == [float(config["isoValue"])]

    for index in range(9):
        renderer._get_or_create_mesh(
            image_data,
            volume_token,
            {**config, "isoValue": 300.0 + index},
        )

    assert len(renderer._mesh_cache) == 8


def test_surface_material_update_keeps_existing_geometry(monkeypatch) -> None:
    renderer = VtkSurfaceRenderer()
    renderer._executor = None
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    volume[2:5, 2:5, 2:5] = 600.0
    initial_config = create_default_surface_render_config("bone")
    progress_messages: list[dict[str, object]] = []
    request = _build_surface_request()
    request = replace(
        request,
        volume=volume,
        surface_config=initial_config,
        volume_token="material-volume",
        progress_callback=progress_messages.append,
    )
    session = renderer._get_or_create_session(request, volume)
    initial_geometry_token = session.geometry_token
    assert [message["message"] for message in progress_messages] == [
        "正在提取 Surface 等值面...",
        "Surface 表面优化完成",
    ]

    def fail_mesh_rebuild(*_args, **_kwargs):
        raise AssertionError("material-only updates must not rebuild the surface mesh")

    monkeypatch.setattr(renderer, "_get_or_create_mesh", fail_mesh_rebuild)
    updated_config = {
        **initial_config,
        "color": "#336699",
        "ambient": 0.41,
    }
    updated = renderer._get_or_create_session(replace(request, surface_config=updated_config), volume)

    assert updated is session
    assert updated.geometry_token == initial_geometry_token
    assert updated.config_token != VtkSurfaceRenderer._build_config_token(initial_config)
    assert updated.actor.GetProperty().GetColor() == pytest.approx((0x33 / 255.0, 0x66 / 255.0, 0x99 / 255.0))


def test_surface_session_reuses_cached_mesh_when_switching_back_to_a_preset(monkeypatch) -> None:
    renderer = VtkSurfaceRenderer()
    renderer._executor = None
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    builds: list[float] = []

    def fake_build_surface_mesh(_image_data, config):
        builds.append(float(config["isoValue"]))
        return vtkPolyData()

    monkeypatch.setattr(renderer, "_build_surface_mesh", fake_build_surface_mesh)
    bone = create_default_surface_render_config("bone")
    soft_tissue = create_default_surface_render_config("softTissue")
    request = replace(
        _build_surface_request(),
        volume=volume,
        volume_token="preset-switch-volume",
        surface_config=bone,
    )

    first_session = renderer._get_or_create_session(request, volume)
    soft_session = renderer._get_or_create_session(replace(request, surface_config=soft_tissue), volume)
    restored_session = renderer._get_or_create_session(request, volume)

    assert first_session is soft_session is restored_session
    assert builds == [float(bone["isoValue"]), float(soft_tissue["isoValue"])]


def test_surface_fast_preview_preserves_mesh_source_and_uses_lower_render_size() -> None:
    volume = np.zeros((64, 512, 384), dtype=np.float32)

    sampled, spacing = VtkSurfaceRenderer._prepare_surface_volume(volume, (0.5, 0.6, 1.2))

    assert sampled is volume
    assert sampled.shape == volume.shape
    assert spacing == pytest.approx((0.5, 0.6, 1.2))

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


def test_surface_preview_and_final_share_session_key(monkeypatch) -> None:
    renderer = VtkSurfaceRenderer()
    created: list[object] = []

    def fake_create_session(volume, spacing_xyz, volume_token, config, config_token):
        del volume, spacing_xyz, config
        created.append(config_token)
        return SimpleNamespace(
            volume_token=volume_token,
            config_token=config_token,
            render_window=SimpleNamespace(Finalize=lambda: None),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    request = _build_surface_request("surface-view")
    renderer._get_or_create_session(request, request.volume)
    renderer._get_or_create_session(replace(request, fast_preview=True), request.volume)
    renderer._get_or_create_session(request, request.volume)

    assert len(created) == 1
    assert list(renderer._sessions.keys()) == [("surface-view", "shared")]


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
    assert view.surface_render_config_source == "manual"
    assert view.is_initialized is True

    service._handle_surface_config(
        view,
        ViewOperationRequest(
            viewId=view.view_id,
            opType=VIEW_OP_TYPE_SURFACE_CONFIG,
            subOpType="surfacePreset:softTissue",
            surfaceConfig=SurfaceRenderConfig(preset="softTissue"),
        ),
    )

    assert view.surface_render_config["preset"] == "softTissue"
    assert view.surface_render_config_source == "preset"
    assert view.surface_render_config_token is None


def test_surface_config_resolver_adapts_preset_and_preserves_manual_config() -> None:
    service = ViewerService()
    series = _build_series()
    volume = np.concatenate(
        [
            np.full(500, -1000, dtype=np.float32),
            np.linspace(60, 240, 500, dtype=np.float32),
        ]
    ).reshape(10, 10, 10)
    view = _build_surface_view()
    view.surface_render_config = create_default_surface_render_config("bone")
    view.surface_render_config_source = "preset"
    view.surface_render_config_token = None

    adapted = service._resolve_surface_render_config_for_render(
        view,
        series=series,
        volume=volume,
        volume_token="surface-volume-token",
    )

    assert adapted["preset"] == "bone"
    assert adapted["isoValue"] != create_default_surface_render_config("bone")["isoValue"]
    assert view.surface_render_config_source == "preset"
    assert view.surface_render_config_token is not None

    view.surface_render_config = {
        **create_default_surface_render_config("highDensity"),
        "isoValue": 720.0,
    }
    view.surface_render_config_source = "manual"
    view.surface_render_config_token = "stale-token"

    manual = service._resolve_surface_render_config_for_render(
        view,
        series=series,
        volume=volume,
        volume_token="surface-volume-token",
    )

    assert manual["preset"] == "highDensity"
    assert manual["isoValue"] == 720.0
    assert view.surface_render_config_source == "manual"
    assert view.surface_render_config_token is None


def test_surface_mode_3d_rotation_updates_quaternion_without_vtk(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(
        view_id="surface-view",
        series_id="surface-series",
        view_type="3D",
        width=200,
        height=100,
    )
    view.render_3d_mode = "surface"
    view.surface_render_config = create_default_surface_render_config("bone")

    def fake_apply_trackball_camera_delta(request, *, delta_x_pixels: float, delta_y_pixels: float):
        del request, delta_x_pixels, delta_y_pixels
        raise AssertionError("rotate move should update quaternion without entering VTK")

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

    assert view.rotation_quaternion != pytest.approx((0.0, 0.0, 0.0, 1.0))
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
    progress_messages: list[dict[str, object]] = []

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_surface_renderer.warm_preview_session",
        lambda request: warm_requests.append(request),
    )

    service._render_3d_view(view, fast_preview=False, progress_callback=progress_messages.append)

    assert len(warm_requests) == 1
    assert warm_requests[0].view_id == view.view_id
    assert warm_requests[0].fast_preview is False
    assert any(message.get("message") == "正在准备 Surface 数据..." for message in progress_messages)


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

    def fake_create_session(volume, spacing_xyz, volume_token, config, config_token):
        del volume, spacing_xyz, config
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
    assert ("surface-view-0", "shared") not in renderer._sessions


def test_surface_drop_session_cleans_matching_view_session_and_warm_keys() -> None:
    renderer = VtkSurfaceRenderer()
    finalized = []
    renderer._sessions[("view-a", "shared")] = SimpleNamespace(render_window=SimpleNamespace(Finalize=lambda: finalized.append("a-shared")))
    renderer._sessions[("view-b", "shared")] = SimpleNamespace(render_window=SimpleNamespace(Finalize=lambda: finalized.append("b-shared")))
    renderer._warm_preview_keys = {
        ("view-a", "warm"),
        ("view-b", "warm"),
    }

    renderer._drop_session_in_executor("view-a")

    assert finalized == ["a-shared"]
    assert list(renderer._sessions.keys()) == [("view-b", "shared")]
    assert renderer._warm_preview_keys == {("view-b", "warm")}
