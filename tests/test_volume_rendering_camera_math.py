from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.models.viewer import ViewRecord
from app.core import VIEW_OP_TYPE_ROTATE_3D
from app.schemas.view import ViewOperationRequest
from app.services.volume_rendering.camera_math import (
    apply_trackball_delta_to_quaternion,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from app.services.volume_rendering.contracts import VolumeRenderRequest
from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer
from app.services.viewer_service import ViewerService


def test_camera_quaternion_math_round_trips_rotation_matrix() -> None:
    quaternion = normalize_quaternion((0.2, 0.3, 0.1, 0.9))
    matrix = quaternion_to_rotation_matrix(quaternion)

    assert rotation_matrix_to_quaternion(matrix) == pytest.approx(quaternion)


def test_trackball_delta_updates_quaternion_without_vtk() -> None:
    updated = apply_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=24.0,
        delta_y_pixels=-12.0,
        canvas_width=240.0,
        canvas_height=120.0,
    )

    assert updated != pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert np.linalg.norm(np.asarray(updated)) == pytest.approx(1.0)


def test_trackball_delta_uses_screen_grab_model_direction() -> None:
    right_drag = apply_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=24.0,
        delta_y_pixels=0.0,
        canvas_width=240.0,
        canvas_height=120.0,
    )
    up_drag = apply_trackball_delta_to_quaternion(
        (0.0, 0.0, 0.0, 1.0),
        delta_x_pixels=0.0,
        delta_y_pixels=-12.0,
        canvas_width=240.0,
        canvas_height=120.0,
    )

    model_forward_after_right_drag = quaternion_to_rotation_matrix(right_drag) @ np.asarray([0.0, 1.0, 0.0])
    model_forward_after_up_drag = quaternion_to_rotation_matrix(up_drag) @ np.asarray([0.0, 1.0, 0.0])
    assert model_forward_after_right_drag[0] > 0.0
    assert model_forward_after_up_drag[2] > 0.0


def _build_volume_request(view_id: str) -> VolumeRenderRequest:
    return VolumeRenderRequest(
        view_id=view_id,
        volume=np.zeros((4, 5, 6), dtype=np.float32),
        spacing_xyz=(0.7, 0.8, 1.2),
        canvas_width=200,
        canvas_height=100,
        window_width=400.0,
        window_center=40.0,
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
    )


def test_volume_renderer_session_lru_finalizes_oldest_session(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    finalized = []

    def fake_create_session(volume, spacing_xyz, volume_token, fast_preview):
        del fast_preview
        return SimpleNamespace(
            volume_token=volume_token,
            render_window=SimpleNamespace(Finalize=lambda: finalized.append(len(finalized))),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    for index in range(9):
        request = _build_volume_request(f"volume-view-{index}")
        renderer._get_or_create_session(request, request.volume)

    assert len(renderer._sessions) == 8
    assert len(finalized) == 1
    assert ("volume-view-0", "final") not in renderer._sessions


def test_volume_renderer_drop_session_finalizes_final_and_preview() -> None:
    renderer = VtkVolumeRenderer()
    finalized = []
    renderer._sessions[("view-a", "final")] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-a-final"))
    )
    renderer._sessions[("view-a", "preview")] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-a-preview"))
    )
    renderer._sessions[("view-b", "final")] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-b"))
    )

    renderer._drop_session_in_executor("view-a")

    assert sorted(finalized) == ["view-a-final", "view-a-preview"]
    assert list(renderer._sessions.keys()) == [("view-b", "final")]


def test_volume_fast_preview_downsamples_source_volume_and_adjusts_spacing() -> None:
    volume = np.zeros((96, 512, 384), dtype=np.float32)

    sampled, spacing = VtkVolumeRenderer._downsample_preview_volume(volume, (0.5, 0.6, 1.2))

    assert sampled.shape == (24, 128, 96)
    assert max(sampled.shape) <= 144
    assert spacing == pytest.approx((2.0, 2.4, 4.8))


def test_volume_preview_and_final_use_separate_session_keys(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    created: list[bool] = []

    def fake_create_session(volume, spacing_xyz, volume_token, fast_preview):
        del volume, spacing_xyz
        created.append(bool(fast_preview))
        return SimpleNamespace(
            volume_token=volume_token,
            render_window=SimpleNamespace(Finalize=lambda: None),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    request = _build_volume_request("view-a")
    renderer._get_or_create_session(request, request.volume)
    renderer._get_or_create_session(replace(request, fast_preview=True), request.volume)
    renderer._get_or_create_session(request, request.volume)

    assert created == [False, True]
    assert list(renderer._sessions.keys()) == [("view-a", "preview"), ("view-a", "final")]


def test_volume_preview_session_uses_stable_volume_token(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    created: list[object] = []

    def fake_create_session(volume, spacing_xyz, volume_token, fast_preview):
        del volume, spacing_xyz, fast_preview
        created.append(volume_token[0])
        return SimpleNamespace(
            volume_token=volume_token,
            render_window=SimpleNamespace(Finalize=lambda: None),
        )

    monkeypatch.setattr(renderer, "_create_session", fake_create_session)

    request = replace(_build_volume_request("view-a"), fast_preview=True, volume_token="series-token")
    renderer._get_or_create_session(request, np.zeros_like(request.volume))
    renderer._get_or_create_session(request, np.ones_like(request.volume))

    assert created == ["series-token"]


def test_volume_renderer_reuses_transfer_and_sampling_configuration(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    request = _build_volume_request("view-a")
    calls: list[str] = []
    session = SimpleNamespace(
        canvas_size=(request.canvas_width, request.canvas_height),
        transfer_function_token=None,
        sampling_token=None,
    )

    monkeypatch.setattr(renderer, "_update_transfer_functions", lambda *args, **kwargs: calls.append("transfer"))
    monkeypatch.setattr(renderer, "_update_sampling", lambda *args: calls.append("sampling"))
    monkeypatch.setattr(renderer, "_update_camera", lambda *args: calls.append("camera"))

    renderer._configure_session(session, request)
    renderer._configure_session(session, request)
    renderer._configure_session(session, replace(request, fast_preview=True))

    assert calls == ["transfer", "sampling", "camera", "camera", "sampling", "camera"]


def test_volume_preview_request_caps_canvas_without_changing_final_or_gesture_size() -> None:
    service = ViewerService()
    view = ViewRecord(view_id="view-a", series_id="series-a", view_type="3D")
    view.width = 1600
    view.height = 1000
    view.zoom = 1.8
    view.offset_x = 80.0
    view.offset_y = -40.0
    volume = np.zeros((4, 5, 6), dtype=np.float32)

    preview = service._build_volume_render_request(
        view,
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        fast_preview=True,
    )
    gesture = service._build_volume_render_request(
        view,
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        fast_preview=True,
        scale_fast_preview_canvas=False,
    )
    final = service._build_volume_render_request(
        view,
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        fast_preview=False,
    )

    assert preview.canvas_width == 560
    assert preview.canvas_height == 350
    assert preview.zoom == pytest.approx(1.8)
    assert preview.offset_x == pytest.approx(28.0)
    assert preview.offset_y == pytest.approx(-14.0)
    assert gesture.canvas_width == 1600
    assert gesture.canvas_height == 1000
    assert gesture.offset_x == pytest.approx(80.0)
    assert gesture.offset_y == pytest.approx(-40.0)
    assert final.canvas_width == 1600
    assert final.canvas_height == 1000


def test_volume_mode_3d_rotation_updates_quaternion_without_vtk(monkeypatch) -> None:
    service = ViewerService()
    view = ViewRecord(
        view_id="volume-view",
        series_id="series-a",
        view_type="3D",
        width=200,
        height=100,
    )
    view.render_3d_mode = "volume"

    def fake_apply_trackball_camera_delta(request, *, delta_x_pixels: float, delta_y_pixels: float):
        del request, delta_x_pixels, delta_y_pixels
        raise AssertionError("volume rotate move should not enter VTK")

    monkeypatch.setattr(
        "app.services.viewer_service.vtk_volume_renderer.apply_trackball_camera_delta",
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
            x=0.58,
            y=0.42,
        ),
    )

    assert view.rotation_quaternion != pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert view.is_initialized is True
