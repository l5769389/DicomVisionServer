from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.volume_rendering.camera_math import (
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from app.services.volume_rendering.contracts import VolumeRenderRequest
from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer


def test_camera_quaternion_math_round_trips_rotation_matrix() -> None:
    quaternion = normalize_quaternion((0.2, 0.3, 0.1, 0.9))
    matrix = quaternion_to_rotation_matrix(quaternion)

    assert rotation_matrix_to_quaternion(matrix) == pytest.approx(quaternion)


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

    def fake_create_session(volume, spacing_xyz, volume_token):
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
    assert "volume-view-0" not in renderer._sessions


def test_volume_renderer_drop_session_finalizes_matching_view() -> None:
    renderer = VtkVolumeRenderer()
    finalized = []
    renderer._sessions["view-a"] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-a"))
    )
    renderer._sessions["view-b"] = SimpleNamespace(
        render_window=SimpleNamespace(Finalize=lambda: finalized.append("view-b"))
    )

    renderer._drop_session_in_executor("view-a")

    assert finalized == ["view-a"]
    assert list(renderer._sessions.keys()) == ["view-b"]


def test_volume_renderer_reuses_transfer_and_sampling_configuration(monkeypatch) -> None:
    renderer = VtkVolumeRenderer()
    request = _build_volume_request("view-a")
    calls: list[str] = []
    session = SimpleNamespace(
        canvas_size=(request.canvas_width, request.canvas_height),
        transfer_function_token=None,
        sampling_token=None,
    )

    monkeypatch.setattr(renderer, "_update_transfer_functions", lambda *args: calls.append("transfer"))
    monkeypatch.setattr(renderer, "_update_sampling", lambda *args: calls.append("sampling"))
    monkeypatch.setattr(renderer, "_update_camera", lambda *args: calls.append("camera"))

    renderer._configure_session(session, request)
    renderer._configure_session(session, request)
    renderer._configure_session(session, replace(request, fast_preview=True))

    assert calls == ["transfer", "sampling", "camera", "camera", "sampling", "camera"]
