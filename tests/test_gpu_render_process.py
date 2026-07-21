from __future__ import annotations

import numpy as np

from app.services.volume_rendering.contracts import VolumeRenderRequest
from app.services.volume_rendering.gpu_render_process import (
    _SharedVolumeStore,
    GpuRenderProcessClient,
    _is_surface_command,
)


def _request(volume: np.ndarray) -> VolumeRenderRequest:
    return VolumeRenderRequest(
        view_id="view-1",
        volume=volume,
        spacing_xyz=(1.0, 1.0, 1.0),
        canvas_width=256,
        canvas_height=256,
        window_width=400.0,
        window_center=40.0,
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
        volume_token="series-token",
    )


def test_shared_volume_store_registers_integral_ct_as_int16_and_reuses_block(monkeypatch) -> None:
    class FakeSharedMemory:
        counter = 0

        def __init__(self, *, create: bool, size: int) -> None:
            assert create
            FakeSharedMemory.counter += 1
            self.name = f"test-{FakeSharedMemory.counter}"
            self.buf = bytearray(size)

        def close(self) -> None:
            return None

        def unlink(self) -> None:
            return None

    monkeypatch.setattr(
        "app.services.volume_rendering.gpu_render_process.SharedMemory",
        FakeSharedMemory,
    )
    store = _SharedVolumeStore(max_bytes=64 * 1024 * 1024)
    volume = np.array([[[-1000.0, 40.0, 3071.0]]], dtype=np.float32)
    try:
        first = store.register(volume, "series-token")
        second = store.register(volume, "series-token")

        assert first == second
        assert np.dtype(str(first["dtype"])) == np.dtype(np.int16)
        assert int(first["nbytes"]) == volume.size * np.dtype(np.int16).itemsize
    finally:
        store.close()


def test_process_request_serialization_excludes_volume() -> None:
    request = _request(np.zeros((2, 2, 2), dtype=np.int16))

    payload = GpuRenderProcessClient._serialize_request(request)

    assert "volume" not in payload
    assert payload["view_id"] == "view-1"
    assert payload["volume_token"] == "series-token"


def test_gpu_worker_routes_render_surface_to_surface_request() -> None:
    assert _is_surface_command("render_surface") is True
    assert _is_surface_command("surface_trackball") is True
    assert _is_surface_command("render_volume") is False
    assert _is_surface_command("volume_trackball") is False
