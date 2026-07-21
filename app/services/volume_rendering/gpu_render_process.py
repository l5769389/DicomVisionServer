from __future__ import annotations

import atexit
from collections import OrderedDict
from dataclasses import fields
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from threading import RLock
from time import perf_counter
import traceback
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.volume_rendering.contracts import SurfaceRenderRequest, VolumeRenderRequest, VtkRenderTimings
from app.services.volume_rendering.volume_dtype import prepare_vtk_volume


logger = get_logger(__name__)


class _SharedVolumeStore:
    def __init__(self, max_bytes: int) -> None:
        self._entries: OrderedDict[tuple[object, ...], tuple[SharedMemory, dict[str, object]]] = OrderedDict()
        self._bytes = 0
        self._max_bytes = max(64 * 1024 * 1024, int(max_bytes))

    def register(self, volume: np.ndarray, volume_token: str | None) -> dict[str, object]:
        source = np.asarray(volume)
        source_key = (
            str(volume_token) if volume_token else f"object:{id(source)}",
            tuple(int(value) for value in source.shape),
            str(source.dtype),
        )
        existing = self._entries.get(source_key)
        if existing is not None:
            self._entries.move_to_end(source_key)
            return existing[1]

        prepared = prepare_vtk_volume(source)
        shared = SharedMemory(create=True, size=int(prepared.nbytes))
        shared_array = np.ndarray(prepared.shape, dtype=prepared.dtype, buffer=shared.buf)
        shared_array[...] = prepared
        descriptor: dict[str, object] = {
            "name": shared.name,
            "shape": tuple(int(value) for value in prepared.shape),
            "dtype": prepared.dtype.str,
            "source_dtype": str(source.dtype),
            "nbytes": int(prepared.nbytes),
        }
        self._entries[source_key] = (shared, descriptor)
        self._bytes += int(prepared.nbytes)
        self._evict_if_needed()
        return descriptor

    def _evict_if_needed(self) -> None:
        while self._bytes > self._max_bytes and len(self._entries) > 1:
            _, (shared, descriptor) = self._entries.popitem(last=False)
            self._bytes -= int(descriptor["nbytes"])
            shared.close()
            shared.unlink()

    def close(self) -> None:
        while self._entries:
            _, (shared, _) = self._entries.popitem(last=False)
            try:
                shared.close()
                shared.unlink()
            except FileNotFoundError:
                pass
        self._bytes = 0


class GpuRenderProcessClient:
    def __init__(self, *, max_shared_memory_bytes: int) -> None:
        self._context = mp.get_context("spawn")
        self._connection: Connection | None = None
        self._process: mp.Process | None = None
        self._lock = RLock()
        self._volumes = _SharedVolumeStore(max_shared_memory_bytes)
        self._diagnostics: dict[str, object] | None = None

    @property
    def diagnostics(self) -> dict[str, object] | None:
        return self._diagnostics

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._process is not None and self._process.is_alive() and self._connection is not None:
                return self._diagnostics or {}
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._process is not None:
                self._process.join(timeout=0.2)
                self._process = None
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_gpu_render_worker_main,
                args=(child,),
                name="dicomvision-vtk-gpu",
                daemon=True,
            )
            process.start()
            child.close()
            ready = parent.recv()
            if ready.get("kind") != "ready":
                process.terminate()
                process.join(timeout=2.0)
                raise RuntimeError(str(ready.get("error") or "GPU render worker failed to start"))
            self._connection = parent
            self._process = process
            self._diagnostics = dict(ready.get("diagnostics") or {})
            return self._diagnostics

    def render_volume(self, request: VolumeRenderRequest) -> tuple[Image.Image, VtkRenderTimings]:
        return self._render("render_volume", request)

    def render_surface(self, request: SurfaceRenderRequest) -> tuple[Image.Image, VtkRenderTimings]:
        return self._render("render_surface", request)

    def apply_volume_trackball(
        self,
        request: VolumeRenderRequest,
        delta_x_pixels: float,
        delta_y_pixels: float,
    ) -> tuple[float, float, float, float]:
        return tuple(self._request({
            "command": "volume_trackball",
            "request": self._serialize_request(request),
            "volume": self._volumes.register(request.volume, request.volume_token),
            "delta_x_pixels": float(delta_x_pixels),
            "delta_y_pixels": float(delta_y_pixels),
        }))  # type: ignore[return-value]

    def apply_surface_trackball(
        self,
        request: SurfaceRenderRequest,
        delta_x_pixels: float,
        delta_y_pixels: float,
    ) -> tuple[float, float, float, float]:
        return tuple(self._request({
            "command": "surface_trackball",
            "request": self._serialize_request(request),
            "volume": self._volumes.register(request.volume, request.volume_token),
            "delta_x_pixels": float(delta_x_pixels),
            "delta_y_pixels": float(delta_y_pixels),
        }))  # type: ignore[return-value]

    def drop_session(self, view_id: str) -> None:
        self._request({"command": "drop_session", "view_id": view_id})

    def _render(
        self,
        command: str,
        request: VolumeRenderRequest | SurfaceRenderRequest,
    ) -> tuple[Image.Image, VtkRenderTimings]:
        started_at = perf_counter()
        volume_descriptor = self._volumes.register(request.volume, request.volume_token)
        response = self._request({
            "command": command,
            "request": self._serialize_request(request),
            "volume": volume_descriptor,
        }, progress_callback=getattr(request, "progress_callback", None))
        ipc_ms = (perf_counter() - started_at) * 1000.0
        image = Image.frombytes(
            str(response["image_mode"]),
            tuple(int(value) for value in response["image_size"]),
            response["image_bytes"],
        )
        raw_timings = dict(response.get("timings") or {})
        raw_timings["source_dtype"] = str(volume_descriptor.get("source_dtype") or "")
        raw_timings["vtk_dtype"] = np.dtype(str(volume_descriptor["dtype"])).name
        measured_worker_ms = sum(
            float(raw_timings.get(key, 0.0))
            for key in ("vtk_render_ms", "gpu_readback_ms", "session_ms", "configure_ms")
        )
        raw_timings["ipc_ms"] = max(0.0, ipc_ms - measured_worker_ms)
        return image, VtkRenderTimings(**raw_timings)

    def _request(self, message: dict[str, object], progress_callback=None):
        with self._lock:
            self.start()
            assert self._connection is not None
            self._connection.send(message)
            while True:
                response = self._connection.recv()
                kind = response.get("kind")
                if kind == "progress":
                    if callable(progress_callback):
                        progress_callback(dict(response.get("payload") or {}))
                    continue
                if kind == "error":
                    raise RuntimeError(str(response.get("error") or "GPU render worker failed"))
                if kind == "result":
                    return response.get("result")

    @staticmethod
    def _serialize_request(request: VolumeRenderRequest | SurfaceRenderRequest) -> dict[str, object]:
        excluded = {"volume", "progress_callback"}
        return {
            field.name: getattr(request, field.name)
            for field in fields(request)
            if field.name not in excluded
        }

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and self._process is not None and self._process.is_alive():
                try:
                    self._connection.send({"command": "shutdown"})
                    self._connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if self._connection is not None:
                self._connection.close()
            if self._process is not None:
                self._process.join(timeout=3.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
            self._connection = None
            self._process = None
            self._volumes.close()


def _open_shared_volume(descriptor: dict[str, object]) -> tuple[SharedMemory, np.ndarray]:
    try:
        shared = SharedMemory(name=str(descriptor["name"]), track=False)
    except TypeError:  # Python 3.12 compatibility
        shared = SharedMemory(name=str(descriptor["name"]))
    volume = np.ndarray(
        tuple(int(value) for value in descriptor["shape"]),
        dtype=np.dtype(str(descriptor["dtype"])),
        buffer=shared.buf,
    )
    volume.setflags(write=False)
    return shared, volume


def _is_surface_command(command: str) -> bool:
    return command in {"render_surface", "surface_trackball"}


def _gpu_render_worker_main(connection: Connection) -> None:
    try:
        from app.core.logging import setup_logging
        from app.services.volume_rendering.diagnostics import collect_vtk_render_diagnostics
        from app.services.volume_rendering.vtk_surface_renderer import VtkSurfaceRenderer
        from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer

        setup_logging()
        volume_renderer = VtkVolumeRenderer(use_process=False)
        surface_renderer = VtkSurfaceRenderer(use_process=False)
        diagnostics = collect_vtk_render_diagnostics()
        connection.send({"kind": "ready", "diagnostics": diagnostics})
    except Exception:
        connection.send({"kind": "error", "error": traceback.format_exc()})
        connection.close()
        return

    while True:
        try:
            message = connection.recv()
            command = str(message.get("command") or "")
            if command == "shutdown":
                connection.send({"kind": "result", "result": None})
                break
            if command == "drop_session":
                view_id = str(message.get("view_id") or "")
                volume_renderer.drop_session(view_id)
                surface_renderer.drop_session(view_id)
                connection.send({"kind": "result", "result": None})
                continue

            shared, volume = _open_shared_volume(dict(message["volume"]))
            try:
                request_payload = dict(message["request"])
                if _is_surface_command(command):
                    request_payload["progress_callback"] = lambda payload: connection.send({
                        "kind": "progress",
                        "payload": payload,
                    })
                    request = SurfaceRenderRequest(volume=volume, **request_payload)
                else:
                    request = VolumeRenderRequest(volume=volume, **request_payload)

                if command == "render_volume":
                    image = volume_renderer.render(request)
                    timings = volume_renderer.get_last_timings(request.view_id).as_dict()
                    result: object = _serialize_image_result(image, timings)
                elif command == "render_surface":
                    image = surface_renderer.render(request)
                    timings = surface_renderer.get_last_timings(request.view_id).as_dict()
                    result = _serialize_image_result(image, timings)
                elif command == "volume_trackball":
                    result = volume_renderer.apply_trackball_camera_delta(
                        request,
                        delta_x_pixels=float(message.get("delta_x_pixels", 0.0)),
                        delta_y_pixels=float(message.get("delta_y_pixels", 0.0)),
                    )
                elif command == "surface_trackball":
                    result = surface_renderer.apply_trackball_camera_delta(
                        request,
                        delta_x_pixels=float(message.get("delta_x_pixels", 0.0)),
                        delta_y_pixels=float(message.get("delta_y_pixels", 0.0)),
                    )
                else:
                    raise ValueError(f"unknown GPU worker command: {command}")
            finally:
                shared.close()
            connection.send({"kind": "result", "result": result})
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            connection.send({"kind": "error", "error": traceback.format_exc()})
    connection.close()


def _serialize_image_result(image: Image.Image, timings: dict[str, object]) -> dict[str, object]:
    return {
        "image_mode": image.mode,
        "image_size": image.size,
        "image_bytes": image.tobytes(),
        "timings": timings,
    }


_gpu_client: GpuRenderProcessClient | None = None
_gpu_client_lock = RLock()


def get_gpu_render_process_client() -> GpuRenderProcessClient:
    global _gpu_client
    with _gpu_client_lock:
        if _gpu_client is None:
            _gpu_client = GpuRenderProcessClient(
                max_shared_memory_bytes=get_settings().vtk_shared_memory_max_bytes,
            )
        return _gpu_client


def start_gpu_render_process_if_enabled() -> dict[str, object] | None:
    if not get_settings().vtk_render_process_enabled:
        return None
    return get_gpu_render_process_client().start()


def shutdown_gpu_render_process() -> None:
    global _gpu_client
    with _gpu_client_lock:
        if _gpu_client is not None:
            _gpu_client.close()
            _gpu_client = None


atexit.register(shutdown_gpu_render_process)
