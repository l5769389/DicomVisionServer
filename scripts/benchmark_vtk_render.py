from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import statistics
import sys
from threading import Thread
from time import perf_counter

import numpy as np
from PIL import Image
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.volume_rendering.contracts import SurfaceRenderRequest, VolumeRenderRequest
from app.services.volume_rendering.diagnostics import collect_vtk_render_diagnostics
from app.services.volume_rendering.gpu_render_process import get_gpu_render_process_client, shutdown_gpu_render_process
from app.services.volume_rendering.vtk_surface_renderer import VtkSurfaceRenderer
from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeatable DicomVision VTK pipeline benchmark")
    parser.add_argument("--dicom-path", type=Path, help="Optional DICOM directory")
    parser.add_argument("--mode", choices=("volume", "surface"), default="volume")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--process", action=argparse.BooleanOptionalAction, default=sys.platform == "darwin")
    args = parser.parse_args()

    volume, spacing = load_volume(args.dicom_path) if args.dicom_path else build_synthetic_volume()
    diagnostics = collect_vtk_render_diagnostics() if not args.process else None
    renderer = VtkSurfaceRenderer(use_process=args.process) if args.mode == "surface" else VtkVolumeRenderer(use_process=args.process)
    records: list[dict[str, float]] = []

    try:
        total_iterations = max(1, args.warmup) + max(1, args.iterations)
        for index in range(total_iterations):
            request = build_request(args, volume, spacing)
            image = renderer.render(request)
            timings = renderer.get_last_timings("benchmark").as_dict()
            encode_started_at = perf_counter()
            payload = encode_webp(image, preview=args.preview)
            encode_ms = (perf_counter() - encode_started_at) * 1000.0
            socket_ms = measure_local_socket_send(payload)
            if index >= args.warmup:
                records.append({
                    "vtk_render_ms": float(timings["vtk_render_ms"]),
                    "gpu_readback_ms": float(timings["gpu_readback_ms"]),
                    "webp_encode_ms": encode_ms,
                    "local_socket_send_ms": socket_ms,
                    "gpu_ipc_ms": float(timings["ipc_ms"]),
                    "bytes": float(len(payload)),
                })
        if args.process:
            diagnostics = get_gpu_render_process_client().diagnostics
        print(json.dumps({
            "mode": args.mode,
            "process": args.process,
            "preview": args.preview,
            "volume_shape": volume.shape,
            "source_dtype": str(volume.dtype),
            "viewport": [args.width, args.height],
            "diagnostics": diagnostics,
            "summary": summarize(records),
            "samples": records,
            "note": "local_socket_send_ms measures repeatable local kernel transfer; production Socket.IO timing is logged by the server.",
        }, ensure_ascii=False, indent=2, default=str))
    finally:
        shutdown_gpu_render_process()


def build_request(args, volume: np.ndarray, spacing: tuple[float, float, float]):
    common = dict(
        view_id="benchmark",
        volume=volume,
        spacing_xyz=spacing,
        canvas_width=args.width,
        canvas_height=args.height,
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0,
        rotation_quaternion=(0.0, 0.0, 0.0, 1.0),
        fast_preview=args.preview,
        volume_token="benchmark-volume",
    )
    if args.mode == "surface":
        return SurfaceRenderRequest(
            **common,
            surface_config={"preset": "bone", "isoValue": 300.0, "smoothing": 0.2, "decimation": 0.25},
        )
    return VolumeRenderRequest(
        **common,
        window_width=600.0,
        window_center=150.0,
        volume_preset="bone",
    )


def build_synthetic_volume() -> tuple[np.ndarray, tuple[float, float, float]]:
    shape = (160, 192, 192)
    z, y, x = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    radius = np.sqrt(((z - center[0]) / 0.8) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    volume = np.full(shape, -1000, dtype=np.int16)
    volume[radius < 70] = 80
    volume[radius < 52] = 180
    volume[radius < 34] = 650
    return volume, (0.8, 0.8, 1.0)


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    files = [item for item in path.rglob("*") if item.is_file()]
    slices: list[tuple[float, np.ndarray, object]] = []
    for index, file_path in enumerate(files):
        try:
            dataset = pydicom.dcmread(str(file_path), force=True)
            pixels = np.asarray(dataset.pixel_array)
            if pixels.ndim != 2:
                continue
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            pixels = pixels.astype(np.float32) * slope + intercept
            position = getattr(dataset, "ImagePositionPatient", None)
            sort_key = float(position[2]) if position is not None and len(position) >= 3 else float(getattr(dataset, "InstanceNumber", index))
            slices.append((sort_key, pixels, dataset))
        except Exception:
            continue
    if not slices:
        raise RuntimeError(f"No readable DICOM slices found in {path}")
    slices.sort(key=lambda item: item[0])
    dataset = slices[0][2]
    pixel_spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
    slice_spacing = float(getattr(dataset, "SpacingBetweenSlices", getattr(dataset, "SliceThickness", 1.0)))
    return np.stack([item[1] for item in slices], axis=0), (
        float(pixel_spacing[1]),
        float(pixel_spacing[0]),
        abs(slice_spacing),
    )


def encode_webp(image: Image.Image, *, preview: bool) -> bytes:
    from io import BytesIO

    output = BytesIO()
    if preview:
        image.save(output, format="WEBP", lossless=False, quality=80, method=0)
    else:
        image.save(output, format="WEBP", lossless=False, quality=94, method=2)
    return output.getvalue()


def measure_local_socket_send(payload: bytes) -> float:
    sender, receiver = socket.socketpair()
    drained = 0

    def drain() -> None:
        nonlocal drained
        while drained < len(payload):
            chunk = receiver.recv(min(1024 * 1024, len(payload) - drained))
            if not chunk:
                break
            drained += len(chunk)

    thread = Thread(target=drain, daemon=True)
    thread.start()
    started_at = perf_counter()
    sender.sendall(payload)
    sender.shutdown(socket.SHUT_WR)
    thread.join()
    elapsed_ms = (perf_counter() - started_at) * 1000.0
    sender.close()
    receiver.close()
    return elapsed_ms


def summarize(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in records[0]:
        values = sorted(record[key] for record in records)
        p95_index = min(len(values) - 1, max(0, round((len(values) - 1) * 0.95)))
        summary[key] = {
            "p50": round(statistics.median(values), 3),
            "p95": round(values[p95_index], 3),
            "mean": round(statistics.fmean(values), 3),
        }
    return summary


if __name__ == "__main__":
    main()
