from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import RLock

import numpy as np
import pydicom
from fastapi import HTTPException
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.pixels import convert_color_space

from app.core.logging import get_logger


logger = get_logger(__name__)

DEFAULT_MAX_CACHE_ENTRIES = 128
DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024


@dataclass
class CachedDicom:
    dataset: Dataset
    source_pixels: np.ndarray
    window_width: float | None
    window_center: float | None
    pixel_min: float
    pixel_max: float
    byte_size: int


class DicomCache:
    def __init__(self, max_entries: int = DEFAULT_MAX_CACHE_ENTRIES, max_bytes: int = DEFAULT_MAX_CACHE_BYTES) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._cache: OrderedDict[str, CachedDicom] = OrderedDict()
        self._fingerprints_by_path: dict[str, tuple[int, int, str]] = {}
        self._current_bytes = 0
        self._lock = RLock()

    def get(self, instance_uid: str | None, path: Path) -> CachedDicom:
        cache_key = self._build_cache_key(instance_uid, path)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                logger.debug("dicom cache hit key=%s", cache_key)
                return cached

        logger.info("dicom cache miss key=%s path=%s", cache_key, path)
        dataset = pydicom.dcmread(str(path), force=True)
        source_pixels = self._extract_source_pixels(dataset)
        cached = CachedDicom(
            dataset=dataset,
            source_pixels=source_pixels,
            window_width=self._get_first_number(getattr(dataset, "WindowWidth", None)),
            window_center=self._get_first_number(getattr(dataset, "WindowCenter", None)),
            pixel_min=float(np.min(source_pixels)),
            pixel_max=float(np.max(source_pixels)),
            byte_size=int(source_pixels.nbytes),
        )
        with self._lock:
            existing = self._cache.get(cache_key)
            if existing is not None:
                self._cache.move_to_end(cache_key)
                return existing

            self._cache[cache_key] = cached
            self._current_bytes += cached.byte_size
            self._evict_if_needed()
        return cached

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_entries or (self._current_bytes > self.max_bytes and len(self._cache) > 1):
            evicted_key, evicted = self._cache.popitem(last=False)
            self._current_bytes = max(0, self._current_bytes - evicted.byte_size)
            logger.debug("dicom cache evict key=%s bytes=%s", evicted_key, evicted.byte_size)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "current_bytes": self._current_bytes,
                "max_bytes": self.max_bytes,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._fingerprints_by_path.clear()
            self._current_bytes = 0
        logger.info("dicom cache cleared")

    def _extract_source_pixels(self, dataset: Dataset) -> np.ndarray:
        if "PixelData" not in dataset:
            raise HTTPException(status_code=400, detail="DICOM file does not contain pixel data")

        try:
            pixels = dataset.pixel_array
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode pixel data: {exc}") from exc

        # The viewer currently treats enhanced/multi-frame files as a single 2D frame.
        # Keep this choice explicit so adding true multi-frame support has one obvious branch.
        if pixels.ndim == 4:
            pixels = pixels[0]

        if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            pixels = self._normalize_color_pixels(dataset, pixels)
            logger.debug(
                "color source pixels extracted rows=%s cols=%s channels=%s min=%.3f max=%.3f",
                pixels.shape[0],
                pixels.shape[1],
                pixels.shape[2],
                float(np.min(pixels)),
                float(np.max(pixels)),
            )
            return pixels

        if pixels.ndim == 3:
            pixels = pixels[0]

        pixels = pixels.astype(np.float32)

        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept

        # MONOCHROME1 stores lower values as visually brighter. Negating keeps the
        # downstream windowing path using the same "larger value is brighter" model.
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = -pixels
        logger.debug(
            "source pixels extracted rows=%s cols=%s slope=%s intercept=%s min=%.3f max=%.3f",
            pixels.shape[0],
            pixels.shape[1],
            slope,
            intercept,
            float(np.min(pixels)),
            float(np.max(pixels)),
        )
        return pixels

    @staticmethod
    def _normalize_color_pixels(dataset: Dataset, pixels: np.ndarray) -> np.ndarray:
        photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "").upper()
        color_pixels = pixels
        if photometric.startswith("YBR"):
            try:
                color_pixels = convert_color_space(color_pixels, photometric, "RGB")
            except Exception as exc:
                logger.debug("failed to convert color space %s to RGB: %s", photometric, exc)

        if color_pixels.shape[-1] == 4:
            color_pixels = color_pixels[..., :3]

        if color_pixels.dtype == np.uint8:
            return np.ascontiguousarray(color_pixels)

        color_pixels = np.asarray(color_pixels, dtype=np.float32)
        bits_stored = DicomCache._get_first_number(getattr(dataset, "BitsStored", None))
        upper = (2.0 ** bits_stored - 1.0) if bits_stored is not None and bits_stored > 0 else float(np.max(color_pixels))
        if upper > 0:
            color_pixels = color_pixels * (255.0 / upper)
        return np.ascontiguousarray(np.clip(color_pixels, 0.0, 255.0).astype(np.uint8))

    def build_instance_content_key(self, instance_uid: str | None, path: Path) -> str:
        return self._build_cache_key(instance_uid, path)

    def _build_cache_key(self, instance_uid: str | None, path: Path) -> str:
        resolved_path = path.resolve()
        fingerprint = self._get_file_fingerprint(resolved_path)
        return f"{instance_uid or resolved_path.as_posix()}::{fingerprint}"

    def _get_file_fingerprint(self, path: Path) -> str:
        stat = path.stat()
        path_key = path.as_posix()
        cached = self._fingerprints_by_path.get(path_key)
        if cached is not None:
            cached_size, cached_mtime_ns, cached_fingerprint = cached
            if cached_size == stat.st_size and cached_mtime_ns == stat.st_mtime_ns:
                return cached_fingerprint

        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            while True:
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        fingerprint = f"{stat.st_size}:{digest.hexdigest()}"
        self._fingerprints_by_path[path_key] = (stat.st_size, stat.st_mtime_ns, fingerprint)
        return fingerprint

    @staticmethod
    def _get_first_number(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, MultiValue):
            if not value:
                return None
            value = value[0]
        try:
            parsed_value = float(value)
        except (TypeError, ValueError):
            return None
        return parsed_value if np.isfinite(parsed_value) else None


dicom_cache = DicomCache()
