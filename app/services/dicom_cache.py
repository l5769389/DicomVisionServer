from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import pydicom
from fastapi import HTTPException
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue

from app.core.logging import get_logger


logger = get_logger(__name__)


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
    def __init__(self, max_entries: int = 128, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._cache: OrderedDict[str, CachedDicom] = OrderedDict()
        self._current_bytes = 0
        self._lock = RLock()

    def get(self, instance_uid: str | None, path: Path) -> CachedDicom:
        cache_key = str(instance_uid or path.resolve().as_posix())
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
            self._current_bytes = 0
        logger.info("dicom cache cleared")

    def _extract_source_pixels(self, dataset: Dataset) -> np.ndarray:
        if "PixelData" not in dataset:
            raise HTTPException(status_code=400, detail="DICOM file does not contain pixel data")

        try:
            pixels = dataset.pixel_array.astype(np.float32)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode pixel data: {exc}") from exc

        if pixels.ndim == 3:
            pixels = pixels[0]

        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept

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
    def _get_first_number(value: float | MultiValue | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, MultiValue):
            return float(value[0])
        return float(value)


dicom_cache = DicomCache()
