from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

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


class DicomCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[str, CachedDicom] = OrderedDict()

    def get(self, instance_uid: str, path: Path) -> CachedDicom:
        cache_key = instance_uid
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            logger.debug("dicom cache hit instance_uid=%s", instance_uid)
            return cached

        logger.info("dicom cache miss instance_uid=%s path=%s", instance_uid, path)
        dataset = pydicom.dcmread(str(path), force=True)
        source_pixels = self._extract_source_pixels(dataset)
        cached = CachedDicom(
            dataset=dataset,
            source_pixels=source_pixels,
            window_width=self._get_first_number(getattr(dataset, "WindowWidth", None)),
            window_center=self._get_first_number(getattr(dataset, "WindowCenter", None)),
            pixel_min=float(np.min(source_pixels)),
            pixel_max=float(np.max(source_pixels)),
        )
        self._cache[cache_key] = cached
        if len(self._cache) > self.max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("dicom cache evict instance_uid=%s", evicted_key)
        return cached

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
        }

    def clear(self) -> None:
        self._cache.clear()
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
