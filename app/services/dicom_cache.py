from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from fastapi import HTTPException
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue


@dataclass
class CachedDicom:
    dataset: Dataset
    image_array: np.ndarray
    window_width: float | None
    window_center: float | None


class DicomCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[str, CachedDicom] = OrderedDict()

    def get(self, path: Path) -> CachedDicom:
        cache_key = str(path.resolve())
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        dataset = pydicom.dcmread(str(path), force=True)
        image_array = self._extract_image_array(dataset)
        cached = CachedDicom(
            dataset=dataset,
            image_array=image_array,
            window_width=self._get_first_number(getattr(dataset, "WindowWidth", None)),
            window_center=self._get_first_number(getattr(dataset, "WindowCenter", None)),
        )
        self._cache[cache_key] = cached
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return cached

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
        }

    def clear(self) -> None:
        self._cache.clear()

    def _extract_image_array(self, dataset: Dataset) -> np.ndarray:
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

        ww = self._get_first_number(getattr(dataset, "WindowWidth", None))
        wl = self._get_first_number(getattr(dataset, "WindowCenter", None))
        if ww and wl:
            lower = wl - ww / 2.0
            upper = wl + ww / 2.0
            pixels = np.clip(pixels, lower, upper)

        pixels = pixels - float(np.min(pixels))
        scale = float(np.max(pixels))
        if scale > 0:
            pixels = pixels / scale
        pixels = (pixels * 255.0).astype(np.uint8)

        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 255 - pixels
        return pixels

    @staticmethod
    def _get_first_number(value: float | MultiValue | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, MultiValue):
            return float(value[0])
        return float(value)


dicom_cache = DicomCache()
