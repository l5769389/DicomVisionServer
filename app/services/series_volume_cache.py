from __future__ import annotations

from collections import OrderedDict
from threading import Lock, RLock
from typing import Callable

import numpy as np

from app.services.volume_rendering.volume_dtype import prepare_vtk_volume


class SeriesVolumeCache:
    def __init__(self, *, max_bytes: int, on_evict: Callable[[str, np.ndarray], None] | None = None) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._max_bytes = int(max_bytes)
        self._lock = RLock()
        self._build_locks: dict[str, Lock] = {}
        self._on_evict = on_evict

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "current_bytes": self._bytes,
                "max_bytes": self._max_bytes,
            }

    def get_build_lock(self, series_id: str) -> Lock:
        with self._lock:
            lock = self._build_locks.get(series_id)
            if lock is None:
                lock = Lock()
                self._build_locks[series_id] = lock
            return lock

    def get(self, series_id: str) -> np.ndarray | None:
        with self._lock:
            cached_volume = self._cache.get(series_id)
            if cached_volume is None:
                return None
            self._cache.move_to_end(series_id)
            return cached_volume

    def store(self, series_id: str, volume: np.ndarray) -> np.ndarray:
        # Integral CT/CBCT values remain 16-bit in the shared series cache.
        # MPR reslicing explicitly promotes interpolated output to float32.
        normalized = prepare_vtk_volume(volume)
        with self._lock:
            existing = self._cache.get(series_id)
            if existing is not None:
                self._cache.move_to_end(series_id)
                return existing

            self._cache[series_id] = normalized
            self._bytes += int(normalized.nbytes)
            self._evict_if_needed()
            return normalized

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._bytes = 0
            self._build_locks.clear()

    def _evict_if_needed(self) -> None:
        while self._bytes > self._max_bytes and len(self._cache) > 1:
            evicted_series_id, evicted_volume = self._cache.popitem(last=False)
            self._bytes = max(0, self._bytes - int(evicted_volume.nbytes))
            self._build_locks.pop(evicted_series_id, None)
            if self._on_evict is not None:
                self._on_evict(evicted_series_id, evicted_volume)
