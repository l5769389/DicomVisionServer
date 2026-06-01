import numpy as np

from app.services.series_volume_cache import SeriesVolumeCache


def test_series_volume_cache_stores_contiguous_float32_volume() -> None:
    cache = SeriesVolumeCache(max_bytes=1024)
    volume = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 2, 2))

    stored = cache.store("series-1", volume)

    assert stored.dtype == np.float32
    assert stored.flags.c_contiguous
    assert cache.get("series-1") is stored


def test_series_volume_cache_evicts_lru_entries_and_reports_callback() -> None:
    evicted: list[str] = []
    cache = SeriesVolumeCache(max_bytes=32, on_evict=lambda series_id, volume: evicted.append(series_id))

    first = cache.store("series-1", np.zeros((2, 2, 2), dtype=np.float32))
    cache.store("series-2", np.ones((2, 2, 2), dtype=np.float32))

    assert cache.get("series-1") is None
    assert cache.get("series-2") is not None
    assert evicted == ["series-1"]
    assert int(first.nbytes) == 32


def test_series_volume_cache_reuses_per_series_build_lock() -> None:
    cache = SeriesVolumeCache(max_bytes=1024)

    assert cache.get_build_lock("series-1") is cache.get_build_lock("series-1")
    assert cache.get_build_lock("series-1") is not cache.get_build_lock("series-2")


def test_series_volume_cache_clear_drops_cached_volumes_and_locks() -> None:
    cache = SeriesVolumeCache(max_bytes=1024)
    lock = cache.get_build_lock("series-1")
    cache.store("series-1", np.zeros((2, 2, 2), dtype=np.float32))

    cache.clear()

    assert cache.get("series-1") is None
    assert cache.current_bytes == 0
    assert cache.get_build_lock("series-1") is not lock
