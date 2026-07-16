from __future__ import annotations

import numpy as np


VTK_INTEGER_DTYPES = (np.dtype(np.int16), np.dtype(np.uint16))


def prepare_vtk_volume(volume: np.ndarray) -> np.ndarray:
    """Return the smallest lossless scalar representation supported by the 3D path."""

    source = np.asarray(volume)
    if source.ndim != 3 or source.size == 0:
        raise ValueError("volume must be a non-empty 3D array")

    if source.dtype in VTK_INTEGER_DTYPES:
        return np.ascontiguousarray(source)

    if np.issubdtype(source.dtype, np.integer):
        minimum = int(np.min(source))
        maximum = int(np.max(source))
        if np.iinfo(np.int16).min <= minimum and maximum <= np.iinfo(np.int16).max:
            return np.ascontiguousarray(source, dtype=np.int16)
        if 0 <= minimum and maximum <= np.iinfo(np.uint16).max:
            return np.ascontiguousarray(source, dtype=np.uint16)
        return np.ascontiguousarray(source, dtype=np.float32)

    finite = np.isfinite(source)
    if not bool(np.all(finite)):
        return np.ascontiguousarray(source, dtype=np.float32)

    minimum = float(np.min(source))
    maximum = float(np.max(source))
    if np.iinfo(np.int16).min <= minimum and maximum <= np.iinfo(np.int16).max:
        rounded = np.rint(source)
        if bool(np.allclose(source, rounded, rtol=0.0, atol=1e-5)):
            return np.ascontiguousarray(rounded, dtype=np.int16)
    if 0.0 <= minimum and maximum <= np.iinfo(np.uint16).max:
        rounded = np.rint(source)
        if bool(np.allclose(source, rounded, rtol=0.0, atol=1e-5)):
            return np.ascontiguousarray(rounded, dtype=np.uint16)
    return np.ascontiguousarray(source, dtype=np.float32)
