from __future__ import annotations

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

from app.services.volume_rendering.volume_dtype import prepare_vtk_volume
from app.services.volume_rendering.vtk_surface_renderer import VtkSurfaceRenderer
from app.services.volume_rendering.vtk_volume_renderer import VtkVolumeRenderer


def test_integral_ct_float_volume_uses_int16_without_value_loss() -> None:
    source = np.array([[[-1000.0, -120.0], [80.0, 3071.0]]], dtype=np.float32)

    prepared = prepare_vtk_volume(source)

    assert prepared.dtype == np.int16
    np.testing.assert_array_equal(prepared, source.astype(np.int16))


def test_fractional_volume_keeps_float32_precision() -> None:
    source = np.array([[[0.25, 1.5]]], dtype=np.float64)

    prepared = prepare_vtk_volume(source)

    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, source)


def test_unsigned_16_bit_volume_uses_uint16_when_int16_is_too_small() -> None:
    source = np.array([[[0, 65535]]], dtype=np.int64)

    prepared = prepare_vtk_volume(source)

    assert prepared.dtype == np.uint16
    np.testing.assert_array_equal(prepared, source.astype(np.uint16))


def test_non_finite_volume_safely_falls_back_to_float32() -> None:
    source = np.array([[[0.0, np.nan]]], dtype=np.float64)

    prepared = prepare_vtk_volume(source)

    assert prepared.dtype == np.float32
    assert np.isnan(prepared[0, 0, 1])


def test_volume_vtk_image_preserves_int16_scalars() -> None:
    source = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    image_data = VtkVolumeRenderer._build_image_data(source, (0.5, 0.6, 1.2))
    restored = vtk_to_numpy(image_data.GetPointData().GetScalars())

    assert restored.dtype == np.int16
    np.testing.assert_array_equal(restored, source.ravel(order="C"))


def test_surface_vtk_image_preserves_uint16_scalars() -> None:
    source = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

    image_data = VtkSurfaceRenderer._build_image_data(source, (0.5, 0.6, 1.2))
    restored = vtk_to_numpy(image_data.GetPointData().GetScalars())

    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, source.ravel(order="C"))
