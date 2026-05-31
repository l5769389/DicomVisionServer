import math

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.multival import MultiValue
from pydicom.uid import ExplicitVRLittleEndian

from app.services.dicom_cache import DicomCache


def test_get_first_number_accepts_scalar_and_multivalue_inputs() -> None:
    assert DicomCache._get_first_number("40.5") == 40.5
    assert DicomCache._get_first_number(MultiValue(float, ["80", "120"])) == 80.0


def test_get_first_number_ignores_empty_invalid_and_non_finite_values() -> None:
    assert DicomCache._get_first_number(None) is None
    assert DicomCache._get_first_number(MultiValue(float, [])) is None
    assert DicomCache._get_first_number("not-a-number") is None
    assert DicomCache._get_first_number(math.nan) is None


def _build_rgb_dataset(pixels: np.ndarray) -> Dataset:
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.Rows = pixels.shape[-3]
    dataset.Columns = pixels.shape[-2]
    dataset.SamplesPerPixel = 3
    dataset.PhotometricInterpretation = "RGB"
    dataset.PlanarConfiguration = 0
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    if pixels.ndim == 4:
        dataset.NumberOfFrames = pixels.shape[0]
    dataset.PixelData = pixels.tobytes()
    return dataset


def test_extract_source_pixels_preserves_rgb_secondary_capture_pixels() -> None:
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [180, 90, 45]],
        ],
        dtype=np.uint8,
    )

    source_pixels = DicomCache()._extract_source_pixels(_build_rgb_dataset(pixels))

    assert source_pixels.dtype == np.uint8
    assert source_pixels.shape == (2, 2, 3)
    np.testing.assert_array_equal(source_pixels, pixels)


def test_extract_source_pixels_uses_first_rgb_frame_without_collapsing_channels() -> None:
    first_frame = np.full((2, 2, 3), [20, 40, 80], dtype=np.uint8)
    second_frame = np.full((2, 2, 3), [200, 180, 120], dtype=np.uint8)
    pixels = np.stack([first_frame, second_frame], axis=0)

    source_pixels = DicomCache()._extract_source_pixels(_build_rgb_dataset(pixels))

    assert source_pixels.shape == (2, 2, 3)
    np.testing.assert_array_equal(source_pixels, first_frame)
