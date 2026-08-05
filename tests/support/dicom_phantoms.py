from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)


@dataclass(frozen=True)
class SyntheticDicomSeries:
    paths: tuple[Path, ...]
    volume_hu: np.ndarray
    stored_volume: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    orientation: tuple[float, float, float, float, float, float]
    origin_patient_mm: tuple[float, float, float]
    rescale_slope: float
    rescale_intercept: float


@dataclass(frozen=True)
class SyntheticPetDicomSeries:
    paths: tuple[Path, ...]
    volume_bqml: np.ndarray
    stored_volume: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    patient_weight_kg: float
    patient_height_m: float
    injected_dose_bq: float
    radionuclide_half_life_s: float


def write_ct_series(
    root: Path,
    volume_hu: np.ndarray,
    *,
    spacing_zyx_mm: tuple[float, float, float] = (2.5, 1.0, 0.5),
    orientation: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    origin_patient_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    series_description: str = "DicomVision synthetic physical truth",
    file_order: tuple[int, ...] | None = None,
) -> SyntheticDicomSeries:
    """Write a deterministic, privacy-free CT series with analytically known HU."""

    requested_hu = np.asarray(volume_hu, dtype=np.float64)
    if requested_hu.ndim != 3 or not all(int(value) > 0 for value in requested_hu.shape):
        raise ValueError("volume_hu must be a non-empty 3D array")
    if not np.all(np.isfinite(requested_hu)):
        raise ValueError("volume_hu must contain only finite values")

    slope = float(rescale_slope)
    intercept = float(rescale_intercept)
    if not np.isfinite(slope) or abs(slope) <= 1e-12 or not np.isfinite(intercept):
        raise ValueError("rescale_slope must be finite and non-zero and rescale_intercept must be finite")
    stored_float = (requested_hu - intercept) / slope
    stored_rounded = np.rint(stored_float)
    if not np.allclose(stored_float, stored_rounded, atol=1e-6, rtol=0.0):
        raise ValueError("volume_hu values must be exactly representable by the requested rescale transform")
    int16_info = np.iinfo(np.int16)
    if float(np.min(stored_rounded)) < int16_info.min or float(np.max(stored_rounded)) > int16_info.max:
        raise ValueError("stored CT pixels must fit signed 16-bit storage")
    stored_volume = stored_rounded.astype(np.int16)
    decoded_hu = (stored_volume.astype(np.float32) * slope + intercept).astype(np.float32)

    root.mkdir(parents=True, exist_ok=True)
    slice_spacing, row_spacing, col_spacing = (float(value) for value in spacing_zyx_mm)
    row_direction = np.asarray(orientation[:3], dtype=np.float64)
    column_direction = np.asarray(orientation[3:], dtype=np.float64)
    normal_direction = np.cross(row_direction, column_direction)
    normal_direction /= max(float(np.linalg.norm(normal_direction)), 1e-12)
    origin = np.asarray(origin_patient_mm, dtype=np.float64)
    study_uid = generate_uid()
    series_uid = generate_uid()
    order = file_order or tuple(range(stored_volume.shape[0]))
    if sorted(order) != list(range(stored_volume.shape[0])):
        raise ValueError("file_order must contain every slice index exactly once")

    paths: list[Path] = []
    for file_index, slice_index in enumerate(order, start=1):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        path = root / f"IM{file_index:04d}.dcm"
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.PatientName = "Synthetic^Truth"
        dataset.PatientID = "SYNTHETIC-TRUTH"
        dataset.Modality = "CT"
        dataset.SeriesDescription = series_description
        dataset.InstanceNumber = int(stored_volume.shape[0] - slice_index)
        dataset.Rows = int(stored_volume.shape[1])
        dataset.Columns = int(stored_volume.shape[2])
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelRepresentation = 1
        dataset.BitsStored = 16
        dataset.BitsAllocated = 16
        dataset.HighBit = 15
        dataset.PixelSpacing = [row_spacing, col_spacing]
        dataset.SliceThickness = slice_spacing
        dataset.SpacingBetweenSlices = slice_spacing
        dataset.ImageOrientationPatient = [float(value) for value in orientation]
        position = origin + normal_direction * slice_spacing * float(slice_index)
        dataset.ImagePositionPatient = [float(value) for value in position]
        dataset.RescaleSlope = slope
        dataset.RescaleIntercept = intercept
        dataset.RescaleType = "HU"
        dataset.WindowWidth = 2000.0
        dataset.WindowCenter = 0.0
        dataset.PixelData = np.ascontiguousarray(stored_volume[slice_index]).tobytes()
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)

    return SyntheticDicomSeries(
        paths=tuple(paths),
        volume_hu=decoded_hu,
        stored_volume=stored_volume,
        spacing_zyx_mm=(slice_spacing, row_spacing, col_spacing),
        orientation=orientation,
        origin_patient_mm=origin_patient_mm,
        rescale_slope=slope,
        rescale_intercept=intercept,
    )


def write_pet_bqml_series(
    root: Path,
    volume_bqml: np.ndarray,
    *,
    spacing_zyx_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    patient_weight_kg: float = 70.0,
    patient_height_m: float = 1.75,
    patient_sex: str = "M",
    injected_dose_bq: float = 350_000_000.0,
    radionuclide_half_life_s: float = 6586.2,
    injection_datetime: str = "20260101235000",
    acquisition_datetime: str = "20260102001000",
    radiopharmaceutical: str = "F-18 FDG",
) -> SyntheticPetDicomSeries:
    """Write a privacy-free quantitative PET phantom with known BQML geometry."""

    requested = np.asarray(volume_bqml, dtype=np.float64)
    if requested.ndim != 3 or not all(int(value) > 0 for value in requested.shape):
        raise ValueError("volume_bqml must be a non-empty 3D array")
    if not np.all(np.isfinite(requested)) or float(np.min(requested)) < 0.0:
        raise ValueError("volume_bqml must contain finite non-negative values")
    stored_rounded = np.rint(requested)
    if not np.allclose(requested, stored_rounded, atol=1e-6, rtol=0.0):
        raise ValueError("volume_bqml must be exactly representable as integer BQML")
    if float(np.max(stored_rounded)) > np.iinfo(np.uint32).max:
        raise ValueError("stored PET pixels must fit unsigned 32-bit storage")
    stored_volume = stored_rounded.astype(np.uint32)
    decoded = stored_volume.astype(np.float32)

    root.mkdir(parents=True, exist_ok=True)
    slice_spacing, row_spacing, col_spacing = (float(value) for value in spacing_zyx_mm)
    study_uid = generate_uid()
    series_uid = generate_uid()
    paths: list[Path] = []
    for slice_index in range(stored_volume.shape[0]):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = PositronEmissionTomographyImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        path = root / f"PT{slice_index + 1:04d}.dcm"
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = PositronEmissionTomographyImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.PatientName = "Synthetic^PET"
        dataset.PatientID = "SYNTHETIC-PET"
        dataset.PatientWeight = float(patient_weight_kg)
        dataset.PatientSize = float(patient_height_m)
        dataset.PatientSex = str(patient_sex)
        dataset.Modality = "PT"
        dataset.SeriesDescription = "DicomVision quantitative PET truth"
        dataset.InstanceNumber = slice_index + 1
        dataset.Rows = int(stored_volume.shape[1])
        dataset.Columns = int(stored_volume.shape[2])
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelRepresentation = 0
        dataset.BitsStored = 32
        dataset.BitsAllocated = 32
        dataset.HighBit = 31
        dataset.PixelSpacing = [row_spacing, col_spacing]
        dataset.SliceThickness = slice_spacing
        dataset.SpacingBetweenSlices = slice_spacing
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.ImagePositionPatient = [0.0, 0.0, slice_spacing * slice_index]
        dataset.RescaleSlope = 1.0
        dataset.RescaleIntercept = 0.0
        dataset.Units = "BQML"
        dataset.CorrectedImage = ["DECY", "ATTN"]
        dataset.DecayCorrection = "START"
        dataset.AcquisitionDateTime = acquisition_datetime

        tracer = Dataset()
        tracer.Radiopharmaceutical = radiopharmaceutical
        tracer.RadionuclideTotalDose = float(injected_dose_bq)
        tracer.RadionuclideHalfLife = float(radionuclide_half_life_s)
        tracer.RadiopharmaceuticalStartDateTime = injection_datetime
        dataset.RadiopharmaceuticalInformationSequence = [tracer]
        dataset.PixelData = np.ascontiguousarray(stored_volume[slice_index]).tobytes()
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)

    return SyntheticPetDicomSeries(
        paths=tuple(paths),
        volume_bqml=decoded,
        stored_volume=stored_volume,
        spacing_zyx_mm=(slice_spacing, row_spacing, col_spacing),
        patient_weight_kg=float(patient_weight_kg),
        patient_height_m=float(patient_height_m),
        injected_dose_bq=float(injected_dose_bq),
        radionuclide_half_life_s=float(radionuclide_half_life_s),
    )


def build_asymmetric_landmark_volume(shape: tuple[int, int, int] = (9, 11, 13)) -> np.ndarray:
    """Build an asymmetric CT phantom with distinguishable L/A/S landmarks."""

    depth, height, width = shape
    volume = np.full(shape, -1000, dtype=np.int16)
    volume[depth // 2, height // 2, min(width - 1, width // 2 + 4)] = 1200
    volume[depth // 2, min(height - 1, height // 2 + 3), width // 2] = 800
    volume[min(depth - 1, depth // 2 + 2), height // 2, width // 2] = 400
    return volume
