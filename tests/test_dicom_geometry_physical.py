from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.schemas.dicom import LoadFolderRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry
from app.services.viewer_service import viewer_service


def _write_oriented_slice(
    path: Path,
    *,
    series_uid: str,
    instance_number: int,
    z_position_mm: float,
    stored_value: int,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientName = "Geometry^Truth"
    dataset.PatientID = "GEOMETRY-TRUTH"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Physical geometry truth"
    dataset.InstanceNumber = instance_number
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [2.0, 0.5]
    dataset.SliceThickness = 9.0
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [10.0, 20.0, float(z_position_mm)]
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1000.0
    dataset.PixelData = np.full((4, 5), stored_value, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_real_dicom_series_uses_patient_position_order_and_physical_voxel_geometry(tmp_path: Path) -> None:
    series_registry.clear()
    dicom_cache.clear()
    viewer_service._series_volume_cache.clear()
    viewer_service._series_patient_transform_cache.clear()
    viewer_service._series_volume_geometry_cache.clear()
    series_uid = generate_uid()

    # InstanceNumber intentionally disagrees with patient-space z ordering.
    _write_oriented_slice(
        tmp_path / "instance-1-z4.dcm",
        series_uid=series_uid,
        instance_number=1,
        z_position_mm=4.0,
        stored_value=1040,
    )
    _write_oriented_slice(
        tmp_path / "instance-2-z0.dcm",
        series_uid=series_uid,
        instance_number=2,
        z_position_mm=0.0,
        stored_value=1000,
    )
    _write_oriented_slice(
        tmp_path / "instance-3-z2.dcm",
        series_uid=series_uid,
        instance_number=3,
        z_position_mm=2.0,
        stored_value=1020,
    )

    try:
        loaded = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
        series = series_registry.get(loaded.series_list[0].series_id)
        volume = viewer_service._build_series_volume(series)
        transform = viewer_service._get_series_patient_transform(series)
        geometry = viewer_service._get_series_volume_geometry(series, volume.shape)

        assert transform is not None
        assert volume.shape == (3, 4, 5)
        # Stored values are converted to HU, then slices are ordered by
        # ImagePositionPatient rather than InstanceNumber.
        np.testing.assert_array_equal(volume[:, 0, 0], np.array([0, 20, 40], dtype=np.int16))

        assert transform.shape == (3, 4, 5)
        assert transform.spacing_xyz() == pytest.approx((0.5, 2.0, 2.0))
        np.testing.assert_allclose(transform.origin, np.array([10.0, 20.0, 0.0]))
        np.testing.assert_allclose(
            transform.point_to_patient((2.0, 3.0, 4.0)),
            np.array([12.0, 26.0, 4.0]),
        )

        # The voxel volume is 0.5 mm * 2.0 mm * 2.0 mm = 2 mm3.
        assert abs(float(np.linalg.det(geometry.ijk_to_world[:3, :3]))) == pytest.approx(2.0)
    finally:
        viewer_service._series_volume_cache.clear()
        viewer_service._series_patient_transform_cache.clear()
        viewer_service._series_volume_geometry_cache.clear()
        series_registry.clear()
        dicom_cache.clear()
