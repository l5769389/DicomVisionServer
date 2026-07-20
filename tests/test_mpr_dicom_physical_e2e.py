from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.schemas.dicom import LoadFolderRequest
from app.schemas.view import ViewCreateRequest, ViewOperationRequest
from app.services.dicom_cache import dicom_cache
from app.services.series_registry import series_registry
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service


def _write_physical_mpr_slice(
    path: Path,
    *,
    study_uid: str,
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
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientName = "MPR^Geometry"
    dataset.PatientID = "MPR-GEOMETRY"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Anisotropic MPR geometry truth"
    dataset.InstanceNumber = instance_number
    dataset.Rows = 6
    dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [2.0, 0.5]
    dataset.SliceThickness = 3.0
    dataset.SpacingBetweenSlices = 3.0
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [10.0, 20.0, z_position_mm]
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1000.0
    dataset.WindowWidth = 500.0
    dataset.WindowCenter = 100.0
    pixels = np.full((dataset.Rows, dataset.Columns), stored_value, dtype=np.uint16)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _clear_viewer_state() -> None:
    view_registry._view_by_id.clear()
    for group in view_group_registry.list_all():
        view_group_registry.delete(group.group_id)
    viewer_service._series_volume_cache.clear()
    viewer_service._series_patient_transform_cache.clear()
    viewer_service._series_volume_geometry_cache.clear()
    viewer_service._mpr_plane_cache.clear()
    series_registry.clear()
    dicom_cache.clear()


def test_real_dicom_geometry_reaches_all_mpr_planes_and_shared_cursor(tmp_path: Path) -> None:
    _clear_viewer_state()
    study_uid = generate_uid()
    series_uid = generate_uid()

    # File/InstanceNumber order deliberately disagrees with patient z position.
    for file_index, (instance_number, z_position, stored_value) in enumerate(
        [(50, 12.0, 1200), (10, 0.0, 1000), (40, 9.0, 1150), (20, 3.0, 1050), (30, 6.0, 1100)],
        start=1,
    ):
        _write_physical_mpr_slice(
            tmp_path / f"slice-{file_index}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            instance_number=instance_number,
            z_position_mm=z_position,
            stored_value=stored_value,
        )

    try:
        loaded = series_registry.load_folder(LoadFolderRequest(folderPath=str(tmp_path)))
        series_id = loaded.series_list[0].series_id
        series = series_registry.get(series_id)
        volume = viewer_service._build_series_volume(series)

        assert volume.shape == (5, 6, 8)
        np.testing.assert_array_equal(volume[:, 0, 0], np.array([0, 50, 100, 150, 200], dtype=np.int16))

        views = {}
        for view_type in ("AX", "COR", "SAG"):
            created = view_registry.create(
                ViewCreateRequest(seriesId=series_id, viewType=view_type, viewGroupKey="physical-mpr")
            )
            view = view_registry.get(created.view_id)
            view.width = 320
            view.height = 240
            views[view_type] = view

        rendered = {
            view_type: viewer_service.render_view_by_id(view.view_id, image_format="png")
            for view_type, view in views.items()
        }

        expected_geometry = {
            "AX": ("mpr-ax", (6, 8), (2.0, 0.5, 3.0)),
            "COR": ("mpr-cor", (5, 8), (3.0, 0.5, 2.0)),
            "SAG": ("mpr-sag", (5, 6), (3.0, 2.0, 0.5)),
        }
        for view_type, result in rendered.items():
            plane = result.meta.mpr_plane
            cursor = result.meta.mpr_cursor
            assert result.image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            assert plane is not None
            assert cursor is not None
            expected_viewport, expected_shape, expected_spacing = expected_geometry[view_type]
            assert plane.viewport == expected_viewport
            assert plane.output_shape == expected_shape
            assert (
                plane.pixel_spacing_row_mm,
                plane.pixel_spacing_col_mm,
                plane.pixel_spacing_normal_mm,
            ) == pytest.approx(expected_spacing)
            assert plane.cursor_center_world == pytest.approx((12.0, 26.0, 6.0))
            assert cursor.center_world == pytest.approx((12.0, 26.0, 6.0))
            assert plane.is_oblique is False
            assert result.meta.scale_bar is not None
            assert result.meta.scale_bar.label == "10 cm"

        initial_axial_cursor = rendered["AX"].meta.mpr_cursor
        assert initial_axial_cursor is not None
        viewer_service.handle_view_operation(
            ViewOperationRequest(viewId=views["AX"].view_id, opType="scroll", delta=1, imageFormat="png")
        )

        after_scroll = {
            view_type: viewer_service.render_view_by_id(view.view_id, image_format="png")
            for view_type, view in views.items()
        }
        shared_centers = []
        for result in after_scroll.values():
            assert result.meta.mpr_cursor is not None
            shared_centers.append(result.meta.mpr_cursor.center_world)
        for center in shared_centers:
            assert center == pytest.approx((12.0, 26.0, 9.0))
        assert after_scroll["AX"].meta.slice_info.current == rendered["AX"].meta.slice_info.current + 1
        assert after_scroll["COR"].meta.slice_info.current == rendered["COR"].meta.slice_info.current
        assert after_scroll["SAG"].meta.slice_info.current == rendered["SAG"].meta.slice_info.current
    finally:
        _clear_viewer_state()
