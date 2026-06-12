import json
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from pydicom import dcmread
from pydicom.dataset import Dataset

from app.core import (
    FUSION_PANE_CT_AXIAL,
    FUSION_PANE_OVERLAY_AXIAL,
    FUSION_PANE_PET_AXIAL,
    FUSION_PANE_PET_CORONAL_MIP,
)
from app.models.viewer import FusionRegistrationState, InstanceRecord, SeriesRecord, ViewGroupRecord, ViewRecord
from app.services.mpr import VolumeGeometry, build_identity_geometry
from app.services.render_layers.render_context import CornerInfoOverlay
from app.services.viewer_fusion import render_fusion_pixels
from app.services.viewer_operation_handlers import _handle_fusion_registration_operation
from app.services.viewer_service import FusionPetDisplayVolume, ViewerService
from app.schemas.view import FusionRegistrationArtifactExportRequest, FusionRegistrationExportRequest, ViewOperationRequest, ViewSetSizeRequest, ViewSize


def _volume(shape: tuple[int, int, int] = (5, 6, 7)) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def _geometry_with_axes(
    shape: tuple[int, int, int],
    axis_i: tuple[float, float, float],
    axis_j: tuple[float, float, float],
    axis_k: tuple[float, float, float],
) -> VolumeGeometry:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = np.asarray(axis_i, dtype=np.float64)
    affine[:3, 1] = np.asarray(axis_j, dtype=np.float64)
    affine[:3, 2] = np.asarray(axis_k, dtype=np.float64)
    return VolumeGeometry(
        shape_ijk=shape,
        ijk_to_world=affine,
        world_to_ijk=np.linalg.inv(affine),
        spacing_hint_mm=(
            float(np.linalg.norm(affine[:3, 0])),
            float(np.linalg.norm(affine[:3, 1])),
            float(np.linalg.norm(affine[:3, 2])),
        ),
    )


def _render(role: str, *, registration: FusionRegistrationState | None = None, has_geometry: bool = True):
    ct_volume = _volume()
    pet_volume = _volume()
    geometry = build_identity_geometry(tuple(int(value) for value in ct_volume.shape))
    return render_fusion_pixels(
        pane_role=role,
        ct_volume=ct_volume,
        ct_geometry=geometry,
        pet_volume=pet_volume,
        pet_geometry=geometry,
        axial_index=2,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=8,
        pet_window_center=4,
        pet_pseudocolor_preset="pet",
        registration=registration or FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=has_geometry,
        pet_has_patient_geometry=has_geometry,
    )


@pytest.mark.parametrize(
    "role",
    [
        FUSION_PANE_CT_AXIAL,
        FUSION_PANE_PET_AXIAL,
        FUSION_PANE_OVERLAY_AXIAL,
        FUSION_PANE_PET_CORONAL_MIP,
    ],
)
def test_fusion_render_result_includes_orientation_directions(role: str) -> None:
    result = _render(role)

    assert result.row_world is not None
    assert result.col_world is not None
    assert result.spacing_xy[0] > 0
    assert result.spacing_xy[1] > 0
    if role == FUSION_PANE_PET_CORONAL_MIP:
        assert result.slice_index == 0
        assert result.slice_total == 1
    else:
        assert result.slice_index == 2
        assert result.slice_total == 5

    overlay = ViewerService()._build_direction_orientation_overlay(
        ViewRecord(view_id="fusion-view", series_id="series", view_type="FusionOverlayAxial"),
        result.row_world,
        result.col_world,
    )
    assert overlay is not None
    assert all(value for value in (overlay.top, overlay.right, overlay.bottom, overlay.left))


def test_pet_only_axial_reports_pet_slice_index_and_total() -> None:
    ct_volume = _volume((5, 6, 7))
    pet_volume = _volume((9, 6, 7))
    ct_geometry = build_identity_geometry(tuple(int(value) for value in ct_volume.shape))
    pet_geometry = _geometry_with_axes(
        tuple(int(value) for value in pet_volume.shape),
        axis_i=(0.5, 0.0, 0.0),
        axis_j=(0.0, 1.0, 0.0),
        axis_k=(0.0, 0.0, 1.0),
    )

    result = render_fusion_pixels(
        pane_role=FUSION_PANE_PET_AXIAL,
        ct_volume=ct_volume,
        ct_geometry=ct_geometry,
        pet_volume=pet_volume,
        pet_geometry=pet_geometry,
        axial_index=2,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=8,
        pet_window_center=4,
        pet_pseudocolor_preset="petct-rainbow",
        registration=FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=True,
        pet_has_patient_geometry=True,
    )

    assert result.slice_index == 4
    assert result.slice_total == 9


@pytest.mark.parametrize(
    "role",
    [
        FUSION_PANE_CT_AXIAL,
        FUSION_PANE_PET_AXIAL,
        FUSION_PANE_OVERLAY_AXIAL,
        FUSION_PANE_PET_CORONAL_MIP,
    ],
)
def test_fusion_render_result_does_not_fabricate_orientation_without_patient_geometry(role: str) -> None:
    result = _render(role, has_geometry=False)

    assert result.row_world is None
    assert result.col_world is None


def test_fusion_orientation_overlay_tracks_horizontal_flip() -> None:
    result = _render(FUSION_PANE_OVERLAY_AXIAL)
    service = ViewerService()
    normal_view = ViewRecord(view_id="fusion-view", series_id="series", view_type="FusionOverlayAxial")
    flipped_view = ViewRecord(view_id="fusion-view", series_id="series", view_type="FusionOverlayAxial")
    flipped_view.hor_flip = True

    normal = service._build_direction_orientation_overlay(normal_view, result.row_world, result.col_world)
    flipped = service._build_direction_orientation_overlay(flipped_view, result.row_world, result.col_world)

    assert normal is not None
    assert flipped is not None
    assert flipped.right == normal.left
    assert flipped.left == normal.right
    assert flipped.top == normal.top
    assert flipped.bottom == normal.bottom


def test_pet_axial_orientation_tracks_manual_registration_rotation() -> None:
    before = _render(FUSION_PANE_PET_AXIAL)
    after = _render(
        FUSION_PANE_PET_AXIAL,
        registration=FusionRegistrationState(rotation_degrees=90),
    )

    assert before.row_world is not None
    assert before.col_world is not None
    assert after.row_world is not None
    assert after.col_world is not None
    assert not np.allclose(before.row_world, after.row_world)
    assert not np.allclose(before.col_world, after.col_world)


@pytest.mark.parametrize(
    ("role", "expected_preset"),
    [
        (FUSION_PANE_CT_AXIAL, "bw"),
        (FUSION_PANE_PET_AXIAL, "bwinverse"),
        (FUSION_PANE_OVERLAY_AXIAL, "petct-rainbow"),
        (FUSION_PANE_PET_CORONAL_MIP, "bwinverse"),
    ],
)
def test_fusion_result_reports_actual_rendered_pseudocolor(role: str, expected_preset: str) -> None:
    result = render_fusion_pixels(
        pane_role=role,
        ct_volume=_volume(),
        ct_geometry=build_identity_geometry(tuple(int(value) for value in _volume().shape)),
        pet_volume=_volume(),
        pet_geometry=build_identity_geometry(tuple(int(value) for value in _volume().shape)),
        axial_index=2,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=8,
        pet_window_center=4,
        pet_pseudocolor_preset="petct-rainbow",
        registration=FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=True,
        pet_has_patient_geometry=True,
    )

    assert result.pseudocolor_preset == expected_preset


def test_pet_only_views_use_inverse_grayscale_independent_of_fusion_pet_pseudocolor() -> None:
    ct_volume = _volume()
    pet_volume = _volume()
    geometry = build_identity_geometry(tuple(int(value) for value in ct_volume.shape))

    result = render_fusion_pixels(
        pane_role=FUSION_PANE_PET_AXIAL,
        ct_volume=ct_volume,
        ct_geometry=geometry,
        pet_volume=pet_volume,
        pet_geometry=geometry,
        axial_index=2,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=8,
        pet_window_center=4,
        pet_pseudocolor_preset="hotiron",
        registration=FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=True,
        pet_has_patient_geometry=True,
    )

    assert result.pseudocolor_preset == "bwinverse"


def test_pet_only_zero_background_maps_to_white_with_inverse_grayscale() -> None:
    ct_volume = np.zeros((3, 4, 4), dtype=np.float32)
    pet_volume = np.zeros((3, 4, 4), dtype=np.float32)
    geometry = build_identity_geometry(tuple(int(value) for value in ct_volume.shape))

    result = render_fusion_pixels(
        pane_role=FUSION_PANE_PET_AXIAL,
        ct_volume=ct_volume,
        ct_geometry=geometry,
        pet_volume=pet_volume,
        pet_geometry=geometry,
        axial_index=1,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=4.5,
        pet_window_center=2.25,
        pet_pseudocolor_preset="petct-rainbow",
        registration=FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=True,
        pet_has_patient_geometry=True,
    )

    assert tuple(int(channel) for channel in result.pixels[0, 0]) == (255, 255, 255)


def test_pet_coronal_mip_uses_physical_spacing_and_head_first_direction() -> None:
    volume = _volume((9, 6, 7))
    geometry = _geometry_with_axes(
        volume.shape,
        axis_i=(0.0, 0.0, 3.0),
        axis_j=(0.0, 2.0, 0.0),
        axis_k=(4.0, 0.0, 0.0),
    )

    result = render_fusion_pixels(
        pane_role=FUSION_PANE_PET_CORONAL_MIP,
        ct_volume=volume,
        ct_geometry=geometry,
        pet_volume=volume,
        pet_geometry=geometry,
        axial_index=4,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=8,
        pet_window_center=4,
        pet_pseudocolor_preset="pet",
        registration=FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=True,
        pet_has_patient_geometry=True,
    )

    assert result.pixels.shape[:2] == (9, 7)
    assert result.spacing_xy == pytest.approx((4.0, 3.0))
    assert result.row_world is not None
    assert result.col_world is not None
    assert np.allclose(result.row_world, (0.0, 0.0, -1.0))
    assert np.allclose(result.col_world, (1.0, 0.0, 0.0))

    overlay = ViewerService()._build_direction_orientation_overlay(
        ViewRecord(view_id="fusion-mip", series_id="series", view_type="FusionPETCoronalMip"),
        result.row_world,
        result.col_world,
    )
    assert overlay is not None
    assert overlay.top == "S"
    assert overlay.bottom == "I"
    assert overlay.left == "R"
    assert overlay.right == "L"


def test_pet_coronal_mip_projection_uses_pet_volume_world_coordinates() -> None:
    volume = _volume((9, 6, 7))
    geometry = _geometry_with_axes(
        volume.shape,
        axis_i=(0.0, 0.0, 3.0),
        axis_j=(0.0, 2.0, 0.0),
        axis_k=(4.0, 0.0, 0.0),
    )

    result = render_fusion_pixels(
        pane_role=FUSION_PANE_PET_CORONAL_MIP,
        ct_volume=volume,
        ct_geometry=geometry,
        pet_volume=volume,
        pet_geometry=geometry,
        axial_index=4,
        ct_window_width=400,
        ct_window_center=40,
        pet_window_width=8,
        pet_window_center=4,
        pet_pseudocolor_preset="pet",
        registration=FusionRegistrationState(),
        alpha=0.52,
        ct_has_patient_geometry=True,
        pet_has_patient_geometry=True,
    )
    projection = ViewerService._build_fusion_projection_info(
        pane_role=FUSION_PANE_PET_CORONAL_MIP,
        source_projection=result.source_projection,
        image_transform=SimpleNamespace(matrix=np.eye(3, dtype=np.float64)),
        image_width=result.pixels.shape[1],
        image_height=result.pixels.shape[0],
    )

    assert projection is not None
    assert projection.reference_world == pytest.approx((12.0, 5.0, 12.0))
    assert projection.reference_x == pytest.approx(3.0 / 7.0)
    assert projection.reference_y == pytest.approx(4.0 / 9.0)


def test_fusion_coronal_mip_initial_fit_uses_physical_aspect() -> None:
    volume = _volume((267, 6, 128))
    geometry = _geometry_with_axes(
        volume.shape,
        axis_i=(0.0, 0.0, 3.0),
        axis_j=(0.0, 2.0, 0.0),
        axis_k=(4.0, 0.0, 0.0),
    )
    view = ViewRecord(
        view_id="fusion-mip",
        series_id="ct",
        view_type="FusionPETCoronalMip",
        width=900,
        height=600,
    )

    ViewerService()._fit_fusion_view_to_source(
        view,
        ct_volume=volume,
        ct_geometry=geometry,
        pet_volume=volume,
        pet_geometry=geometry,
    )

    assert view.zoom == pytest.approx(600.0 / (267.0 * 3.0))


@pytest.mark.parametrize("view_type", ["FusionCTAxial", "FusionPETAxial", "FusionOverlayAxial"])
def test_fusion_axial_initial_fit_uses_shared_ct_pet_physical_extent(view_type: str) -> None:
    ct_volume = _volume((20, 100, 100))
    pet_volume = _volume((20, 200, 200))
    ct_geometry = build_identity_geometry(tuple(int(value) for value in ct_volume.shape))
    pet_geometry = build_identity_geometry(tuple(int(value) for value in pet_volume.shape))
    view = ViewRecord(
        view_id="fusion-axial",
        series_id="ct",
        view_type=view_type,
        width=600,
        height=600,
    )

    ViewerService()._fit_fusion_view_to_source(
        view,
        ct_volume=ct_volume,
        ct_geometry=ct_geometry,
        pet_volume=pet_volume,
        pet_geometry=pet_geometry,
    )

    assert view.zoom == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("role", "expected_label"),
    [
        (FUSION_PANE_CT_AXIAL, "CT Axial"),
        (FUSION_PANE_PET_AXIAL, "PET Axial"),
        (FUSION_PANE_OVERLAY_AXIAL, "PET/CT"),
        (FUSION_PANE_PET_CORONAL_MIP, "PET Coronal MIP"),
    ],
)
def test_fusion_corner_info_label_only_marks_overlay_as_fusion(role: str, expected_label: str) -> None:
    assert ViewerService._build_fusion_viewport_label(role) == expected_label


@pytest.mark.parametrize(
    ("role", "expected_label"),
    [
        (FUSION_PANE_CT_AXIAL, "Axial"),
        (FUSION_PANE_PET_AXIAL, "Axial"),
        (FUSION_PANE_OVERLAY_AXIAL, "Axial"),
        (FUSION_PANE_PET_CORONAL_MIP, "MIP"),
    ],
)
def test_fusion_corner_info_uses_anatomic_axis_label_for_physical_location(role: str, expected_label: str) -> None:
    assert ViewerService._build_fusion_corner_viewport_label(role) == expected_label


def test_fusion_mip_corner_info_omits_single_slice_location_and_index() -> None:
    dataset = Dataset()
    dataset.ImagePositionPatient = [1.0, 2.0, 3.0]
    dataset.InstanceNumber = 134
    series = SeriesRecord(
        series_id="pet",
        folder_path="",
        series_instance_uid="pet-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="PT",
        series_description="PET",
    )

    corner_info = ViewerService()._build_slice_corner_info_overlay(
        ViewRecord(view_id="fusion-mip", series_id="pet", view_type="FusionPETCoronalMip"),
        series,
        dataset,
        current_index=133,
        total_slices=267,
        viewport_label="MIP",
        show_physical_location=False,
        show_image_index=False,
    )

    assert corner_info.top_left == ("MIP",)
    assert corner_info.tags["viewportLocation"] == ("MIP",)
    assert "imageIndex" not in corner_info.tags
    assert all("Im:" not in line for line in corner_info.top_left)


def test_fusion_corner_info_uses_current_indexed_instance(monkeypatch) -> None:
    instances = [
        InstanceRecord(
            path=Path(f"slice-{index}.dcm"),
            sop_instance_uid=f"sop-{index}",
            instance_number=index + 1,
            rows=None,
            columns=None,
        )
        for index in range(3)
    ]
    series = SeriesRecord(
        series_id="ct",
        folder_path="",
        series_instance_uid="ct-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="CT",
        instances=instances,
    )
    calls: list[str] = []

    def fake_cache_get(sop_instance_uid, path):
        calls.append(sop_instance_uid)
        return SimpleNamespace(dataset=Dataset())

    monkeypatch.setattr("app.services.viewer_service.dicom_cache.get", fake_cache_get)

    instance, cached = ViewerService._get_indexed_instance_and_cache(series, 2)

    assert instance is instances[2]
    assert cached is not None
    assert calls == ["sop-2"]


def test_fusion_set_size_initializes_shared_fusion_group(monkeypatch) -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.fusion_ct_series_id = "ct"
    group.fusion_pet_series_id = "pet"
    ct_series = SeriesRecord(
        series_id="ct",
        folder_path="",
        series_instance_uid="ct-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="CT",
    )
    pet_series = SeriesRecord(
        series_id="pet",
        folder_path="",
        series_instance_uid="pet-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="PT",
        series_description="PET",
    )
    ct_volume = np.arange(7 * 4 * 4, dtype=np.float32).reshape(7, 4, 4)
    pet_volume = np.arange(7 * 4 * 4, dtype=np.float32).reshape(7, 4, 4)
    geometry = build_identity_geometry(tuple(int(value) for value in ct_volume.shape))
    view = ViewRecord(
        view_id="fusion-pet",
        series_id="ct",
        secondary_series_id="pet",
        view_type="FusionPETAxial",
        fusion_pane_role=FUSION_PANE_PET_AXIAL,
        view_group=group,
    )

    monkeypatch.setattr("app.services.viewer_service.view_registry.get", lambda view_id, workspace_id=None: view)
    monkeypatch.setattr(service, "_resolve_fusion_group_series", lambda current_view: (group, ct_series, pet_series))
    monkeypatch.setattr(
        service,
        "_get_series_volume",
        lambda series, **_: ct_volume if series.series_id == "ct" else pet_volume,
    )
    monkeypatch.setattr(service, "_get_series_volume_geometry", lambda series, shape: geometry)
    monkeypatch.setattr(
        service,
        "_build_fusion_pet_display_volume",
        lambda series, volume, unit: FusionPetDisplayVolume(
            volume=np.asarray(volume, dtype=np.float32),
            unit="SUVbw",
            unit_label="g/ml (SUVbw)",
        ),
    )

    service.set_view_size(
        ViewSetSizeRequest(
            viewId="fusion-pet",
            opType="setSize",
            size=ViewSize(width=512, height=512),
        )
    )

    assert group.fusion_initialized is True
    assert group.fusion_axial_index == 3
    assert view.current_index == 3
    assert view.is_initialized is True
    assert view.pseudocolor_preset == "bwinverse"


def test_fusion_view_windows_are_scoped_by_pane_role() -> None:
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.window.window_width = 400.0
    group.window.window_center = 40.0
    group.fusion_pet_window.window_width = 9.0
    group.fusion_pet_window.window_center = 4.5
    ct_view = ViewRecord(
        view_id="ct",
        series_id="ct",
        view_type="FusionCTAxial",
        fusion_pane_role=FUSION_PANE_CT_AXIAL,
        view_group=group,
    )
    overlay_view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    pet_view = ViewRecord(
        view_id="pet",
        series_id="pet",
        view_type="FusionPETAxial",
        fusion_pane_role=FUSION_PANE_PET_AXIAL,
        view_group=group,
    )

    assert ct_view.window_width == 400.0
    assert overlay_view.window_center == 40.0
    assert pet_view.window_width == 9.0
    assert pet_view.window_center == 4.5

    pet_view.window_width = 12.0
    pet_view.window_center = 6.0

    assert group.fusion_pet_window.window_width == 12.0
    assert group.fusion_pet_window.window_center == 6.0
    assert group.window.window_width == 400.0
    assert group.window.window_center == 40.0


def test_fusion_info_reports_pet_window_for_overlay_pane(monkeypatch) -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.window.window_width = 400.0
    group.window.window_center = 40.0
    group.fusion_pet_window.window_width = 4.5
    group.fusion_pet_window.window_center = 2.25
    group.fusion_ct_series_id = "ct"
    group.fusion_pet_series_id = "pet"
    ct_series = SeriesRecord(
        series_id="ct",
        folder_path="",
        series_instance_uid="ct-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="CT",
    )
    pet_series = SeriesRecord(
        series_id="pet",
        folder_path="",
        series_instance_uid="pet-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="PT",
        series_description="PET",
    )
    ct_volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    pet_volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    geometry = build_identity_geometry((4, 4, 4))

    monkeypatch.setattr(service, "_resolve_fusion_group_series", lambda view: (group, ct_series, pet_series))
    monkeypatch.setattr(
        service,
        "_get_series_volume",
        lambda series, **_: ct_volume if series.series_id == "ct" else pet_volume,
    )
    monkeypatch.setattr(service, "_get_series_volume_geometry", lambda series, shape: geometry)
    monkeypatch.setattr(
        service,
        "_build_fusion_pet_display_volume",
        lambda series, volume, unit: FusionPetDisplayVolume(
            volume=np.asarray(volume, dtype=np.float32),
            unit="SUVbw",
            unit_label="g/ml (SUVbw)",
        ),
    )
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
        width=64,
        height=64,
    )
    view.window_width = 400.0
    view.window_center = 40.0

    result = service._render_fusion_view(view)

    assert result.meta.fusion_info is not None
    assert result.meta.fusion_info.pet_window_min == pytest.approx(0.0)
    assert result.meta.fusion_info.pet_window_max == pytest.approx(4.49)


def test_fusion_pet_bqml_can_be_displayed_as_suvbw() -> None:
    dataset = Dataset()
    dataset.Units = "BQML"
    dataset.PatientWeight = 70.0
    dataset.CorrectedImage = ["DECY"]
    dataset.DecayCorrection = "START"
    dataset.AcquisitionDate = "20200101"
    dataset.AcquisitionTime = "120000"
    radiopharmaceutical = Dataset()
    radiopharmaceutical.RadionuclideTotalDose = 350_000_000.0
    radiopharmaceutical.RadionuclideHalfLife = 6586.2
    radiopharmaceutical.RadiopharmaceuticalStartTime = "120000"
    dataset.RadiopharmaceuticalInformationSequence = [radiopharmaceutical]

    scale, unit, label = ViewerService()._resolve_pet_display_scale(dataset, "SUVbw")

    assert unit == "SUVbw"
    assert label == "g/ml (SUVbw)"
    assert scale == pytest.approx(0.0002)


def test_fusion_pet_missing_required_suv_fields_falls_back_to_source() -> None:
    dataset = Dataset()
    dataset.Units = "BQML"

    scale, unit, label = ViewerService()._resolve_pet_display_scale(dataset, "SUVbw")

    assert scale == pytest.approx(1.0)
    assert unit == "source"
    assert label == "BQML"


def test_fusion_registration_move_uses_immediate_overlay_fast_preview() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    series = SeriesRecord(
        series_id="ct",
        folder_path="",
        series_instance_uid="ct-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="CT",
    )

    move = _handle_fusion_registration_operation(
        service,
        view,
        series,
        ViewOperationRequest(
            viewId="overlay",
            opType="fusionRegistration",
            actionType="move",
            subOpType="rotate",
            x=10,
            y=0,
        ),
        False,
    )
    end = _handle_fusion_registration_operation(
        service,
        view,
        series,
        ViewOperationRequest(
            viewId="overlay",
            opType="fusionRegistration",
            actionType="end",
            subOpType="rotate",
            x=10,
            y=0,
        ),
        False,
    )

    assert move.mode == "single"
    assert move.fast_preview is True
    assert move.fast_preview_full_resolution is False
    assert move.broadcast_viewports is None
    assert end.mode == "broadcast"
    assert end.fast_preview is False
    assert end.broadcast_viewports is None


def test_fusion_registration_end_applies_final_delta_without_move() -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    series = SeriesRecord(
        series_id="ct",
        folder_path="",
        series_instance_uid="ct-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="CT",
    )

    _handle_fusion_registration_operation(
        service,
        view,
        series,
        ViewOperationRequest(
            viewId="overlay",
            opType="fusionRegistration",
            actionType="start",
            subOpType="rotate",
            x=0,
            y=0,
        ),
        False,
    )
    end = _handle_fusion_registration_operation(
        service,
        view,
        series,
        ViewOperationRequest(
            viewId="overlay",
            opType="fusionRegistration",
            actionType="end",
            subOpType="rotate",
            x=10,
            y=0,
        ),
        False,
    )

    assert end.mode == "broadcast"
    assert group.fusion_registration.rotation_degrees == pytest.approx(3.5)
    assert group.crosshair_drag_origin_center is None


def test_fusion_config_pet_unit_updates_group_and_resets_pet_window(monkeypatch) -> None:
    service = ViewerService()
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.fusion_ct_series_id = "ct"
    group.fusion_pet_series_id = "pet"
    group.fusion_pet_unit = "SUVbw"
    group.fusion_pet_window.window_width = 4.49
    group.fusion_pet_window.window_center = 2.245
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    ct_series = SeriesRecord(
        series_id="ct",
        folder_path="",
        series_instance_uid="ct-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="CT",
    )
    pet_series = SeriesRecord(
        series_id="pet",
        folder_path="",
        series_instance_uid="pet-uid",
        study_instance_uid="study",
        patient_id=None,
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="PT",
        series_description="PET",
    )

    monkeypatch.setattr(service, "_resolve_fusion_group_series", lambda _view: (group, ct_series, pet_series))
    monkeypatch.setattr(service, "_get_series_volume", lambda _series, **_: np.ones((2, 2, 2), dtype=np.float32))
    monkeypatch.setattr(
        service,
        "_build_fusion_pet_display_volume",
        lambda _series, volume, unit: FusionPetDisplayVolume(
            volume=np.asarray(volume, dtype=np.float32) * 0.001,
            unit="kBqml",
            unit_label="kBq/ml (uptake)",
            source_units="BQML",
            scale=0.001,
        ),
    )
    monkeypatch.setattr(service, "_derive_default_pet_window_for_display_volume", lambda _display: (2.0, 1.0))

    changed = service._handle_fusion_config(
        view,
        ViewOperationRequest(viewId="overlay", opType="fusionConfig", fusionPetUnit="kBqml"),
    )

    assert changed is True
    assert group.fusion_pet_unit == "kBqml"
    assert group.fusion_pet_window.window_width == pytest.approx(2.0)
    assert group.fusion_pet_window.window_center == pytest.approx(1.0)
    assert group.fusion_revision == 1


def test_fusion_pet_corner_info_uses_pet_window_label() -> None:
    display = FusionPetDisplayVolume(
        volume=np.zeros((2, 2, 2), dtype=np.float32),
        unit="SUVbw",
        unit_label="g/ml (SUVbw)",
    )
    corner_info = CornerInfoOverlay(
        bottom_left=("W: 8 L: 4", "2006.04.27"),
        tags={"windowLevel": ("W: 8 L: 4",)},
    )

    updated = ViewerService()._with_pet_window_corner_info(corner_info, display, 8.0, 4.0)

    assert updated.tags["windowLevel"] == ("SUV:0.00--8.00g/ml",)
    assert updated.bottom_left[0] == "SUV:0.00--8.00g/ml"
    assert "W: 8 L: 4" not in updated.bottom_left


def test_fusion_pet_corner_info_replaces_leaked_ct_window_label() -> None:
    display = FusionPetDisplayVolume(
        volume=np.zeros((2, 2, 2), dtype=np.float32),
        unit="SUVbw",
        unit_label="g/ml (SUVbw)",
    )
    corner_info = CornerInfoOverlay(
        bottom_left=("W: 400 L: 40", "2006.04.27"),
        tags={"windowLevel": ("W: 400 L: 40",)},
    )

    updated = ViewerService()._with_pet_window_corner_info(corner_info, display, 4.49, 2.245)

    assert updated.tags["windowLevel"] == ("SUV:0.00--4.49g/ml",)
    assert updated.bottom_left[0] == "SUV:0.00--4.49g/ml"
    assert "W: 400 L: 40" not in updated.bottom_left


def test_fusion_suv_default_window_matches_pet_display_range() -> None:
    display = FusionPetDisplayVolume(
        volume=np.linspace(0.0, 18.0, num=32, dtype=np.float32).reshape(2, 4, 4),
        unit="SUVbw",
        unit_label="g/ml (SUVbw)",
    )

    ww, wl = ViewerService()._derive_default_pet_window_for_display_volume(display)

    assert ww == pytest.approx(4.49)
    assert wl == pytest.approx(2.245)


def test_fusion_suv_default_window_uses_reference_suv_range_for_low_activity_pet() -> None:
    display = FusionPetDisplayVolume(
        volume=np.linspace(0.0, 0.8, num=32, dtype=np.float32).reshape(2, 4, 4),
        unit="SUVbw",
        unit_label="g/ml (SUVbw)",
    )

    ww, wl = ViewerService()._derive_default_pet_window_for_display_volume(display)

    assert ww == pytest.approx(4.49)
    assert wl == pytest.approx(2.245)


def test_fusion_config_updates_pet_window_range() -> None:
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.fusion_pet_window.window_width = 4.5
    group.fusion_pet_window.window_center = 2.25
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    payload = ViewOperationRequest(
        viewId="overlay",
        opType="fusionConfig",
        fusionPetWindowMin=0.0,
        fusionPetWindowMax=12.5,
    )

    changed = ViewerService()._handle_fusion_config(view, payload)

    assert changed is True
    assert group.fusion_pet_window.window_width == pytest.approx(12.5)
    assert group.fusion_pet_window.window_center == pytest.approx(6.25)


def test_fusion_config_allows_subunit_pet_window_range() -> None:
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.fusion_pet_window.window_width = 4.5
    group.fusion_pet_window.window_center = 2.25
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    payload = ViewOperationRequest(
        viewId="overlay",
        opType="fusionConfig",
        fusionPetWindowMin=0.0,
        fusionPetWindowMax=0.5,
    )

    changed = ViewerService()._handle_fusion_config(view, payload)

    assert changed is True
    assert group.fusion_pet_window.window_width == pytest.approx(0.5)
    assert group.fusion_pet_window.window_center == pytest.approx(0.25)


def test_fusion_pet_window_drag_keeps_lower_bound_at_zero() -> None:
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.fusion_pet_window.window_width = 4.5
    group.fusion_pet_window.window_center = 2.25
    view = ViewRecord(
        view_id="pet",
        series_id="pet",
        view_type="FusionPETAxial",
        fusion_pane_role=FUSION_PANE_PET_AXIAL,
        view_group=group,
    )
    service = ViewerService()

    service._handle_fusion_window(
        view,
        ViewOperationRequest(viewId="pet", opType="window", actionType="start"),
    )
    changed = service._handle_fusion_window(
        view,
        ViewOperationRequest(viewId="pet", opType="window", actionType="move", x=10.0, y=0.0),
    )

    assert changed is True
    assert service._resolve_window_min(
        group.fusion_pet_window.window_width,
        group.fusion_pet_window.window_center,
    ) == pytest.approx(0.0)
    assert service._resolve_window_max(
        group.fusion_pet_window.window_width,
        group.fusion_pet_window.window_center,
    ) == pytest.approx(4.95)


def test_fusion_pet_window_drag_sensitivity_scales_with_current_range() -> None:
    assert ViewerService._resolve_fusion_pet_window_drag_sensitivity(4.5) == pytest.approx(0.045)
    assert ViewerService._resolve_fusion_pet_window_drag_sensitivity(30.0) == pytest.approx(0.3)


def _export_test_series(series_id: str, modality: str, description: str) -> SeriesRecord:
    uid_suffix = {"ct": "1", "pet": "2"}.get(series_id, "9")
    return SeriesRecord(
        series_id=series_id,
        folder_path="",
        series_instance_uid=f"1.2.826.0.1.3680043.10.5432.{uid_suffix}",
        study_instance_uid="study",
        patient_id="patient",
        patient_name="Patient",
        study_date="20200101",
        study_description="Study",
        accession_number="ACC",
        modality=modality,
        series_description=description,
    )


def _export_test_group_and_view() -> tuple[ViewGroupRecord, ViewRecord]:
    group = ViewGroupRecord(group_id="fusion-group", group_type="fusion", series_id="ct")
    group.fusion_ct_series_id = "ct"
    group.fusion_pet_series_id = "pet"
    group.fusion_pet_unit = "SUVbw"
    group.fusion_pet_window.window_width = 4.49
    group.fusion_pet_window.window_center = 2.245
    group.fusion_registration.translate_row_mm = 1.5
    group.fusion_registration.translate_col_mm = -2.25
    group.fusion_registration.rotation_degrees = 6.0
    view = ViewRecord(
        view_id="overlay",
        series_id="ct",
        secondary_series_id="pet",
        view_type="FusionOverlayAxial",
        fusion_pane_role=FUSION_PANE_OVERLAY_AXIAL,
        view_group=group,
    )
    return (group, view)


def _patch_export_dependencies(
    monkeypatch,
    service: ViewerService,
    group: ViewGroupRecord,
    view: ViewRecord,
    *,
    ct_volume: np.ndarray | None = None,
    pet_volume: np.ndarray | None = None,
) -> tuple[SeriesRecord, SeriesRecord]:
    ct_series = _export_test_series("ct", "CT", "CT")
    pet_series = _export_test_series("pet", "PT", "PET FDG SUV")
    ct_data = ct_volume if ct_volume is not None else np.zeros((2, 3, 4), dtype=np.float32)
    pet_data = pet_volume if pet_volume is not None else np.arange(np.prod(ct_data.shape), dtype=np.float32).reshape(ct_data.shape)
    geometry = build_identity_geometry(tuple(int(value) for value in ct_data.shape))

    monkeypatch.setattr("app.services.viewer_service.view_registry.get", lambda *_args, **_kwargs: view)
    monkeypatch.setattr(service, "_resolve_fusion_group_series", lambda _view: (group, ct_series, pet_series))
    monkeypatch.setattr(
        service,
        "_get_series_volume",
        lambda series, **_: ct_data if series.series_id == "ct" else pet_data,
    )
    monkeypatch.setattr(service, "_get_series_volume_geometry", lambda _series, _shape: geometry)
    monkeypatch.setattr(
        service,
        "_build_fusion_pet_display_volume",
        lambda _series, volume, _unit: FusionPetDisplayVolume(
            volume=np.asarray(volume, dtype=np.float32),
            unit="SUVbw",
            unit_label="g/ml (SUVbw)",
            source_units="BQML",
            scale=0.0002,
        ),
    )
    return (ct_series, pet_series)


def _registration_sidecar_payload(
    ct_series: SeriesRecord,
    pet_series: SeriesRecord,
    *,
    translate_row_mm: float = 7.5,
    translate_col_mm: float = -3.25,
    rotation_degrees: float = 12.0,
    pet_unit: str = "kBqml",
) -> dict[str, object]:
    return {
        "format": "DicomVisionFusionRegistration",
        "version": 1,
        "seriesDescription": "PET FDG SUV_Reg",
        "ct": {
            "seriesId": ct_series.series_id,
            "seriesInstanceUid": ct_series.series_instance_uid,
        },
        "pet": {
            "seriesId": pet_series.series_id,
            "seriesInstanceUid": pet_series.series_instance_uid,
            "unit": pet_unit,
            "window": {"min": 0.25, "max": 9.5},
        },
        "registration": {
            "translateRowMm": translate_row_mm,
            "translateColMm": translate_col_mm,
            "rotationDegrees": rotation_degrees,
        },
    }


def test_fusion_registration_export_writes_br_sidecar(tmp_path, monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    _patch_export_dependencies(monkeypatch, service, group, view)

    response = service.export_fusion_registration(
        FusionRegistrationExportRequest(
            viewId="overlay",
            mode="br",
            seriesDescription="PET FDG SUV_Reg",
            outputDirectory=str(tmp_path),
        )
    )

    assert response.mode == "br"
    assert response.file_count == 1
    assert response.pet_unit == "SUVbw"
    assert group.fusion_registration.saved is True
    file_path = Path(response.file_path or "")
    assert file_path.exists()
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    assert payload["format"] == "DicomVisionFusionRegistration"
    assert payload["ct"]["seriesId"] == "ct"
    assert payload["pet"]["seriesId"] == "pet"
    assert payload["pet"]["unit"] == "SUVbw"
    assert payload["registration"]["translateRowMm"] == pytest.approx(1.5)
    assert payload["registration"]["translateColMm"] == pytest.approx(-2.25)
    assert payload["registration"]["rotationDegrees"] == pytest.approx(6.0)


def test_fusion_registration_export_writes_derived_dicom_series(tmp_path, monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    _, pet_series = _patch_export_dependencies(
        monkeypatch,
        service,
        group,
        view,
        ct_volume=np.zeros((3, 4, 5), dtype=np.float32),
        pet_volume=np.arange(60, dtype=np.float32).reshape(3, 4, 5),
    )
    reference = Dataset()
    reference.SOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
    reference.SOPInstanceUID = "1.2.3.4.5"
    reference.SeriesInstanceUID = pet_series.series_instance_uid
    reference.Modality = "PT"
    reference.SeriesNumber = "7"
    monkeypatch.setattr(service, "_get_reference_instance_and_cache", lambda _series: (None, SimpleNamespace(dataset=reference)))

    response = service.export_fusion_registration(
        FusionRegistrationExportRequest(
            viewId="overlay",
            mode="newDicom",
            seriesDescription="PET FDG SUV_Reg",
            outputDirectory=str(tmp_path),
        )
    )

    output_dir = Path(response.directory_path)
    files = sorted(output_dir.glob("*.dcm"))
    assert response.mode == "newDicom"
    assert response.file_count == 3
    assert len(files) == 3
    assert response.file_path is None
    assert group.fusion_registration.saved is True

    dataset = dcmread(str(files[0]))
    assert dataset.SeriesDescription == "PET FDG SUV_Reg"
    assert dataset.SeriesInstanceUID != pet_series.series_instance_uid
    assert dataset.SOPInstanceUID != reference.SOPInstanceUID
    assert dataset.Rows == 4
    assert dataset.Columns == 5
    assert int(dataset.SeriesNumber) == 1007
    assert dataset.Units == "GML"
    assert dataset[(0x0011, 0x1001)].value == "SUVbw"
    assert dataset[(0x0011, 0x1002)].value == "g/ml (SUVbw)"
    assert float(dataset[(0x0011, 0x1003)].value) == pytest.approx(1.5)
    assert float(dataset[(0x0011, 0x1004)].value) == pytest.approx(-2.25)
    assert float(dataset[(0x0011, 0x1005)].value) == pytest.approx(6.0)


def test_fusion_registration_artifact_exports_br_json(monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    _patch_export_dependencies(monkeypatch, service, group, view)

    result = service.export_fusion_registration_artifact(
        FusionRegistrationArtifactExportRequest(
            viewId="overlay",
            mode="br",
            seriesDescription="PET FDG SUV_Reg",
        )
    )

    payload = json.loads(result.file_bytes.decode("utf-8"))
    assert result.file_name == "PET-FDG-SUV_Reg.br"
    assert result.media_type == "application/json"
    assert result.extra_headers == {"x-dicomvision-artifact-kind": "br", "x-dicomvision-file-count": "1"}
    assert payload["format"] == "DicomVisionFusionRegistration"
    assert payload["pet"]["unit"] == "SUVbw"
    assert payload["registration"]["translateRowMm"] == pytest.approx(1.5)
    assert group.fusion_registration.saved is True


def test_fusion_registration_artifact_exports_derived_dicom_zip(monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    _, pet_series = _patch_export_dependencies(
        monkeypatch,
        service,
        group,
        view,
        ct_volume=np.zeros((2, 3, 4), dtype=np.float32),
        pet_volume=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    )
    reference = Dataset()
    reference.SOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
    reference.SOPInstanceUID = "1.2.3.4.5"
    reference.SeriesInstanceUID = pet_series.series_instance_uid
    reference.Modality = "PT"
    monkeypatch.setattr(service, "_get_reference_instance_and_cache", lambda _series: (None, SimpleNamespace(dataset=reference)))

    result = service.export_fusion_registration_artifact(
        FusionRegistrationArtifactExportRequest(
            viewId="overlay",
            mode="newDicom",
            seriesDescription="PET FDG SUV_Reg",
        )
    )

    assert result.file_name == "PET-FDG-SUV_Reg.zip"
    assert result.media_type == "application/zip"
    assert result.extra_headers == {"x-dicomvision-artifact-kind": "zip", "x-dicomvision-file-count": "2"}
    with zipfile.ZipFile(io.BytesIO(result.file_bytes)) as archive:
        names = archive.namelist()
        assert names == ["PET-FDG-SUV_Reg/IM000001.dcm", "PET-FDG-SUV_Reg/IM000002.dcm"]
        dataset = dcmread(io.BytesIO(archive.read(names[0])))
    assert dataset.SeriesDescription == "PET FDG SUV_Reg"
    assert dataset.SeriesInstanceUID != pet_series.series_instance_uid
    assert dataset.Rows == 3
    assert dataset.Columns == 4
    assert dataset[(0x0011, 0x1001)].value == "SUVbw"
    assert group.fusion_registration.saved is True


def test_fusion_registration_load_applies_matching_br_payload(monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    ct_series, pet_series = _patch_export_dependencies(monkeypatch, service, group, view)
    payload = _registration_sidecar_payload(ct_series, pet_series)

    changed = service._handle_fusion_registration(
        view,
        ViewOperationRequest(
            viewId="overlay",
            opType="fusionRegistration",
            subOpType="load",
            fusionRegistrationFile=payload,
        ),
    )

    assert changed is True
    assert group.fusion_registration.translate_row_mm == pytest.approx(7.5)
    assert group.fusion_registration.translate_col_mm == pytest.approx(-3.25)
    assert group.fusion_registration.rotation_degrees == pytest.approx(12.0)
    assert group.fusion_registration.saved is True
    assert group.fusion_pet_unit == "kBqml"
    assert group.fusion_pet_window.window_width == pytest.approx(9.25)
    assert group.fusion_pet_window.window_center == pytest.approx(4.875)
    assert group.fusion_revision == 1


def test_fusion_registration_load_rejects_mismatched_series(monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    ct_series, pet_series = _patch_export_dependencies(monkeypatch, service, group, view)
    payload = _registration_sidecar_payload(ct_series, pet_series)
    payload["pet"] = {**payload["pet"], "seriesId": "other-pet", "seriesInstanceUid": "9.9.9"}

    with pytest.raises(HTTPException, match="PET series"):
        service._handle_fusion_registration(
            view,
            ViewOperationRequest(
                viewId="overlay",
                opType="fusionRegistration",
                subOpType="load",
                fusionRegistrationFile=payload,
            ),
        )


def test_fusion_registration_load_rejects_invalid_payload(monkeypatch) -> None:
    service = ViewerService()
    group, view = _export_test_group_and_view()
    ct_series, pet_series = _patch_export_dependencies(monkeypatch, service, group, view)
    payload = _registration_sidecar_payload(ct_series, pet_series, rotation_degrees=float("nan"))

    with pytest.raises(HTTPException, match="finite number"):
        service._handle_fusion_registration(
            view,
            ViewOperationRequest(
                viewId="overlay",
                opType="fusionRegistration",
                subOpType="load",
                fusionRegistrationFile=payload,
            ),
        )

    with pytest.raises(HTTPException, match="Unsupported registration file format"):
        service._handle_fusion_registration(
            view,
            ViewOperationRequest(
                viewId="overlay",
                opType="fusionRegistration",
                subOpType="load",
                fusionRegistrationFile={"format": "Other"},
            ),
        )
