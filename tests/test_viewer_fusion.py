from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
from app.services.viewer_service import FusionPetDisplayVolume, ViewerService
from app.schemas.view import ViewOperationRequest, ViewSetSizeRequest, ViewSize


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
