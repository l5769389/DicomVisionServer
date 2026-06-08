import numpy as np
import pytest

from app.core import (
    FUSION_PANE_CT_AXIAL,
    FUSION_PANE_OVERLAY_AXIAL,
    FUSION_PANE_PET_AXIAL,
    FUSION_PANE_PET_CORONAL_MIP,
)
from app.models.viewer import FusionRegistrationState, ViewRecord
from app.services.mpr import VolumeGeometry, build_identity_geometry
from app.services.viewer_fusion import render_fusion_pixels
from app.services.viewer_service import ViewerService


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
    assert result.slice_index == 2
    assert result.slice_total == 5

    overlay = ViewerService()._build_direction_orientation_overlay(
        ViewRecord(view_id="fusion-view", series_id="series", view_type="FusionOverlayAxial"),
        result.row_world,
        result.col_world,
    )
    assert overlay is not None
    assert all(value for value in (overlay.top, overlay.right, overlay.bottom, overlay.left))


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
        (FUSION_PANE_OVERLAY_AXIAL, "pet"),
        (FUSION_PANE_PET_CORONAL_MIP, "bwinverse"),
    ],
)
def test_fusion_result_reports_actual_rendered_pseudocolor(role: str, expected_preset: str) -> None:
    result = _render(role)

    assert result.pseudocolor_preset == expected_preset


def test_pet_only_views_preserve_user_selected_non_default_pseudocolor() -> None:
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

    assert result.pseudocolor_preset == "hotiron"


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
