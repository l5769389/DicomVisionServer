from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from app.core import (
    FUSION_PANE_CT_AXIAL,
    FUSION_PANE_OVERLAY_AXIAL,
    FUSION_PANE_PET_AXIAL,
    FUSION_PANE_PET_CORONAL_MIP,
)
from app.models.viewer import FusionRegistrationState, MprFrameState
from app.services.mpr import (
    MipConfig,
    PlanePose,
    VolumeGeometry,
    legacy_frame_to_cursor,
    reslice_plane,
)
from app.services.pseudocolor import DEFAULT_PSEUDOCOLOR_PRESET, apply_pseudocolor, normalize_pseudocolor_preset


FUSION_VIEW_TYPES = {
    "FusionCTAxial",
    "FusionPETAxial",
    "FusionOverlayAxial",
    "FusionPETCoronalMip",
}

FUSION_VIEW_TYPE_TO_PANE_ROLE = {
    "FusionCTAxial": FUSION_PANE_CT_AXIAL,
    "FusionPETAxial": FUSION_PANE_PET_AXIAL,
    "FusionOverlayAxial": FUSION_PANE_OVERLAY_AXIAL,
    "FusionPETCoronalMip": FUSION_PANE_PET_CORONAL_MIP,
}

FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET = "bwinverse"


@dataclass(frozen=True)
class FusionSourceProjection:
    source_shape: tuple[int, int]
    source_to_world_origin: np.ndarray
    source_to_world_x: np.ndarray
    source_to_world_y: np.ndarray
    world_to_source_x: np.ndarray
    world_to_source_y: np.ndarray
    reference_world: np.ndarray


@dataclass(frozen=True)
class FusionRenderResult:
    pixels: np.ndarray
    spacing_xy: tuple[float, float]
    slice_index: int
    slice_total: int
    row_world: np.ndarray | None
    col_world: np.ndarray | None
    pseudocolor_preset: str
    source_projection: FusionSourceProjection | None


def _axis_direction_and_spacing(geometry: VolumeGeometry, axis_index: int) -> tuple[np.ndarray, float]:
    axis = np.asarray(geometry.ijk_to_world[:3, axis_index], dtype=np.float64)
    spacing = max(float(np.linalg.norm(axis)), 1e-6)
    return axis / spacing, spacing


def clamp_fusion_axial_index(index: int, ct_shape: tuple[int, int, int]) -> int:
    return max(0, min(int(index), int(ct_shape[0]) - 1))


def _map_plane_center_to_volume_axis_index(
    plane: PlanePose,
    geometry: VolumeGeometry,
    volume_shape: tuple[int, int, int],
    *,
    axis_index: int = 0,
) -> int:
    axis_size = int(volume_shape[axis_index])
    if axis_size <= 0:
        return 0
    try:
        center = np.asarray([*plane.center_world, 1.0], dtype=np.float64)
        ijk = np.asarray(geometry.world_to_ijk, dtype=np.float64) @ center
        coordinate = float(ijk[axis_index])
    except Exception:
        coordinate = 0.0
    if not np.isfinite(coordinate):
        coordinate = 0.0
    return max(0, min(int(round(coordinate)), axis_size - 1))


def _plane_source_projection(
    plane: PlanePose,
    *,
    has_patient_geometry: bool,
    reference_world: np.ndarray | None = None,
) -> FusionSourceProjection | None:
    if not has_patient_geometry:
        return None
    height, width = (int(plane.output_shape[0]), int(plane.output_shape[1]))
    if height <= 0 or width <= 0:
        return None
    center_world = np.asarray(plane.center_world, dtype=np.float64)
    row_world = np.asarray(plane.row_world, dtype=np.float64)
    col_world = np.asarray(plane.col_world, dtype=np.float64)
    row_spacing = max(float(plane.pixel_spacing_row_mm), 1e-6)
    col_spacing = max(float(plane.pixel_spacing_col_mm), 1e-6)
    row_center = (float(height) - 1.0) / 2.0
    col_center = (float(width) - 1.0) / 2.0
    source_to_world_x = col_world * col_spacing
    source_to_world_y = row_world * row_spacing
    source_to_world_origin = center_world - source_to_world_x * col_center - source_to_world_y * row_center
    world_to_source_x = np.asarray(
        [
            float(col_world[0]) / col_spacing,
            float(col_world[1]) / col_spacing,
            float(col_world[2]) / col_spacing,
            col_center - float(np.dot(col_world, center_world)) / col_spacing,
        ],
        dtype=np.float64,
    )
    world_to_source_y = np.asarray(
        [
            float(row_world[0]) / row_spacing,
            float(row_world[1]) / row_spacing,
            float(row_world[2]) / row_spacing,
            row_center - float(np.dot(row_world, center_world)) / row_spacing,
        ],
        dtype=np.float64,
    )
    return FusionSourceProjection(
        source_shape=(height, width),
        source_to_world_origin=source_to_world_origin,
        source_to_world_x=source_to_world_x,
        source_to_world_y=source_to_world_y,
        world_to_source_x=world_to_source_x,
        world_to_source_y=world_to_source_y,
        reference_world=np.asarray(
            reference_world if reference_world is not None else plane.cursor_center_world,
            dtype=np.float64,
        ),
    )


def _mip_source_projection(
    pet_geometry: VolumeGeometry,
    pet_shape: tuple[int, int, int],
    *,
    has_patient_geometry: bool,
    reference_world: np.ndarray,
) -> FusionSourceProjection | None:
    if not has_patient_geometry:
        return None
    height = int(pet_shape[0])
    width = int(pet_shape[2])
    depth = int(pet_shape[1])
    if height <= 0 or width <= 0 or depth <= 0:
        return None
    j_mid = (float(depth) - 1.0) / 2.0
    origin_ijk = np.asarray([float(height) - 1.0, j_mid, 0.0, 1.0], dtype=np.float64)
    source_to_world_origin = np.asarray(pet_geometry.ijk_to_world, dtype=np.float64) @ origin_ijk
    source_to_world_x = np.asarray(pet_geometry.ijk_to_world[:3, 2], dtype=np.float64)
    source_to_world_y = -np.asarray(pet_geometry.ijk_to_world[:3, 0], dtype=np.float64)
    world_to_ijk = np.asarray(pet_geometry.world_to_ijk, dtype=np.float64)
    world_to_source_x = np.asarray(
        [
            float(world_to_ijk[2, 0]),
            float(world_to_ijk[2, 1]),
            float(world_to_ijk[2, 2]),
            float(world_to_ijk[2, 3]),
        ],
        dtype=np.float64,
    )
    world_to_source_y = np.asarray(
        [
            -float(world_to_ijk[0, 0]),
            -float(world_to_ijk[0, 1]),
            -float(world_to_ijk[0, 2]),
            float(height) - 1.0 - float(world_to_ijk[0, 3]),
        ],
        dtype=np.float64,
    )
    return FusionSourceProjection(
        source_shape=(height, width),
        source_to_world_origin=source_to_world_origin[:3],
        source_to_world_x=source_to_world_x,
        source_to_world_y=source_to_world_y,
        world_to_source_x=world_to_source_x,
        world_to_source_y=world_to_source_y,
        reference_world=np.asarray(reference_world, dtype=np.float64),
    )


def build_ct_axial_plane(
    ct_geometry: VolumeGeometry,
    ct_shape: tuple[int, int, int],
    index: int,
) -> PlanePose:
    axial_index = clamp_fusion_axial_index(index, ct_shape)
    frame = MprFrameState(
        center=(float(axial_index), (float(ct_shape[1]) - 1.0) / 2.0, (float(ct_shape[2]) - 1.0) / 2.0),
        axis_slice=(1.0, 0.0, 0.0),
        axis_row=(0.0, 1.0, 0.0),
        axis_col=(0.0, 0.0, 1.0),
    )
    cursor = legacy_frame_to_cursor(frame, ct_geometry, reference_center=frame.center)
    orientation = np.asarray(cursor.orientation_world, dtype=np.float64)
    row_world = orientation[:, 1]
    col_world = orientation[:, 2]
    normal_world = orientation[:, 0]
    center_world = np.asarray(cursor.center_world, dtype=np.float64)
    row_spacing = max(float(np.linalg.norm(ct_geometry.ijk_to_world[:3, 1])), 1e-6)
    col_spacing = max(float(np.linalg.norm(ct_geometry.ijk_to_world[:3, 2])), 1e-6)
    return PlanePose(
        viewport="fusion-axial",
        center_world=center_world,
        cursor_center_world=center_world,
        row_world=row_world / max(float(np.linalg.norm(row_world)), 1e-6),
        col_world=col_world / max(float(np.linalg.norm(col_world)), 1e-6),
        normal_world=normal_world / max(float(np.linalg.norm(normal_world)), 1e-6),
        pixel_spacing_row_mm=row_spacing,
        pixel_spacing_col_mm=col_spacing,
        output_shape=(int(ct_shape[1]), int(ct_shape[2])),
        is_oblique=False,
    )


def transform_pet_sampling_plane(plane: PlanePose, registration: FusionRegistrationState) -> PlanePose:
    angle_rad = -np.deg2rad(float(registration.rotation_degrees))
    cos_angle = float(np.cos(angle_rad))
    sin_angle = float(np.sin(angle_rad))
    normal = np.asarray(plane.normal_world, dtype=np.float64)
    row = np.asarray(plane.row_world, dtype=np.float64)
    col = np.asarray(plane.col_world, dtype=np.float64)
    rotation = (
        np.eye(3, dtype=np.float64) * cos_angle
        + (1.0 - cos_angle) * np.outer(normal, normal)
        + sin_angle
        * np.asarray(
            [
                [0.0, -normal[2], normal[1]],
                [normal[2], 0.0, -normal[0]],
                [-normal[1], normal[0], 0.0],
            ],
            dtype=np.float64,
        )
    )
    translation_world = (
        row * float(registration.translate_row_mm)
        + col * float(registration.translate_col_mm)
    )
    pivot = np.asarray(plane.cursor_center_world, dtype=np.float64)
    center = np.asarray(plane.center_world, dtype=np.float64)
    next_center = pivot + rotation @ (center - pivot - translation_world)
    return replace(
        plane,
        center_world=next_center,
        row_world=rotation @ row,
        col_world=rotation @ col,
        normal_world=rotation @ normal,
    )


def window_to_uint8(pixels: np.ndarray, ww: float | None, wl: float | None) -> np.ndarray:
    source = np.asarray(pixels, dtype=np.float32)
    if ww is None or wl is None or float(ww) <= 0.0:
        pixel_min = float(np.nanmin(source)) if source.size else 0.0
        pixel_max = float(np.nanmax(source)) if source.size else 1.0
        ww = max(pixel_max - pixel_min, 1.0)
        wl = (pixel_min + pixel_max) / 2.0
    low = float(wl) - float(ww) / 2.0
    high = float(wl) + float(ww) / 2.0
    if high <= low:
        high = low + 1.0
    return np.clip((source - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)


def render_fusion_pixels(
    *,
    pane_role: str,
    ct_volume: np.ndarray,
    ct_geometry: VolumeGeometry,
    pet_volume: np.ndarray,
    pet_geometry: VolumeGeometry,
    axial_index: int,
    ct_window_width: float | None,
    ct_window_center: float | None,
    pet_window_width: float | None,
    pet_window_center: float | None,
    pet_pseudocolor_preset: str,
    registration: FusionRegistrationState,
    alpha: float,
    ct_has_patient_geometry: bool,
    pet_has_patient_geometry: bool,
    interpolation_order: int = 1,
) -> FusionRenderResult:
    ct_shape = tuple(int(value) for value in ct_volume.shape)
    pet_shape = tuple(int(value) for value in pet_volume.shape)
    axial_index = clamp_fusion_axial_index(axial_index, ct_shape)
    plane = build_ct_axial_plane(ct_geometry, ct_shape, axial_index)
    pet_plane = transform_pet_sampling_plane(plane, registration)
    reference_world = np.asarray(plane.cursor_center_world, dtype=np.float64)
    pet_slice_index = _map_plane_center_to_volume_axis_index(pet_plane, pet_geometry, pet_shape)

    if pane_role == FUSION_PANE_PET_CORONAL_MIP:
        pet_mip = np.max(np.asarray(pet_volume, dtype=np.float32), axis=1)
        pet_mip = np.flipud(pet_mip)
        pet_uint8 = window_to_uint8(pet_mip, pet_window_width, pet_window_center)
        pet_display_preset = FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET
        pet_rgb = apply_pseudocolor(pet_uint8, pet_display_preset)
        row_world, row_spacing = _axis_direction_and_spacing(pet_geometry, 0)
        col_world, col_spacing = _axis_direction_and_spacing(pet_geometry, 2)
        return FusionRenderResult(
            pixels=pet_rgb,
            spacing_xy=(col_spacing, row_spacing),
            slice_index=0,
            slice_total=1,
            row_world=-row_world if pet_has_patient_geometry else None,
            col_world=col_world if pet_has_patient_geometry else None,
            pseudocolor_preset=pet_display_preset,
            source_projection=_mip_source_projection(
                pet_geometry,
                pet_shape,
                has_patient_geometry=pet_has_patient_geometry,
                reference_world=reference_world,
            ),
        )

    ct_slice = np.asarray(ct_volume[axial_index, :, :], dtype=np.float32)
    pet_slice = reslice_plane(
        pet_volume,
        pet_geometry,
        pet_plane,
        MipConfig(enabled=False),
        interpolation_order=interpolation_order,
    )
    if pane_role == FUSION_PANE_CT_AXIAL:
        ct_uint8 = window_to_uint8(ct_slice, ct_window_width, ct_window_center)
        return FusionRenderResult(
            pixels=ct_uint8,
            spacing_xy=(plane.pixel_spacing_col_mm, plane.pixel_spacing_row_mm),
            slice_index=axial_index,
            slice_total=ct_shape[0],
            row_world=plane.row_world if ct_has_patient_geometry else None,
            col_world=plane.col_world if ct_has_patient_geometry else None,
            pseudocolor_preset=DEFAULT_PSEUDOCOLOR_PRESET,
            source_projection=_plane_source_projection(
                plane,
                has_patient_geometry=ct_has_patient_geometry,
                reference_world=reference_world,
            ),
        )

    pet_uint8 = window_to_uint8(pet_slice, pet_window_width, pet_window_center)
    if pane_role == FUSION_PANE_PET_AXIAL:
        pet_display_preset = FUSION_PET_STANDALONE_PSEUDOCOLOR_PRESET
        pet_rgb = apply_pseudocolor(pet_uint8, pet_display_preset)
        return FusionRenderResult(
            pixels=pet_rgb,
            spacing_xy=(plane.pixel_spacing_col_mm, plane.pixel_spacing_row_mm),
            slice_index=pet_slice_index,
            slice_total=pet_shape[0],
            row_world=pet_plane.row_world if pet_has_patient_geometry else None,
            col_world=pet_plane.col_world if pet_has_patient_geometry else None,
            pseudocolor_preset=pet_display_preset,
            source_projection=_plane_source_projection(
                pet_plane,
                has_patient_geometry=pet_has_patient_geometry,
                reference_world=reference_world,
            ),
        )

    overlay_preset = normalize_pseudocolor_preset(pet_pseudocolor_preset)
    pet_rgb = apply_pseudocolor(pet_uint8, overlay_preset)
    ct_uint8 = window_to_uint8(ct_slice, ct_window_width, ct_window_center)
    ct_rgb = np.repeat(ct_uint8[..., None], 3, axis=-1)
    pet_alpha = np.clip(float(alpha), 0.0, 1.0)
    pet_mask = (pet_uint8.astype(np.float32) / 255.0)[..., None]
    blend_alpha = pet_alpha * pet_mask
    fused = ct_rgb.astype(np.float32) * (1.0 - blend_alpha) + pet_rgb.astype(np.float32) * blend_alpha
    return FusionRenderResult(
        pixels=np.clip(fused, 0.0, 255.0).astype(np.uint8),
        spacing_xy=(plane.pixel_spacing_col_mm, plane.pixel_spacing_row_mm),
        slice_index=axial_index,
        slice_total=ct_shape[0],
        row_world=plane.row_world if ct_has_patient_geometry else None,
        col_world=plane.col_world if ct_has_patient_geometry else None,
        pseudocolor_preset=overlay_preset,
        source_projection=_plane_source_projection(
            plane,
            has_patient_geometry=ct_has_patient_geometry,
            reference_world=reference_world,
        ),
    )


def image_from_pixels(pixels: np.ndarray) -> Image.Image:
    array = np.asarray(pixels)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8, copy=False), mode="L")
    return Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB")
