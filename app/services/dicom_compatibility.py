from dataclasses import dataclass

from app.models.viewer import InstanceRecord, SeriesRecord
import numpy as np

from app.schemas.dicom import DicomCompatibilityIssue, DicomCompatibilitySeverity, SeriesViewCapability


@dataclass(frozen=True)
class SeriesVolumeCompatibility:
    supported: bool
    blocked_code: str | None = None
    blocked_reason: str | None = None

    def to_view_capability(self) -> SeriesViewCapability:
        return SeriesViewCapability(
            supported=self.supported,
            blockedCode=self.blocked_code,
            blockedReason=self.blocked_reason,
        )


def _finite_vector(value: object, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not bool(np.all(np.isfinite(vector))):
        return None
    return vector.copy()


def _normalized_orientation(value: object) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    orientation = _finite_vector(value, 6)
    if orientation is None:
        return None
    row = orientation[:3]
    column = orientation[3:]
    row_norm = float(np.linalg.norm(row))
    column_norm = float(np.linalg.norm(column))
    if row_norm <= 1e-6 or column_norm <= 1e-6:
        return None
    row /= row_norm
    column /= column_norm
    if abs(float(np.dot(row, column))) > 1e-3:
        return None
    normal = np.cross(row, column)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-6:
        return None
    return row, column, normal / normal_norm


def _normalized_pixel_spacing(value: object) -> np.ndarray | None:
    spacing = _finite_vector(value, 2)
    if spacing is None or bool(np.any(spacing <= 0.0)):
        return None
    return np.abs(spacing)


def get_instances_volume_compatibility(instances: list[InstanceRecord]) -> SeriesVolumeCompatibility:
    if any((instance.number_of_frames or 1) > 1 for instance in instances):
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="multiframe-unsupported",
            blocked_reason="Enhanced or multi-frame DICOM is not supported yet.",
        )
    if len(instances) <= 1:
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="insufficient-slices",
            blocked_reason="MPR, 3D, and 4D require at least two image slices.",
        )

    dimensions = {(instance.rows, instance.columns) for instance in instances}
    if any(
        not isinstance(instance.rows, int)
        or isinstance(instance.rows, bool)
        or instance.rows <= 0
        or not isinstance(instance.columns, int)
        or isinstance(instance.columns, bool)
        or instance.columns <= 0
        for instance in instances
    ):
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="missing-image-size",
            blocked_reason="MPR, 3D, and 4D require valid Rows and Columns on every slice.",
        )
    if len(dimensions) > 1:
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="mixed-image-size",
            blocked_reason="The series contains slices with different Rows or Columns values.",
        )

    pixel_spacings = [_normalized_pixel_spacing(instance.pixel_spacing) for instance in instances]
    if any(spacing is None for spacing in pixel_spacings):
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="missing-pixel-spacing",
            blocked_reason="MPR, 3D, and 4D require valid PixelSpacing on every slice.",
        )
    reference_spacing = pixel_spacings[0]
    assert reference_spacing is not None
    if any(
        spacing is not None
        and not bool(np.allclose(spacing, reference_spacing, rtol=1e-3, atol=1e-4))
        for spacing in pixel_spacings[1:]
    ):
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="mixed-pixel-spacing",
            blocked_reason="The series contains inconsistent PixelSpacing values and requires resampling before volume display.",
        )

    if any(instance.image_orientation_patient is None or instance.image_position_patient is None for instance in instances):
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="missing-spatial-geometry",
            blocked_reason="MPR, 3D, and fusion require ImageOrientationPatient and ImagePositionPatient on every slice.",
        )

    reference_axes = _normalized_orientation(instances[0].image_orientation_patient)
    if reference_axes is None:
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="invalid-slice-orientation",
            blocked_reason="The series contains an invalid slice orientation.",
        )
    reference_row, reference_column, reference_normal = reference_axes

    projections: list[float] = []
    for instance in instances:
        axes = _normalized_orientation(instance.image_orientation_patient)
        position = _finite_vector(instance.image_position_patient, 3)
        if axes is None or position is None:
            return SeriesVolumeCompatibility(
                supported=False,
                blocked_code="invalid-spatial-geometry",
                blocked_reason="The series contains invalid orientation or position coordinates.",
            )
        row, column, normal = axes
        if (
            float(np.dot(row, reference_row)) < 0.999
            or float(np.dot(column, reference_column)) < 0.999
            or float(np.dot(normal, reference_normal)) < 0.999
        ):
            return SeriesVolumeCompatibility(
                supported=False,
                blocked_code="mixed-slice-orientations",
                blocked_reason="The series contains mixed in-plane or slice orientations and requires resampling before volume display.",
            )
        projections.append(float(np.dot(position, reference_normal)))

    projections.sort()
    spacing = np.diff(np.asarray(projections, dtype=np.float64))
    if spacing.size and bool(np.any(spacing <= 1e-3)):
        return SeriesVolumeCompatibility(
            supported=False,
            blocked_code="duplicate-slice-positions",
            blocked_reason="The series contains duplicate slice positions.",
        )
    if spacing.size > 1:
        median_spacing = float(np.median(spacing))
        tolerance = max(0.1, median_spacing * 0.05)
        if bool(np.any(np.abs(spacing - median_spacing) > tolerance)):
            return SeriesVolumeCompatibility(
                supported=False,
                blocked_code="irregular-slice-spacing",
                blocked_reason="The series contains irregular slice spacing and requires resampling before volume display.",
            )
    return SeriesVolumeCompatibility(supported=True)


def get_series_volume_compatibility(series: SeriesRecord) -> SeriesVolumeCompatibility:
    return get_instances_volume_compatibility(series.instances)


def get_series_volume_block_reason(series: SeriesRecord) -> str | None:
    return get_series_volume_compatibility(series).blocked_reason


def build_series_view_capabilities(series: SeriesRecord) -> dict[str, SeriesViewCapability]:
    multiframe_reason = (
        "Enhanced or multi-frame DICOM is not supported yet."
        if any((instance.number_of_frames or 1) > 1 for instance in series.instances)
        else None
    )
    volume_capability = get_series_volume_compatibility(series).to_view_capability()
    return {
        "stack": SeriesViewCapability(
            supported=multiframe_reason is None,
            blockedCode="multiframe-unsupported" if multiframe_reason else None,
            blockedReason=multiframe_reason,
        ),
        "montage": SeriesViewCapability(
            supported=multiframe_reason is None,
            blockedCode="multiframe-unsupported" if multiframe_reason else None,
            blockedReason=multiframe_reason,
        ),
        "mpr": volume_capability.model_copy(),
        "3d": volume_capability.model_copy(),
        "fusion": volume_capability.model_copy(),
        "4d": SeriesViewCapability(
            supported=False,
            blockedCode="not-four-d-series",
            blockedReason="The series does not contain at least two detectable 4D phases.",
        ),
    }


def build_dicom_compatibility_issues(series: SeriesRecord) -> list[DicomCompatibilityIssue]:
    instances = series.instances
    if not instances:
        return []

    total_count = len(instances)
    issues: list[DicomCompatibilityIssue] = []

    def add_issue(
        code: str,
        severity: DicomCompatibilitySeverity,
        title: str,
        detail: str,
        affected_instances: int,
    ) -> None:
        if affected_instances <= 0:
            return
        issues.append(
            DicomCompatibilityIssue(
                code=code,
                severity=severity,
                title=title,
                detail=detail,
                affectedInstances=affected_instances,
            )
        )

    invalid_size_count = sum(
        1
        for instance in instances
        if _safe_positive_int(instance.rows) is None or _safe_positive_int(instance.columns) is None
    )
    add_issue(
        "missing-image-size",
        "error",
        "Missing image dimensions",
        "Rows or Columns are absent or invalid; this series may fail to display.",
        invalid_size_count,
    )

    dimensions = {
        (
            _safe_positive_int(instance.rows),
            _safe_positive_int(instance.columns),
        )
        for instance in instances
        if _safe_positive_int(instance.rows) is not None and _safe_positive_int(instance.columns) is not None
    }
    if len(dimensions) > 1:
        add_issue(
            "mixed-image-size",
            "warning",
            "Mixed image dimensions",
            "Instances in this series use different Rows/Columns values; stack and MPR geometry may be inconsistent.",
            total_count,
        )

    compressed_instances = [instance for instance in instances if instance.transfer_syntax_is_compressed]
    if compressed_instances:
        transfer_names = sorted(
            {
                instance.transfer_syntax_name or instance.transfer_syntax_uid or "compressed transfer syntax"
                for instance in compressed_instances
            }
        )
        add_issue(
            "compressed-transfer-syntax",
            "warning",
            "Compressed transfer syntax",
            f"Pixel decoding depends on installed DICOM codecs: {', '.join(transfer_names[:3])}.",
            len(compressed_instances),
        )

    missing_transfer_syntax_count = sum(1 for instance in instances if not instance.transfer_syntax_uid)
    add_issue(
        "missing-transfer-syntax",
        "warning",
        "Missing transfer syntax",
        "File meta TransferSyntaxUID is missing; decoding behavior may vary by reader.",
        missing_transfer_syntax_count,
    )

    unsupported_photometric_instances = [
        instance
        for instance in instances
        if (
            instance.photometric_interpretation
            and instance.photometric_interpretation.upper() not in {"MONOCHROME1", "MONOCHROME2"}
        )
        or (instance.samples_per_pixel is not None and instance.samples_per_pixel > 1)
    ]
    if unsupported_photometric_instances:
        photometric_values = sorted(
            {
                instance.photometric_interpretation or f"{instance.samples_per_pixel} samples per pixel"
                for instance in unsupported_photometric_instances
            }
        )
        add_issue(
            "unsupported-photometric",
            "warning",
            "Non-monochrome pixel data",
            f"The viewer is optimized for MONOCHROME images; found {', '.join(photometric_values[:3])}.",
            len(unsupported_photometric_instances),
        )

    multi_frame_instances = [
        instance for instance in instances if instance.number_of_frames is not None and instance.number_of_frames > 1
    ]
    add_issue(
        "multiframe-unsupported",
        "error",
        "Unsupported multi-frame instances",
        "Enhanced and multi-frame instances are blocked until per-frame functional groups are supported.",
        len(multi_frame_instances),
    )

    missing_spacing_count = sum(
        1 for instance in instances if instance.pixel_spacing is None and instance.imager_pixel_spacing is None
    )
    add_issue(
        "missing-pixel-spacing",
        "warning",
        "Missing pixel spacing",
        "Distance measurements may fall back to pixel units because PixelSpacing/ImagerPixelSpacing is unavailable.",
        missing_spacing_count,
    )

    if total_count > 1:
        missing_geometry_count = sum(
            1
            for instance in instances
            if not instance.has_image_orientation_patient or not instance.has_image_position_patient
        )
        add_issue(
            "missing-spatial-geometry",
            "warning",
            "Missing spatial geometry",
            "ImageOrientationPatient or ImagePositionPatient is missing; stack order, MPR, and 3D geometry may be approximate.",
            missing_geometry_count,
        )

        volume_block_reason = get_series_volume_block_reason(series)
        if volume_block_reason and not missing_geometry_count and not multi_frame_instances:
            add_issue(
                "unsupported-volume-geometry",
                "error",
                "Unsupported volume geometry",
                volume_block_reason,
                total_count,
            )

    modality = (series.modality or "").upper()
    if modality in {"CT", "PT", "PET"}:
        missing_rescale_count = sum(
            1 for instance in instances if not instance.has_rescale_slope or not instance.has_rescale_intercept
        )
        add_issue(
            "missing-rescale",
            "warning",
            "Missing rescale metadata",
            "RescaleSlope or RescaleIntercept is missing; quantitative pixel values may remain in stored units.",
            missing_rescale_count,
        )

    return issues


def _safe_positive_int(value) -> int | None:
    try:
        resolved = int(float(str(value).strip()))
    except (OverflowError, TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
