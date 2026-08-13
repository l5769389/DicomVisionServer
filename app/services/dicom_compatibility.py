from app.models.viewer import SeriesRecord
import numpy as np

from app.schemas.dicom import DicomCompatibilityIssue, DicomCompatibilitySeverity, SeriesViewCapability


def get_series_volume_block_reason(series: SeriesRecord) -> str | None:
    instances = series.instances
    if any((instance.number_of_frames or 1) > 1 for instance in instances):
        return "Enhanced or multi-frame DICOM is not supported yet."
    if len(instances) <= 1:
        return None
    if any(instance.image_orientation_patient is None or instance.image_position_patient is None for instance in instances):
        return "MPR, 3D, and fusion require ImageOrientationPatient and ImagePositionPatient on every slice."

    reference_orientation = np.asarray(instances[0].image_orientation_patient, dtype=np.float64)
    reference_normal = np.cross(reference_orientation[:3], reference_orientation[3:])
    reference_norm = float(np.linalg.norm(reference_normal))
    if reference_norm <= 1e-6:
        return "The series contains an invalid slice orientation."
    reference_normal /= reference_norm

    projections: list[float] = []
    for instance in instances:
        orientation = np.asarray(instance.image_orientation_patient, dtype=np.float64)
        normal = np.cross(orientation[:3], orientation[3:])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-6 or abs(float(np.dot(normal / normal_norm, reference_normal))) < 0.999:
            return "The series contains mixed slice orientations."
        projections.append(float(np.dot(np.asarray(instance.image_position_patient, dtype=np.float64), reference_normal)))

    projections.sort()
    spacing = np.diff(np.asarray(projections, dtype=np.float64))
    if spacing.size and bool(np.any(spacing <= 1e-3)):
        return "The series contains duplicate slice positions."
    if spacing.size > 1:
        median_spacing = float(np.median(spacing))
        tolerance = max(0.1, median_spacing * 0.05)
        if bool(np.any(np.abs(spacing - median_spacing) > tolerance)):
            return "The series contains irregular slice spacing and requires resampling before volume display."
    return None


def build_series_view_capabilities(series: SeriesRecord) -> dict[str, SeriesViewCapability]:
    multiframe_reason = (
        "Enhanced or multi-frame DICOM is not supported yet."
        if any((instance.number_of_frames or 1) > 1 for instance in series.instances)
        else None
    )
    volume_reason = get_series_volume_block_reason(series)
    return {
        "stack": SeriesViewCapability(supported=multiframe_reason is None, blockedReason=multiframe_reason),
        "montage": SeriesViewCapability(supported=multiframe_reason is None, blockedReason=multiframe_reason),
        "mpr": SeriesViewCapability(supported=volume_reason is None, blockedReason=volume_reason),
        "3d": SeriesViewCapability(supported=volume_reason is None, blockedReason=volume_reason),
        "fusion": SeriesViewCapability(supported=volume_reason is None, blockedReason=volume_reason),
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
