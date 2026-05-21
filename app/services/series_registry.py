import io
from pathlib import Path
from threading import RLock
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import pydicom
from fastapi import HTTPException
from PIL import Image, ImageOps

from app.core.logging import get_logger
from app.models.viewer import InstanceRecord, SeriesRecord
from app.schemas.dicom import DicomCompatibilityIssue, LoadFolderRequest, LoadFolderResponse, SeriesSummary
from app.services.dicom_cache import dicom_cache
from app.services.four_d_service import four_d_service


logger = get_logger(__name__)
SERIES_THUMBNAIL_SIZE = (96, 96)


class SeriesRegistry:
    def __init__(self) -> None:
        self._series_by_id: dict[str, SeriesRecord] = {}
        self._series_id_by_key: dict[str, str] = {}
        self._series_id_by_instance_path: dict[str, str] = {}
        self._lock = RLock()

    @staticmethod
    def _build_series_folder_partition_key(folder: Path, path: Path) -> str:
        try:
            relative_parent = path.parent.relative_to(folder)
            relative_parent_text = relative_parent.as_posix().strip()
        except ValueError:
            relative_parent_text = path.parent.as_posix().strip()
        return "" if relative_parent_text in {"", "."} else relative_parent_text

    @staticmethod
    def _build_series_phase_partition_key(folder: Path, path: Path, dataset) -> str:
        try:
            relative_parent = path.parent.relative_to(folder)
            relative_parent_text = " ".join(part for part in relative_parent.parts if part not in ("", "."))
        except ValueError:
            relative_parent_text = path.parent.name

        for value in (
            getattr(dataset, "SeriesDescription", None),
            relative_parent_text,
            path.stem,
        ):
            phase = four_d_service.extract_phase_from_text(value)
            if phase is not None:
                return phase.label_value
        return ""

    @classmethod
    def _build_series_key(cls, folder: Path, series_instance_uid: str | None, fallback_path: Path, dataset) -> str:
        normalized_folder = folder.as_posix()
        if series_instance_uid:
            folder_partition_key = cls._build_series_folder_partition_key(folder, fallback_path)
            phase_partition_key = cls._build_series_phase_partition_key(folder, fallback_path, dataset)
            partition_tokens: list[str] = []
            if folder_partition_key:
                partition_tokens.append(f"dir={folder_partition_key}")
            if phase_partition_key:
                partition_tokens.append(f"phase={phase_partition_key}")
            if partition_tokens:
                return f"{normalized_folder}::{series_instance_uid}::{'::'.join(partition_tokens)}"
            return f"{normalized_folder}::{series_instance_uid}"
        return f"{normalized_folder}::{fallback_path.parent.as_posix()}"

    @staticmethod
    def _build_instance_path_key(path: Path) -> str:
        return path.resolve().as_posix()

    @staticmethod
    def _resolve_scan_target(folder_path: str) -> tuple[Path, list[Path]]:
        """Normalize an input file or folder path into a scan root and file list."""

        target = Path(folder_path).expanduser().resolve()
        if not target.exists():
            raise HTTPException(status_code=404, detail="DICOM path not found")
        if target.is_file():
            return target.parent, [target]
        if target.is_dir():
            return target, sorted(target.rglob("*"))
        raise HTTPException(status_code=404, detail="DICOM path is not a file or folder")

    @staticmethod
    def _read_dataset_header(path: Path):
        try:
            return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return None

    @staticmethod
    def _safe_header_text(value) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _has_header_value(dataset, keyword: str) -> bool:
        value = getattr(dataset, keyword, None)
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        try:
            return len(value) > 0
        except TypeError:
            return True

    @staticmethod
    def _safe_positive_int(value) -> int | None:
        try:
            resolved = int(float(str(value).strip()))
        except (OverflowError, TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

    @staticmethod
    def _safe_numeric_pair(value) -> tuple[float, float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            parts = value.replace("\\", " ").split()
        else:
            try:
                parts = list(value)
            except TypeError:
                return None
        if len(parts) < 2:
            return None
        try:
            return (float(parts[0]), float(parts[1]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_transfer_syntax_info(dataset) -> tuple[str | None, str | None, bool]:
        file_meta = getattr(dataset, "file_meta", None)
        transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
        if transfer_syntax is None:
            return (None, None, False)

        transfer_syntax_uid = str(transfer_syntax).strip() or None
        transfer_syntax_name = str(getattr(transfer_syntax, "name", None) or transfer_syntax_uid or "").strip() or None
        return (
            transfer_syntax_uid,
            transfer_syntax_name,
            bool(getattr(transfer_syntax, "is_compressed", False)),
        )

    @staticmethod
    def _is_readable_dicom(dataset) -> bool:
        return bool(getattr(dataset, "SeriesInstanceUID", None) or "PixelData" in dataset)

    def _resolve_existing_series_id(self, series_key: str, path: Path) -> str | None:
        return self._series_id_by_key.get(series_key) or self._series_id_by_instance_path.get(
            self._build_instance_path_key(path)
        )

    def _get_or_create_grouped_series(
        self,
        *,
        grouped: dict[str, SeriesRecord],
        instance_keys_by_series_key: dict[str, set[str]],
        folder: Path,
        path: Path,
        dataset,
    ) -> tuple[str, SeriesRecord]:
        series_instance_uid = getattr(dataset, "SeriesInstanceUID", None)
        series_key = self._build_series_key(folder, series_instance_uid, path, dataset)
        series = grouped.get(series_key)
        if series is not None:
            return (series_key, series)

        existing_series_id = self._resolve_existing_series_id(series_key, path)
        series = SeriesRecord(
            series_id=existing_series_id or str(uuid4()),
            folder_path=str(folder),
            series_instance_uid=series_instance_uid,
            study_instance_uid=self._safe_header_text(getattr(dataset, "StudyInstanceUID", None)),
            patient_id=self._safe_header_text(getattr(dataset, "PatientID", None)),
            patient_name=self._safe_header_text(getattr(dataset, "PatientName", None)),
            study_date=self._safe_header_text(getattr(dataset, "StudyDate", None)),
            study_description=self._safe_header_text(getattr(dataset, "StudyDescription", None)),
            accession_number=self._safe_header_text(getattr(dataset, "AccessionNumber", None)),
            modality=self._safe_header_text(getattr(dataset, "Modality", None)),
            series_description=self._safe_header_text(getattr(dataset, "SeriesDescription", None)),
        )
        grouped[series_key] = series
        instance_keys_by_series_key[series_key] = set()
        return (series_key, series)

    @staticmethod
    def _build_instance_record(path: Path, dataset, default_instance_number: int) -> InstanceRecord:
        instance_number = SeriesRegistry._resolve_instance_number(
            getattr(dataset, "InstanceNumber", None),
            default_instance_number,
        )
        transfer_syntax_uid, transfer_syntax_name, transfer_syntax_is_compressed = (
            SeriesRegistry._get_transfer_syntax_info(dataset)
        )
        return InstanceRecord(
            path=path,
            sop_instance_uid=getattr(dataset, "SOPInstanceUID", None),
            instance_number=instance_number,
            rows=getattr(dataset, "Rows", None),
            columns=getattr(dataset, "Columns", None),
            transfer_syntax_uid=transfer_syntax_uid,
            transfer_syntax_name=transfer_syntax_name,
            transfer_syntax_is_compressed=transfer_syntax_is_compressed,
            photometric_interpretation=SeriesRegistry._safe_header_text(
                getattr(dataset, "PhotometricInterpretation", None)
            ),
            samples_per_pixel=SeriesRegistry._safe_positive_int(getattr(dataset, "SamplesPerPixel", None)),
            pixel_spacing=SeriesRegistry._safe_numeric_pair(getattr(dataset, "PixelSpacing", None)),
            imager_pixel_spacing=SeriesRegistry._safe_numeric_pair(getattr(dataset, "ImagerPixelSpacing", None)),
            has_image_orientation_patient=SeriesRegistry._has_header_value(dataset, "ImageOrientationPatient"),
            has_image_position_patient=SeriesRegistry._has_header_value(dataset, "ImagePositionPatient"),
            has_rescale_slope=SeriesRegistry._has_header_value(dataset, "RescaleSlope"),
            has_rescale_intercept=SeriesRegistry._has_header_value(dataset, "RescaleIntercept"),
            has_window_width=SeriesRegistry._has_header_value(dataset, "WindowWidth"),
            has_window_center=SeriesRegistry._has_header_value(dataset, "WindowCenter"),
            number_of_frames=SeriesRegistry._safe_positive_int(getattr(dataset, "NumberOfFrames", None)),
        )

    @staticmethod
    def _resolve_instance_number(value, fallback: int) -> int:
        try:
            raw_value = str(value).strip()
            if not raw_value:
                return fallback
            return int(float(raw_value))
        except (OverflowError, TypeError, ValueError):
            return fallback

    def _build_series_instance_key(self, path: Path, dataset) -> str:
        sop_instance_uid = getattr(dataset, "SOPInstanceUID", None)
        return str(sop_instance_uid or self._build_instance_path_key(path))

    def _collect_grouped_series(self, folder: Path, scan_paths: list[Path]) -> dict[str, SeriesRecord]:
        grouped: dict[str, SeriesRecord] = {}
        instance_keys_by_series_key: dict[str, set[str]] = {}

        for path in scan_paths:
            if not path.is_file():
                continue

            dataset = self._read_dataset_header(path)
            if dataset is None or not self._is_readable_dicom(dataset):
                continue

            series_key, series = self._get_or_create_grouped_series(
                grouped=grouped,
                instance_keys_by_series_key=instance_keys_by_series_key,
                folder=folder,
                path=path,
                dataset=dataset,
            )

            instance_key = self._build_series_instance_key(path, dataset)
            if instance_key in instance_keys_by_series_key[series_key]:
                continue
            instance_keys_by_series_key[series_key].add(instance_key)

            series.instances.append(
                self._build_instance_record(
                    path,
                    dataset,
                    len(series.instances) + 1,
                )
            )

        return grouped

    def _index_series_instances(self, series: SeriesRecord) -> None:
        for instance in series.instances:
            self._series_id_by_instance_path[self._build_instance_path_key(instance.path)] = series.series_id

    @staticmethod
    def _build_compatibility_issues(series: SeriesRecord) -> list[DicomCompatibilityIssue]:
        instances = series.instances
        if not instances:
            return []

        total_count = len(instances)
        issues: list[DicomCompatibilityIssue] = []

        def add_issue(
            code: str,
            severity: str,
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
            if SeriesRegistry._safe_positive_int(instance.rows) is None
            or SeriesRegistry._safe_positive_int(instance.columns) is None
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
                SeriesRegistry._safe_positive_int(instance.rows),
                SeriesRegistry._safe_positive_int(instance.columns),
            )
            for instance in instances
            if SeriesRegistry._safe_positive_int(instance.rows) is not None
            and SeriesRegistry._safe_positive_int(instance.columns) is not None
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
            "multiframe-first-frame",
            "warning",
            "Multi-frame instances",
            "Only the decoded first frame is used by the current image pipeline.",
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

    def _build_series_summary(self, series_key: str, series: SeriesRecord) -> SeriesSummary:
        series.instances.sort(key=lambda item: item.instance_number)
        self._series_by_id[series.series_id] = series
        self._series_id_by_key[series_key] = series.series_id
        self._index_series_instances(series)

        first = series.instances[0]
        return SeriesSummary(
            seriesId=series.series_id,
            seriesInstanceUid=series.series_instance_uid,
            studyInstanceUid=series.study_instance_uid,
            patientId=series.patient_id,
            patientName=series.patient_name,
            studyDate=series.study_date,
            studyDescription=series.study_description,
            accessionNumber=series.accession_number,
            modality=series.modality,
            seriesDescription=series.series_description,
            instanceCount=len(series.instances),
            width=first.columns,
            height=first.rows,
            thumbnailSrc="",
            thumbnailUrl=self._build_series_thumbnail_url(series.series_id),
            folderPath=series.folder_path,
            compatibilityIssues=self._build_compatibility_issues(series),
        )

    @staticmethod
    def _build_series_thumbnail_url(series_id: str) -> str:
        return f"/api/v1/dicom/thumbnail?seriesId={quote(series_id, safe='')}"

    def get_series_thumbnail_png(self, series_id: str) -> bytes:
        with self._lock:
            series = self.get(series_id)
        thumbnail = self._build_series_thumbnail_png(series)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Series thumbnail is not available")
        return thumbnail

    def _build_series_thumbnail_png(self, series: SeriesRecord) -> bytes | None:
        if not series.instances:
            return None

        thumbnail_instance = series.instances[len(series.instances) // 2]
        cache_key = thumbnail_instance.sop_instance_uid or self._build_instance_path_key(thumbnail_instance.path)

        try:
            cached = dicom_cache.get(cache_key, thumbnail_instance.path)
            pixels = np.asarray(cached.source_pixels, dtype=np.float32)
            if pixels.ndim != 2:
                return None

            low, high = self._resolve_thumbnail_window(
                pixels,
                cached.window_width,
                cached.window_center,
            )
            clipped = np.clip(pixels, low, high)
            scale = high - low
            if scale <= 0:
                return None

            normalized = ((clipped - low) * (255.0 / scale)).astype(np.uint8)
            image = Image.fromarray(normalized)
            image = ImageOps.contain(image, SERIES_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            canvas = Image.new("L", SERIES_THUMBNAIL_SIZE, 0)
            canvas.paste(image, ((SERIES_THUMBNAIL_SIZE[0] - image.width) // 2, (SERIES_THUMBNAIL_SIZE[1] - image.height) // 2))

            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
        except Exception as exc:
            logger.debug("failed to build series thumbnail series_id=%s error=%s", series.series_id, exc)
            return None

    @staticmethod
    def _resolve_thumbnail_window(
        pixels: np.ndarray,
        window_width: float | None,
        window_center: float | None,
    ) -> tuple[float, float]:
        if window_width is not None and window_width > 0 and window_center is not None:
            return (float(window_center - window_width / 2.0), float(window_center + window_width / 2.0))

        finite_values = np.asarray(pixels[np.isfinite(pixels)], dtype=np.float32)
        if finite_values.size == 0:
            return (0.0, 1.0)

        low = float(np.percentile(finite_values, 1.0))
        high = float(np.percentile(finite_values, 99.0))
        if high <= low:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
        if high <= low:
            high = low + 1.0
        return (low, high)

    @staticmethod
    def _build_virtual_four_d_phase_series_id(series_id: str, phase_index: int) -> str:
        return f"{series_id}::phase::{phase_index}"

    def _register_virtual_four_d_phase_series(self, series_records: list[SeriesRecord]) -> list[SeriesRecord]:
        virtual_series_records: list[SeriesRecord] = []

        for series in series_records:
            if series.is_virtual:
                continue
            phase_groups = four_d_service.get_single_series_phase_groups(series)
            if len(phase_groups) < 2:
                continue

            for phase_index, (phase, instances) in enumerate(phase_groups):
                virtual_series = SeriesRecord(
                    series_id=self._build_virtual_four_d_phase_series_id(series.series_id, phase_index),
                    folder_path=series.folder_path,
                    series_instance_uid=series.series_instance_uid,
                    study_instance_uid=series.study_instance_uid,
                    patient_id=series.patient_id,
                    patient_name=series.patient_name,
                    study_date=series.study_date,
                    study_description=series.study_description,
                    accession_number=series.accession_number,
                    modality=series.modality,
                    series_description=series.series_description,
                    is_virtual=True,
                    source_series_id=series.series_id,
                    four_d_phase_sort_value=phase.sort_value,
                    four_d_phase_label_value=phase.label_value,
                    four_d_phase_source=phase.source,
                    instances=sorted(instances, key=lambda item: item.instance_number),
                )
                self._series_by_id[virtual_series.series_id] = virtual_series
                virtual_series_records.append(virtual_series)

        return virtual_series_records

    def _list_real_series_records(self) -> list[SeriesRecord]:
        return [series for series in self._series_by_id.values() if not series.is_virtual]

    def _apply_loaded_four_d_metadata(self, series_list: list[SeriesSummary]) -> None:
        real_series_records = self._list_real_series_records()
        virtual_series_records = self._register_virtual_four_d_phase_series(real_series_records)
        four_d_service.apply_four_d_metadata(series_list, [*real_series_records, *virtual_series_records])

    def ensure_four_d_phase_series(self, series_id: str) -> SeriesRecord:
        with self._lock:
            series = self.get(series_id)
            source_series = self.get(series.source_series_id) if series.is_virtual and series.source_series_id else series
            self._register_virtual_four_d_phase_series([source_series])
            return series

    def load_folder(self, payload: LoadFolderRequest) -> LoadFolderResponse:
        with self._lock:
            folder, scan_paths = self._resolve_scan_target(payload.folder_path)
            grouped = self._collect_grouped_series(folder, scan_paths)
            if not grouped:
                raise HTTPException(status_code=404, detail="No readable DICOM series found in path")

            series_list = [self._build_series_summary(series_key, series) for series_key, series in grouped.items()]
            self._apply_loaded_four_d_metadata(series_list)
            series_list.sort(key=lambda item: item.series_id)
            return LoadFolderResponse(seriesId=series_list[0].series_id, seriesList=series_list)

    def get(self, series_id: str) -> SeriesRecord:
        with self._lock:
            series = self._series_by_id.get(series_id)
            if series is None:
                raise HTTPException(status_code=404, detail="seriesId not found")
            return series

    def list_all(self) -> list[SeriesRecord]:
        with self._lock:
            return list(self._series_by_id.values())

    def clear(self) -> None:
        with self._lock:
            self._series_by_id.clear()
            self._series_id_by_key.clear()
            self._series_id_by_instance_path.clear()


series_registry = SeriesRegistry()
