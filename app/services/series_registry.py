import io
import os
from collections.abc import Iterable
from pathlib import Path
from threading import RLock
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import pydicom
from fastapi import HTTPException
from PIL import Image, ImageOps
from pydicom.uid import (
    BasicTextSRStorage,
    Comprehensive3DSRStorage,
    ComprehensiveSRStorage,
    EnhancedSRStorage,
    ExtensibleSRStorage,
    KeyObjectSelectionDocumentStorage,
    MammographyCADSRStorage,
    ProcedureLogStorage,
)

from app.core.logging import get_logger
from app.core.workspace import DEFAULT_WORKSPACE_ID, WORKSPACE_QUERY_PARAM, normalize_workspace_id
from app.models.viewer import InstanceRecord, SeriesRecord
from app.schemas.dicom import DicomCompatibilityIssue, LoadFolderRequest, LoadFolderResponse, SeriesSummary
from app.services.dicom_cache import dicom_cache
from app.services.dicom_compatibility import build_dicom_compatibility_issues
from app.services.dicom_gsps_import_service import is_gsps_dataset, parse_gsps_dataset
from app.services.four_d_service import four_d_service


logger = get_logger(__name__)
SERIES_THUMBNAIL_SIZE = (96, 96)
DICOM_SR_SOP_CLASS_UIDS = {
    str(BasicTextSRStorage),
    str(EnhancedSRStorage),
    str(ComprehensiveSRStorage),
    str(Comprehensive3DSRStorage),
    str(ExtensibleSRStorage),
    str(ProcedureLogStorage),
    str(MammographyCADSRStorage),
    str(KeyObjectSelectionDocumentStorage),
}

# DICOM instances frequently have no filename suffix, so accept unknown suffixes and
# only skip the common archive, metadata, and operating-system byproducts that can
# never be an image instance.
NON_DICOM_SCAN_SUFFIXES = frozenset({
    ".7z",
    ".bz2",
    ".csv",
    ".db",
    ".gz",
    ".html",
    ".htm",
    ".json",
    ".log",
    ".md",
    ".pdf",
    ".rar",
    ".tar",
    ".tgz",
    ".txt",
    ".webp",
    ".xml",
    ".xz",
    ".yaml",
    ".yml",
    ".zip",
    ".zst",
})

# Folder and archive imports only need these fields to group instances and build
# lightweight series cards. Pixel data is intentionally excluded and richer
# metadata is read on demand by the viewer APIs.
SCAN_HEADER_TAGS = (
    "SOPClassUID",
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "PatientID",
    "PatientName",
    "StudyDate",
    "StudyDescription",
    "AccessionNumber",
    "Modality",
    "SeriesDescription",
    "InstanceNumber",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowWidth",
    "WindowCenter",
    "NumberOfFrames",
)


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
        if series_instance_uid:
            folder_partition_key = cls._build_series_folder_partition_key(folder, fallback_path)
            phase_partition_key = cls._build_series_phase_partition_key(folder, fallback_path, dataset)
            partition_tokens: list[str] = []
            if folder_partition_key:
                partition_tokens.append(f"dir={folder_partition_key}")
            if phase_partition_key:
                partition_tokens.append(f"phase={phase_partition_key}")
            base_key = f"uid={series_instance_uid}"
            if partition_tokens:
                return f"{base_key}::{'::'.join(partition_tokens)}"
            return base_key
        return f"path::{fallback_path.parent.resolve().as_posix()}"

    @staticmethod
    def _build_instance_path_key(path: Path) -> str:
        return path.resolve().as_posix()

    @staticmethod
    def _build_workspace_scoped_key(workspace_id: str, key: str) -> str:
        return f"{normalize_workspace_id(workspace_id)}::{key}"

    @staticmethod
    def is_dicom_candidate_name(path: Path) -> bool:
        """Return whether a filename is worth passing to the DICOM header reader."""

        if path.name.startswith(".") or "__MACOSX" in path.parts:
            return False
        return path.suffix.lower() not in NON_DICOM_SCAN_SUFFIXES

    @classmethod
    def is_dicom_scan_candidate(cls, path: Path) -> bool:
        """Return whether an existing file is worth passing to the DICOM header reader."""

        return path.is_file() and cls.is_dicom_candidate_name(path)

    @classmethod
    def _resolve_scan_target(cls, folder_path: str) -> tuple[Path, list[Path]]:
        """Normalize an input file or folder path into a scan root and file list."""

        target = Path(folder_path).expanduser().resolve()
        if not target.exists():
            raise HTTPException(status_code=404, detail="DICOM path not found")
        if target.is_file():
            return target.parent, [target] if cls.is_dicom_scan_candidate(target) else []
        if target.is_dir():
            return target, cls._collect_scan_candidates(target)
        raise HTTPException(status_code=404, detail="DICOM path is not a file or folder")

    @classmethod
    def _collect_scan_candidates(cls, root: Path) -> list[Path]:
        """Recursively collect candidates without materializing every directory entry first."""

        candidates: list[Path] = []
        pending = [root]
        while pending:
            current_dir = pending.pop()
            try:
                with os.scandir(current_dir) as entries:
                    for entry in entries:
                        if entry.name.startswith(".") or entry.name == "__MACOSX":
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                path = Path(entry.path)
                                if cls.is_dicom_candidate_name(path):
                                    candidates.append(path)
                        except OSError:
                            continue
            except OSError:
                continue
        return sorted(candidates)

    @staticmethod
    def _read_dataset_header(path: Path):
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                specific_tags=SCAN_HEADER_TAGS,
                force=True,
            )
            # GSPS stores its references in sequences outside the normal scan
            # set. It is uncommon, so pay for the full header only when needed.
            if is_gsps_dataset(dataset):
                return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            return dataset
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
    def _is_sr_dataset(dataset) -> bool:
        sop_class_uid = str(getattr(dataset, "SOPClassUID", "") or "").strip()
        modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
        return sop_class_uid in DICOM_SR_SOP_CLASS_UIDS or modality == "SR"

    @classmethod
    def _is_readable_dicom(cls, dataset) -> bool:
        if is_gsps_dataset(dataset) or cls._is_sr_dataset(dataset):
            return False
        return bool("PixelData" in dataset or (getattr(dataset, "Rows", None) and getattr(dataset, "Columns", None)))

    def _resolve_existing_series_id(self, workspace_id: str, series_key: str, path: Path) -> str | None:
        return self._series_id_by_key.get(
            self._build_workspace_scoped_key(workspace_id, series_key)
        ) or self._series_id_by_instance_path.get(
            self._build_workspace_scoped_key(workspace_id, self._build_instance_path_key(path))
        )

    def _get_or_create_grouped_series(
        self,
        *,
        workspace_id: str,
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

        existing_series_id = self._resolve_existing_series_id(workspace_id, series_key, path)
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
            workspace_id=workspace_id,
        )
        grouped[series_key] = series
        instance_keys_by_series_key[series_key] = set()
        return (series_key, series)

    def _get_or_create_grouped_document_series(
        self,
        *,
        workspace_id: str,
        grouped: dict[str, SeriesRecord],
        instance_keys_by_series_key: dict[str, set[str]],
        folder: Path,
        path: Path,
        dataset,
        fallback_modality: str,
        fallback_series_description: str,
        standard_object_type: str,
    ) -> tuple[str, SeriesRecord]:
        series_instance_uid = getattr(dataset, "SeriesInstanceUID", None)
        series_key = self._build_series_key(folder, series_instance_uid, path, dataset)
        series = grouped.get(series_key)
        if series is not None:
            return (series_key, series)

        existing_series_id = self._resolve_existing_series_id(workspace_id, series_key, path)
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
            modality=self._safe_header_text(getattr(dataset, "Modality", None)) or fallback_modality,
            series_description=self._safe_header_text(getattr(dataset, "SeriesDescription", None))
            or fallback_series_description,
            workspace_id=workspace_id,
            is_image_series=False,
            standard_object_type=standard_object_type,
            preferred_view_type="Tag",
        )
        grouped[series_key] = series
        instance_keys_by_series_key[series_key] = set()
        return (series_key, series)

    def _add_instance_to_grouped_series(
        self,
        *,
        series_key: str,
        series: SeriesRecord,
        instance_keys_by_series_key: dict[str, set[str]],
        path: Path,
        dataset,
    ) -> None:
        instance_key = self._build_series_instance_key(path, dataset)
        if instance_key in instance_keys_by_series_key[series_key]:
            return
        instance_keys_by_series_key[series_key].add(instance_key)
        series.instances.append(
            self._build_instance_record(
                path,
                dataset,
                len(series.instances) + 1,
            )
        )

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

    def _collect_grouped_series(self, folder: Path, scan_paths: list[Path], workspace_id: str) -> dict[str, SeriesRecord]:
        grouped: dict[str, SeriesRecord] = {}
        instance_keys_by_series_key: dict[str, set[str]] = {}
        gsps_datasets: list[tuple[Path, object]] = []

        for path in scan_paths:

            dataset = self._read_dataset_header(path)
            if dataset is None:
                continue

            if is_gsps_dataset(dataset):
                gsps_datasets.append((path, dataset))
                continue

            if self._is_sr_dataset(dataset):
                series_key, series = self._get_or_create_grouped_document_series(
                    workspace_id=workspace_id,
                    grouped=grouped,
                    instance_keys_by_series_key=instance_keys_by_series_key,
                    folder=folder,
                    path=path,
                    dataset=dataset,
                    fallback_modality="SR",
                    fallback_series_description="DICOM SR",
                    standard_object_type="DICOM_SR",
                )
                self._add_instance_to_grouped_series(
                    series_key=series_key,
                    series=series,
                    instance_keys_by_series_key=instance_keys_by_series_key,
                    path=path,
                    dataset=dataset,
                )
                continue

            if not self._is_readable_dicom(dataset):
                continue

            series_key, series = self._get_or_create_grouped_series(
                workspace_id=workspace_id,
                grouped=grouped,
                instance_keys_by_series_key=instance_keys_by_series_key,
                folder=folder,
                path=path,
                dataset=dataset,
            )

            self._add_instance_to_grouped_series(
                series_key=series_key,
                series=series,
                instance_keys_by_series_key=instance_keys_by_series_key,
                path=path,
                dataset=dataset,
            )

        attached_gsps_paths = self._attach_presentation_states(grouped.values(), gsps_datasets)
        self._register_unattached_gsps_documents(
            workspace_id=workspace_id,
            grouped=grouped,
            instance_keys_by_series_key=instance_keys_by_series_key,
            folder=folder,
            gsps_datasets=gsps_datasets,
            attached_gsps_paths=attached_gsps_paths,
        )
        return grouped

    def _attach_presentation_states(
        self,
        series_records: Iterable[SeriesRecord],
        gsps_datasets: list[tuple[Path, object]],
    ) -> set[Path]:
        series_by_sop_uid: dict[str, SeriesRecord] = {}
        for series in series_records:
            if not series.is_image_series:
                continue
            for instance in series.instances:
                if instance.sop_instance_uid:
                    series_by_sop_uid[str(instance.sop_instance_uid)] = series

        attached_paths: set[Path] = set()
        for path, dataset in gsps_datasets:
            for presentation_state in parse_gsps_dataset(dataset, path):
                target_series = series_by_sop_uid.get(presentation_state.referenced_sop_instance_uid)
                if target_series is None:
                    continue
                attached_paths.add(path)
                target_series.presentation_states_by_sop_uid.setdefault(
                    presentation_state.referenced_sop_instance_uid,
                    [],
                ).append(presentation_state)
        return attached_paths

    def _register_unattached_gsps_documents(
        self,
        *,
        workspace_id: str,
        grouped: dict[str, SeriesRecord],
        instance_keys_by_series_key: dict[str, set[str]],
        folder: Path,
        gsps_datasets: list[tuple[Path, object]],
        attached_gsps_paths: set[Path],
    ) -> None:
        for path, dataset in gsps_datasets:
            if path in attached_gsps_paths:
                continue

            series_key, series = self._get_or_create_grouped_document_series(
                workspace_id=workspace_id,
                grouped=grouped,
                instance_keys_by_series_key=instance_keys_by_series_key,
                folder=folder,
                path=path,
                dataset=dataset,
                fallback_modality="PR",
                fallback_series_description="DICOM GSPS",
                standard_object_type="DICOM_GSPS",
            )
            self._add_instance_to_grouped_series(
                series_key=series_key,
                series=series,
                instance_keys_by_series_key=instance_keys_by_series_key,
                path=path,
                dataset=dataset,
            )

    def _index_series_instances(self, series: SeriesRecord) -> None:
        for instance in series.instances:
            self._series_id_by_instance_path[
                self._build_workspace_scoped_key(series.workspace_id, self._build_instance_path_key(instance.path))
            ] = series.series_id

    def _build_series_summary(self, series_key: str, series: SeriesRecord) -> SeriesSummary:
        series.instances.sort(key=lambda item: item.instance_number)
        self._series_by_id[series.series_id] = series
        self._series_id_by_key[self._build_workspace_scoped_key(series.workspace_id, series_key)] = series.series_id
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
            thumbnailUrl=(
                self._build_series_thumbnail_url(series.series_id, series.workspace_id)
                if series.is_image_series
                else ""
            ),
            folderPath=series.folder_path,
            isImageSeries=series.is_image_series,
            standardObjectType=series.standard_object_type,
            preferredViewType=series.preferred_view_type,
            compatibilityIssues=[],
        )

    @staticmethod
    def _build_series_thumbnail_url(series_id: str, workspace_id: str = DEFAULT_WORKSPACE_ID) -> str:
        query = f"seriesId={quote(series_id, safe='')}"
        if normalize_workspace_id(workspace_id) != DEFAULT_WORKSPACE_ID:
            query = f"{query}&{WORKSPACE_QUERY_PARAM}={quote(workspace_id, safe='')}"
        return f"/api/v1/dicom/thumbnail?{query}"

    def get_series_thumbnail_png(self, series_id: str, workspace_id: str | None = None) -> bytes:
        with self._lock:
            series = self.get(series_id, workspace_id=workspace_id)
        thumbnail = self._build_series_thumbnail_png(series)
        if thumbnail is None:
            raise HTTPException(status_code=404, detail="Series thumbnail is not available")
        return thumbnail

    def _build_series_thumbnail_png(self, series: SeriesRecord) -> bytes | None:
        if not series.is_image_series or not series.instances:
            return None

        thumbnail_instance = series.instances[len(series.instances) // 2]
        cache_key = thumbnail_instance.sop_instance_uid or self._build_instance_path_key(thumbnail_instance.path)

        try:
            cached = dicom_cache.get(cache_key, thumbnail_instance.path)
            if cached.source_pixels.ndim == 3 and cached.source_pixels.shape[-1] in (3, 4):
                image = Image.fromarray(np.asarray(cached.source_pixels[..., :3], dtype=np.uint8))
                image = ImageOps.contain(image, SERIES_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", SERIES_THUMBNAIL_SIZE, (0, 0, 0))
                canvas.paste(image, ((SERIES_THUMBNAIL_SIZE[0] - image.width) // 2, (SERIES_THUMBNAIL_SIZE[1] - image.height) // 2))

                buffer = io.BytesIO()
                canvas.save(buffer, format="PNG", optimize=True)
                return buffer.getvalue()

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
                    workspace_id=series.workspace_id,
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

    def _list_real_series_records(self, workspace_id: str | None = None) -> list[SeriesRecord]:
        normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
        return [
            series
            for series in self._series_by_id.values()
            if not series.is_virtual
            and (normalized_workspace_id is None or series.workspace_id == normalized_workspace_id)
        ]

    def _apply_loaded_four_d_metadata(self, series_list: list[SeriesSummary], workspace_id: str) -> None:
        real_series_records = self._list_real_series_records(workspace_id)
        virtual_series_records = self._register_virtual_four_d_phase_series(real_series_records)
        four_d_service.apply_four_d_metadata(series_list, [*real_series_records, *virtual_series_records])

    def ensure_four_d_phase_series(self, series_id: str, workspace_id: str | None = None) -> SeriesRecord:
        with self._lock:
            series = self.get(series_id, workspace_id=workspace_id)
            source_series = (
                self.get(series.source_series_id, workspace_id=workspace_id)
                if series.is_virtual and series.source_series_id
                else series
            )
            self._register_virtual_four_d_phase_series([source_series])
            return series

    def load_folder(self, payload: LoadFolderRequest, workspace_id: str | None = None) -> LoadFolderResponse:
        with self._lock:
            normalized_workspace_id = normalize_workspace_id(workspace_id)
            folder, scan_paths = self._resolve_scan_target(payload.folder_path)
            grouped = self._collect_grouped_series(folder, scan_paths, normalized_workspace_id)
            if not grouped:
                raise HTTPException(status_code=404, detail="No readable DICOM series found in path")

            series_list = [self._build_series_summary(series_key, series) for series_key, series in grouped.items()]
            self._apply_loaded_four_d_metadata(series_list, normalized_workspace_id)
            series_list.sort(key=lambda item: (not item.is_image_series, item.series_id))
            return LoadFolderResponse(seriesId=series_list[0].series_id, seriesList=series_list)

    def get(self, series_id: str, workspace_id: str | None = None) -> SeriesRecord:
        with self._lock:
            series = self._series_by_id.get(series_id)
            normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
            if series is None or (
                normalized_workspace_id is not None and series.workspace_id != normalized_workspace_id
            ):
                raise HTTPException(status_code=404, detail="seriesId not found")
            return series

    def list_all(self, workspace_id: str | None = None) -> list[SeriesRecord]:
        with self._lock:
            normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
            return [
                series
                for series in self._series_by_id.values()
                if normalized_workspace_id is None or series.workspace_id == normalized_workspace_id
            ]

    def check_compatibility(self, series_id: str, workspace_id: str | None = None) -> list[DicomCompatibilityIssue]:
        series = self.get(series_id, workspace_id=workspace_id)
        if not series.is_image_series:
            return []
        return build_dicom_compatibility_issues(series)

    def clear(self, workspace_id: str | None = None) -> None:
        with self._lock:
            if workspace_id is not None:
                normalized_workspace_id = normalize_workspace_id(workspace_id)
                series_ids = {
                    series.series_id
                    for series in self._series_by_id.values()
                    if series.workspace_id == normalized_workspace_id
                }
                self._series_by_id = {
                    series_id: series
                    for series_id, series in self._series_by_id.items()
                    if series_id not in series_ids
                }
                self._series_id_by_key = {
                    key: series_id
                    for key, series_id in self._series_id_by_key.items()
                    if series_id not in series_ids
                }
                self._series_id_by_instance_path = {
                    key: series_id
                    for key, series_id in self._series_id_by_instance_path.items()
                    if series_id not in series_ids
                }
                return
            self._series_by_id.clear()
            self._series_id_by_key.clear()
            self._series_id_by_instance_path.clear()


series_registry = SeriesRegistry()
