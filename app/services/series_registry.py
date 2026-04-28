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
from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse, SeriesSummary
from app.services.dicom_cache import dicom_cache
from app.services.four_d_service import four_d_service


logger = get_logger(__name__)
SERIES_THUMBNAIL_SIZE = (96, 96)


class SeriesRegistry:
    def __init__(self) -> None:
        self._series_by_id: dict[str, SeriesRecord] = {}
        self._series_id_by_key: dict[str, str] = {}
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
    def _resolve_folder(folder_path: str) -> Path:
        """Normalize the input path so registry keys remain stable across calls."""

        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=404, detail="DICOM folder not found")
        return folder

    @staticmethod
    def _read_dataset_header(path: Path):
        try:
            return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return None

    @staticmethod
    def _is_readable_dicom(dataset) -> bool:
        return bool(getattr(dataset, "SeriesInstanceUID", None) or "PixelData" in dataset)

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

        existing_series_id = self._series_id_by_key.get(series_key)
        series = SeriesRecord(
            series_id=existing_series_id or str(uuid4()),
            folder_path=str(folder),
            series_instance_uid=series_instance_uid,
            study_instance_uid=getattr(dataset, "StudyInstanceUID", None),
            patient_id=getattr(dataset, "PatientID", None),
            modality=getattr(dataset, "Modality", None),
            series_description=getattr(dataset, "SeriesDescription", None),
        )
        grouped[series_key] = series
        instance_keys_by_series_key[series_key] = set()
        return (series_key, series)

    @staticmethod
    def _build_instance_record(path: Path, dataset, default_instance_number: int) -> InstanceRecord:
        instance_number = int(getattr(dataset, "InstanceNumber", default_instance_number) or default_instance_number)
        return InstanceRecord(
            path=path,
            sop_instance_uid=getattr(dataset, "SOPInstanceUID", None),
            instance_number=instance_number,
            rows=getattr(dataset, "Rows", None),
            columns=getattr(dataset, "Columns", None),
        )

    def _collect_grouped_series(self, folder: Path) -> dict[str, SeriesRecord]:
        grouped: dict[str, SeriesRecord] = {}
        instance_keys_by_series_key: dict[str, set[str]] = {}

        for path in sorted(folder.rglob("*")):
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

            sop_instance_uid = getattr(dataset, "SOPInstanceUID", None)
            instance_key = str(sop_instance_uid or path.resolve().as_posix())
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

    def _build_series_summary(self, series_key: str, series: SeriesRecord) -> SeriesSummary:
        series.instances.sort(key=lambda item: item.instance_number)
        self._series_by_id[series.series_id] = series
        self._series_id_by_key[series_key] = series.series_id

        first = series.instances[0]
        return SeriesSummary(
            seriesId=series.series_id,
            seriesInstanceUid=series.series_instance_uid,
            studyInstanceUid=series.study_instance_uid,
            patientId=series.patient_id,
            modality=series.modality,
            seriesDescription=series.series_description,
            instanceCount=len(series.instances),
            width=first.columns,
            height=first.rows,
            thumbnailSrc="",
            thumbnailUrl=self._build_series_thumbnail_url(series.series_id),
            folderPath=series.folder_path,
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
        cache_key = thumbnail_instance.sop_instance_uid or thumbnail_instance.path.resolve().as_posix()

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

    def ensure_four_d_phase_series(self, series_id: str) -> SeriesRecord:
        with self._lock:
            series = self.get(series_id)
            source_series = self.get(series.source_series_id) if series.is_virtual and series.source_series_id else series
            self._register_virtual_four_d_phase_series([source_series])
            return series

    def load_folder(self, payload: LoadFolderRequest) -> LoadFolderResponse:
        with self._lock:
            folder = self._resolve_folder(payload.folder_path)
            grouped = self._collect_grouped_series(folder)
            if not grouped:
                raise HTTPException(status_code=404, detail="No readable DICOM series found in folder")

            real_series_records = list(grouped.values())
            series_list = [self._build_series_summary(series_key, series) for series_key, series in grouped.items()]
            virtual_series_records = self._register_virtual_four_d_phase_series(real_series_records)
            four_d_service.apply_four_d_metadata(series_list, [*real_series_records, *virtual_series_records])
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


series_registry = SeriesRegistry()
