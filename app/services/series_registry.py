from pathlib import Path
from uuid import uuid4

import pydicom
from fastapi import HTTPException

from app.models.viewer import InstanceRecord, SeriesRecord
from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse, SeriesSummary


class SeriesRegistry:
    def __init__(self) -> None:
        self._series_by_id: dict[str, SeriesRecord] = {}
        self._series_id_by_key: dict[str, str] = {}

    @staticmethod
    def _build_series_key(folder: Path, series_instance_uid: str | None, fallback_path: Path) -> str:
        normalized_folder = folder.as_posix()
        if series_instance_uid:
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
        series_key = self._build_series_key(folder, series_instance_uid, path)
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
            folderPath=series.folder_path,
        )

    def load_folder(self, payload: LoadFolderRequest) -> LoadFolderResponse:
        folder = self._resolve_folder(payload.folder_path)
        grouped = self._collect_grouped_series(folder)
        if not grouped:
            raise HTTPException(status_code=404, detail="No readable DICOM series found in folder")

        series_list = [self._build_series_summary(series_key, series) for series_key, series in grouped.items()]
        series_list.sort(key=lambda item: item.series_id)
        return LoadFolderResponse(seriesId=series_list[0].series_id, seriesList=series_list)

    def get(self, series_id: str) -> SeriesRecord:
        series = self._series_by_id.get(series_id)
        if series is None:
            raise HTTPException(status_code=404, detail="seriesId not found")
        return series

    def list_all(self) -> list[SeriesRecord]:
        return list(self._series_by_id.values())

    def clear(self) -> None:
        self._series_by_id.clear()
        self._series_id_by_key.clear()


series_registry = SeriesRegistry()
