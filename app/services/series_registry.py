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

    def load_folder(self, payload: LoadFolderRequest) -> LoadFolderResponse:
        # expanduser()：把 ~ 展开成用户家目录
        # resolve()：转成绝对路径，并规范化路径
        folder = Path(payload.folder_path).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=404, detail="DICOM folder not found")

        grouped: dict[str, SeriesRecord] = {}
        instance_keys_by_series_key: dict[str, set[str]] = {}
        # 递归遍历 folder 下面所有文件和子目录
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not getattr(dataset, "SeriesInstanceUID", None) and "PixelData" not in dataset:
                continue
            series_instance_uid = getattr(dataset, "SeriesInstanceUID", None)
            series_key = self._build_series_key(folder, series_instance_uid, path)
            series = grouped.get(series_key)
            if series is None:
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

            sop_instance_uid = getattr(dataset, "SOPInstanceUID", None)
            instance_key = str(sop_instance_uid or path.resolve().as_posix())
            if instance_key in instance_keys_by_series_key[series_key]:
                continue
            instance_keys_by_series_key[series_key].add(instance_key)
            instance_number = int(getattr(dataset, "InstanceNumber", len(series.instances) + 1) or len(series.instances) + 1)
            series.instances.append(
                InstanceRecord(
                    path=path,
                    sop_instance_uid=sop_instance_uid,
                    instance_number=instance_number,
                    rows=getattr(dataset, "Rows", None),
                    columns=getattr(dataset, "Columns", None),
                )
            )

        if not grouped:
            raise HTTPException(status_code=404, detail="No readable DICOM series found in folder")

        series_list: list[SeriesSummary] = []
        for series_key, series in grouped.items():
            series.instances.sort(key=lambda item: item.instance_number)
            self._series_by_id[series.series_id] = series
            self._series_id_by_key[series_key] = series.series_id
            first = series.instances[0]
            series_list.append(
                SeriesSummary(
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
            )

        series_list.sort(key=lambda item: item.series_id)
        return LoadFolderResponse(seriesId=series_list[0].series_id, seriesList=series_list)

    def get(self, series_id: str) -> SeriesRecord:
        series = self._series_by_id.get(series_id)
        if series is None:
            raise HTTPException(status_code=404, detail="seriesId not found")
        return series

    def list_all(self) -> list[SeriesRecord]:
        return list(self._series_by_id.values())


series_registry = SeriesRegistry()
