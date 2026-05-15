from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import re
from zipfile import ZIP_DEFLATED, ZipFile

import pydicom
from fastapi import HTTPException
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import generate_uid

from app.schemas.dicom import DicomDeidentifyFieldKey, DicomDeidentifyRequest
from app.services.series_registry import series_registry


DicomDeidentifyProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DicomDeidentifyArtifact:
    content: bytes
    file_name: str
    media_type: str
    modified_count: int
    artifact_kind: str
    series_folder: str


class DicomDeidentifyService:
    _SAFE_FILE_NAME_PATTERN = re.compile(r'[\\/:*?"<>|\s]+')
    _DEFAULT_REPLACEMENT_PREFIX = "ANON"
    _MAX_PREFIX_LENGTH = 24

    _CLEAR_TAGS_BY_FIELD: dict[DicomDeidentifyFieldKey, set[int]] = {
        "patientIdentity": {
            0x00100021,  # Issuer of Patient ID
            0x00101000,  # Other Patient IDs
            0x00101001,  # Other Patient Names
            0x00101005,  # Patient's Birth Name
            0x00101060,  # Patient's Mother's Birth Name
            0x00101080,  # Military Rank
            0x00101081,  # Branch of Service
            0x00102150,  # Country of Residence
            0x00102152,  # Region of Residence
            0x00102154,  # Patient's Telephone Numbers
            0x00102155,  # Patient's Telecom Information
            0x00104000,  # Patient Comments
            0x00101040,  # Patient Address
        },
        "patientDemographics": {
            0x00100030,  # Patient's Birth Date
            0x00100032,  # Patient's Birth Time
            0x00100040,  # Patient's Sex
            0x00101010,  # Patient's Age
            0x00101020,  # Patient's Size
            0x00101030,  # Patient's Weight
            0x00102160,  # Ethnic Group
            0x00102180,  # Occupation
        },
        "accessionInstitution": {
            0x00080050,  # Accession Number
            0x00080080,  # Institution Name
            0x00080081,  # Institution Address
            0x00081040,  # Institutional Department Name
            0x00200010,  # Study ID
            0x00321032,  # Requesting Physician
            0x00321033,  # Requesting Service
        },
        "physiciansOperators": {
            0x00080090,  # Referring Physician's Name
            0x00081048,  # Physician(s) of Record
            0x00081050,  # Performing Physician's Name
            0x00081060,  # Name of Physician(s) Reading Study
            0x00081070,  # Operators' Name
            0x00321070,  # Requested Contrast Agent
        },
        "descriptions": {
            0x00081030,  # Study Description
            0x0008103E,  # Series Description
            0x00181030,  # Protocol Name
            0x00321060,  # Requested Procedure Description
            0x00400254,  # Performed Procedure Step Description
        },
        "deviceInfo": {
            0x00081010,  # Station Name
            0x00081090,  # Manufacturer's Model Name
            0x00181000,  # Device Serial Number
            0x00181020,  # Software Versions
            0x00181050,  # Spatial Resolution
        },
        "datesAndTimes": set(),
        "privateTags": set(),
        "uids": set(),
    }

    _PATIENT_NAME_TAG = Tag(0x00100010)
    _PATIENT_ID_TAG = Tag(0x00100020)
    _UID_TAGS = (
        Tag(0x0020000D),  # Study Instance UID
        Tag(0x0020000E),  # Series Instance UID
        Tag(0x00080018),  # SOP Instance UID
    )

    def deidentify_series(
        self,
        payload: DicomDeidentifyRequest,
        progress_callback: DicomDeidentifyProgressCallback | None = None,
    ) -> DicomDeidentifyArtifact:
        series = series_registry.get(payload.series_id)
        if not series.instances:
            raise HTTPException(status_code=404, detail="No instances found for seriesId")

        total_count = len(series.instances)
        if progress_callback is not None:
            progress_callback(0, total_count)

        field_keys = self._normalize_field_keys(payload.field_keys)
        replacement_prefix = self._normalize_replacement_prefix(payload.replacement_prefix)
        uid_map = self._build_uid_map([instance.path for instance in series.instances]) if "uids" in field_keys else {}
        modified_files: list[tuple[str, bytes]] = []
        used_file_names: set[str] = set()
        series_folder = f"{self._safe_file_name_part(series.series_id)[:24] or 'series'}-deidentified"
        patient_token = self._build_patient_token(replacement_prefix, series.series_id)

        for write_index, instance in enumerate(series.instances, start=1):
            try:
                dataset = pydicom.dcmread(str(instance.path), force=True)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to read DICOM file: {exc}") from exc

            self._apply_deidentification(dataset, field_keys=field_keys, patient_token=patient_token, uid_map=uid_map)
            file_name = self._resolve_output_file_name(
                source_path=instance.path,
                instance_number=instance.instance_number,
                fallback_index=write_index,
                used_file_names=used_file_names,
            )
            modified_files.append((file_name, self._serialize_dataset(dataset)))
            if progress_callback is not None:
                progress_callback(write_index, total_count)

        return DicomDeidentifyArtifact(
            content=self._create_zip_artifact(modified_files, series_folder=series_folder),
            file_name=f"{series_folder}.zip",
            media_type="application/zip",
            modified_count=len(modified_files),
            artifact_kind="zip",
            series_folder=series_folder,
        )

    def _normalize_field_keys(self, field_keys: list[DicomDeidentifyFieldKey]) -> set[DicomDeidentifyFieldKey]:
        normalized = {field_key for field_key in field_keys if field_key in self._CLEAR_TAGS_BY_FIELD}
        if not normalized:
            raise HTTPException(status_code=400, detail="At least one de-identification option is required")
        return normalized

    def _normalize_replacement_prefix(self, value: str) -> str:
        normalized = self._safe_file_name_part(value).upper()[: self._MAX_PREFIX_LENGTH]
        return normalized or self._DEFAULT_REPLACEMENT_PREFIX

    def _build_uid_map(self, paths: Iterable[Path]) -> dict[str, str]:
        uid_map: dict[str, str] = {}
        for path in paths:
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to read DICOM UID headers: {exc}") from exc

            for tag in self._UID_TAGS:
                value = self._get_dataset_value(dataset, tag)
                if value:
                    uid_map.setdefault(value, generate_uid())

            file_meta_uid = self._get_dataset_value(getattr(dataset, "file_meta", None), Tag(0x00020003))
            if file_meta_uid:
                uid_map.setdefault(file_meta_uid, generate_uid())

        return uid_map

    @staticmethod
    def _get_dataset_value(dataset: Dataset | None, tag: Tag) -> str:
        if dataset is None or tag not in dataset:
            return ""
        value = dataset[tag].value
        return str(value).strip() if value is not None else ""

    def _apply_deidentification(
        self,
        dataset: Dataset,
        *,
        field_keys: set[DicomDeidentifyFieldKey],
        patient_token: str,
        uid_map: dict[str, str],
    ) -> None:
        if "privateTags" in field_keys:
            self._remove_private_tags(dataset)

        for nested_dataset in self._walk_datasets(dataset):
            if "patientIdentity" in field_keys:
                self._replace_if_present(nested_dataset, self._PATIENT_NAME_TAG, "ANONYMIZED")
                self._replace_if_present(nested_dataset, self._PATIENT_ID_TAG, patient_token)

            if "datesAndTimes" in field_keys:
                self._clear_date_time_elements(nested_dataset)

            for field_key in field_keys:
                for tag in self._CLEAR_TAGS_BY_FIELD[field_key]:
                    self._clear_if_present(nested_dataset, Tag(tag))

            if "uids" in field_keys:
                self._replace_uid_values(nested_dataset, uid_map)

        dataset.PatientIdentityRemoved = "YES"
        dataset.DeidentificationMethod = "DicomVision basic de-identification"
        self._sync_file_meta(dataset, uid_map)

    def _walk_datasets(self, dataset: Dataset) -> Iterable[Dataset]:
        yield dataset
        for element in dataset:
            if element.VR != "SQ" or not isinstance(element.value, Sequence):
                continue
            for item in element.value:
                if isinstance(item, Dataset):
                    yield from self._walk_datasets(item)

    def _remove_private_tags(self, dataset: Dataset) -> None:
        for nested_dataset in self._walk_datasets(dataset):
            for tag in list(nested_dataset.keys()):
                if Tag(tag).is_private:
                    del nested_dataset[tag]

    @staticmethod
    def _clear_if_present(dataset: Dataset, tag: Tag) -> None:
        if tag in dataset:
            dataset[tag].value = ""

    @staticmethod
    def _replace_if_present(dataset: Dataset, tag: Tag, value: str) -> None:
        if tag in dataset:
            dataset[tag].value = value

    @staticmethod
    def _clear_date_time_elements(dataset: Dataset) -> None:
        for element in dataset:
            if element.VR in {"DA", "DT", "TM"}:
                element.value = ""

    def _replace_uid_values(self, dataset: Dataset, uid_map: dict[str, str]) -> None:
        if not uid_map:
            return

        for element in dataset:
            if element.VR != "UI":
                continue
            element.value = self._map_uid_value(element.value, uid_map)

    def _map_uid_value(self, value: object, uid_map: dict[str, str]) -> object:
        if isinstance(value, MultiValue):
            return [uid_map.get(str(item), str(item)) for item in value]
        if isinstance(value, (list, tuple)):
            return [uid_map.get(str(item), str(item)) for item in value]
        return uid_map.get(str(value), value)

    @staticmethod
    def _sync_file_meta(dataset: Dataset, uid_map: dict[str, str]) -> None:
        file_meta = getattr(dataset, "file_meta", None)
        if file_meta is None or not uid_map:
            return

        media_storage_uid = getattr(file_meta, "MediaStorageSOPInstanceUID", None)
        if media_storage_uid:
            file_meta.MediaStorageSOPInstanceUID = uid_map.get(str(media_storage_uid), str(media_storage_uid))

    @staticmethod
    def _serialize_dataset(dataset: Dataset) -> bytes:
        buffer = BytesIO()
        try:
            dataset.save_as(buffer, write_like_original=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to serialize de-identified DICOM file: {exc}") from exc
        return buffer.getvalue()

    @staticmethod
    def _create_zip_artifact(modified_files: list[tuple[str, bytes]], *, series_folder: str) -> bytes:
        buffer = BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            for file_name, content in modified_files:
                archive.writestr(f"{series_folder}/{file_name}", content)
        return buffer.getvalue()

    def _resolve_output_file_name(
        self,
        *,
        source_path: Path,
        instance_number: int,
        fallback_index: int,
        used_file_names: set[str],
    ) -> str:
        source_stem = self._safe_file_name_part(source_path.stem) or "dicom"
        instance_label = instance_number if instance_number > 0 else fallback_index
        candidate = f"{source_stem}-deidentified-i{instance_label}.dcm"
        suffix = 1
        while candidate in used_file_names:
            candidate = f"{source_stem}-deidentified-i{instance_label}-{suffix}.dcm"
            suffix += 1
        used_file_names.add(candidate)
        return candidate

    def _build_patient_token(self, prefix: str, series_id: str) -> str:
        digest = hashlib.sha1(series_id.encode("utf-8")).hexdigest()[:8].upper()
        return f"{prefix}-{digest}"

    def _safe_file_name_part(self, value: object) -> str:
        return self._SAFE_FILE_NAME_PATTERN.sub("-", str(value)).strip(".-_ ")


dicom_deidentify_service = DicomDeidentifyService()
