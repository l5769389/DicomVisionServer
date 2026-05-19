from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

import pydicom
from fastapi import HTTPException
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.valuerep import DSfloat, DSdecimal, IS

from app.schemas.dicom import DicomTagItem, DicomTagModifyRequest, DicomTagsRequest, DicomTagsResponse
from app.services.series_registry import series_registry


DicomTagModifyProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class DicomTagModifyArtifact:
    content: bytes
    file_name: str
    media_type: str
    modified_count: int
    artifact_kind: str
    series_folder: str
    tag: str
    keyword: str
    vr: str


class DicomTagService:
    _MAX_TEXT_LENGTH = 512
    _MAX_MULTI_VALUE_ITEMS = 12
    _BINARY_VR_VALUES = {"OB", "OD", "OF", "OL", "OV", "OW", "UN"}
    _INTEGER_VR_VALUES = {"IS", "SL", "SS", "UL", "US", "SV", "UV"}
    _DECIMAL_VR_VALUES = {"DS", "FL", "FD"}
    _MAX_TEXT_LENGTH_BY_VR = {
        "AE": 16,
        "AS": 4,
        "CS": 16,
        "DA": 8,
        "DT": 26,
        "LO": 64,
        "PN": 64,
        "SH": 16,
        "TM": 16,
        "UI": 64,
    }
    _DA_PATTERN = re.compile(r"^\d{8}$")
    _TM_PATTERN = re.compile(r"^(\d{2})(\d{2})?(\d{2})?(?:\.\d{1,6})?$")
    _AS_PATTERN = re.compile(r"^\d{3}[DWMY]$")
    _CS_PATTERN = re.compile(r"^[A-Z0-9_ ]*$")
    _UI_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
    _SAFE_FILE_NAME_PATTERN = re.compile(r'[\\/:*?"<>|\s]+')

    def get_series_tags(self, payload: DicomTagsRequest) -> DicomTagsResponse:
        series = series_registry.get(payload.series_id)
        if not series.instances:
            raise HTTPException(status_code=404, detail="No instances found for seriesId")

        index = max(0, min(payload.index, len(series.instances) - 1))
        instance = series.instances[index]

        try:
            dataset = pydicom.dcmread(str(instance.path), defer_size=1024, force=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read DICOM tags: {exc}") from exc

        return DicomTagsResponse(
            seriesId=series.series_id,
            index=index,
            total=len(series.instances),
            instanceNumber=instance.instance_number,
            sopInstanceUid=instance.sop_instance_uid,
            filePath=str(instance.path),
            items=self._dataset_to_items(dataset),
        )

    def _dataset_to_items(self, dataset: Dataset) -> list[DicomTagItem]:
        items: list[DicomTagItem] = []
        for element in dataset:
            items.extend(self._element_to_items(element, depth=0, tag_path=[]))
        return items

    def _element_to_items(self, element: DataElement, *, depth: int, tag_path: list[str]) -> list[DicomTagItem]:
        tag_label = str(element.tag)
        current_tag_path = [*tag_path, self._tag_to_path_token(element.tag)]
        keyword = element.keyword or ""
        name = element.name or keyword or tag_label
        vr = element.VR or ""

        if vr == "SQ":
            sequence_items = list(element.value) if isinstance(element.value, Sequence) else []
            items = [
                DicomTagItem(
                    tag=tag_label,
                    keyword=keyword,
                    name=name,
                    vr=vr,
                    value=f"Sequence with {len(sequence_items)} item(s)",
                    depth=depth,
                    tagPath=current_tag_path,
                )
            ]
            for item_index, nested_dataset in enumerate(sequence_items):
                items.append(
                    DicomTagItem(
                        tag="",
                        keyword=f"Item{item_index + 1}",
                        name=f"Item #{item_index + 1}",
                        vr="ITEM",
                        value="",
                        depth=depth + 1,
                        tagPath=[*current_tag_path, str(item_index)],
                    )
                )
                for nested_element in nested_dataset:
                    items.extend(
                        self._element_to_items(
                            nested_element,
                            depth=depth + 2,
                            tag_path=[*current_tag_path, str(item_index)],
                        )
                    )
            return items

        return [
            DicomTagItem(
                tag=tag_label,
                keyword=keyword,
                name=name,
                vr=vr,
                value=self._format_value(element),
                depth=depth,
                tagPath=current_tag_path,
            )
        ]

    def modify_series_tag(
        self,
        payload: DicomTagModifyRequest,
        progress_callback: DicomTagModifyProgressCallback | None = None,
    ) -> DicomTagModifyArtifact:
        series = series_registry.get(payload.series_id)
        if not series.instances:
            raise HTTPException(status_code=404, detail="No instances found for seriesId")

        target_index = max(0, min(payload.index, len(series.instances) - 1))
        target_instances = series.instances if payload.scope == "series" else [series.instances[target_index]]
        total_count = len(target_instances)
        modified_files: list[tuple[str, bytes]] = []
        used_file_names: set[str] = set()
        series_folder = self._safe_file_name_part(series.series_id)[:24] or "series"
        tag_label = ""
        keyword = ""
        vr = ""
        if progress_callback is not None:
            progress_callback(0, total_count)

        for write_index, instance in enumerate(target_instances, start=1):
            try:
                dataset = pydicom.dcmread(str(instance.path), force=True)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to read DICOM file: {exc}") from exc

            element = self._resolve_element_by_path(dataset, payload.tag_path)
            self._assert_editable_element(element)
            tag_label = str(element.tag)
            keyword = element.keyword or ""
            vr = element.VR or ""
            element.value = self._coerce_value_for_element(element, payload.value)
            self._sync_file_meta_after_edit(dataset, element)

            file_name = self._resolve_output_file_name(
                source_path=instance.path,
                instance_number=instance.instance_number,
                fallback_index=write_index,
                used_file_names=used_file_names,
            )
            modified_files.append((file_name, self._serialize_dataset(dataset)))
            if progress_callback is not None:
                progress_callback(write_index, total_count)

        if len(modified_files) == 1 and payload.scope == "current":
            file_name, content = modified_files[0]
            return DicomTagModifyArtifact(
                content=content,
                file_name=file_name,
                media_type="application/dicom",
                modified_count=1,
                artifact_kind="dicom",
                series_folder=series_folder,
                tag=tag_label,
                keyword=keyword,
                vr=vr,
            )

        archive_name = f"{series_folder}-tag-edits.zip"
        return DicomTagModifyArtifact(
            content=self._create_zip_artifact(modified_files, series_folder=series_folder),
            file_name=archive_name,
            media_type="application/zip",
            modified_count=len(modified_files),
            artifact_kind="zip",
            series_folder=series_folder,
            tag=tag_label,
            keyword=keyword,
            vr=vr,
        )

    def _format_value(self, element: DataElement) -> str:
        if element.tag == 0x7FE00010:
            return "<Pixel Data omitted>"

        value = element.value
        if value is None:
            return ""

        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"

        if isinstance(value, MultiValue):
            rendered = [self._format_scalar(item) for item in list(value)[: self._MAX_MULTI_VALUE_ITEMS]]
            suffix = " ..." if len(value) > self._MAX_MULTI_VALUE_ITEMS else ""
            return ", ".join(rendered) + suffix

        if isinstance(value, Sequence):
            return f"Sequence with {len(value)} item(s)"

        if isinstance(value, Dataset):
            return "Dataset"

        return self._truncate(self._format_scalar(value))

    def _format_scalar(self, value: object) -> str:
        if isinstance(value, str):
            return self._truncate(value)

        if isinstance(value, (IS, DSfloat, DSdecimal, int, float, bool)):
            return str(value)

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, MultiValue, Sequence, Dataset)):
            return ", ".join(self._truncate(str(item)) for item in value)

        return self._truncate(str(value))

    def _truncate(self, value: str) -> str:
        if len(value) <= self._MAX_TEXT_LENGTH:
            return value
        return f"{value[: self._MAX_TEXT_LENGTH]}..."

    @staticmethod
    def _tag_to_path_token(tag: object) -> str:
        return f"{int(Tag(tag)):08X}"

    def _resolve_element_by_path(self, dataset: Dataset, tag_path: list[str]) -> DataElement:
        if not tag_path:
            raise HTTPException(status_code=400, detail="tagPath is required")

        current_dataset = dataset
        index = 0
        while index < len(tag_path):
            try:
                tag = Tag(int(tag_path[index], 16))
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid tagPath token") from exc

            if tag not in current_dataset:
                raise HTTPException(status_code=404, detail="Target tag was not found in DICOM dataset")

            element = current_dataset[tag]
            if index == len(tag_path) - 1:
                return element

            if element.VR != "SQ" or not isinstance(element.value, Sequence):
                raise HTTPException(status_code=400, detail="tagPath traverses through a non-sequence tag")

            try:
                item_index = int(tag_path[index + 1])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid sequence item index in tagPath") from exc

            if item_index < 0 or item_index >= len(element.value):
                raise HTTPException(status_code=404, detail="Sequence item index was not found")

            nested_dataset = element.value[item_index]
            if not isinstance(nested_dataset, Dataset):
                raise HTTPException(status_code=400, detail="Sequence item is not a DICOM dataset")

            current_dataset = nested_dataset
            index += 2

        raise HTTPException(status_code=400, detail="tagPath does not point to a DICOM element")

    def _assert_editable_element(self, element: DataElement) -> None:
        if element.tag == 0x7FE00010 or element.VR in self._BINARY_VR_VALUES:
            raise HTTPException(status_code=400, detail="Binary DICOM tags cannot be edited here")
        if element.VR == "SQ":
            raise HTTPException(status_code=400, detail="Sequence tags cannot be edited directly")

    @staticmethod
    def _coerce_value_for_element(element: DataElement, value: str) -> object:
        vr = (element.VR or "").upper()
        normalized_value = value.strip()
        values = [part.strip() for part in normalized_value.split("\\")]
        coerced_values = [DicomTagService._normalize_value_for_vr(vr, part) for part in values]
        for part in coerced_values:
            DicomTagService._validate_value_for_vr(vr, part)
        return coerced_values if len(coerced_values) > 1 else coerced_values[0]

    @staticmethod
    def _normalize_value_for_vr(vr: str, value: str) -> str:
        if vr == "CS":
            return value.upper()
        return value

    @staticmethod
    def _validate_value_for_vr(vr: str, value: str) -> None:
        max_length = DicomTagService._MAX_TEXT_LENGTH_BY_VR.get(vr)
        if max_length is not None and len(value) > max_length:
            raise HTTPException(status_code=400, detail=f"{vr} value must be at most {max_length} characters")

        if value == "":
            return

        if vr in DicomTagService._INTEGER_VR_VALUES and not re.fullmatch(r"[+-]?\d+", value):
            raise HTTPException(status_code=400, detail=f"{vr} value must be an integer")
        if vr in DicomTagService._DECIMAL_VR_VALUES:
            try:
                numeric_value = float(value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{vr} value must be numeric") from exc
            if not numeric_value == numeric_value or numeric_value in {float("inf"), float("-inf")}:
                raise HTTPException(status_code=400, detail=f"{vr} value must be finite")
        if vr == "DA" and not DicomTagService._is_valid_dicom_date(value):
            raise HTTPException(status_code=400, detail="DA value must use YYYYMMDD")
        if vr == "TM" and not DicomTagService._is_valid_dicom_time(value):
            raise HTTPException(status_code=400, detail="TM value must use HHMMSS")
        if vr == "AS" and not DicomTagService._AS_PATTERN.fullmatch(value.upper()):
            raise HTTPException(status_code=400, detail="AS value must use 3 digits plus D/W/M/Y")
        if vr == "CS" and not DicomTagService._CS_PATTERN.fullmatch(value.upper()):
            raise HTTPException(status_code=400, detail="CS value contains unsupported characters")
        if vr == "UI" and not DicomTagService._UI_PATTERN.fullmatch(value):
            raise HTTPException(status_code=400, detail="UI value must contain digits and dots only")

    @staticmethod
    def _is_valid_dicom_date(value: str) -> bool:
        if not DicomTagService._DA_PATTERN.fullmatch(value):
            return False
        year = int(value[:4])
        month = int(value[4:6])
        day = int(value[6:8])
        if month < 1 or month > 12 or day < 1 or day > 31:
            return False
        try:
            from datetime import date

            date(year, month, day)
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_valid_dicom_time(value: str) -> bool:
        match = DicomTagService._TM_PATTERN.fullmatch(value)
        if not match:
            return False
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        second = int(match.group(3) or "0")
        return 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59

    @staticmethod
    def _sync_file_meta_after_edit(dataset: Dataset, element: DataElement) -> None:
        if element.tag == Tag(0x00080018) and getattr(dataset, "file_meta", None) is not None:
            dataset.file_meta.MediaStorageSOPInstanceUID = str(element.value)
        if element.tag == Tag(0x00080016) and getattr(dataset, "file_meta", None) is not None:
            dataset.file_meta.MediaStorageSOPClassUID = str(element.value)

    @staticmethod
    def _serialize_dataset(dataset: Dataset) -> bytes:
        buffer = BytesIO()
        try:
            dataset.save_as(buffer, write_like_original=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to serialize modified DICOM file: {exc}") from exc
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
        candidate = f"{source_stem}-tag-edit-i{instance_label}.dcm"
        suffix = 1
        while candidate in used_file_names:
            candidate = f"{source_stem}-tag-edit-i{instance_label}-{suffix}.dcm"
            suffix += 1
        used_file_names.add(candidate)
        return candidate

    def _safe_file_name_part(self, value: object) -> str:
        return self._SAFE_FILE_NAME_PATTERN.sub("-", str(value)).strip(".-_ ")


dicom_tag_service = DicomTagService()
