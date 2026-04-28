from collections.abc import Iterable

import pydicom
from fastapi import HTTPException
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.valuerep import DSfloat, DSdecimal, IS

from app.schemas.dicom import DicomTagItem, DicomTagsRequest, DicomTagsResponse
from app.services.series_registry import series_registry


class DicomTagService:
    _MAX_TEXT_LENGTH = 512
    _MAX_MULTI_VALUE_ITEMS = 12

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
            items.extend(self._element_to_items(element, depth=0))
        return items

    def _element_to_items(self, element: DataElement, *, depth: int) -> list[DicomTagItem]:
        tag_label = str(element.tag)
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
                )
            ]
            for item_index, nested_dataset in enumerate(sequence_items, start=1):
                items.append(
                    DicomTagItem(
                        tag="",
                        keyword=f"Item{item_index}",
                        name=f"Item #{item_index}",
                        vr="ITEM",
                        value="",
                        depth=depth + 1,
                    )
                )
                for nested_element in nested_dataset:
                    items.extend(self._element_to_items(nested_element, depth=depth + 2))
            return items

        return [
            DicomTagItem(
                tag=tag_label,
                keyword=keyword,
                name=name,
                vr=vr,
                value=self._format_value(element),
                depth=depth,
            )
        ]

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


dicom_tag_service = DicomTagService()
