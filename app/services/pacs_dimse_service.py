from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import re
from typing import Any

from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pynetdicom import AE, StoragePresentationContexts, build_role, evt
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelGet,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelGet,
    Verification,
)

from app.schemas.pacs import (
    PacsDimseProfile,
    PacsDimseSeriesQueryRequest,
    PacsDimseStudyQueryRequest,
    PacsDicomwebTestResponse,
    PacsQidoSeriesQueryResponse,
    PacsQidoStudyQueryResponse,
    PacsSeriesItem,
    PacsStudyItem,
)


DIMSE_PENDING_STATUSES = {0xFF00, 0xFF01}
DIMSE_SUCCESS_STATUS = 0x0000
DIMSE_WARNING_STATUSES = {0xB000, 0xB006, 0xB007}


class PacsDimseError(RuntimeError):
    pass


class PacsDimseService:
    def __init__(self, ae_factory: Callable[[str], Any] | None = None) -> None:
        self._ae_factory = ae_factory or self._create_ae

    def test_connection(self, profile: PacsDimseProfile) -> PacsDicomwebTestResponse:
        try:
            association = self._associate(profile, [Verification])
            if not association.is_established:
                return PacsDicomwebTestResponse(ok=False, statusCode=None, message="DIMSE association was rejected or timed out.")

            try:
                status = association.send_c_echo()
                status_code = self._status_code(status)
                if status_code == DIMSE_SUCCESS_STATUS:
                    return PacsDicomwebTestResponse(ok=True, statusCode=status_code, message="DIMSE C-ECHO succeeded.")
                status_text = self._format_status(status_code)
                return PacsDicomwebTestResponse(ok=False, statusCode=status_code, message=f"DIMSE C-ECHO returned {status_text}.")
            finally:
                association.release()
        except PacsDimseError as exc:
            return PacsDicomwebTestResponse(ok=False, statusCode=None, message=str(exc))
        except Exception as exc:
            return PacsDicomwebTestResponse(ok=False, statusCode=None, message=f"DIMSE request failed: {exc}")

    def query_studies(self, payload: PacsDimseStudyQueryRequest) -> PacsQidoStudyQueryResponse:
        dataset = self._study_query_dataset(payload)
        records = self._send_c_find(payload.profile, dataset, limit=payload.limit, offset=payload.offset)
        return PacsQidoStudyQueryResponse(items=[self._parse_study(record) for record in records])

    def query_series(self, payload: PacsDimseSeriesQueryRequest) -> PacsQidoSeriesQueryResponse:
        dataset = self._series_query_dataset(payload)
        records = self._send_c_find(payload.profile, dataset, limit=payload.limit, offset=payload.offset)
        return PacsQidoSeriesQueryResponse(items=[self._parse_series(record, payload.study_instance_uid) for record in records])

    def retrieve_series(
        self,
        profile: PacsDimseProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> int:
        dataset = Dataset()
        dataset.QueryRetrieveLevel = "SERIES"
        dataset.StudyInstanceUID = study_instance_uid.strip()
        dataset.SeriesInstanceUID = series_instance_uid.strip()
        context = self._query_model_get_context(profile)
        output_dir.mkdir(parents=True, exist_ok=True)

        received_count = 0
        expected_total = 0

        def handle_store(event: Any) -> int:
            nonlocal received_count, expected_total
            if should_cancel is not None and should_cancel():
                return 0xC000

            try:
                received_count += 1
                self._store_c_get_dataset(event, output_dir, received_count)
            except Exception:
                return 0xA701

            expected_total = max(expected_total, received_count)
            if progress_callback is not None:
                progress_callback(received_count, expected_total)
            return 0x0000

        ext_neg = [build_role(cx.abstract_syntax, scp_role=True) for cx in StoragePresentationContexts]
        association = self._associate(
            profile,
            [context, *(cx.abstract_syntax for cx in StoragePresentationContexts)],
            ext_neg=ext_neg,
            evt_handlers=[(evt.EVT_C_STORE, handle_store)],
        )
        if not association.is_established:
            raise PacsDimseError("DIMSE association was rejected or timed out.")

        try:
            for status, _identifier in association.send_c_get(dataset, context):
                status_code = self._status_code(status)
                total = self._suboperation_total(status)
                if total:
                    expected_total = max(expected_total, total)
                    if progress_callback is not None:
                        progress_callback(received_count, expected_total)

                if status_code in DIMSE_PENDING_STATUSES:
                    continue
                if status_code in {DIMSE_SUCCESS_STATUS, *DIMSE_WARNING_STATUSES}:
                    break
                raise PacsDimseError(f"DIMSE C-GET returned {self._format_status(status_code)}.")
        finally:
            association.release()

        if should_cancel is not None and should_cancel():
            raise PacsDimseError("PACS download was cancelled.")
        if received_count <= 0:
            raise PacsDimseError("DIMSE C-GET returned no instances for this series.")
        if progress_callback is not None:
            progress_callback(received_count, max(expected_total, received_count))
        return received_count

    def _send_c_find(
        self,
        profile: PacsDimseProfile,
        dataset: Dataset,
        *,
        limit: int,
        offset: int,
    ) -> list[Dataset]:
        context = self._query_model_context(profile)
        association = self._associate(profile, [context])
        if not association.is_established:
            raise PacsDimseError("DIMSE association was rejected or timed out.")

        records: list[Dataset] = []
        matched_count = 0
        try:
            for status, identifier in association.send_c_find(dataset, context):
                status_code = self._status_code(status)
                if status_code in DIMSE_PENDING_STATUSES:
                    if identifier is None:
                        continue
                    if matched_count >= offset and len(records) < limit:
                        records.append(identifier)
                    matched_count += 1
                    if len(records) >= limit:
                        break
                    continue
                if status_code == DIMSE_SUCCESS_STATUS:
                    break
                raise PacsDimseError(f"DIMSE C-FIND returned {self._format_status(status_code)}.")
        finally:
            association.release()

        return records

    def _associate(
        self,
        profile: PacsDimseProfile,
        contexts: Iterable[Any],
        *,
        ext_neg: list[Any] | None = None,
        evt_handlers: list[Any] | None = None,
    ) -> Any:
        try:
            ae = self._ae_factory(profile.client_ae_title.strip())
            ae.acse_timeout = profile.timeout_seconds
            ae.dimse_timeout = profile.timeout_seconds
            ae.network_timeout = profile.timeout_seconds
            for context in contexts:
                ae.add_requested_context(context)
            return ae.associate(
                profile.host.strip(),
                profile.port,
                ae_title=profile.called_ae_title.strip(),
                ext_neg=ext_neg,
                evt_handlers=evt_handlers,
            )
        except Exception as exc:
            raise PacsDimseError(f"DIMSE association failed: {exc}") from exc

    @staticmethod
    def _create_ae(client_ae_title: str) -> AE:
        return AE(ae_title=client_ae_title)

    @staticmethod
    def _query_model_context(profile: PacsDimseProfile) -> Any:
        if profile.query_model == "patient-root":
            return PatientRootQueryRetrieveInformationModelFind
        return StudyRootQueryRetrieveInformationModelFind

    @staticmethod
    def _query_model_get_context(profile: PacsDimseProfile) -> Any:
        if profile.query_model == "patient-root":
            return PatientRootQueryRetrieveInformationModelGet
        return StudyRootQueryRetrieveInformationModelGet

    @staticmethod
    def _store_c_get_dataset(event: Any, output_dir: Path, index: int) -> Path:
        dataset = event.dataset
        dataset.file_meta = event.file_meta
        sop_instance_uid = PacsDimseService._safe_sop_instance_uid(getattr(dataset, "SOPInstanceUID", "unknown"))
        output_path = output_dir / f"IM_{index:04d}_{sop_instance_uid}.dcm"
        dataset.save_as(output_path, write_like_original=False)
        return output_path

    @staticmethod
    def _safe_sop_instance_uid(value: Any) -> str:
        safe_uid = re.sub(r"[^\d.]", "_", str(value or "unknown"))
        return safe_uid[:128] or "unknown"

    @staticmethod
    def _study_query_dataset(payload: PacsDimseStudyQueryRequest) -> Dataset:
        dataset = Dataset()
        dataset.QueryRetrieveLevel = "STUDY"
        PacsDimseService._set_value(dataset, "StudyInstanceUID", payload.study_instance_uid)
        PacsDimseService._set_value(dataset, "PatientID", payload.patient_id)
        PacsDimseService._set_value(dataset, "PatientName", payload.patient_name)
        PacsDimseService._set_value(dataset, "AccessionNumber", payload.accession_number)
        PacsDimseService._set_value(dataset, "StudyDescription", payload.study_description)
        PacsDimseService._set_value(dataset, "ModalitiesInStudy", payload.modality)
        PacsDimseService._set_value(dataset, "StudyDate", PacsDimseService._dicom_date_range(payload.study_date_from, payload.study_date_to))
        PacsDimseService._ensure_return_keys(
            dataset,
            [
                "StudyTime",
                "NumberOfStudyRelatedSeries",
                "NumberOfStudyRelatedInstances",
            ],
        )
        return dataset

    @staticmethod
    def _series_query_dataset(payload: PacsDimseSeriesQueryRequest) -> Dataset:
        dataset = Dataset()
        dataset.QueryRetrieveLevel = "SERIES"
        dataset.StudyInstanceUID = payload.study_instance_uid.strip()
        PacsDimseService._set_value(dataset, "SeriesInstanceUID", payload.series_instance_uid)
        PacsDimseService._set_value(dataset, "Modality", payload.modality)
        PacsDimseService._set_value(dataset, "SeriesDescription", payload.series_description)
        PacsDimseService._set_value(dataset, "BodyPartExamined", payload.body_part_examined)
        PacsDimseService._ensure_return_keys(
            dataset,
            [
                "SeriesNumber",
                "NumberOfSeriesRelatedInstances",
            ],
        )
        return dataset

    @staticmethod
    def _set_value(dataset: Dataset, keyword: str, value: str | None) -> None:
        normalized = value.strip() if isinstance(value, str) else ""
        setattr(dataset, keyword, normalized)

    @staticmethod
    def _ensure_return_keys(dataset: Dataset, keywords: Iterable[str]) -> None:
        for keyword in keywords:
            if keyword not in dataset:
                setattr(dataset, keyword, "")

    @staticmethod
    def _dicom_date_range(start: str | None, end: str | None) -> str | None:
        start_value = PacsDimseService._normalize_dicom_date(start)
        end_value = PacsDimseService._normalize_dicom_date(end)
        if start_value and end_value:
            return f"{start_value}-{end_value}"
        if start_value:
            return f"{start_value}-"
        if end_value:
            return f"-{end_value}"
        return None

    @staticmethod
    def _normalize_dicom_date(value: str | None) -> str | None:
        if not value:
            return None
        stripped = value.strip().replace("-", "")
        return stripped if len(stripped) == 8 and stripped.isdigit() else None

    @staticmethod
    def _parse_study(record: Dataset) -> PacsStudyItem:
        return PacsStudyItem(
            studyInstanceUid=PacsDimseService._text_value(record, "StudyInstanceUID") or "",
            patientName=PacsDimseService._text_value(record, "PatientName"),
            patientId=PacsDimseService._text_value(record, "PatientID"),
            studyDate=PacsDimseService._text_value(record, "StudyDate"),
            studyTime=PacsDimseService._text_value(record, "StudyTime"),
            accessionNumber=PacsDimseService._text_value(record, "AccessionNumber"),
            studyDescription=PacsDimseService._text_value(record, "StudyDescription"),
            modalitiesInStudy=PacsDimseService._list_values(record, "ModalitiesInStudy"),
            numberOfStudyRelatedSeries=PacsDimseService._int_value(record, "NumberOfStudyRelatedSeries"),
            numberOfStudyRelatedInstances=PacsDimseService._int_value(record, "NumberOfStudyRelatedInstances"),
            raw=PacsDimseService._raw_dataset(record),
        )

    @staticmethod
    def _parse_series(record: Dataset, fallback_study_uid: str) -> PacsSeriesItem:
        return PacsSeriesItem(
            studyInstanceUid=PacsDimseService._text_value(record, "StudyInstanceUID") or fallback_study_uid,
            seriesInstanceUid=PacsDimseService._text_value(record, "SeriesInstanceUID") or "",
            seriesNumber=PacsDimseService._text_value(record, "SeriesNumber"),
            modality=PacsDimseService._text_value(record, "Modality"),
            seriesDescription=PacsDimseService._text_value(record, "SeriesDescription"),
            bodyPartExamined=PacsDimseService._text_value(record, "BodyPartExamined"),
            numberOfSeriesRelatedInstances=PacsDimseService._int_value(record, "NumberOfSeriesRelatedInstances"),
            raw=PacsDimseService._raw_dataset(record),
        )

    @staticmethod
    def _text_value(record: Dataset, keyword: str) -> str | None:
        value = record.get(keyword)
        if value is None:
            return None
        if isinstance(value, MultiValue):
            return "\\".join(str(item) for item in value if item is not None) or None
        text = str(value)
        return text if text else None

    @staticmethod
    def _list_values(record: Dataset, keyword: str) -> list[str]:
        value = record.get(keyword)
        if value is None:
            return []
        if isinstance(value, MultiValue):
            return [str(item) for item in value if item is not None and str(item)]
        text = str(value)
        if not text:
            return []
        return [item for item in text.split("\\") if item]

    @staticmethod
    def _int_value(record: Dataset, keyword: str) -> int | None:
        value = PacsDimseService._text_value(record, keyword)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _raw_dataset(record: Dataset) -> dict[str, str]:
        raw: dict[str, str] = {}
        for element in record:
            keyword = element.keyword or str(element.tag)
            raw[keyword] = str(element.value)
        return raw

    @staticmethod
    def _status_code(status: Any) -> int | None:
        if status is None:
            return None
        value = getattr(status, "Status", status)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_status(status_code: int | None) -> str:
        return "no status" if status_code is None else f"0x{status_code:04X}"

    @staticmethod
    def _suboperation_total(status: Any) -> int:
        if status is None:
            return 0
        total = 0
        for keyword in (
            "NumberOfRemainingSuboperations",
            "NumberOfCompletedSuboperations",
            "NumberOfFailedSuboperations",
            "NumberOfWarningSuboperations",
        ):
            try:
                total += int(getattr(status, keyword, 0) or 0)
            except (TypeError, ValueError):
                continue
        return total


pacs_dimse_service = PacsDimseService()
