import re
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from app.schemas.pacs import (
    PacsDicomwebProfile,
    PacsDicomwebTestResponse,
    PacsQidoSeriesQueryRequest,
    PacsQidoSeriesQueryResponse,
    PacsQidoStudyQueryRequest,
    PacsQidoStudyQueryResponse,
    PacsSeriesItem,
    PacsStudyItem,
)


DICOMWEB_ACCEPT_HEADER = "application/dicom+json, application/json"
DICOMWEB_DICOM_ACCEPT_HEADER = 'application/dicom, multipart/related; type="application/dicom"'


class PacsDicomwebError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PacsDicomwebService:
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def test_connection(self, profile: PacsDicomwebProfile) -> PacsDicomwebTestResponse:
        try:
            response = self._get(profile, self._qido_url(profile, "studies"), params={"limit": "1"})
            if 200 <= response.status_code < 300:
                return PacsDicomwebTestResponse(ok=True, statusCode=response.status_code, message="DICOMweb QIDO connection succeeded.")
            return PacsDicomwebTestResponse(
                ok=False,
                statusCode=response.status_code,
                message=f"DICOMweb QIDO returned HTTP {response.status_code}.",
            )
        except PacsDicomwebError as exc:
            return PacsDicomwebTestResponse(ok=False, statusCode=exc.status_code, message=str(exc))

    def query_studies(self, payload: PacsQidoStudyQueryRequest) -> PacsQidoStudyQueryResponse:
        params = self._study_query_params(payload)
        records = self._get_dicom_json(payload.profile, self._qido_url(payload.profile, "studies"), params=params)
        return PacsQidoStudyQueryResponse(items=[self._parse_study(record) for record in records])

    def query_series(self, payload: PacsQidoSeriesQueryRequest) -> PacsQidoSeriesQueryResponse:
        params: dict[str, str] = {"limit": str(payload.limit)}
        if payload.modality:
            params["Modality"] = payload.modality.strip()
        study_uid = quote(payload.study_instance_uid, safe="")
        path = f"studies/{study_uid}/series"
        records = self._get_dicom_json(payload.profile, self._qido_url(payload.profile, path), params=params)
        return PacsQidoSeriesQueryResponse(items=[self._parse_series(record, payload.study_instance_uid) for record in records])

    def query_instance_uids(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
    ) -> list[str]:
        study_uid = quote(study_instance_uid, safe="")
        series_uid = quote(series_instance_uid, safe="")
        path = f"studies/{study_uid}/series/{series_uid}/instances"
        records = self._get_dicom_json(profile, self._qido_url(profile, path), params={})
        return [uid for record in records if (uid := self._value(record, "00080018"))]

    def download_instance(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
    ) -> bytes:
        study_uid = quote(study_instance_uid, safe="")
        series_uid = quote(series_instance_uid, safe="")
        sop_uid = quote(sop_instance_uid, safe="")
        url = self._wado_url(profile, f"studies/{study_uid}/series/{series_uid}/instances/{sop_uid}")
        response = self._get(profile, url, params={}, accept_header=DICOMWEB_DICOM_ACCEPT_HEADER)
        if not 200 <= response.status_code < 300:
            raise PacsDicomwebError(f"DICOMweb WADO returned HTTP {response.status_code}.", status_code=response.status_code)
        return self._extract_dicom_payload(response.content, response.headers.get("content-type", ""))

    def _get_dicom_json(
        self,
        profile: PacsDicomwebProfile,
        url: str,
        *,
        params: dict[str, str],
    ) -> list[dict[str, Any]]:
        response = self._get(profile, url, params=params)
        if response.status_code == 204:
            return []
        if not 200 <= response.status_code < 300:
            raise PacsDicomwebError(f"DICOMweb QIDO returned HTTP {response.status_code}.", status_code=response.status_code)
        try:
            payload = response.json()
        except ValueError as exc:
            raise PacsDicomwebError("DICOMweb QIDO did not return valid JSON.", status_code=response.status_code) from exc
        if not isinstance(payload, list):
            raise PacsDicomwebError("DICOMweb QIDO response must be a JSON array.", status_code=response.status_code)
        return [record for record in payload if isinstance(record, dict)]

    def _get(
        self,
        profile: PacsDicomwebProfile,
        url: str,
        *,
        params: dict[str, str],
        accept_header: str = DICOMWEB_ACCEPT_HEADER,
    ) -> httpx.Response:
        headers = {"Accept": accept_header}
        if profile.auth_type == "bearer" and profile.bearer_token:
            headers["Authorization"] = f"Bearer {profile.bearer_token}"
        auth: httpx.Auth | None = None
        if profile.auth_type == "basic":
            auth = httpx.BasicAuth(profile.username or "", profile.password or "")
        try:
            with httpx.Client(
                headers=headers,
                auth=auth,
                timeout=profile.timeout_seconds,
                follow_redirects=True,
                transport=self._transport,
            ) as client:
                return client.get(url, params=params)
        except httpx.RequestError as exc:
            raise PacsDicomwebError(f"DICOMweb request failed: {exc}") from exc

    def _qido_url(self, profile: PacsDicomwebProfile, path: str) -> str:
        base = profile.base_url.strip().rstrip("/") + "/"
        qido_path = (profile.qido_path or "").strip().strip("/")
        if qido_path:
            base = urljoin(base, qido_path + "/")
        return urljoin(base, path.lstrip("/"))

    def _wado_url(self, profile: PacsDicomwebProfile, path: str) -> str:
        base = profile.base_url.strip().rstrip("/") + "/"
        wado_path = (profile.wado_path or "").strip().strip("/")
        if wado_path:
            base = urljoin(base, wado_path + "/")
        return urljoin(base, path.lstrip("/"))

    @staticmethod
    def _study_query_params(payload: PacsQidoStudyQueryRequest) -> dict[str, str]:
        params: dict[str, str] = {"limit": str(payload.limit)}
        if payload.patient_id:
            params["PatientID"] = payload.patient_id.strip()
        if payload.patient_name:
            params["PatientName"] = payload.patient_name.strip()
        if payload.accession_number:
            params["AccessionNumber"] = payload.accession_number.strip()
        if payload.modality:
            params["ModalitiesInStudy"] = payload.modality.strip()
        date_range = PacsDicomwebService._dicom_date_range(payload.study_date_from, payload.study_date_to)
        if date_range:
            params["StudyDate"] = date_range
        return params

    @staticmethod
    def _dicom_date_range(start: str | None, end: str | None) -> str | None:
        start_value = PacsDicomwebService._normalize_dicom_date(start)
        end_value = PacsDicomwebService._normalize_dicom_date(end)
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
    def _parse_study(record: dict[str, Any]) -> PacsStudyItem:
        return PacsStudyItem(
            studyInstanceUid=PacsDicomwebService._value(record, "0020000D") or "",
            patientName=PacsDicomwebService._value(record, "00100010"),
            patientId=PacsDicomwebService._value(record, "00100020"),
            studyDate=PacsDicomwebService._value(record, "00080020"),
            studyTime=PacsDicomwebService._value(record, "00080030"),
            accessionNumber=PacsDicomwebService._value(record, "00080050"),
            studyDescription=PacsDicomwebService._value(record, "00081030"),
            modalitiesInStudy=PacsDicomwebService._values(record, "00080061"),
            numberOfStudyRelatedSeries=PacsDicomwebService._int_value(record, "00201206"),
            numberOfStudyRelatedInstances=PacsDicomwebService._int_value(record, "00201208"),
            raw=record,
        )

    @staticmethod
    def _parse_series(record: dict[str, Any], fallback_study_uid: str) -> PacsSeriesItem:
        return PacsSeriesItem(
            studyInstanceUid=PacsDicomwebService._value(record, "0020000D") or fallback_study_uid,
            seriesInstanceUid=PacsDicomwebService._value(record, "0020000E") or "",
            seriesNumber=PacsDicomwebService._value(record, "00200011"),
            modality=PacsDicomwebService._value(record, "00080060"),
            seriesDescription=PacsDicomwebService._value(record, "0008103E"),
            bodyPartExamined=PacsDicomwebService._value(record, "00180015"),
            numberOfSeriesRelatedInstances=PacsDicomwebService._int_value(record, "00201209"),
            raw=record,
        )

    @staticmethod
    def _value(record: dict[str, Any], tag: str) -> str | None:
        values = PacsDicomwebService._values(record, tag)
        return values[0] if values else None

    @staticmethod
    def _values(record: dict[str, Any], tag: str) -> list[str]:
        attribute = record.get(tag)
        if not isinstance(attribute, dict):
            return []
        raw_values = attribute.get("Value")
        if not isinstance(raw_values, list):
            return []
        values: list[str] = []
        for item in raw_values:
            value = PacsDicomwebService._stringify_value(item)
            if value:
                values.append(value)
        return values

    @staticmethod
    def _int_value(record: dict[str, Any], tag: str) -> int | None:
        value = PacsDicomwebService._value(record, tag)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _stringify_value(value: Any) -> str | None:
        if isinstance(value, dict):
            person_name = value.get("Alphabetic") or value.get("Ideographic") or value.get("Phonetic")
            return str(person_name) if person_name is not None else None
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _extract_dicom_payload(content: bytes, content_type: str) -> bytes:
        if "multipart/related" not in content_type.lower():
            return content

        boundary_match = re.search(r'boundary="?([^";]+)"?', content_type, re.IGNORECASE)
        if not boundary_match:
            raise PacsDicomwebError("DICOMweb WADO multipart response is missing a boundary.")

        boundary = boundary_match.group(1).encode()
        delimiter = b"--" + boundary
        for raw_part in content.split(delimiter):
            part = raw_part.strip()
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].rstrip()

            _, separator, body = part.partition(b"\r\n\r\n")
            if not separator:
                _, separator, body = part.partition(b"\n\n")
            if separator and body:
                return body.strip(b"\r\n")

        raise PacsDicomwebError("DICOMweb WADO multipart response did not contain a DICOM part.")


pacs_dicomweb_service = PacsDicomwebService()
