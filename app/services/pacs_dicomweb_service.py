import base64
import io
import re
from typing import Any
from urllib.parse import quote, urljoin

import httpx
import numpy as np
import pydicom
from PIL import Image, ImageOps
from pydicom.multival import MultiValue
from pydicom.uid import UID

from app.schemas.pacs import (
    PacsDicomwebProfile,
    PacsDicomwebTestResponse,
    PacsQidoSeriesQueryRequest,
    PacsQidoSeriesQueryResponse,
    PacsQidoStudyQueryRequest,
    PacsQidoStudyQueryResponse,
    PacsSeriesPreviewRequest,
    PacsSeriesPreviewResponse,
    PacsSeriesItem,
    PacsStudyItem,
)


DICOMWEB_ACCEPT_HEADER = "application/dicom+json, application/json"
DICOMWEB_DICOM_ACCEPT_HEADER = 'application/dicom, multipart/related; type="application/dicom"'
DICOMWEB_RENDERED_ACCEPT_HEADER = "image/png, image/jpeg;q=0.9, image/*;q=0.8"
PACS_THUMBNAIL_SIZE = (192, 192)
TAG_TRANSFER_SYNTAX_UID = "00020010"
TAG_SOP_INSTANCE_UID = "00080018"
TAG_MODALITY = "00080060"
TAG_MODALITIES_IN_STUDY = "00080061"
TAG_STUDY_DATE = "00080020"
TAG_STUDY_TIME = "00080030"
TAG_ACCESSION_NUMBER = "00080050"
TAG_STUDY_DESCRIPTION = "00081030"
TAG_SERIES_DESCRIPTION = "0008103E"
TAG_PATIENT_NAME = "00100010"
TAG_PATIENT_ID = "00100020"
TAG_BODY_PART_EXAMINED = "00180015"
TAG_STUDY_INSTANCE_UID = "0020000D"
TAG_SERIES_INSTANCE_UID = "0020000E"
TAG_SERIES_NUMBER = "00200011"
TAG_STUDY_RELATED_SERIES = "00201206"
TAG_STUDY_RELATED_INSTANCES = "00201208"
TAG_SERIES_RELATED_INSTANCES = "00201209"
TAG_NUMBER_OF_FRAMES = "00280008"
TAG_ROWS = "00280010"
TAG_COLUMNS = "00280011"
TAG_PHOTOMETRIC_INTERPRETATION = "00280004"


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
        params = self._series_query_params(payload)
        study_uid = quote(payload.study_instance_uid, safe="")
        path = f"studies/{study_uid}/series"
        records = self._get_dicom_json(payload.profile, self._qido_url(payload.profile, path), params=params)
        return PacsQidoSeriesQueryResponse(items=[self._parse_series(record, payload.study_instance_uid) for record in records])

    def preview_series(self, payload: PacsSeriesPreviewRequest) -> PacsSeriesPreviewResponse:
        records = self.query_instance_records(
            payload.profile,
            study_instance_uid=payload.study_instance_uid,
            series_instance_uid=payload.series_instance_uid,
        )
        first_record = records[0] if records else {}
        sop_instance_uid = self._value(first_record, TAG_SOP_INSTANCE_UID)
        metadata_record: dict[str, Any] = {}
        if sop_instance_uid:
            try:
                metadata_record = self.get_instance_metadata(
                    payload.profile,
                    study_instance_uid=payload.study_instance_uid,
                    series_instance_uid=payload.series_instance_uid,
                    sop_instance_uid=sop_instance_uid,
                )
            except PacsDicomwebError:
                metadata_record = {}

        summary_record = {**first_record, **metadata_record}
        summary_records = records + ([metadata_record] if metadata_record else [])
        frame_counts = [frame_count for record in summary_records if (frame_count := self._int_value(record, TAG_NUMBER_OF_FRAMES)) is not None]
        number_of_frames = max(frame_counts) if frame_counts else None
        transfer_syntaxes = self._unique_values(summary_records, TAG_TRANSFER_SYNTAX_UID)
        photometric_interpretations = self._unique_values(summary_records, TAG_PHOTOMETRIC_INTERPRETATION)
        thumbnail_src: str | None = None
        thumbnail_error: str | None = None

        if payload.thumbnail and sop_instance_uid:
            try:
                media_type, content = self.render_instance_thumbnail(
                    payload.profile,
                    study_instance_uid=payload.study_instance_uid,
                    series_instance_uid=payload.series_instance_uid,
                    sop_instance_uid=sop_instance_uid,
                )
                if content:
                    encoded = base64.b64encode(content).decode("ascii")
                    thumbnail_src = f"data:{media_type};base64,{encoded}"
            except PacsDicomwebError as exc:
                thumbnail_error = str(exc)

        return PacsSeriesPreviewResponse(
            studyInstanceUid=payload.study_instance_uid,
            seriesInstanceUid=payload.series_instance_uid,
            instanceCount=len(records),
            rows=self._int_value(summary_record, TAG_ROWS),
            columns=self._int_value(summary_record, TAG_COLUMNS),
            numberOfFrames=number_of_frames,
            hasMultiFrameInstances=any(frame_count > 1 for frame_count in frame_counts),
            transferSyntaxes=transfer_syntaxes,
            isCompressed=any(self._is_compressed_transfer_syntax(value) for value in transfer_syntaxes),
            photometricInterpretations=photometric_interpretations,
            sopInstanceUid=sop_instance_uid,
            thumbnailSrc=thumbnail_src,
            thumbnailError=thumbnail_error,
        )

    def query_instance_uids(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
    ) -> list[str]:
        records = self.query_instance_records(
            profile,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
        )
        return [uid for record in records if (uid := self._value(record, TAG_SOP_INSTANCE_UID))]

    def query_instance_records(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
    ) -> list[dict[str, Any]]:
        study_uid = quote(study_instance_uid, safe="")
        series_uid = quote(series_instance_uid, safe="")
        path = f"studies/{study_uid}/series/{series_uid}/instances"
        return self._get_dicom_json(profile, self._qido_url(profile, path), params={})

    def get_instance_metadata(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
    ) -> dict[str, Any]:
        study_uid = quote(study_instance_uid, safe="")
        series_uid = quote(series_instance_uid, safe="")
        sop_uid = quote(sop_instance_uid, safe="")
        path = f"studies/{study_uid}/series/{series_uid}/instances/{sop_uid}/metadata"
        records = self._get_dicom_json(profile, self._wado_url(profile, path), params={})
        return records[0] if records else {}

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

    def render_instance_thumbnail(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
    ) -> tuple[str, bytes]:
        study_uid = quote(study_instance_uid, safe="")
        series_uid = quote(series_instance_uid, safe="")
        sop_uid = quote(sop_instance_uid, safe="")
        path = f"studies/{study_uid}/series/{series_uid}/instances/{sop_uid}/rendered"
        response = self._get(profile, self._wado_url(profile, path), params={}, accept_header=DICOMWEB_RENDERED_ACCEPT_HEADER)
        if not 200 <= response.status_code < 300:
            rendered_error = PacsDicomwebError(f"DICOMweb rendered thumbnail returned HTTP {response.status_code}.", status_code=response.status_code)
            try:
                return self.render_downloaded_instance_thumbnail(
                    profile,
                    study_instance_uid=study_instance_uid,
                    series_instance_uid=series_instance_uid,
                    sop_instance_uid=sop_instance_uid,
                )
            except PacsDicomwebError as exc:
                raise PacsDicomwebError(f"{rendered_error} Local DICOM thumbnail fallback failed: {exc}", status_code=response.status_code) from exc
        media_type = response.headers.get("content-type", "image/png").split(";", 1)[0].strip().lower()
        if not media_type.startswith("image/"):
            media_type = "image/png"
        return media_type, response.content

    def render_downloaded_instance_thumbnail(
        self,
        profile: PacsDicomwebProfile,
        *,
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
    ) -> tuple[str, bytes]:
        content = self.download_instance(
            profile,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            sop_instance_uid=sop_instance_uid,
        )
        return "image/png", self._render_dicom_thumbnail(content)

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
                trust_env=False,
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
        if payload.offset:
            params["offset"] = str(payload.offset)
        if payload.study_instance_uid:
            params["StudyInstanceUID"] = payload.study_instance_uid.strip()
        if payload.patient_id:
            params["PatientID"] = payload.patient_id.strip()
        if payload.patient_name:
            params["PatientName"] = payload.patient_name.strip()
        if payload.accession_number:
            params["AccessionNumber"] = payload.accession_number.strip()
        if payload.study_description:
            params["StudyDescription"] = payload.study_description.strip()
        if payload.modality:
            params["ModalitiesInStudy"] = payload.modality.strip()
        date_range = PacsDicomwebService._dicom_date_range(payload.study_date_from, payload.study_date_to)
        if date_range:
            params["StudyDate"] = date_range
        return params

    @staticmethod
    def _series_query_params(payload: PacsQidoSeriesQueryRequest) -> dict[str, str]:
        params: dict[str, str] = {"limit": str(payload.limit)}
        if payload.offset:
            params["offset"] = str(payload.offset)
        if payload.series_instance_uid:
            params["SeriesInstanceUID"] = payload.series_instance_uid.strip()
        if payload.modality:
            params["Modality"] = payload.modality.strip()
        if payload.series_description:
            params["SeriesDescription"] = payload.series_description.strip()
        if payload.body_part_examined:
            params["BodyPartExamined"] = payload.body_part_examined.strip()
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
            studyInstanceUid=PacsDicomwebService._value(record, TAG_STUDY_INSTANCE_UID) or "",
            patientName=PacsDicomwebService._value(record, TAG_PATIENT_NAME),
            patientId=PacsDicomwebService._value(record, TAG_PATIENT_ID),
            studyDate=PacsDicomwebService._value(record, TAG_STUDY_DATE),
            studyTime=PacsDicomwebService._value(record, TAG_STUDY_TIME),
            accessionNumber=PacsDicomwebService._value(record, TAG_ACCESSION_NUMBER),
            studyDescription=PacsDicomwebService._value(record, TAG_STUDY_DESCRIPTION),
            modalitiesInStudy=PacsDicomwebService._values(record, TAG_MODALITIES_IN_STUDY),
            numberOfStudyRelatedSeries=PacsDicomwebService._int_value(record, TAG_STUDY_RELATED_SERIES),
            numberOfStudyRelatedInstances=PacsDicomwebService._int_value(record, TAG_STUDY_RELATED_INSTANCES),
            raw=record,
        )

    @staticmethod
    def _parse_series(record: dict[str, Any], fallback_study_uid: str) -> PacsSeriesItem:
        return PacsSeriesItem(
            studyInstanceUid=PacsDicomwebService._value(record, TAG_STUDY_INSTANCE_UID) or fallback_study_uid,
            seriesInstanceUid=PacsDicomwebService._value(record, TAG_SERIES_INSTANCE_UID) or "",
            seriesNumber=PacsDicomwebService._value(record, TAG_SERIES_NUMBER),
            modality=PacsDicomwebService._value(record, TAG_MODALITY),
            seriesDescription=PacsDicomwebService._value(record, TAG_SERIES_DESCRIPTION),
            bodyPartExamined=PacsDicomwebService._value(record, TAG_BODY_PART_EXAMINED),
            numberOfSeriesRelatedInstances=PacsDicomwebService._int_value(record, TAG_SERIES_RELATED_INSTANCES),
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
    def _unique_values(records: list[dict[str, Any]], tag: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for record in records:
            for value in PacsDicomwebService._values(record, tag):
                if value in seen:
                    continue
                seen.add(value)
                values.append(value)
        return values

    @staticmethod
    def _is_compressed_transfer_syntax(value: str) -> bool:
        try:
            return bool(UID(value).is_compressed)
        except (TypeError, ValueError):
            return False

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

    @staticmethod
    def _render_dicom_thumbnail(content: bytes) -> bytes:
        try:
            dataset = pydicom.dcmread(io.BytesIO(content), force=True)
            pixels = dataset.pixel_array
        except Exception as exc:
            raise PacsDicomwebError(f"Failed to decode DICOM pixels: {exc}") from exc

        if pixels.ndim == 4:
            pixels = pixels[0]
        if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
            pixels = pixels[0]

        if pixels.ndim == 2:
            image = PacsDicomwebService._render_grayscale_dicom_thumbnail(dataset, np.asarray(pixels, dtype=np.float32))
        elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            image = PacsDicomwebService._render_color_dicom_thumbnail(np.asarray(pixels))
        else:
            raise PacsDicomwebError("DICOM pixel data is not a renderable 2D image.")

        image = ImageOps.contain(image, PACS_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        canvas = Image.new(image.mode, PACS_THUMBNAIL_SIZE, 0)
        canvas.paste(image, ((PACS_THUMBNAIL_SIZE[0] - image.width) // 2, (PACS_THUMBNAIL_SIZE[1] - image.height) // 2))

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    @staticmethod
    def _render_grayscale_dicom_thumbnail(dataset: Any, pixels: np.ndarray) -> Image.Image:
        slope = PacsDicomwebService._get_first_number(getattr(dataset, "RescaleSlope", None))
        intercept = PacsDicomwebService._get_first_number(getattr(dataset, "RescaleIntercept", None))
        pixels = pixels * (slope if slope is not None else 1.0) + (intercept if intercept is not None else 0.0)
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = -pixels

        low, high = PacsDicomwebService._resolve_thumbnail_window(
            pixels,
            PacsDicomwebService._get_first_number(getattr(dataset, "WindowWidth", None)),
            PacsDicomwebService._get_first_number(getattr(dataset, "WindowCenter", None)),
        )
        scale = high - low
        if scale <= 0:
            raise PacsDicomwebError("DICOM thumbnail window has no display range.")

        clipped = np.clip(pixels, low, high)
        normalized = ((clipped - low) * (255.0 / scale)).astype(np.uint8)
        return Image.fromarray(normalized)

    @staticmethod
    def _render_color_dicom_thumbnail(pixels: np.ndarray) -> Image.Image:
        if pixels.dtype != np.uint8:
            finite_values = np.asarray(pixels[np.isfinite(pixels)], dtype=np.float32)
            if finite_values.size == 0:
                raise PacsDicomwebError("DICOM color pixel data has no finite values.")
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
            scale = high - low
            if scale <= 0:
                raise PacsDicomwebError("DICOM color pixel data has no display range.")
            pixels = ((np.clip(pixels, low, high) - low) * (255.0 / scale)).astype(np.uint8)
        return Image.fromarray(pixels[..., :3])

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
    def _get_first_number(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, MultiValue):
            if not value:
                return None
            value = value[0]
        try:
            parsed_value = float(value)
        except (TypeError, ValueError):
            return None
        return parsed_value if np.isfinite(parsed_value) else None


pacs_dicomweb_service = PacsDicomwebService()
