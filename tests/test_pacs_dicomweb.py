import os
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.api.routes import pacs as pacs_route
from app.main import fastapi_app
from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse, SeriesSummary
from app.schemas.pacs import (
    PacsDicomwebProfile,
    PacsDicomwebTestResponse,
    PacsQidoSeriesQueryRequest,
    PacsQidoStudyQueryRequest,
    PacsWadoSeriesDownloadRequest,
)
from app.services.pacs_dicomweb_service import PacsDicomwebError, PacsDicomwebService
from app.services.pacs_wado_job_service import PacsWadoDownloadJobService


def _profile() -> PacsDicomwebProfile:
    return PacsDicomwebProfile(
        id="orthanc-local",
        name="Orthanc Local",
        baseUrl="http://pacs.local",
        qidoPath="/dicom-web",
        authType="none",
    )


def test_qido_studies_maps_filters_and_dicom_json() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json=[
                {
                    "0020000D": {"vr": "UI", "Value": ["1.2.3"]},
                    "00100010": {"vr": "PN", "Value": [{"Alphabetic": "Patient^Demo"}]},
                    "00100020": {"vr": "LO", "Value": ["P001"]},
                    "00080020": {"vr": "DA", "Value": ["20260522"]},
                    "00080061": {"vr": "CS", "Value": ["CT", "MR"]},
                    "00201206": {"vr": "IS", "Value": [2]},
                    "00201208": {"vr": "IS", "Value": [42]},
                }
            ],
        )

    service = PacsDicomwebService(transport=httpx.MockTransport(handler))
    response = service.query_studies(
        PacsQidoStudyQueryRequest(
            profile=_profile(),
            patientName="Patient*",
            modality="CT",
            studyDateFrom="2026-05-01",
            studyDateTo="2026-05-22",
            limit=25,
        )
    )

    assert seen_request is not None
    assert str(seen_request.url).startswith("http://pacs.local/dicom-web/studies")
    assert seen_request.url.params["PatientName"] == "Patient*"
    assert seen_request.url.params["ModalitiesInStudy"] == "CT"
    assert seen_request.url.params["StudyDate"] == "20260501-20260522"
    assert response.items[0].study_instance_uid == "1.2.3"
    assert response.items[0].patient_name == "Patient^Demo"
    assert response.items[0].modalities_in_study == ["CT", "MR"]
    assert response.items[0].number_of_study_related_instances == 42


def test_qido_series_maps_dicom_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/dicom-web/studies/1.2.3/series"
        return httpx.Response(
            200,
            json=[
                {
                    "0020000E": {"vr": "UI", "Value": ["4.5.6"]},
                    "00200011": {"vr": "IS", "Value": [7]},
                    "00080060": {"vr": "CS", "Value": ["CT"]},
                    "0008103E": {"vr": "LO", "Value": ["Portal Venous"]},
                    "00201209": {"vr": "IS", "Value": [99]},
                }
            ],
        )

    service = PacsDicomwebService(transport=httpx.MockTransport(handler))
    response = service.query_series(PacsQidoSeriesQueryRequest(profile=_profile(), studyInstanceUid="1.2.3"))

    assert response.items[0].study_instance_uid == "1.2.3"
    assert response.items[0].series_instance_uid == "4.5.6"
    assert response.items[0].series_number == "7"
    assert response.items[0].number_of_series_related_instances == 99


def test_qido_instances_and_wado_download_use_dicomweb_paths() -> None:
    seen_paths: list[str] = []
    multipart_body = (
        b"--dicom-boundary\r\n"
        b"Content-Type: application/dicom\r\n\r\n"
        b"DICM-BINARY\r\n"
        b"--dicom-boundary--\r\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path.endswith("/instances/1.2.840.1"):
            return httpx.Response(
                200,
                headers={"content-type": 'multipart/related; type="application/dicom"; boundary="dicom-boundary"'},
                content=multipart_body,
            )
        return httpx.Response(
            200,
            json=[
                {
                    "00080018": {"vr": "UI", "Value": ["1.2.840.1"]},
                }
            ],
        )

    service = PacsDicomwebService(transport=httpx.MockTransport(handler))

    instance_uids = service.query_instance_uids(_profile(), study_instance_uid="1.2.3", series_instance_uid="4.5.6")
    content = service.download_instance(
        _profile(),
        study_instance_uid="1.2.3",
        series_instance_uid="4.5.6",
        sop_instance_uid=instance_uids[0],
    )

    assert seen_paths == [
        "/dicom-web/studies/1.2.3/series/4.5.6/instances",
        "/dicom-web/studies/1.2.3/series/4.5.6/instances/1.2.840.1",
    ]
    assert content == b"DICM-BINARY"


def test_wado_download_job_registers_downloaded_series(monkeypatch, tmp_path: Path) -> None:
    class FakeDicomwebService:
        def query_instance_uids(self, profile: PacsDicomwebProfile, *, study_instance_uid: str, series_instance_uid: str) -> list[str]:
            assert profile.id == "orthanc-local"
            assert study_instance_uid == "1.2.3"
            assert series_instance_uid == "4.5.6"
            return ["1.2.3.1", "1.2.3.2"]

        def download_instance(
            self,
            profile: PacsDicomwebProfile,
            *,
            study_instance_uid: str,
            series_instance_uid: str,
            sop_instance_uid: str,
        ) -> bytes:
            return f"DICOM {sop_instance_uid}".encode()

    def fake_load_folder(payload: LoadFolderRequest) -> LoadFolderResponse:
        folder = Path(payload.folder_path)
        assert (folder / "IM_0001.dcm").read_bytes() == b"DICOM 1.2.3.1"
        assert (folder / "IM_0002.dcm").read_bytes() == b"DICOM 1.2.3.2"
        return LoadFolderResponse(
            seriesId="series-1",
            seriesList=[
                SeriesSummary(
                    seriesId="series-1",
                    seriesInstanceUid="4.5.6",
                    instanceCount=2,
                    folderPath=str(folder),
                )
            ],
        )

    monkeypatch.setattr("app.services.pacs_wado_job_service.series_registry.load_folder", fake_load_folder)
    service = PacsWadoDownloadJobService(dicomweb_service=FakeDicomwebService(), cache_root=tmp_path)  # type: ignore[arg-type]
    initial = service.create_job(
        PacsWadoSeriesDownloadRequest(
            profile=_profile(),
            studyInstanceUid="1.2.3",
            seriesInstanceUid="4.5.6",
        )
    )

    status = initial
    for _ in range(50):
        status = service.get_status(initial.job_id)
        if status.status in {"succeeded", "failed"}:
            break
        time.sleep(0.02)

    assert status.status == "succeeded"
    assert status.progress_percent == 100
    assert status.processed_count == 2
    assert status.total_count == 2
    assert status.series_id == "series-1"
    assert status.series_list[0].series_instance_uid == "4.5.6"


def test_wado_download_job_deletes_failed_partial_cache(tmp_path: Path) -> None:
    class FakeDicomwebService:
        def query_instance_uids(self, profile: PacsDicomwebProfile, *, study_instance_uid: str, series_instance_uid: str) -> list[str]:
            return ["1.2.3.1", "1.2.3.2"]

        def download_instance(
            self,
            profile: PacsDicomwebProfile,
            *,
            study_instance_uid: str,
            series_instance_uid: str,
            sop_instance_uid: str,
        ) -> bytes:
            if sop_instance_uid == "1.2.3.2":
                raise PacsDicomwebError("download failed")
            return b"DICOM 1.2.3.1"

    service = PacsWadoDownloadJobService(dicomweb_service=FakeDicomwebService(), cache_root=tmp_path)  # type: ignore[arg-type]
    initial = service.create_job(
        PacsWadoSeriesDownloadRequest(
            profile=_profile(),
            studyInstanceUid="1.2.3",
            seriesInstanceUid="4.5.6",
        )
    )

    status = initial
    for _ in range(50):
        status = service.get_status(initial.job_id)
        if status.status in {"succeeded", "failed"}:
            break
        time.sleep(0.02)

    assert status.status == "failed"
    assert status.error == "download failed"
    assert not (tmp_path / initial.job_id).exists()


def test_wado_download_job_removes_stale_cache_dirs(tmp_path: Path) -> None:
    stale_cache_dir = tmp_path / "stale-job"
    stale_cache_dir.mkdir()
    (stale_cache_dir / "IM_0001.dcm").write_bytes(b"stale")
    fresh_cache_dir = tmp_path / "fresh-job"
    fresh_cache_dir.mkdir()
    (fresh_cache_dir / "IM_0001.dcm").write_bytes(b"fresh")

    stale_timestamp = time.time() - 120
    os.utime(stale_cache_dir / "IM_0001.dcm", (stale_timestamp, stale_timestamp))
    os.utime(stale_cache_dir, (stale_timestamp, stale_timestamp))

    service = PacsWadoDownloadJobService(
        dicomweb_service=object(),  # type: ignore[arg-type]
        cache_root=tmp_path,
        cache_max_age_seconds=60,
    )

    service.cleanup_cache()

    assert not stale_cache_dir.exists()
    assert fresh_cache_dir.exists()


def test_pacs_test_connection_endpoint_uses_service(monkeypatch) -> None:
    def fake_test_connection(profile: PacsDicomwebProfile) -> PacsDicomwebTestResponse:
        assert profile.base_url == "http://pacs.local"
        return PacsDicomwebTestResponse(ok=True, statusCode=200, message="ok")

    monkeypatch.setattr(pacs_route.pacs_dicomweb_service, "test_connection", fake_test_connection)

    client = TestClient(fastapi_app)
    response = client.post(
        "/api/v1/pacs/dicomweb/test",
        json={"profile": {"id": "p1", "name": "PACS", "baseUrl": "http://pacs.local"}},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "statusCode": 200, "message": "ok"}
