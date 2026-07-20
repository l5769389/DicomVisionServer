from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time

import httpx
import numpy as np
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.api.routes import pacs as pacs_route
from app.main import fastapi_app
from app.services.dicom_cache import dicom_cache
from app.services.pacs_dicomweb_service import PacsDicomwebService
from app.services.pacs_wado_job_service import PacsWadoDownloadJobService
from app.services.series_registry import series_registry
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service


def _dicom_bytes(
    *,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    instance_number: int,
    z_position_mm: float,
    stored_value: int,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientName = "PACS^Roundtrip"
    dataset.PatientID = "PACS-ROUNDTRIP"
    dataset.StudyDate = "20260719"
    dataset.StudyDescription = "Local mock PACS study"
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Downloaded physical CT"
    dataset.SeriesNumber = 7
    dataset.InstanceNumber = instance_number
    dataset.Rows = 4
    dataset.Columns = 6
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelSpacing = [1.25, 0.75]
    dataset.SliceThickness = 2.5
    dataset.SpacingBetweenSlices = 2.5
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [0.0, 0.0, z_position_mm]
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1000.0
    dataset.WindowWidth = 400.0
    dataset.WindowCenter = 100.0
    dataset.PixelData = np.full((dataset.Rows, dataset.Columns), stored_value, dtype=np.uint16).tobytes()
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _dicom_json(tag: str, vr: str, value: object) -> dict[str, object]:
    return {tag: {"vr": vr, "Value": value if isinstance(value, list) else [value]}}


def _clear_state() -> None:
    view_registry._view_by_id.clear()
    for group in view_group_registry.list_all():
        view_group_registry.delete(group.group_id)
    viewer_service._series_volume_cache.clear()
    viewer_service._series_patient_transform_cache.clear()
    viewer_service._series_volume_geometry_cache.clear()
    viewer_service._mpr_plane_cache.clear()
    series_registry.clear()
    dicom_cache.clear()


def test_local_dicomweb_query_download_register_and_render_roundtrip(monkeypatch, tmp_path: Path) -> None:
    _clear_state()
    study_uid = generate_uid()
    series_uid = generate_uid()
    sop_uids = [generate_uid() for _ in range(3)]
    instance_payloads = {
        sop_uids[0]: _dicom_bytes(
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=sop_uids[0],
            instance_number=30,
            z_position_mm=5.0,
            stored_value=1200,
        ),
        sop_uids[1]: _dicom_bytes(
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=sop_uids[1],
            instance_number=10,
            z_position_mm=0.0,
            stored_value=1000,
        ),
        sop_uids[2]: _dicom_bytes(
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=sop_uids[2],
            instance_number=20,
            z_position_mm=2.5,
            stored_value=1100,
        ),
    }
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        seen_paths.append(path)
        instances_path = f"/dicom-web/studies/{study_uid}/series/{series_uid}/instances"
        if path == "/dicom-web/studies":
            record = {
                **_dicom_json("0020000D", "UI", study_uid),
                **_dicom_json("00100010", "PN", {"Alphabetic": "PACS^Roundtrip"}),
                **_dicom_json("00100020", "LO", "PACS-ROUNDTRIP"),
                **_dicom_json("00080020", "DA", "20260719"),
                **_dicom_json("00080061", "CS", ["CT"]),
                **_dicom_json("00201206", "IS", 1),
                **_dicom_json("00201208", "IS", 3),
            }
            return httpx.Response(200, json=[record])
        if path == f"/dicom-web/studies/{study_uid}/series":
            record = {
                **_dicom_json("0020000E", "UI", series_uid),
                **_dicom_json("00200011", "IS", 7),
                **_dicom_json("00080060", "CS", "CT"),
                **_dicom_json("0008103E", "LO", "Downloaded physical CT"),
                **_dicom_json("00201209", "IS", 3),
            }
            return httpx.Response(200, json=[record])
        if path == instances_path:
            records = [
                {
                    **_dicom_json("00080018", "UI", sop_uid),
                    **_dicom_json("00020010", "UI", str(ExplicitVRLittleEndian)),
                    **_dicom_json("00280004", "CS", "MONOCHROME2"),
                    **_dicom_json("00280010", "US", 4),
                    **_dicom_json("00280011", "US", 6),
                }
                for sop_uid in sop_uids
            ]
            return httpx.Response(200, json=records)
        for sop_uid, content in instance_payloads.items():
            if path == f"{instances_path}/{sop_uid}":
                boundary = b"local-pacs-boundary"
                body = (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: application/dicom\r\n\r\n"
                    + content
                    + b"\r\n--" + boundary + b"--\r\n"
                )
                return httpx.Response(
                    200,
                    headers={
                        "content-type": 'multipart/related; type="application/dicom"; boundary="local-pacs-boundary"'
                    },
                    content=body,
                )
        return httpx.Response(404, text=f"Unhandled mock PACS path: {path}")

    dicomweb_service = PacsDicomwebService(transport=httpx.MockTransport(handler))
    download_service = PacsWadoDownloadJobService(
        dicomweb_service=dicomweb_service,
        cache_root=tmp_path / "pacs-cache",
    )
    monkeypatch.setattr(pacs_route, "pacs_dicomweb_service", dicomweb_service)
    monkeypatch.setattr(pacs_route, "pacs_wado_download_job_service", download_service)
    client = TestClient(fastapi_app)
    profile = {
        "id": "local-mock",
        "name": "Local mock PACS",
        "baseUrl": "http://local-pacs.test",
        "qidoPath": "/dicom-web",
        "wadoPath": "/dicom-web",
        "authType": "none",
    }

    try:
        connection = client.post("/api/v1/pacs/dicomweb/test", json={"profile": profile})
        assert connection.status_code == 200
        assert connection.json()["ok"] is True

        studies = client.post(
            "/api/v1/pacs/dicomweb/studies",
            json={"profile": profile, "patientId": "PACS-ROUNDTRIP", "limit": 10},
        )
        assert studies.status_code == 200
        assert studies.json()["items"][0]["studyInstanceUid"] == study_uid
        assert studies.json()["items"][0]["numberOfStudyRelatedInstances"] == 3

        series = client.post(
            "/api/v1/pacs/dicomweb/series",
            json={"profile": profile, "studyInstanceUid": study_uid, "limit": 10},
        )
        assert series.status_code == 200
        assert series.json()["items"][0]["seriesInstanceUid"] == series_uid
        assert series.json()["items"][0]["numberOfSeriesRelatedInstances"] == 3

        preview = client.post(
            "/api/v1/pacs/dicomweb/seriesPreview",
            json={
                "profile": profile,
                "studyInstanceUid": study_uid,
                "seriesInstanceUid": series_uid,
                "thumbnail": False,
            },
        )
        assert preview.status_code == 200
        assert preview.json()["instanceCount"] == 3
        assert preview.json()["rows"] == 4
        assert preview.json()["columns"] == 6

        created_job = client.post(
            "/api/v1/pacs/dicomweb/downloadSeries/jobs",
            json={"profile": profile, "studyInstanceUid": study_uid, "seriesInstanceUid": series_uid},
        )
        assert created_job.status_code == 200
        job_id = created_job.json()["jobId"]

        job = created_job.json()
        for _ in range(100):
            status_response = client.get(f"/api/v1/pacs/dicomweb/downloadSeries/jobs/{job_id}")
            assert status_response.status_code == 200
            job = status_response.json()
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.01)

        assert job["status"] == "succeeded", job
        assert job["processedCount"] == 3
        assert job["totalCount"] == 3
        assert job["progressPercent"] == 100
        assert job["seriesList"][0]["instanceCount"] == 3
        registered_series_id = job["seriesId"]
        registered = series_registry.get(registered_series_id)
        volume = viewer_service._build_series_volume(registered)
        assert volume.shape == (3, 4, 6)
        np.testing.assert_array_equal(volume[:, 0, 0], np.array([0, 100, 200], dtype=np.int16))

        created_view = client.post(
            "/api/v1/view/create",
            json={"seriesId": registered_series_id, "viewType": "MPR"},
        )
        assert created_view.status_code == 200
        view_id = created_view.json()["viewId"]
        view = view_registry.get(view_id)
        view.width = 240
        view.height = 180
        rendered = viewer_service.render_view_by_id(view_id, image_format="png")
        assert rendered.image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert rendered.meta.mpr_plane is not None
        assert rendered.meta.mpr_plane.output_shape == (4, 6)
        assert (
            rendered.meta.mpr_plane.pixel_spacing_row_mm,
            rendered.meta.mpr_plane.pixel_spacing_col_mm,
            rendered.meta.mpr_plane.pixel_spacing_normal_mm,
        ) == (1.25, 0.75, 2.5)

        assert seen_paths.count(f"/dicom-web/studies/{study_uid}/series/{series_uid}/instances") >= 2
        for sop_uid in sop_uids:
            assert f"/dicom-web/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}" in seen_paths
    finally:
        download_service.shutdown()
        _clear_state()
