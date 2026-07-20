from pathlib import Path
import threading
import time

from fastapi import HTTPException
import pytest

from app.schemas.dicom import DicomDeidentifyRequest, DicomTagModifyRequest
from app.services.dicom_deidentify_job_service import DicomDeidentifyJobService
from app.services.dicom_tag_job_service import DicomTagModifyJobService
from app.services.dicom_tag_service import DicomTagModifyArtifact


def _wait_for_final_status(service, job_id: str, workspace_id: str):
    deadline = time.monotonic() + 3.0
    while True:
        status = service.get_status(job_id, workspace_id=workspace_id)
        if status.status in {"succeeded", "failed"}:
            return status
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_deidentify_job_failure_is_workspace_isolated_and_never_exposes_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def fail_deidentification(payload, progress_callback=None, workspace_id=None):
        assert workspace_id == "job-a"
        if progress_callback is not None:
            progress_callback(1, 2)
        started.set()
        assert release.wait(timeout=2.0)
        raise HTTPException(status_code=400, detail="synthetic de-identification failure")

    monkeypatch.setattr(
        "app.services.dicom_deidentify_job_service.dicom_deidentify_service.deidentify_series",
        fail_deidentification,
    )
    service = DicomDeidentifyJobService(temp_root=tmp_path)

    try:
        created = service.create_job(
            DicomDeidentifyRequest(
                seriesId="series-does-not-need-to-exist",
                fieldKeys=["patientIdentity"],
                replacementPrefix="safe",
            ),
            workspace_id="job-a",
        )
        assert started.wait(timeout=2.0)

        running = service.get_status(created.job_id, workspace_id="job-a")
        assert running.status == "running"
        assert running.processed_count == 1
        assert running.total_count == 2
        assert running.progress_percent == 50
        assert running.artifact_url is None

        with pytest.raises(HTTPException) as wrong_workspace:
            service.get_status(created.job_id, workspace_id="job-b")
        assert wrong_workspace.value.status_code == 404

        with pytest.raises(HTTPException) as incomplete_artifact:
            service.get_completed_artifact(created.job_id, workspace_id="job-a")
        assert incomplete_artifact.value.status_code == 409

        release.set()
        failed = _wait_for_final_status(service, created.job_id, "job-a")
        assert failed.status == "failed"
        assert failed.error == "synthetic de-identification failure"
        assert failed.completed_at is not None
        assert failed.artifact_url is None
        assert failed.artifact_kind is None
        assert failed.progress_percent == 50

        with pytest.raises(HTTPException) as failed_artifact:
            service.get_completed_artifact(created.job_id, workspace_id="job-a")
        assert failed_artifact.value.status_code == 409
    finally:
        release.set()
        service.clear()
        service._executor.shutdown(wait=True)


def test_tag_edit_job_reports_gone_when_completed_artifact_was_removed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def create_artifact(payload, progress_callback=None, workspace_id=None):
        assert workspace_id == "job-a"
        if progress_callback is not None:
            progress_callback(1, 1)
        return DicomTagModifyArtifact(
            content=b"synthetic dicom bytes",
            file_name="edited.dcm",
            media_type="application/dicom",
            modified_count=1,
            artifact_kind="dicom",
            series_folder="series-edited",
            tag="(0010,0020)",
            keyword="PatientID",
            vr="LO",
        )

    monkeypatch.setattr(
        "app.services.dicom_tag_job_service.dicom_tag_service.modify_series_tag",
        create_artifact,
    )
    service = DicomTagModifyJobService(temp_root=tmp_path)

    try:
        created = service.create_job(
            DicomTagModifyRequest(
                seriesId="series-does-not-need-to-exist",
                index=0,
                tagPath=["00100020"],
                value="EDITED",
                scope="current",
            ),
            workspace_id="job-a",
        )
        succeeded = _wait_for_final_status(service, created.job_id, "job-a")
        assert succeeded.status == "succeeded"
        assert succeeded.progress_percent == 100
        assert succeeded.processed_count == 1
        assert succeeded.total_count == 1
        assert succeeded.artifact_url is not None

        artifact = service.get_completed_artifact(created.job_id, workspace_id="job-a")
        assert artifact.path.read_bytes() == b"synthetic dicom bytes"
        artifact.path.unlink()

        with pytest.raises(HTTPException) as removed_artifact:
            service.get_completed_artifact(created.job_id, workspace_id="job-a")
        assert removed_artifact.value.status_code == 410
        assert "no longer available" in str(removed_artifact.value.detail)
    finally:
        service.clear()
        service._executor.shutdown(wait=True)
