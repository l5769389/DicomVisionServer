from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import threading
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException

from app.schemas.dicom import DicomTagModifyJobStatusResponse, DicomTagModifyRequest
from app.services.dicom_tag_service import DicomTagModifyArtifact, dicom_tag_service


DicomTagModifyJobState = Literal["pending", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class DicomTagModifyJobArtifact:
    path: Path
    file_name: str
    media_type: str
    modified_count: int
    artifact_kind: Literal["dicom", "zip"]
    series_folder: str
    tag: str
    keyword: str
    vr: str


@dataclass
class DicomTagModifyJob:
    job_id: str
    status: DicomTagModifyJobState
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    artifact: DicomTagModifyJobArtifact | None = None


class DicomTagModifyJobService:
    _MAX_RETAINED_JOBS = 64
    _ARTIFACT_MAX_AGE_SECONDS = 24 * 60 * 60
    _TEMP_ARTIFACT_SUFFIXES = {".dcm", ".zip"}

    def __init__(self, *, temp_root: Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dicom-tag-edit")
        self._jobs: dict[str, DicomTagModifyJob] = {}
        self._lock = threading.Lock()
        self._temp_root = temp_root or Path(tempfile.gettempdir()) / "dicomvision-tag-edit-jobs"
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._delete_stale_temp_files()

    def create_job(self, payload: DicomTagModifyRequest) -> DicomTagModifyJobStatusResponse:
        job_id = uuid4().hex
        job = DicomTagModifyJob(job_id=job_id, status="pending", created_at=self._utc_now())
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()

        self._executor.submit(self._run_job, job_id, payload)
        return self.get_status(job_id)

    def get_status(self, job_id: str) -> DicomTagModifyJobStatusResponse:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="DICOM tag edit job was not found")
            return self._to_response(job)

    def get_completed_artifact(self, job_id: str) -> DicomTagModifyJobArtifact:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="DICOM tag edit job was not found")
            if job.status != "succeeded" or job.artifact is None:
                raise HTTPException(status_code=409, detail="DICOM tag edit job is not complete")
            artifact = job.artifact

        if not artifact.path.is_file():
            raise HTTPException(status_code=410, detail="DICOM tag edit artifact is no longer available")
        return artifact

    def clear(self) -> None:
        with self._lock:
            artifacts = [job.artifact for job in self._jobs.values() if job.artifact is not None]
            self._jobs.clear()
        for artifact in artifacts:
            self._delete_artifact(artifact)

    def _run_job(self, job_id: str, payload: DicomTagModifyRequest) -> None:
        self._mark_running(job_id)
        try:
            artifact = dicom_tag_service.modify_series_tag(payload)
            stored_artifact = self._store_artifact(job_id, artifact)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            self._mark_failed(job_id, detail)
            return
        except Exception as exc:
            self._mark_failed(job_id, str(exc) or "DICOM tag edit job failed")
            return

        self._mark_succeeded(job_id, stored_artifact)

    def _mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "running"

    def _mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "failed"
                job.error = error
                job.completed_at = self._utc_now()
            self._prune_locked()

    def _mark_succeeded(self, job_id: str, artifact: DicomTagModifyJobArtifact) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "succeeded"
                job.artifact = artifact
                job.completed_at = self._utc_now()
            self._prune_locked()

    def _store_artifact(self, job_id: str, artifact: DicomTagModifyArtifact) -> DicomTagModifyJobArtifact:
        suffix = ".zip" if artifact.artifact_kind == "zip" else ".dcm"
        artifact_path = self._temp_root / f"{job_id}{suffix}"
        artifact_path.write_bytes(artifact.content)
        return DicomTagModifyJobArtifact(
            path=artifact_path,
            file_name=artifact.file_name,
            media_type=artifact.media_type,
            modified_count=artifact.modified_count,
            artifact_kind="zip" if artifact.artifact_kind == "zip" else "dicom",
            series_folder=artifact.series_folder,
            tag=artifact.tag,
            keyword=artifact.keyword,
            vr=artifact.vr,
        )

    def _to_response(self, job: DicomTagModifyJob) -> DicomTagModifyJobStatusResponse:
        artifact = job.artifact
        artifact_url = self._artifact_url(job.job_id) if artifact is not None and job.status == "succeeded" else None
        return DicomTagModifyJobStatusResponse(
            jobId=job.job_id,
            status=job.status,
            statusUrl=self._status_url(job.job_id),
            artifactUrl=artifact_url,
            error=job.error,
            artifactKind=artifact.artifact_kind if artifact is not None else None,
            fileName=artifact.file_name if artifact is not None else None,
            mediaType=artifact.media_type if artifact is not None else None,
            modifiedCount=artifact.modified_count if artifact is not None else None,
            seriesFolder=artifact.series_folder if artifact is not None else None,
            createdAt=self._format_datetime(job.created_at),
            completedAt=self._format_datetime(job.completed_at),
        )

    def _prune_locked(self) -> None:
        now = self._utc_now()
        expired_jobs = [
            job
            for job in self._jobs.values()
            if job.status in {"succeeded", "failed"}
            and job.completed_at is not None
            and (now - job.completed_at).total_seconds() > self._ARTIFACT_MAX_AGE_SECONDS
        ]
        for job in expired_jobs:
            self._remove_job_locked(job)

        if len(self._jobs) <= self._MAX_RETAINED_JOBS:
            return

        final_jobs = sorted(
            (job for job in self._jobs.values() if job.status in {"succeeded", "failed"}),
            key=lambda item: item.completed_at or item.created_at,
        )
        for job in final_jobs[: max(0, len(self._jobs) - self._MAX_RETAINED_JOBS)]:
            self._remove_job_locked(job)

    def _remove_job_locked(self, job: DicomTagModifyJob) -> None:
        self._jobs.pop(job.job_id, None)
        if job.artifact is not None:
            self._delete_artifact(job.artifact)

    @staticmethod
    def _status_url(job_id: str) -> str:
        return f"/api/v1/dicom/modifyTag/jobs/{job_id}"

    @staticmethod
    def _artifact_url(job_id: str) -> str:
        return f"/api/v1/dicom/modifyTag/jobs/{job_id}/artifact"

    @staticmethod
    def _format_datetime(value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _delete_artifact(artifact: DicomTagModifyJobArtifact) -> None:
        try:
            artifact.path.unlink(missing_ok=True)
        except OSError:
            pass

    def _delete_stale_temp_files(self) -> None:
        cutoff_timestamp = self._utc_now().timestamp() - self._ARTIFACT_MAX_AGE_SECONDS
        for path in self._temp_root.iterdir():
            try:
                if (
                    not path.is_file()
                    or path.suffix.lower() not in self._TEMP_ARTIFACT_SUFFIXES
                    or len(path.stem) != 32
                    or path.stat().st_mtime >= cutoff_timestamp
                ):
                    continue
                path.unlink(missing_ok=True)
            except OSError:
                pass


dicom_tag_job_service = DicomTagModifyJobService()
