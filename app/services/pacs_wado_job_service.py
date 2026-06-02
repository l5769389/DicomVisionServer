from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.workspace import DEFAULT_WORKSPACE_ID, WORKSPACE_QUERY_PARAM, normalize_workspace_id
from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse
from app.schemas.pacs import PacsWadoSeriesDownloadJobStatusResponse, PacsWadoSeriesDownloadRequest
from app.services.dicom_job_utils import format_datetime, normalize_progress_counts, resolve_progress_percent, utc_now
from app.services.pacs_dicomweb_service import PacsDicomwebError, PacsDicomwebService, pacs_dicomweb_service
from app.services.series_registry import series_registry


PacsWadoDownloadJobState = Literal["pending", "running", "succeeded", "failed", "cancelled"]
logger = get_logger(__name__)


@dataclass
class PacsWadoDownloadJob:
    job_id: str
    workspace_id: str
    status: PacsWadoDownloadJobState
    created_at: datetime
    processed_count: int = 0
    total_count: int = 0
    completed_at: datetime | None = None
    error: str | None = None
    folder_path: str | None = None
    load_response: LoadFolderResponse | None = None
    cancel_requested: bool = False


class PacsWadoDownloadJobService:
    _MAX_RETAINED_JOBS = 64
    _FINAL_STATUSES = {"succeeded", "failed", "cancelled"}

    def __init__(
        self,
        *,
        dicomweb_service: PacsDicomwebService | None = None,
        cache_root: Path | None = None,
        cache_max_age_seconds: int | None = None,
        cleanup_interval_seconds: int | None = None,
        start_cleanup_worker: bool = False,
    ) -> None:
        settings = get_settings()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pacs-wado-download")
        self._jobs: dict[str, PacsWadoDownloadJob] = {}
        self._lock = threading.Lock()
        self._dicomweb_service = dicomweb_service or pacs_dicomweb_service
        configured_cache_root = Path(settings.pacs_wado_cache_root).expanduser() if settings.pacs_wado_cache_root else None
        self._cache_root = cache_root or configured_cache_root or Path(tempfile.gettempdir()) / "dicomvision-pacs-cache"
        self._cache_max_age_seconds = max(
            1,
            cache_max_age_seconds
            if cache_max_age_seconds is not None
            else settings.pacs_wado_cache_max_age_seconds,
        )
        self._cleanup_interval_seconds = max(
            1,
            cleanup_interval_seconds
            if cleanup_interval_seconds is not None
            else settings.pacs_wado_cache_cleanup_interval_seconds,
        )
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self.cleanup_cache()
        if start_cleanup_worker:
            self._start_cleanup_worker()

    def create_job(
        self,
        payload: PacsWadoSeriesDownloadRequest,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> PacsWadoSeriesDownloadJobStatusResponse:
        self.cleanup_cache()
        job_id = uuid4().hex
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        job = PacsWadoDownloadJob(
            job_id=job_id,
            workspace_id=normalized_workspace_id,
            status="pending",
            created_at=utc_now(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()

        self._executor.submit(self._run_job, job_id, payload)
        return self.get_status(job_id, workspace_id=normalized_workspace_id)

    def get_status(
        self,
        job_id: str,
        workspace_id: str | None = None,
    ) -> PacsWadoSeriesDownloadJobStatusResponse:
        with self._lock:
            job = self._get_job_locked(job_id, workspace_id=workspace_id)
            if job is None:
                raise HTTPException(status_code=404, detail="PACS download job was not found")
            return self._to_response(job)

    def cancel_job(
        self,
        job_id: str,
        workspace_id: str | None = None,
    ) -> PacsWadoSeriesDownloadJobStatusResponse:
        should_delete_cache = False
        with self._lock:
            job = self._get_job_locked(job_id, workspace_id=workspace_id)
            if job is None:
                raise HTTPException(status_code=404, detail="PACS download job was not found")
            if job.status not in self._FINAL_STATUSES:
                job.cancel_requested = True
                job.status = "cancelled"
                job.error = "PACS download was cancelled."
                job.completed_at = utc_now()
                should_delete_cache = True
            response = self._to_response(job)
            self._prune_locked()

        if should_delete_cache:
            self._delete_job_cache_dir(job_id)
        return response

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def cleanup_cache(self) -> None:
        self._prune_expired_jobs()
        self._delete_stale_cache_dirs()

    def shutdown(self) -> None:
        self._cleanup_stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_job(self, job_id: str, payload: PacsWadoSeriesDownloadRequest) -> None:
        self._mark_running(job_id)
        download_dir = self._job_cache_dir(job_id)
        try:
            if self._should_stop_cancelled_job(job_id):
                self._delete_job_cache_dir(job_id)
                return

            instance_uids = self._dicomweb_service.query_instance_uids(
                payload.profile,
                study_instance_uid=payload.study_instance_uid,
                series_instance_uid=payload.series_instance_uid,
            )
            if self._should_stop_cancelled_job(job_id):
                self._delete_job_cache_dir(job_id)
                return
            if not instance_uids:
                raise PacsDicomwebError("DICOMweb QIDO returned no instances for this series.")

            self._update_progress(job_id, 0, len(instance_uids))
            download_dir.mkdir(parents=True, exist_ok=True)

            for index, sop_instance_uid in enumerate(instance_uids, start=1):
                if self._should_stop_cancelled_job(job_id):
                    self._delete_job_cache_dir(job_id)
                    return
                content = self._dicomweb_service.download_instance(
                    payload.profile,
                    study_instance_uid=payload.study_instance_uid,
                    series_instance_uid=payload.series_instance_uid,
                    sop_instance_uid=sop_instance_uid,
                )
                if self._should_stop_cancelled_job(job_id):
                    self._delete_job_cache_dir(job_id)
                    return
                (download_dir / f"IM_{index:04d}.dcm").write_bytes(content)
                self._update_progress(job_id, index, len(instance_uids))

            if self._should_stop_cancelled_job(job_id):
                self._delete_job_cache_dir(job_id)
                return
            job_workspace_id = self._get_job_workspace_id(job_id)
            load_payload = LoadFolderRequest(folderPath=str(download_dir))
            load_response = (
                series_registry.load_folder(load_payload)
                if job_workspace_id == DEFAULT_WORKSPACE_ID
                else series_registry.load_folder(load_payload, workspace_id=job_workspace_id)
            )
            if not load_response.series_list:
                raise HTTPException(status_code=400, detail="Downloaded PACS series did not contain readable DICOM images")
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            self._mark_failed(job_id, detail)
            self._delete_job_cache_dir(job_id)
            return
        except PacsDicomwebError as exc:
            self._mark_failed(job_id, str(exc))
            self._delete_job_cache_dir(job_id)
            return
        except Exception as exc:
            self._mark_failed(job_id, str(exc) or "PACS series download failed")
            self._delete_job_cache_dir(job_id)
            return

        self._mark_succeeded(job_id, str(download_dir), load_response)

    def _mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status == "pending":
                job.status = "running"

    def _update_progress(self, job_id: str, processed_count: int, total_count: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in self._FINAL_STATUSES:
                return
            job.processed_count, job.total_count = normalize_progress_counts(processed_count, total_count)

    def _mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status != "cancelled":
                job.status = "failed"
                job.error = error
                job.completed_at = utc_now()
            self._prune_locked()

    def _mark_succeeded(self, job_id: str, folder_path: str, load_response: LoadFolderResponse) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status != "cancelled":
                job.status = "succeeded"
                job.folder_path = folder_path
                job.load_response = load_response
                job.processed_count = max(job.processed_count, job.total_count)
                job.completed_at = utc_now()
            self._prune_locked()

    def _mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status not in {"succeeded", "failed"}:
                job.cancel_requested = True
                job.status = "cancelled"
                job.error = job.error or "PACS download was cancelled."
                job.completed_at = job.completed_at or utc_now()
            self._prune_locked()

    def _should_stop_cancelled_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            should_stop = job is None or job.cancel_requested or job.status == "cancelled"
        if should_stop:
            self._mark_cancelled(job_id)
        return should_stop

    def _to_response(self, job: PacsWadoDownloadJob) -> PacsWadoSeriesDownloadJobStatusResponse:
        load_response = job.load_response
        return PacsWadoSeriesDownloadJobStatusResponse(
            jobId=job.job_id,
            status=job.status,
            statusUrl=self._status_url(job.job_id, job.workspace_id),
            error=job.error,
            folderPath=job.folder_path,
            processedCount=job.processed_count,
            progressPercent=resolve_progress_percent(job.status, job.processed_count, job.total_count),
            seriesId=load_response.series_id if load_response is not None else None,
            seriesList=load_response.series_list if load_response is not None else [],
            totalCount=job.total_count,
            createdAt=format_datetime(job.created_at),
            completedAt=format_datetime(job.completed_at),
        )

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._MAX_RETAINED_JOBS:
            return

        final_jobs = sorted(
            (job for job in self._jobs.values() if job.status in self._FINAL_STATUSES),
            key=lambda item: item.completed_at or item.created_at,
        )
        for job in final_jobs[: max(0, len(self._jobs) - self._MAX_RETAINED_JOBS)]:
            self._jobs.pop(job.job_id, None)

    def _prune_expired_jobs(self) -> None:
        cutoff_timestamp = utc_now().timestamp() - self._cache_max_age_seconds
        with self._lock:
            expired_job_ids = [
                job.job_id
                for job in self._jobs.values()
                if job.status in self._FINAL_STATUSES
                and (job.completed_at or job.created_at).timestamp() < cutoff_timestamp
            ]
            for job_id in expired_job_ids:
                self._jobs.pop(job_id, None)

    def _delete_stale_cache_dirs(self) -> None:
        cutoff_timestamp = utc_now().timestamp() - self._cache_max_age_seconds
        protected_job_ids = self._protected_job_ids()
        try:
            candidates = list(self._cache_root.iterdir())
        except OSError as exc:
            logger.debug("failed to list PACS cache root %s: %s", self._cache_root, exc)
            return

        for path in candidates:
            if path.name in protected_job_ids:
                continue
            try:
                if not path.is_dir() or path.stat().st_mtime >= cutoff_timestamp:
                    continue
            except OSError:
                continue
            self._delete_cache_dir(path)

    def _delete_job_cache_dir(self, job_id: str) -> None:
        self._delete_cache_dir(self._job_cache_dir(job_id))

    def _delete_cache_dir(self, path: Path) -> None:
        resolved_path = self._resolve_cache_child(path)
        if resolved_path is None or not resolved_path.exists():
            return
        try:
            shutil.rmtree(resolved_path)
        except OSError as exc:
            logger.debug("failed to delete PACS cache directory %s: %s", resolved_path, exc)

    def _protected_job_ids(self) -> set[str]:
        with self._lock:
            return set(self._jobs)

    def _resolve_cache_child(self, path: Path) -> Path | None:
        try:
            resolved_root = self._cache_root.resolve()
            resolved_path = path.resolve()
            resolved_path.relative_to(resolved_root)
        except (OSError, ValueError):
            return None
        return resolved_path

    def _job_cache_dir(self, job_id: str) -> Path:
        return self._cache_root / job_id

    def _start_cleanup_worker(self) -> None:
        if self._cleanup_thread is not None:
            return
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="pacs-wado-cache-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        while not self._cleanup_stop_event.wait(self._cleanup_interval_seconds):
            self.cleanup_cache()

    def _get_job_locked(self, job_id: str, workspace_id: str | None = None) -> PacsWadoDownloadJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if workspace_id is not None and job.workspace_id != normalize_workspace_id(workspace_id):
            return None
        return job

    def _get_job_workspace_id(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.workspace_id if job is not None else DEFAULT_WORKSPACE_ID

    @staticmethod
    def _workspace_query(workspace_id: str) -> str:
        if normalize_workspace_id(workspace_id) == DEFAULT_WORKSPACE_ID:
            return ""
        return f"?{WORKSPACE_QUERY_PARAM}={quote(workspace_id, safe='')}"

    @classmethod
    def _status_url(cls, job_id: str, workspace_id: str) -> str:
        return f"/api/v1/pacs/dicomweb/downloadSeries/jobs/{job_id}{cls._workspace_query(workspace_id)}"


pacs_wado_download_job_service = PacsWadoDownloadJobService(start_cleanup_worker=True)
