from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamps for background job bookkeeping."""
    return datetime.now(UTC)


def format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def normalize_progress_counts(processed_count: int, total_count: int) -> tuple[int, int]:
    total = max(0, total_count)
    processed = max(0, min(processed_count, total or processed_count))
    return processed, total


def resolve_progress_percent(status: str, processed_count: int, total_count: int) -> int:
    if status == "succeeded":
        return 100
    if total_count <= 0:
        return 0
    progress = max(0, round((processed_count / total_count) * 100))
    return min(100, progress)


def delete_stale_temp_files(
    temp_root: Path,
    *,
    max_age_seconds: int,
    allowed_suffixes: set[str],
) -> None:
    cutoff_timestamp = utc_now().timestamp() - max_age_seconds
    normalized_suffixes = {suffix.lower() for suffix in allowed_suffixes}
    for path in temp_root.iterdir():
        try:
            if not _is_stale_job_artifact(path, cutoff_timestamp, normalized_suffixes):
                continue
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_stale_job_artifact(path: Path, cutoff_timestamp: float, allowed_suffixes: set[str]) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and len(path.stem) == 32
        and path.stat().st_mtime < cutoff_timestamp
    )
