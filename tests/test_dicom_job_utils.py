import os
from pathlib import Path

from app.services.dicom_job_utils import (
    delete_stale_temp_files,
    normalize_progress_counts,
    resolve_progress_percent,
)


def test_normalize_progress_counts_clamps_processed_to_total() -> None:
    assert normalize_progress_counts(12, 10) == (10, 10)
    assert normalize_progress_counts(-1, 10) == (0, 10)
    assert normalize_progress_counts(3, 0) == (3, 0)


def test_resolve_progress_percent_handles_final_and_running_states() -> None:
    assert resolve_progress_percent("succeeded", 0, 0) == 100
    assert resolve_progress_percent("running", 3, 10) == 30
    assert resolve_progress_percent("running", 12, 10) == 100
    assert resolve_progress_percent("running", 0, 0) == 0


def test_delete_stale_temp_files_keeps_fresh_and_unrelated_files(tmp_path: Path) -> None:
    stale_zip = tmp_path / f"{'a' * 32}.zip"
    stale_dicom = tmp_path / f"{'b' * 32}.dcm"
    fresh_zip = tmp_path / f"{'c' * 32}.zip"
    unrelated_zip = tmp_path / "notes.zip"
    stale_txt = tmp_path / f"{'d' * 32}.txt"
    for path in [stale_zip, stale_dicom, fresh_zip, unrelated_zip, stale_txt]:
        path.write_bytes(b"x")
    os.utime(stale_zip, (1, 1))
    os.utime(stale_dicom, (1, 1))
    os.utime(stale_txt, (1, 1))

    delete_stale_temp_files(tmp_path, max_age_seconds=60, allowed_suffixes={".zip", ".dcm"})

    assert not stale_zip.exists()
    assert not stale_dicom.exists()
    assert fresh_zip.exists()
    assert unrelated_zip.exists()
    assert stale_txt.exists()
