import shutil
import stat
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import HTTPException, UploadFile
import py7zr
import rarfile

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.workspace import DEFAULT_WORKSPACE_ID, normalize_workspace_id
from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse
from app.services.series_registry import SeriesRegistry, series_registry


logger = get_logger(__name__)
UPLOAD_CHUNK_SIZE = 1024 * 1024
SUPPORTED_ARCHIVE_SUFFIXES = frozenset({".zip", ".7z", ".rar"})


class DicomUploadService:
    def __init__(
        self,
        *,
        upload_root: Path | None = None,
        max_age_seconds: int | None = None,
        cleanup_interval_seconds: int | None = None,
        start_cleanup_worker: bool = False,
    ) -> None:
        settings = get_settings()
        self._upload_root = upload_root
        self._max_age_seconds = max(
            60,
            max_age_seconds if max_age_seconds is not None else settings.web_upload_max_age_seconds,
        )
        self._cleanup_interval_seconds = max(
            60,
            cleanup_interval_seconds
            if cleanup_interval_seconds is not None
            else settings.web_upload_cleanup_interval_seconds,
        )
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        if start_cleanup_worker:
            self._start_cleanup_worker()

    def _resolve_upload_root(self) -> Path:
        settings = get_settings()
        root = self._upload_root or (
            Path(settings.web_upload_dicom_root)
            if settings.web_upload_dicom_root
            else Path(tempfile.gettempdir()) / "dicomvision-web-uploads"
        )
        root.mkdir(parents=True, exist_ok=True)
        return root

    def cleanup_uploads(self) -> int:
        upload_root = self._resolve_upload_root()
        cutoff_timestamp = time.time() - self._max_age_seconds
        deleted_count = 0
        try:
            candidates = list(upload_root.iterdir())
        except OSError as exc:
            logger.debug("failed to list upload root %s: %s", upload_root, exc)
            return 0

        for path in candidates:
            if not path.is_dir():
                continue
            try:
                modified_at = path.stat().st_mtime
            except OSError:
                continue
            if modified_at > cutoff_timestamp:
                continue
            if self._delete_upload_dir(path):
                deleted_count += 1

        if deleted_count:
            logger.info("web dicom upload cleanup removed %s stale session dirs", deleted_count)
        return deleted_count

    def stop_cleanup_worker(self) -> None:
        self._cleanup_stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2)

    def _start_cleanup_worker(self) -> None:
        if self._cleanup_thread is not None:
            return
        self.cleanup_uploads()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="dicom-upload-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        while not self._cleanup_stop_event.wait(self._cleanup_interval_seconds):
            self.cleanup_uploads()

    def _resolve_upload_child(self, path: Path) -> Path | None:
        try:
            resolved_root = self._resolve_upload_root().resolve()
            resolved_path = path.resolve()
            resolved_path.relative_to(resolved_root)
        except (OSError, ValueError):
            return None
        if resolved_path == resolved_root:
            return None
        return resolved_path

    def _delete_upload_dir(self, path: Path) -> bool:
        resolved_path = self._resolve_upload_child(path)
        if resolved_path is None:
            return False
        try:
            shutil.rmtree(resolved_path)
            return True
        except OSError as exc:
            logger.debug("failed to delete upload directory %s: %s", resolved_path, exc)
            return False

    @staticmethod
    def _safe_relative_path(relative_path: str | None, fallback_name: str | None, index: int) -> Path:
        raw_value = (relative_path or fallback_name or f"dicom-{index + 1}.dcm").replace("\\", "/").strip()
        if not raw_value:
            raw_value = f"dicom-{index + 1}.dcm"

        parts: list[str] = []
        for part in PurePosixPath(raw_value).parts:
            clean_part = part.strip()
            if clean_part in {"", ".", "..", "/"} or clean_part.endswith(":"):
                continue
            parts.append(clean_part)

        if not parts:
            parts.append(f"dicom-{index + 1}.dcm")

        return Path(*parts)

    @staticmethod
    def _deduplicate_target_path(target_path: Path) -> Path:
        if not target_path.exists():
            return target_path

        stem = target_path.stem or "dicom"
        suffix = target_path.suffix
        for index in range(2, 10000):
            candidate = target_path.with_name(f"{stem}-{index}{suffix}")
            if not candidate.exists():
                return candidate
        raise HTTPException(status_code=400, detail="Too many duplicate upload file names")

    async def _save_upload_file(self, upload_file: UploadFile, target_path: Path, total_bytes: int) -> int:
        settings = get_settings()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        try:
            with target_path.open("wb") as output:
                while True:
                    chunk = await upload_file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    written += len(chunk)
                    if total_bytes > settings.web_upload_max_bytes:
                        raise HTTPException(status_code=413, detail="Uploaded DICOM data is too large")
                    output.write(chunk)
        finally:
            await upload_file.close()

        if written <= 0:
            target_path.unlink(missing_ok=True)
        return total_bytes

    @staticmethod
    def _archive_suffix(path: Path) -> str | None:
        suffix = path.suffix.lower()
        return suffix if suffix in SUPPORTED_ARCHIVE_SUFFIXES else None

    @classmethod
    def _is_upload_candidate(cls, path: Path) -> bool:
        return cls._archive_suffix(path) is not None or SeriesRegistry.is_dicom_candidate_name(path)

    @staticmethod
    def _safe_archive_member_path(member_name: str, archive_label: str) -> Path | None:
        raw_path = member_name.replace("\\", "/").strip()
        if not raw_path:
            return None
        pure_path = PurePosixPath(raw_path)
        if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
            raise HTTPException(status_code=400, detail=f"{archive_label} archive contains an unsafe file path")
        return Path(*pure_path.parts)

    @staticmethod
    def _is_archive_symlink(info: zipfile.ZipInfo) -> bool:
        return stat.S_ISLNK(info.external_attr >> 16)

    @staticmethod
    def _archive_limits() -> tuple[int, int, int]:
        settings = get_settings()
        return (
            max(1, int(settings.web_upload_max_archive_entries)),
            max(1, int(settings.web_upload_max_archive_uncompressed_bytes)),
            max(1, int(settings.web_upload_max_archive_compression_ratio)),
        )

    def _validate_archive_member(
        self,
        *,
        member_name: str,
        archive_label: str,
        compressed_size: int | None,
        uncompressed_size: int | None,
        entry_offset: int,
        byte_offset: int,
        is_encrypted: bool = False,
        is_symlink: bool = False,
    ) -> tuple[Path | None, int, int]:
        max_entries, max_uncompressed_bytes, max_compression_ratio = self._archive_limits()
        entry_count = entry_offset + 1
        if entry_count > max_entries:
            raise HTTPException(status_code=413, detail=f"{archive_label} archive contains too many files")
        if is_encrypted:
            raise HTTPException(status_code=400, detail=f"Encrypted {archive_label} archives are not supported")
        if is_symlink:
            raise HTTPException(status_code=400, detail=f"{archive_label} archive contains unsupported symbolic links")

        member_path = self._safe_archive_member_path(member_name, archive_label)
        if member_path is None or not SeriesRegistry.is_dicom_candidate_name(member_path):
            return None, entry_count, byte_offset

        safe_compressed_size = max(0, int(compressed_size or 0))
        safe_uncompressed_size = max(0, int(uncompressed_size or 0))
        # Solid 7z archives do not reliably expose a compressed size for every
        # member. A missing member size is validated against the archive as a
        # whole below instead of being treated as a compression bomb.
        if (
            safe_uncompressed_size > 0
            and safe_compressed_size > 0
            and safe_uncompressed_size / safe_compressed_size > max_compression_ratio
        ):
            raise HTTPException(status_code=413, detail=f"{archive_label} archive compression ratio is too high")

        total_uncompressed_bytes = byte_offset + safe_uncompressed_size
        if total_uncompressed_bytes > max_uncompressed_bytes:
            raise HTTPException(status_code=413, detail=f"{archive_label} archive expands to too much data")
        return member_path, entry_count, total_uncompressed_bytes

    def _validate_archive_compression_ratio(
        self,
        archive_path: Path,
        archive_label: str,
        uncompressed_bytes: int,
    ) -> None:
        """Guard solid archives when members do not publish compressed sizes."""

        if uncompressed_bytes <= 0:
            return
        archive_bytes = max(0, archive_path.stat().st_size)
        _, _, max_compression_ratio = self._archive_limits()
        if archive_bytes <= 0 or uncompressed_bytes / archive_bytes > max_compression_ratio:
            raise HTTPException(status_code=413, detail=f"{archive_label} archive compression ratio is too high")

    def _extract_zip_archive(
        self,
        archive_path: Path,
        destination_dir: Path,
        *,
        entry_offset: int = 0,
        byte_offset: int = 0,
    ) -> tuple[int, int, int]:
        """Extract safe DICOM candidates and return extracted files, entries, and bytes."""

        extracted_files = 0
        inspected_entries = entry_offset
        uncompressed_bytes = byte_offset
        destination_dir.mkdir(parents=True, exist_ok=True)

        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise HTTPException(status_code=400, detail="Invalid ZIP archive") from exc

        with archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path, inspected_entries, uncompressed_bytes = self._validate_archive_member(
                    member_name=info.filename,
                    archive_label="ZIP",
                    compressed_size=info.compress_size,
                    uncompressed_size=info.file_size,
                    entry_offset=inspected_entries,
                    byte_offset=uncompressed_bytes,
                    is_encrypted=bool(info.flag_bits & 0x1),
                    is_symlink=self._is_archive_symlink(info),
                )
                if member_path is None:
                    continue

                target_path = self._deduplicate_target_path(destination_dir / member_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with archive.open(info, "r") as source, target_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=UPLOAD_CHUNK_SIZE)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="Failed to extract ZIP archive") from exc
                extracted_files += 1

        return extracted_files, inspected_entries, uncompressed_bytes

    def _extract_7z_archive(
        self,
        archive_path: Path,
        destination_dir: Path,
        *,
        entry_offset: int = 0,
        byte_offset: int = 0,
    ) -> tuple[int, int, int]:
        extracted_files = 0
        inspected_entries = entry_offset
        uncompressed_bytes = byte_offset
        destination_dir.mkdir(parents=True, exist_ok=True)

        try:
            archive = py7zr.SevenZipFile(archive_path, mode="r")
        except (OSError, py7zr.Bad7zFile) as exc:
            raise HTTPException(status_code=400, detail="Invalid 7z archive") from exc

        with archive:
            if archive.needs_password():
                raise HTTPException(status_code=400, detail="Encrypted 7z archives are not supported")

            targets: list[str] = []
            for info in archive.list():
                if not info.is_file:
                    continue
                member_path, inspected_entries, uncompressed_bytes = self._validate_archive_member(
                    member_name=info.filename,
                    archive_label="7z",
                    compressed_size=info.compressed,
                    uncompressed_size=info.uncompressed,
                    entry_offset=inspected_entries,
                    byte_offset=uncompressed_bytes,
                    is_symlink=bool(info.is_symlink),
                )
                if member_path is not None:
                    targets.append(info.filename)

            self._validate_archive_compression_ratio(archive_path, "7z", uncompressed_bytes - byte_offset)

            if targets:
                try:
                    archive.extract(path=destination_dir, targets=targets)
                except (OSError, py7zr.Bad7zFile) as exc:
                    raise HTTPException(status_code=400, detail="Failed to extract 7z archive") from exc
                extracted_files = sum(
                    1 for member_name in targets if (destination_dir / self._safe_archive_member_path(member_name, "7z")).is_file()
                )

        return extracted_files, inspected_entries, uncompressed_bytes

    def _extract_rar_archive(
        self,
        archive_path: Path,
        destination_dir: Path,
        *,
        entry_offset: int = 0,
        byte_offset: int = 0,
    ) -> tuple[int, int, int]:
        extracted_files = 0
        inspected_entries = entry_offset
        uncompressed_bytes = byte_offset
        destination_dir.mkdir(parents=True, exist_ok=True)

        try:
            archive = rarfile.RarFile(archive_path)
        except (OSError, rarfile.Error) as exc:
            raise HTTPException(status_code=400, detail="Invalid RAR archive") from exc

        with archive:
            if archive.needs_password():
                raise HTTPException(status_code=400, detail="Encrypted RAR archives are not supported")
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path, inspected_entries, uncompressed_bytes = self._validate_archive_member(
                    member_name=info.filename,
                    archive_label="RAR",
                    compressed_size=info.compress_size,
                    uncompressed_size=info.file_size,
                    entry_offset=inspected_entries,
                    byte_offset=uncompressed_bytes,
                    is_encrypted=info.needs_password(),
                    is_symlink=info.is_symlink() or bool(info.file_redir),
                )
                if member_path is None:
                    continue

                target_path = self._deduplicate_target_path(destination_dir / member_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with archive.open(info) as source, target_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=UPLOAD_CHUNK_SIZE)
                except rarfile.RarCannotExec as exc:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=501,
                        detail="RAR extraction requires unar, unrar, 7z, or bsdtar on the server",
                    ) from exc
                except (OSError, rarfile.Error) as exc:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="Failed to extract RAR archive") from exc
                extracted_files += 1

        return extracted_files, inspected_entries, uncompressed_bytes

    def _extract_archive(
        self,
        archive_path: Path,
        destination_dir: Path,
        *,
        entry_offset: int = 0,
        byte_offset: int = 0,
    ) -> tuple[int, int, int]:
        suffix = self._archive_suffix(archive_path)
        if suffix == ".zip":
            return self._extract_zip_archive(archive_path, destination_dir, entry_offset=entry_offset, byte_offset=byte_offset)
        if suffix == ".7z":
            return self._extract_7z_archive(archive_path, destination_dir, entry_offset=entry_offset, byte_offset=byte_offset)
        if suffix == ".rar":
            return self._extract_rar_archive(archive_path, destination_dir, entry_offset=entry_offset, byte_offset=byte_offset)
        raise HTTPException(status_code=400, detail="Unsupported compressed DICOM format")

    def _load_uploaded_session(self, session_dir: Path, workspace_id: str) -> LoadFolderResponse:
        response = series_registry.load_folder(
            LoadFolderRequest(folderPath=str(session_dir)),
            workspace_id=workspace_id,
        )
        session_dir.touch()
        return response

    def load_archive_path(
        self,
        archive_path: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> LoadFolderResponse:
        source_path = Path(archive_path).expanduser().resolve()
        if not source_path.is_file() or self._archive_suffix(source_path) is None:
            raise HTTPException(status_code=400, detail="Only ZIP, 7z, and RAR archives can be loaded as compressed DICOM data")

        upload_root = self._resolve_upload_root()
        self.cleanup_uploads()
        session_id = uuid4().hex
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        session_dir = upload_root / f"{normalized_workspace_id.replace(':', '_')}-{session_id}"
        try:
            extracted_files, entry_count, extracted_bytes = self._extract_archive(
                source_path,
                session_dir / "extracted",
            )
            if not extracted_files:
                raise HTTPException(status_code=400, detail="Archive does not contain DICOM candidate files")
            logger.info(
                "local compressed DICOM import session=%s format=%s files=%s entries=%s extracted_bytes=%s root=%s",
                session_id,
                source_path.suffix.lower(),
                extracted_files,
                entry_count,
                extracted_bytes,
                session_dir,
            )
            return self._load_uploaded_session(session_dir, normalized_workspace_id)
        except Exception:
            self._delete_upload_dir(session_dir)
            raise

    async def upload_and_load(
        self,
        files: list[UploadFile],
        relative_paths: list[str] | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> LoadFolderResponse:
        settings = get_settings()
        if not files:
            raise HTTPException(status_code=400, detail="No DICOM files were uploaded")
        relative_paths = relative_paths or []
        upload_items: list[tuple[int, UploadFile, Path]] = []
        skipped_count = 0
        for index, upload_file in enumerate(files):
            relative_path = relative_paths[index] if index < len(relative_paths) else None
            candidate_path = self._safe_relative_path(relative_path, upload_file.filename, index)
            if not self._is_upload_candidate(candidate_path):
                skipped_count += 1
                await upload_file.close()
                continue
            upload_items.append((index, upload_file, candidate_path))

        if not upload_items:
            raise HTTPException(status_code=400, detail="No DICOM candidate files were uploaded")
        if len(upload_items) > settings.web_upload_max_files:
            raise HTTPException(status_code=413, detail="Too many DICOM files were uploaded")

        upload_root = self._resolve_upload_root()
        self.cleanup_uploads()
        session_id = uuid4().hex
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        session_dir = upload_root / f"{normalized_workspace_id.replace(':', '_')}-{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        saved_count = 0
        archive_paths: list[Path] = []
        try:
            for index, upload_file, relative_path in upload_items:
                target_path = session_dir / relative_path
                target_path = self._deduplicate_target_path(target_path)
                total_bytes = await self._save_upload_file(upload_file, target_path, total_bytes)
                if target_path.exists():
                    saved_count += 1
                    if self._archive_suffix(target_path) is not None:
                        archive_paths.append(target_path)

            if saved_count <= 0:
                raise HTTPException(status_code=400, detail="No non-empty DICOM files were uploaded")

            archive_entry_count = 0
            archive_uncompressed_bytes = 0
            extracted_count = 0
            for index, archive_path in enumerate(archive_paths, start=1):
                extracted, archive_entry_count, archive_uncompressed_bytes = self._extract_archive(
                    archive_path,
                    session_dir / "extracted" / f"archive-{index}",
                    entry_offset=archive_entry_count,
                    byte_offset=archive_uncompressed_bytes,
                )
                extracted_count += extracted
                archive_path.unlink(missing_ok=True)

            logger.info(
                "web dicom upload session=%s files=%s extracted=%s skipped=%s bytes=%s root=%s",
                session_id,
                saved_count,
                extracted_count,
                skipped_count,
                total_bytes,
                session_dir,
            )
            return self._load_uploaded_session(session_dir, normalized_workspace_id)
        except Exception:
            self._delete_upload_dir(session_dir)
            raise


dicom_upload_service = DicomUploadService(start_cleanup_worker=True)
