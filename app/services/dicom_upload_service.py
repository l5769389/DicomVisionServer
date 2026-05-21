import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.dicom import LoadFolderRequest, LoadFolderResponse
from app.services.series_registry import series_registry


logger = get_logger(__name__)
UPLOAD_CHUNK_SIZE = 1024 * 1024


class DicomUploadService:
    def _resolve_upload_root(self) -> Path:
        settings = get_settings()
        root = Path(settings.web_upload_dicom_root) if settings.web_upload_dicom_root else Path(tempfile.gettempdir()) / "dicomvision-web-uploads"
        root.mkdir(parents=True, exist_ok=True)
        return root

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

    async def upload_and_load(
        self,
        files: list[UploadFile],
        relative_paths: list[str] | None = None,
    ) -> LoadFolderResponse:
        settings = get_settings()
        if not files:
            raise HTTPException(status_code=400, detail="No DICOM files were uploaded")
        if len(files) > settings.web_upload_max_files:
            raise HTTPException(status_code=413, detail="Too many DICOM files were uploaded")

        upload_root = self._resolve_upload_root()
        session_id = uuid4().hex
        session_dir = upload_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        saved_count = 0
        relative_paths = relative_paths or []

        for index, upload_file in enumerate(files):
            relative_path = relative_paths[index] if index < len(relative_paths) else None
            target_path = session_dir / self._safe_relative_path(relative_path, upload_file.filename, index)
            target_path = self._deduplicate_target_path(target_path)
            total_bytes = await self._save_upload_file(upload_file, target_path, total_bytes)
            if target_path.exists():
                saved_count += 1

        if saved_count <= 0:
            raise HTTPException(status_code=400, detail="No non-empty DICOM files were uploaded")

        logger.info(
            "web dicom upload session=%s files=%s bytes=%s root=%s",
            session_id,
            saved_count,
            total_bytes,
            session_dir,
        )
        return series_registry.load_folder(LoadFolderRequest(folderPath=str(session_dir)))


dicom_upload_service = DicomUploadService()
