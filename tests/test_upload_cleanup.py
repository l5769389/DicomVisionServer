import os
import time
from pathlib import Path

from app.core.config import Settings
from app.services.dicom_upload_service import DicomUploadService


def test_upload_cleanup_deletes_stale_session_dirs(tmp_path: Path) -> None:
    service = DicomUploadService(
        upload_root=tmp_path,
        max_age_seconds=60,
        cleanup_interval_seconds=60,
    )
    stale_session = tmp_path / "stale-session"
    fresh_session = tmp_path / "fresh-session"
    stale_session.mkdir()
    fresh_session.mkdir()
    (stale_session / "image.dcm").write_bytes(b"old")
    (fresh_session / "image.dcm").write_bytes(b"new")

    stale_timestamp = time.time() - 120
    os.utime(stale_session, (stale_timestamp, stale_timestamp))

    assert service.cleanup_uploads() == 1
    assert not stale_session.exists()
    assert fresh_session.exists()


def test_api_docs_are_hidden_by_default_in_production() -> None:
    assert Settings(APP_ENV="production").api_docs_enabled is False


def test_api_docs_can_be_enabled_explicitly() -> None:
    assert Settings(APP_ENV="production", EXPOSE_API_DOCS=True).api_docs_enabled is True


def test_3d_transport_settings_are_normalized_and_bounded() -> None:
    settings = Settings(
        DICOMVISION_3D_TRANSPORT="WebRTC",
        DICOMVISION_WEBRTC_VIDEO_CODEC="H264",
        DICOMVISION_WEBRTC_VIDEO_BITRATE_BPS=99_000_000,
        DICOMVISION_WEBRTC_VIDEO_FPS=120,
        DICOMVISION_WEBRTC_INITIAL_BURST_FRAMES=9,
    )

    assert settings.normalized_three_d_transport == "webrtc"
    assert settings.normalized_webrtc_video_codec == "h264"
    assert settings.normalized_webrtc_video_bitrate_bps == 20_000_000
    assert settings.normalized_webrtc_video_fps == 60
    assert settings.normalized_webrtc_initial_burst_frames == 3
