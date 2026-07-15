from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.routes import dicom as dicom_routes
from app.main import fastapi_app
from app.services.series_registry import series_registry


def test_load_sample_falls_back_to_bundled_sample_when_configured_path_is_invalid(monkeypatch) -> None:
    series_registry.clear()
    monkeypatch.setattr(dicom_routes.sys, "platform", "linux")
    monkeypatch.setattr(
        dicom_routes,
        "settings",
        SimpleNamespace(web_sample_dicom_path="web_sample xxx"),
    )

    response = TestClient(fastapi_app).post("/api/v1/dicom/loadSample")

    assert response.status_code == 200
    data = response.json()
    assert data["samplePath"].endswith("sample-data/test")
    assert data["seriesId"]
    assert data["seriesList"]


def test_resolve_sample_prefers_configured_path(monkeypatch, tmp_path: Path) -> None:
    configured_path = tmp_path / "configured"
    mac_local_path = tmp_path / "mac-local"
    bundled_path = tmp_path / "bundled"
    configured_path.mkdir()
    mac_local_path.mkdir()
    bundled_path.mkdir()
    monkeypatch.setattr(dicom_routes.sys, "platform", "darwin")
    monkeypatch.setattr(dicom_routes, "MAC_LOCAL_SAMPLE_DICOM_PATH", mac_local_path)
    monkeypatch.setattr(dicom_routes, "BUNDLED_SAMPLE_DICOM_PATH", bundled_path)
    monkeypatch.setattr(dicom_routes, "settings", SimpleNamespace(web_sample_dicom_path=str(configured_path)))

    assert dicom_routes._resolve_sample_dicom_path() == str(configured_path.resolve())


def test_resolve_sample_uses_mac_local_path_before_bundled_on_macos(monkeypatch, tmp_path: Path) -> None:
    mac_local_path = tmp_path / "mac-local"
    bundled_path = tmp_path / "bundled"
    mac_local_path.mkdir()
    bundled_path.mkdir()
    monkeypatch.setattr(dicom_routes.sys, "platform", "darwin")
    monkeypatch.setattr(dicom_routes, "MAC_LOCAL_SAMPLE_DICOM_PATH", mac_local_path)
    monkeypatch.setattr(dicom_routes, "BUNDLED_SAMPLE_DICOM_PATH", bundled_path)
    monkeypatch.setattr(dicom_routes, "settings", SimpleNamespace(web_sample_dicom_path=None))

    assert dicom_routes._resolve_sample_dicom_path() == str(mac_local_path.resolve())


def test_resolve_sample_uses_mac_local_path_when_configured_path_is_bundled(monkeypatch, tmp_path: Path) -> None:
    mac_local_path = tmp_path / "mac-local"
    bundled_path = tmp_path / "bundled"
    mac_local_path.mkdir()
    bundled_path.mkdir()
    monkeypatch.setattr(dicom_routes.sys, "platform", "darwin")
    monkeypatch.setattr(dicom_routes, "MAC_LOCAL_SAMPLE_DICOM_PATH", mac_local_path)
    monkeypatch.setattr(dicom_routes, "BUNDLED_SAMPLE_DICOM_PATH", bundled_path)
    monkeypatch.setattr(dicom_routes, "settings", SimpleNamespace(web_sample_dicom_path=str(bundled_path)))

    assert dicom_routes._resolve_sample_dicom_path() == str(mac_local_path.resolve())


def test_resolve_sample_uses_bundled_path_on_non_macos(monkeypatch, tmp_path: Path) -> None:
    mac_local_path = tmp_path / "mac-local"
    bundled_path = tmp_path / "bundled"
    mac_local_path.mkdir()
    bundled_path.mkdir()
    monkeypatch.setattr(dicom_routes.sys, "platform", "win32")
    monkeypatch.setattr(dicom_routes, "MAC_LOCAL_SAMPLE_DICOM_PATH", mac_local_path)
    monkeypatch.setattr(dicom_routes, "BUNDLED_SAMPLE_DICOM_PATH", bundled_path)
    monkeypatch.setattr(dicom_routes, "settings", SimpleNamespace(web_sample_dicom_path=None))

    assert dicom_routes._resolve_sample_dicom_path() == str(bundled_path.resolve())
