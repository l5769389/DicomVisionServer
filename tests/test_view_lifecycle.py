from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import fastapi_app
from app.models.viewer import SeriesRecord
from app.schemas.view import ViewOperationRequest
from app.sockets.runtime import view_socket_hub
from app.services.series_registry import series_registry
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry


@pytest.fixture(autouse=True)
def isolated_registries() -> Iterator[None]:
    previous_series_by_id = dict(series_registry._series_by_id)
    previous_series_id_by_key = dict(series_registry._series_id_by_key)
    previous_views = dict(view_registry._view_by_id)
    previous_groups = dict(view_group_registry._view_groups_by_id)
    previous_mpr_group_ids = dict(view_group_registry._mpr_group_id_by_series_id)
    previous_view_sids = {key: set(value) for key, value in view_socket_hub._view_sids.items()}
    previous_sid_views = {key: set(value) for key, value in view_socket_hub._sid_views.items()}

    series_registry._series_by_id.clear()
    series_registry._series_id_by_key.clear()
    view_registry._view_by_id.clear()
    view_group_registry._view_groups_by_id.clear()
    view_group_registry._mpr_group_id_by_series_id.clear()
    view_socket_hub._view_sids.clear()
    view_socket_hub._sid_views.clear()
    view_socket_hub._pending_render_requests.clear()
    view_socket_hub._render_locks.clear()

    try:
        yield
    finally:
        series_registry._series_by_id.clear()
        series_registry._series_by_id.update(previous_series_by_id)
        series_registry._series_id_by_key.clear()
        series_registry._series_id_by_key.update(previous_series_id_by_key)
        view_registry._view_by_id.clear()
        view_registry._view_by_id.update(previous_views)
        view_group_registry._view_groups_by_id.clear()
        view_group_registry._view_groups_by_id.update(previous_groups)
        view_group_registry._mpr_group_id_by_series_id.clear()
        view_group_registry._mpr_group_id_by_series_id.update(previous_mpr_group_ids)
        view_socket_hub._view_sids.clear()
        view_socket_hub._view_sids.update(previous_view_sids)
        view_socket_hub._sid_views.clear()
        view_socket_hub._sid_views.update(previous_sid_views)
        view_socket_hub._pending_render_requests.clear()
        view_socket_hub._render_locks.clear()


def _register_series(series_id: str = "series-1") -> str:
    series_registry._series_by_id[series_id] = SeriesRecord(
        series_id=series_id,
        folder_path="",
        series_instance_uid=None,
        study_instance_uid=None,
        patient_id=None,
        modality="CT",
        series_description="Lifecycle test",
    )
    return series_id


def _create_view(client: TestClient, series_id: str, view_type: str, view_group_key: str | None = None) -> str:
    payload = {"seriesId": series_id, "viewType": view_type}
    if view_group_key:
        payload["viewGroupKey"] = view_group_key

    response = client.post(
        "/api/v1/view/create",
        json=payload,
    )
    assert response.status_code == 200
    return str(response.json()["viewId"])


def test_close_view_releases_view_and_socket_bindings() -> None:
    client = TestClient(fastapi_app)
    series_id = _register_series()
    view_id = _create_view(client, series_id, "Stack")
    view_socket_hub.bind_view("sid-1", view_id)

    response = client.post("/api/v1/view/close", json={"viewId": view_id})

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "View closed", "viewId": view_id}
    with pytest.raises(HTTPException):
        view_registry.get(view_id)
    assert view_id not in view_socket_hub._view_sids
    assert "sid-1" not in view_socket_hub._sid_views


def test_close_last_mpr_view_releases_view_group() -> None:
    client = TestClient(fastapi_app)
    series_id = _register_series()
    axial_view_id = _create_view(client, series_id, "AX")
    coronal_view_id = _create_view(client, series_id, "COR")
    group_id = view_registry.get(axial_view_id).view_group.group_id

    assert view_registry.get(coronal_view_id).view_group.group_id == group_id

    first_response = client.post("/api/v1/view/close", json={"viewId": axial_view_id})
    assert first_response.status_code == 200
    assert view_group_registry.get_view_group(group_id) is not None

    second_response = client.post("/api/v1/view/close", json={"viewId": coronal_view_id})
    assert second_response.status_code == 200
    assert view_group_registry.get_view_group(group_id) is None
    assert view_group_registry._mpr_group_id_by_series_id.get(series_id) is None


def test_mpr_view_group_key_isolates_same_series_groups() -> None:
    client = TestClient(fastapi_app)
    series_id = _register_series()
    default_view_id = _create_view(client, series_id, "AX")
    four_d_axial_view_id = _create_view(client, series_id, "AX", "4d:tab:phase-0")
    four_d_coronal_view_id = _create_view(client, series_id, "COR", "4d:tab:phase-0")
    other_phase_view_id = _create_view(client, series_id, "AX", "4d:tab:phase-1")

    default_group_id = view_registry.get(default_view_id).view_group.group_id
    four_d_group_id = view_registry.get(four_d_axial_view_id).view_group.group_id

    assert four_d_group_id != default_group_id
    assert view_registry.get(four_d_coronal_view_id).view_group.group_id == four_d_group_id
    assert view_registry.get(other_phase_view_id).view_group.group_id not in {default_group_id, four_d_group_id}


def test_mpr_state_sync_operation_accepts_source_view_alias() -> None:
    payload = ViewOperationRequest.model_validate(
        {"viewId": "target-view", "opType": "mprStateSync", "sourceViewId": "source-view"}
    )

    assert payload.view_id == "target-view"
    assert payload.source_view_id == "source-view"
