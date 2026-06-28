from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import fastapi_app
from app.models.viewer import SeriesRecord, ViewRecord
from app.schemas.view import ViewOperationRequest
from app.sockets.runtime import view_socket_hub
from app.services.series_registry import series_registry
from app.services.view_group_registry import view_group_registry
from app.services.view_registry import view_registry
from app.services.viewer_service import viewer_service


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
        patient_name=None,
        study_date=None,
        study_description=None,
        accession_number=None,
        modality="CT",
        series_description="Lifecycle test",
    )
    return series_id


def _register_stack_view(series_id: str, view_id: str = "view-1") -> ViewRecord:
    view = ViewRecord(view_id=view_id, series_id=series_id, view_type="Stack", width=128, height=128)
    view_registry._view_by_id[view.view_id] = view
    return view


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


def test_transform2d_resets_pan_absolute_and_renders(monkeypatch) -> None:
    series_id = _register_series()
    view = _register_stack_view(series_id)
    view.offset_x = 42.0
    view.offset_y = -17.0
    view.zoom = 2.5
    view.rotation_degrees = 90
    view.hor_flip = True
    view.drag_origin_offset_x = 42.0
    view.drag_origin_offset_y = -17.0
    render_result = object()
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: render_result)

    result = viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate({"viewId": view.view_id, "opType": "transform2d", "x": 0, "y": 0})
    )

    assert result.primary_result is render_result
    assert view.offset_x == 0.0
    assert view.offset_y == 0.0
    assert view.zoom == 2.5
    assert view.rotation_degrees == 90
    assert view.hor_flip is True
    assert view.drag_origin_offset_x is None
    assert view.drag_origin_offset_y is None
    assert view.is_initialized is True


def test_transform2d_resets_zoom_absolute_and_renders(monkeypatch) -> None:
    series_id = _register_series()
    view = _register_stack_view(series_id)
    view.offset_x = 12.0
    view.offset_y = 8.0
    view.zoom = 3.25
    view.drag_origin_zoom = 3.25
    render_result = object()
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: render_result)

    result = viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate({"viewId": view.view_id, "opType": "transform2d", "zoom": 1})
    )

    assert result.primary_result is render_result
    assert view.offset_x == 12.0
    assert view.offset_y == 8.0
    assert view.zoom == 1.0
    assert view.drag_origin_zoom is None
    assert view.is_initialized is True


def test_transform2d_applies_all_transform_fields_absolute(monkeypatch) -> None:
    series_id = _register_series()
    view = _register_stack_view(series_id)
    view.offset_x = 10.0
    view.offset_y = 11.0
    view.zoom = 1.5
    view.rotation_degrees = 270
    view.hor_flip = True
    view.ver_flip = False
    render_result = object()
    monkeypatch.setattr(viewer_service, "_render_by_view_type", lambda *args, **kwargs: render_result)

    result = viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate(
            {
                "viewId": view.view_id,
                "opType": "transform2d",
                "x": 5,
                "y": -6,
                "zoom": 2,
                "rotationDegrees": 450,
                "hor_flip": False,
                "ver_flip": True,
            }
        )
    )

    assert result.primary_result is render_result
    assert view.offset_x == 5.0
    assert view.offset_y == -6.0
    assert view.zoom == 2.0
    assert view.rotation_degrees == 90
    assert view.hor_flip is False
    assert view.ver_flip is True


def test_empty_transform2d_does_not_render_or_mutate(monkeypatch) -> None:
    series_id = _register_series()
    view = _register_stack_view(series_id)
    view.offset_x = 7.0
    view.offset_y = -9.0
    view.zoom = 1.75
    render_calls = 0

    def render_stub(*args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        return object()

    monkeypatch.setattr(viewer_service, "_render_by_view_type", render_stub)

    result = viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate({"viewId": view.view_id, "opType": "transform2d"})
    )

    assert render_calls == 0
    assert result.primary_result is None
    assert view.offset_x == 7.0
    assert view.offset_y == -9.0
    assert view.zoom == 1.75
    assert view.is_initialized is False


def test_non_transform2d_xy_mutations_remain_incremental() -> None:
    series_id = _register_series()
    view = _register_stack_view(series_id)
    view.offset_x = 2.0
    view.offset_y = 4.0

    result = viewer_service.handle_view_operation(
        ViewOperationRequest.model_validate({"viewId": view.view_id, "opType": "pseudocolor", "x": 3, "y": -1})
    )

    assert result.primary_result is None
    assert view.offset_x == 5.0
    assert view.offset_y == 3.0
    assert view.is_initialized is True


def test_mpr_state_sync_broadcast_skips_unsized_group_views(monkeypatch) -> None:
    client = TestClient(fastapi_app)
    series_id = _register_series()
    source_view_id = _create_view(client, series_id, "AX", "4d:tab:phase-0")
    target_axial_view_id = _create_view(client, series_id, "AX", "4d:tab:phase-1")
    target_coronal_view_id = _create_view(client, series_id, "COR", "4d:tab:phase-1")

    target_axial_view = view_registry.get(target_axial_view_id)
    target_axial_view.width = 320
    target_axial_view.height = 240

    monkeypatch.setattr(viewer_service, "_sync_mpr_state_from_source_view", lambda target_view, source_view_id: True)

    payload = ViewOperationRequest.model_validate(
        {
            "viewId": target_axial_view_id,
            "opType": "mprStateSync",
            "sourceViewId": source_view_id,
        }
    )
    result = viewer_service.handle_view_operation(payload)

    assert result.broadcast_view_ids == (target_axial_view_id,)
    assert target_coronal_view_id not in result.broadcast_view_ids
