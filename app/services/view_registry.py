from threading import RLock
from uuid import uuid4

from fastapi import HTTPException

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.core.logging import get_logger
from app.core.workspace import DEFAULT_WORKSPACE_ID, normalize_workspace_id
from app.models.viewer import ViewRecord
from app.schemas.view import ViewCreateRequest, ViewCreateResponse
from app.services.series_registry import series_registry
from app.services.view_group_registry import view_group_registry

logger = get_logger(__name__)


def _resolve_mpr_active_viewport(view_type: str) -> str:
    if view_type == "COR":
        return MPR_VIEWPORT_CORONAL
    if view_type == "SAG":
        return MPR_VIEWPORT_SAGITTAL
    return MPR_VIEWPORT_AXIAL


class ViewRegistry:
    def __init__(self) -> None:
        self._view_by_id: dict[str, ViewRecord] = {}
        self._lock = RLock()

    def create(self, payload: ViewCreateRequest, workspace_id: str = DEFAULT_WORKSPACE_ID) -> ViewCreateResponse:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        series_registry.get(payload.series_id, workspace_id=normalized_workspace_id)

        with self._lock:
            view = ViewRecord(
                view_id=str(uuid4()),
                series_id=payload.series_id,
                view_type=payload.view_type,
                workspace_id=normalized_workspace_id,
            )
            if payload.view_type in {"MPR", "AX", "COR", "SAG"}:
                view.view_group = view_group_registry.get_or_create_mpr_group_for_series(
                    payload.series_id,
                    active_viewport=_resolve_mpr_active_viewport(payload.view_type),
                    view_group_key=payload.view_group_key,
                    workspace_id=normalized_workspace_id,
                )
            self._view_by_id[view.view_id] = view
        logger.info(
            "create view view_id=%s series_id=%s view_type=%s workspace_id=%s view_group_key=%s group_id=%s",
            view.view_id,
            view.series_id,
            view.view_type,
            view.workspace_id,
            payload.view_group_key,
            view.view_group.group_id if view.view_group is not None else None,
        )
        return ViewCreateResponse(viewId=view.view_id)

    def get(self, view_id: str, workspace_id: str | None = None) -> ViewRecord:
        with self._lock:
            view = self._view_by_id.get(view_id)
            normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
            if view is None or (
                normalized_workspace_id is not None and view.workspace_id != normalized_workspace_id
            ):
                raise HTTPException(status_code=404, detail="viewId not found")
            return view

    def delete(self, view_id: str, workspace_id: str | None = None) -> ViewRecord:
        with self._lock:
            view = self.get(view_id, workspace_id=workspace_id)
            self._view_by_id.pop(view_id, None)
            return view

    def list_all(self, workspace_id: str | None = None) -> list[ViewRecord]:
        with self._lock:
            normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
            return [
                view
                for view in self._view_by_id.values()
                if normalized_workspace_id is None or view.workspace_id == normalized_workspace_id
            ]

    def list_view_group(self, group_id: str, workspace_id: str | None = None) -> list[ViewRecord]:
        with self._lock:
            normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
            return [
                view
                for view in self._view_by_id.values()
                if view.view_group is not None
                and view.view_group.group_id == group_id
                and (normalized_workspace_id is None or view.workspace_id == normalized_workspace_id)
            ]

    def delete_workspace(self, workspace_id: str) -> list[ViewRecord]:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        with self._lock:
            views = [view for view in self._view_by_id.values() if view.workspace_id == normalized_workspace_id]
            for view in views:
                self._view_by_id.pop(view.view_id, None)
            return views


view_registry = ViewRegistry()
