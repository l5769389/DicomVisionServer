from uuid import uuid4

from fastapi import HTTPException

from app.core import MPR_VIEWPORT_AXIAL, MPR_VIEWPORT_CORONAL, MPR_VIEWPORT_SAGITTAL
from app.core.logging import get_logger
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

    def create(self, payload: ViewCreateRequest) -> ViewCreateResponse:
        series_registry.get(payload.series_id)

        view = ViewRecord(
            view_id=str(uuid4()),
            series_id=payload.series_id,
            view_type=payload.view_type,
        )
        if payload.view_type in {"MPR", "AX", "COR", "SAG"}:
            view.view_group = view_group_registry.get_or_create_mpr_group_for_series(
                payload.series_id,
                active_viewport=_resolve_mpr_active_viewport(payload.view_type),
                view_group_key=payload.view_group_key,
            )
        self._view_by_id[view.view_id] = view
        logger.info(
            "create view view_id=%s series_id=%s view_type=%s view_group_key=%s group_id=%s",
            view.view_id,
            view.series_id,
            view.view_type,
            payload.view_group_key,
            view.view_group.group_id if view.view_group is not None else None,
        )
        return ViewCreateResponse(viewId=view.view_id)

    def get(self, view_id: str) -> ViewRecord:
        view = self._view_by_id.get(view_id)
        if view is None:
            raise HTTPException(status_code=404, detail="viewId not found")
        return view

    def delete(self, view_id: str) -> ViewRecord:
        view = self._view_by_id.pop(view_id, None)
        if view is None:
            raise HTTPException(status_code=404, detail="viewId not found")
        return view

    def list_all(self) -> list[ViewRecord]:
        return list(self._view_by_id.values())

    def list_view_group(self, group_id: str) -> list[ViewRecord]:
        return [view for view in self._view_by_id.values() if view.view_group is not None and view.view_group.group_id == group_id]


view_registry = ViewRegistry()
