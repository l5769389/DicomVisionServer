from uuid import uuid4

from fastapi import HTTPException

from app.models.viewer import ViewRecord
from app.schemas.view import ViewCreateRequest, ViewCreateResponse
from app.services.series_registry import series_registry


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
        self._view_by_id[view.view_id] = view
        return ViewCreateResponse(viewId=view.view_id)

    def get(self, view_id: str) -> ViewRecord:
        view = self._view_by_id.get(view_id)
        if view is None:
            raise HTTPException(status_code=404, detail="viewId not found")
        return view

    def list_all(self) -> list[ViewRecord]:
        return list(self._view_by_id.values())


view_registry = ViewRegistry()
