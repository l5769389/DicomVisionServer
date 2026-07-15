from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock

from app.core.workspace import normalize_workspace_id


class WorkspaceActivityService:
    def __init__(self) -> None:
        self._last_seen_by_workspace: dict[str, datetime] = {}
        self._lock = RLock()

    def touch(self, workspace_id: str) -> str:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        with self._lock:
            self._last_seen_by_workspace[normalized_workspace_id] = datetime.now(timezone.utc)
        return normalized_workspace_id

    def release(self, workspace_id: str) -> dict[str, object]:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        from app.services.series_registry import series_registry
        from app.services.view_group_registry import view_group_registry
        from app.services.view_registry import view_registry
        from app.services.viewer_service import viewer_service
        from app.sockets.runtime import view_socket_hub

        workspace_views = view_registry.list_all(workspace_id=normalized_workspace_id)
        released_view_count = 0
        for view in workspace_views:
            try:
                view_socket_hub.close_view(view.view_id)
                viewer_service.close_view_by_id(view.view_id, workspace_id=normalized_workspace_id)
                released_view_count += 1
            except Exception:
                continue
        view_group_registry.delete_workspace(normalized_workspace_id)
        series_registry.clear(workspace_id=normalized_workspace_id)
        with self._lock:
            self._last_seen_by_workspace.pop(normalized_workspace_id, None)
        return {
            "workspaceId": normalized_workspace_id,
            "releasedViewCount": released_view_count,
        }

    def stats(self, workspace_id: str) -> dict[str, object]:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        from app.services.dicom_cache import dicom_cache
        from app.services.series_registry import series_registry
        from app.services.view_group_registry import view_group_registry
        from app.services.view_registry import view_registry
        from app.services.viewer_service import viewer_service

        series = series_registry.list_all(workspace_id=normalized_workspace_id)
        views = view_registry.list_all(workspace_id=normalized_workspace_id)
        groups = view_group_registry.list_all(workspace_id=normalized_workspace_id)
        with self._lock:
            last_seen = self._last_seen_by_workspace.get(normalized_workspace_id)
        return {
            "workspaceId": normalized_workspace_id,
            "lastSeenAt": last_seen.isoformat() if last_seen is not None else None,
            "series": {
                "total": len(series),
                "real": len([item for item in series if not item.is_virtual]),
                "virtual": len([item for item in series if item.is_virtual]),
            },
            "views": {
                "total": len(views),
                "groups": len(groups),
            },
            "cache": {
                "dicom": dicom_cache.stats(),
                "volume": viewer_service.get_volume_cache_stats(),
            },
        }


workspace_activity_service = WorkspaceActivityService()
