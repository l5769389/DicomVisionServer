from uuid import uuid4

from app.models.viewer import ViewGroupRecord


class ViewGroupRegistry:
    def __init__(self) -> None:
        self._view_groups_by_id: dict[str, ViewGroupRecord] = {}
        self._mpr_group_id_by_series_id: dict[str, str] = {}

    def _get_mpr_registry_key(self, series_id: str, view_group_key: str | None = None) -> str:
        if view_group_key:
            return f"{series_id}::{view_group_key}"
        return series_id

    def get_or_create_mpr_group_for_series(
        self,
        series_id: str,
        *,
        active_viewport: str,
        view_group_key: str | None = None,
    ) -> ViewGroupRecord:
        registry_key = self._get_mpr_registry_key(series_id, view_group_key)
        group_id = self._mpr_group_id_by_series_id.get(registry_key)
        if group_id is not None:
            group = self._view_groups_by_id.get(group_id)
            if group is not None:
                return group
            self._mpr_group_id_by_series_id.pop(registry_key, None)

        group = ViewGroupRecord(
            group_id=str(uuid4()),
            group_type="mpr",
            series_id=series_id,
            active_viewport=active_viewport,
        )
        self._view_groups_by_id[group.group_id] = group
        self._mpr_group_id_by_series_id[registry_key] = group.group_id
        return group

    def get_view_group(self, group_id: str) -> ViewGroupRecord | None:
        return self._view_groups_by_id.get(group_id)

    def delete(self, group_id: str) -> None:
        group = self._view_groups_by_id.pop(group_id, None)
        if group is None:
            return
        stale_keys = [
            registry_key
            for registry_key, candidate_group_id in self._mpr_group_id_by_series_id.items()
            if candidate_group_id == group_id
        ]
        for registry_key in stale_keys:
            self._mpr_group_id_by_series_id.pop(registry_key, None)


view_group_registry = ViewGroupRegistry()
