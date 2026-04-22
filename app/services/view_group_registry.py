from uuid import uuid4

from app.models.viewer import ViewGroupRecord


class ViewGroupRegistry:
    def __init__(self) -> None:
        self._view_groups_by_id: dict[str, ViewGroupRecord] = {}
        self._mpr_group_id_by_series_id: dict[str, str] = {}

    def get_or_create_mpr_group_for_series(
        self,
        series_id: str,
        *,
        active_viewport: str,
    ) -> ViewGroupRecord:
        group_id = self._mpr_group_id_by_series_id.get(series_id)
        if group_id is not None:
            return self._view_groups_by_id[group_id]

        group = ViewGroupRecord(
            group_id=str(uuid4()),
            group_type="mpr",
            series_id=series_id,
            active_viewport=active_viewport,
        )
        self._view_groups_by_id[group.group_id] = group
        self._mpr_group_id_by_series_id[series_id] = group.group_id
        return group

    def get_view_group(self, group_id: str) -> ViewGroupRecord | None:
        return self._view_groups_by_id.get(group_id)

    def delete(self, group_id: str) -> None:
        group = self._view_groups_by_id.pop(group_id, None)
        if group is None:
            return
        if self._mpr_group_id_by_series_id.get(group.series_id) == group_id:
            self._mpr_group_id_by_series_id.pop(group.series_id, None)


view_group_registry = ViewGroupRegistry()
