from threading import RLock
from uuid import uuid4

from fastapi import HTTPException

from app.core.workspace import DEFAULT_WORKSPACE_ID, normalize_workspace_id
from app.models.viewer import FusionRegistrationState, ViewGroupRecord


class ViewGroupRegistry:
    def __init__(self) -> None:
        self._view_groups_by_id: dict[str, ViewGroupRecord] = {}
        self._mpr_group_id_by_series_id: dict[str, str] = {}
        self._fusion_group_id_by_pair_key: dict[str, str] = {}
        self._saved_fusion_registration_by_pair_key: dict[str, FusionRegistrationState] = {}
        self._lock = RLock()

    def _get_mpr_registry_key(
        self,
        workspace_id: str,
        series_id: str,
        view_group_key: str | None = None,
    ) -> str:
        workspace_key = normalize_workspace_id(workspace_id)
        if view_group_key:
            return f"{workspace_key}::{series_id}::{view_group_key}"
        return f"{workspace_key}::{series_id}"

    def get_or_create_mpr_group_for_series(
        self,
        series_id: str,
        *,
        active_viewport: str,
        view_group_key: str | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> ViewGroupRecord:
        with self._lock:
            normalized_workspace_id = normalize_workspace_id(workspace_id)
            registry_key = self._get_mpr_registry_key(normalized_workspace_id, series_id, view_group_key)
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
                workspace_id=normalized_workspace_id,
                active_viewport=active_viewport,
            )
            self._view_groups_by_id[group.group_id] = group
            self._mpr_group_id_by_series_id[registry_key] = group.group_id
            return group

    def _get_fusion_registry_key(
        self,
        workspace_id: str,
        ct_series_id: str,
        pet_series_id: str,
        view_group_key: str | None = None,
    ) -> str:
        workspace_key = normalize_workspace_id(workspace_id)
        pair_key = f"{ct_series_id}::{pet_series_id}"
        if view_group_key:
            return f"{workspace_key}::{pair_key}::{view_group_key}"
        return f"{workspace_key}::{pair_key}"

    def get_or_create_fusion_group_for_pair(
        self,
        ct_series_id: str,
        pet_series_id: str,
        *,
        view_group_key: str | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> ViewGroupRecord:
        with self._lock:
            normalized_workspace_id = normalize_workspace_id(workspace_id)
            registry_key = self._get_fusion_registry_key(
                normalized_workspace_id,
                ct_series_id,
                pet_series_id,
                view_group_key,
            )
            group_id = self._fusion_group_id_by_pair_key.get(registry_key)
            if group_id is not None:
                group = self._view_groups_by_id.get(group_id)
                if group is not None:
                    return group
                self._fusion_group_id_by_pair_key.pop(registry_key, None)

            saved_registration = self._saved_fusion_registration_by_pair_key.get(registry_key)
            group = ViewGroupRecord(
                group_id=str(uuid4()),
                group_type="fusion",
                series_id=ct_series_id,
                secondary_series_id=pet_series_id,
                fusion_ct_series_id=ct_series_id,
                fusion_pet_series_id=pet_series_id,
                fusion_view_group_key=view_group_key,
                workspace_id=normalized_workspace_id,
                fusion_registration=FusionRegistrationState(
                    translate_row_mm=float(saved_registration.translate_row_mm),
                    translate_col_mm=float(saved_registration.translate_col_mm),
                    rotation_degrees=float(saved_registration.rotation_degrees),
                    saved=True,
                ) if saved_registration is not None else FusionRegistrationState(),
            )
            self._view_groups_by_id[group.group_id] = group
            self._fusion_group_id_by_pair_key[registry_key] = group.group_id
            return group

    def save_fusion_registration(
        self,
        group: ViewGroupRecord,
        *,
        view_group_key: str | None = None,
    ) -> None:
        if group.fusion_ct_series_id is None or group.fusion_pet_series_id is None:
            return
        registry_key = self._get_fusion_registry_key(
            group.workspace_id,
            group.fusion_ct_series_id,
            group.fusion_pet_series_id,
            view_group_key if view_group_key is not None else group.fusion_view_group_key,
        )
        group.fusion_registration.saved = True
        self._saved_fusion_registration_by_pair_key[registry_key] = FusionRegistrationState(
            translate_row_mm=float(group.fusion_registration.translate_row_mm),
            translate_col_mm=float(group.fusion_registration.translate_col_mm),
            rotation_degrees=float(group.fusion_registration.rotation_degrees),
            saved=True,
        )

    def get_view_group(self, group_id: str, workspace_id: str | None = None) -> ViewGroupRecord | None:
        with self._lock:
            group = self._view_groups_by_id.get(group_id)
            if group is None:
                return None
            if workspace_id is not None and group.workspace_id != normalize_workspace_id(workspace_id):
                return None
            return group

    def require_view_group(self, group_id: str, workspace_id: str | None = None) -> ViewGroupRecord:
        group = self.get_view_group(group_id, workspace_id=workspace_id)
        if group is None:
            raise HTTPException(status_code=404, detail="viewGroupId not found")
        return group

    def list_all(self, workspace_id: str | None = None) -> list[ViewGroupRecord]:
        normalized_workspace_id = normalize_workspace_id(workspace_id) if workspace_id is not None else None
        with self._lock:
            return [
                group
                for group in self._view_groups_by_id.values()
                if normalized_workspace_id is None or group.workspace_id == normalized_workspace_id
            ]

    def delete(self, group_id: str) -> None:
        with self._lock:
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
            stale_fusion_keys = [
                registry_key
                for registry_key, candidate_group_id in self._fusion_group_id_by_pair_key.items()
                if candidate_group_id == group_id
            ]
            for registry_key in stale_fusion_keys:
                self._fusion_group_id_by_pair_key.pop(registry_key, None)

    def delete_workspace(self, workspace_id: str) -> None:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        with self._lock:
            group_ids = [
                group_id
                for group_id, group in self._view_groups_by_id.items()
                if group.workspace_id == normalized_workspace_id
            ]
        for group_id in group_ids:
            self.delete(group_id)


view_group_registry = ViewGroupRegistry()
