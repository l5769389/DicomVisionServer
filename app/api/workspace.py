from __future__ import annotations

from typing import Annotated

from fastapi import Header, Query

from app.core.workspace import WORKSPACE_HEADER, WORKSPACE_QUERY_PARAM, normalize_workspace_id
from app.services.workspace_activity import workspace_activity_service


def get_request_workspace_id(
    header_workspace_id: Annotated[str | None, Header(alias=WORKSPACE_HEADER)] = None,
    query_workspace_id: Annotated[str | None, Query(alias=WORKSPACE_QUERY_PARAM)] = None,
) -> str:
    # The query fallback exists for image/preview URLs rendered by <img>, where
    # browsers cannot attach custom headers.
    workspace_id = normalize_workspace_id(header_workspace_id or query_workspace_id)
    workspace_activity_service.touch(workspace_id)
    return workspace_id
