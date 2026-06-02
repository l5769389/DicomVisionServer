from __future__ import annotations

import re

from fastapi import HTTPException


WORKSPACE_HEADER = "X-DicomVision-Workspace-Id"
WORKSPACE_QUERY_PARAM = "workspaceId"
DEFAULT_WORKSPACE_ID = "default"

_WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def normalize_workspace_id(value: str | None) -> str:
    workspace_id = str(value or "").strip() or DEFAULT_WORKSPACE_ID
    if not _WORKSPACE_ID_PATTERN.match(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace id")
    return workspace_id
