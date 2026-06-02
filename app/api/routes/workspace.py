from fastapi import APIRouter, Depends

from app.api.workspace import get_request_workspace_id
from app.services.workspace_activity import workspace_activity_service

router = APIRouter(prefix="/workspace", tags=["workspace"])


@router.get("/stats", summary="Get current anonymous workspace stats")
def get_workspace_stats(
    workspace_id: str = Depends(get_request_workspace_id),
) -> dict[str, object]:
    return workspace_activity_service.stats(workspace_id)


@router.post("/release", summary="Release current anonymous workspace resources")
def release_workspace(
    workspace_id: str = Depends(get_request_workspace_id),
) -> dict[str, object]:
    return workspace_activity_service.release(workspace_id)
