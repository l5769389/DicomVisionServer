from fastapi import HTTPException

from app.models.viewer import ViewRecord


def ensure_view_size(view: ViewRecord) -> None:
    """Validate that a server-side view has a concrete frontend viewport size."""

    if not view.width or not view.height:
        raise HTTPException(status_code=400, detail="View size has not been set")
