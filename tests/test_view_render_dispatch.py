import pytest
from fastapi import HTTPException

from app.models.viewer import ViewRecord
from app.services.viewer_render_dispatch import render_by_view_type
from app.services.viewer_render_guards import ensure_view_size


def _view(view_type: str = "Stack", *, width: int | None = 512, height: int | None = 512) -> ViewRecord:
    return ViewRecord(
        view_id=f"{view_type.lower()}-view",
        series_id="series-1",
        view_type=view_type,
        width=width,
        height=height,
    )


def test_ensure_view_size_rejects_missing_dimensions() -> None:
    with pytest.raises(HTTPException) as missing_width:
        ensure_view_size(_view(width=None, height=512))
    with pytest.raises(HTTPException) as zero_height:
        ensure_view_size(_view(width=512, height=0))

    assert missing_width.value.status_code == 400
    assert missing_width.value.detail == "View size has not been set"
    assert zero_height.value.status_code == 400
    assert zero_height.value.detail == "View size has not been set"


def test_ensure_view_size_accepts_concrete_dimensions() -> None:
    ensure_view_size(_view(width=512, height=384))


class _RenderServiceSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []

    def _is_mpr_view_type(self, view_type: str) -> bool:
        return view_type in {"AX", "COR", "SAG"}

    def _is_3d_view_type(self, view_type: str) -> bool:
        return view_type == "3D"

    def _render_mpr_view(self, view: ViewRecord, *, image_format: str, fast_preview: bool, progress_callback=None) -> str:
        self.calls.append(("mpr", view.view_id, image_format, fast_preview))
        return "mpr-result"

    def _render_3d_view(self, view: ViewRecord, *, image_format: str, fast_preview: bool, progress_callback=None) -> str:
        self.calls.append(("3d", view.view_id, image_format, fast_preview))
        return "3d-result"

    def _render_view(self, view: ViewRecord, *, image_format: str, fast_preview: bool) -> str:
        self.calls.append(("stack", view.view_id, image_format, fast_preview))
        return "stack-result"


def test_render_by_view_type_dispatches_to_matching_renderer() -> None:
    service = _RenderServiceSpy()

    assert render_by_view_type(service, _view("AX"), image_format="jpeg", fast_preview=True) == "mpr-result"
    assert render_by_view_type(service, _view("3D"), image_format="png", fast_preview=False) == "3d-result"
    assert render_by_view_type(service, _view("Stack"), image_format="webp", fast_preview=True) == "stack-result"

    assert service.calls == [
        ("mpr", "ax-view", "jpeg", True),
        ("3d", "3d-view", "png", False),
        ("stack", "stack-view", "webp", True),
    ]
