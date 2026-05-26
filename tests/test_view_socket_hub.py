import asyncio

from app.sockets.runtime import RenderRequest, ViewSocketHub


class _SocketServerStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object], str | None]] = []

    async def emit(self, event: str, payload: dict[str, object], to: str | None = None) -> None:
        self.events.append((event, payload, to))


def test_merge_render_request_keeps_full_quality_when_preview_arrives_later() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-2",)),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids == ("sid-1", "sid-2")


def test_merge_render_request_promotes_pending_preview_to_full_quality() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",)),
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1", "sid-2")),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids == ("sid-1", "sid-2")


def test_merge_render_request_keeps_broadcast_target_when_either_request_broadcasts() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=None),
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids is None


def test_emit_progress_message_targets_bound_sids() -> None:
    async def run() -> list[tuple[str, dict[str, object], str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]

        await hub._emit_progress_message(
            "view-1",
            ("sid-1", "sid-2"),
            {"phase": "volume", "progressPercent": 42},
        )
        return server.events

    assert asyncio.run(run()) == [
        ("view_progress", {"viewId": "view-1", "phase": "volume", "progressPercent": 42}, "sid-1"),
        ("view_progress", {"viewId": "view-1", "phase": "volume", "progressPercent": 42}, "sid-2"),
    ]
