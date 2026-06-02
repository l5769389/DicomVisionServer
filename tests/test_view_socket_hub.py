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


def test_mpr_group_queue_drains_latest_pending_requests(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, bool]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        first_render_started = asyncio.Event()
        release_first_render = asyncio.Event()
        calls: list[tuple[str, str, bool]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.image_format, request.fast_preview))
            if len(calls) == 1:
                first_render_started.set()
                await release_first_render.wait()
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        first_task = asyncio.create_task(
            hub.emit_render_for_view("v-ax", image_format="jpeg", fast_preview=True)
        )
        await first_render_started.wait()

        assert await hub.emit_render_for_view("v-cor", image_format="jpeg", fast_preview=True) is False
        assert await hub.emit_render_for_view("v-cor", image_format="png", fast_preview=False) is False
        assert await hub.emit_render_for_view("v-sag", image_format="jpeg", fast_preview=True) is False

        release_first_render.set()
        assert await first_task is True
        return calls

    assert asyncio.run(run()) == [
        ("v-ax", "jpeg", True),
        ("v-cor", "png", False),
        ("v-sag", "jpeg", True),
    ]


def test_mpr_group_queue_renders_pending_batch_in_parallel(monkeypatch) -> None:
    async def run() -> list[str]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        first_render_started = asyncio.Event()
        release_first_render = asyncio.Event()
        coronal_started = asyncio.Event()
        sagittal_started = asyncio.Event()
        release_reference_renders = asyncio.Event()
        calls: list[str] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            del request
            calls.append(view_id)
            if view_id == "v-ax":
                first_render_started.set()
                await release_first_render.wait()
                return True
            if view_id == "v-cor":
                coronal_started.set()
            if view_id == "v-sag":
                sagittal_started.set()
            await release_reference_renders.wait()
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        first_task = asyncio.create_task(
            hub.emit_render_for_view("v-ax", image_format="jpeg", fast_preview=True)
        )
        await first_render_started.wait()

        assert await hub.emit_render_for_view("v-cor", image_format="jpeg", fast_preview=True) is False
        assert await hub.emit_render_for_view("v-sag", image_format="jpeg", fast_preview=True) is False

        release_first_render.set()
        await asyncio.wait_for(coronal_started.wait(), timeout=1.0)
        await asyncio.wait_for(sagittal_started.wait(), timeout=1.0)
        assert not first_task.done()
        release_reference_renders.set()
        assert await first_task is True
        return calls

    assert asyncio.run(run()) == ["v-ax", "v-cor", "v-sag"]


def test_drain_promotes_current_preview_when_final_is_pending(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, bool]]:
        hub = ViewSocketHub()
        calls: list[tuple[str, str, bool]] = []
        hub._pending_render_requests["mpr-group:g"] = {
            "v-cor": RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",))
        }

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.image_format, request.fast_preview))
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub._drain_render_requests(
            "mpr-group:g",
            "v-cor",
            RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",)),
        ) is True
        return calls

    assert asyncio.run(run()) == [("v-cor", "png", False)]
