import asyncio
from types import SimpleNamespace

from app.services.viewer_operation_handlers import OperationRenderOutcome
from app.sockets import handlers


class _SocketServerStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, str | None]] = []
        self.render_error_emitted = asyncio.Event()

    async def emit(self, event: str, payload: object, to: str | None = None) -> None:
        self.events.append((event, payload, to))
        if event == "render_error":
            self.render_error_emitted.set()


def test_handle_operation_schedules_mpr_broadcast_without_waiting(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, bool, tuple[str, ...] | None]]:
        server = _SocketServerStub()
        render_started = asyncio.Event()
        release_render = asyncio.Event()
        render_calls: list[tuple[str, str, bool, tuple[str, ...] | None]] = []

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="AX"),
        )
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(
            handlers.viewer_service,
            "handle_view_operation",
            lambda payload, workspace_id=None: OperationRenderOutcome(
                broadcast_view_ids=("v-ax", "v-cor", "v-sag"),
                broadcast_image_format="jpeg",
                broadcast_fast_preview=True,
            ),
        )

        async def fake_emit_render_for_view(
            view_id: str,
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            target_sids: tuple[str, ...] | None = None,
        ) -> bool:
            render_calls.append((view_id, image_format, fast_preview, target_sids))
            render_started.set()
            await release_render.wait()
            return True

        monkeypatch.setattr(handlers.view_socket_hub, "emit_render_for_view", fake_emit_render_for_view)

        response_task = asyncio.create_task(
            handlers._handle_operation(
                server,  # type: ignore[arg-type]
                "sid-1",
                {"viewId": "v-ax", "opType": "mprOblique", "actionType": "move", "x": 0.5, "y": 0.5},
            )
        )
        response = await asyncio.wait_for(response_task, timeout=0.2)
        assert response == {"ok": True}

        await asyncio.wait_for(render_started.wait(), timeout=1.0)
        release_render.set()
        await asyncio.sleep(0)
        return render_calls

    assert asyncio.run(run()) == [
        ("v-ax", "jpeg", True, None),
        ("v-cor", "jpeg", True, None),
        ("v-sag", "jpeg", True, None),
    ]


def test_mpr_drag_operations_bypass_default_threadpool(monkeypatch) -> None:
    async def run() -> list[tuple[str, str]]:
        server = _SocketServerStub()
        calls: list[tuple[str, str]] = []

        async def fail_to_thread(*args, **kwargs):
            del args, kwargs
            raise AssertionError("MPR move should not wait for the shared render threadpool")

        def fake_handle_view_operation(payload, workspace_id=None):
            calls.append((str(workspace_id), str(payload.action_type)))
            return OperationRenderOutcome()

        monkeypatch.setattr(handlers.asyncio, "to_thread", fail_to_thread)
        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="AX"),
        )
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(handlers.viewer_service, "handle_view_operation", fake_handle_view_operation)

        for action_type in ("start", "move", "end"):
            response = await handlers._handle_operation(
                server,  # type: ignore[arg-type]
                "sid-1",
                {"viewId": "v-ax", "opType": "mprOblique", "actionType": action_type, "x": 0.5, "y": 0.5},
            )
            assert response == {"ok": True}
        return calls

    assert asyncio.run(run()) == [
        ("workspace-a", "start"),
        ("workspace-a", "move"),
        ("workspace-a", "end"),
    ]


def test_background_render_error_is_reported_to_socket(monkeypatch) -> None:
    async def run() -> list[tuple[str, object, str | None]]:
        server = _SocketServerStub()

        async def fake_emit_render_for_view(
            view_id: str,
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            target_sids: tuple[str, ...] | None = None,
        ) -> bool:
            del view_id, image_format, fast_preview, target_sids
            raise RuntimeError("render failed")

        monkeypatch.setattr(handlers.view_socket_hub, "emit_render_for_view", fake_emit_render_for_view)

        handlers._schedule_render_for_view(
            server,  # type: ignore[arg-type]
            "sid-1",
            "v-ax",
            image_format="jpeg",
            fast_preview=True,
        )
        await asyncio.wait_for(server.render_error_emitted.wait(), timeout=1.0)
        return server.events

    events = asyncio.run(run())
    assert ("image_error", {"message": "render failed"}, "sid-1") in events
    assert ("render_error", {"message": "render failed"}, "sid-1") in events
