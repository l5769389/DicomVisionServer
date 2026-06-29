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


async def _wait_for(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not met")
        await asyncio.sleep(0)


def test_bind_view_uses_requested_image_format(monkeypatch) -> None:
    class _RegisteredServer(_SocketServerStub):
        def __init__(self) -> None:
            super().__init__()
            self.handlers: dict[str, object] = {}

        def event(self, func):
            self.handlers[func.__name__] = func
            return func

        def on(self, event_name: str):
            def decorator(func):
                self.handlers[event_name] = func
                return func

            return decorator

    async def run() -> tuple[dict[str, object] | None, list[tuple[str, str, tuple[str, ...] | None]]]:
        server = _RegisteredServer()
        render_calls: list[tuple[str, str, tuple[str, ...] | None]] = []

        handlers.register_socket_handlers(server)  # type: ignore[arg-type]
        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="Stack", width=512, height=512),
        )

        async def fake_emit_render_for_view(
            view_id: str,
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del fast_preview, fast_preview_full_resolution, metadata_mode, mpr_revision
            render_calls.append((view_id, image_format, target_sids))
            return True

        monkeypatch.setattr(handlers.view_socket_hub, "emit_render_for_view", fake_emit_render_for_view)

        bind_view = server.handlers["bind_view"]
        response = await bind_view("sid-1", {"viewId": "v-stack", "imageFormat": "webp"})  # type: ignore[misc]
        return response, render_calls

    response, render_calls = asyncio.run(run())
    assert response == {"ok": True}
    assert render_calls == [("v-stack", "webp", ("sid-1",))]


def test_set_view_size_uses_requested_image_format(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, tuple[str, ...] | None]]:
        server = _SocketServerStub()
        render_calls: list[tuple[str, str, tuple[str, ...] | None]] = []

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="Stack"),
        )
        monkeypatch.setattr(
            handlers.viewer_service,
            "set_view_size",
            lambda payload, workspace_id=None: SimpleNamespace(
                model_dump=lambda by_alias=True: {"success": True, "message": "ok", "viewId": payload.view_id}
            ),
        )

        async def fake_emit_render_for_view(
            view_id: str,
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del fast_preview, fast_preview_full_resolution, metadata_mode, mpr_revision
            render_calls.append((view_id, image_format, target_sids))
            return True

        monkeypatch.setattr(handlers.view_socket_hub, "emit_render_for_view", fake_emit_render_for_view)

        await handlers._handle_set_size(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-stack", "opType": "setSize", "size": {"width": 512, "height": 512}, "imageFormat": "webp"},
        )
        return render_calls

    assert asyncio.run(run()) == [("v-stack", "webp", ("sid-1",))]


def test_view_operation_payload_normalizes_image_format(monkeypatch) -> None:
    async def run() -> tuple[list[str], list[str]]:
        server = _SocketServerStub()
        seen_formats: list[str] = []

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="Stack"),
        )

        def fake_handle_view_operation(payload, workspace_id=None):
            del workspace_id
            seen_formats.append(payload.image_format)
            return OperationRenderOutcome()

        monkeypatch.setattr(handlers.viewer_service, "handle_view_operation", fake_handle_view_operation)

        await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-stack", "opType": "window", "actionType": "start", "imageFormat": "webp"},
        )
        await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-stack", "opType": "window", "actionType": "start", "imageFormat": "avif"},
        )
        return seen_formats, [event for event, _payload, _to in server.events]

    seen_formats, events = asyncio.run(run())
    assert seen_formats == ["webp", "png"]
    assert events == []


def test_handle_operation_schedules_mpr_broadcast_batch_without_waiting(monkeypatch) -> None:
    async def run() -> list[tuple[tuple[str, ...], str, bool, tuple[str, ...] | None]]:
        server = _SocketServerStub()
        render_calls: list[tuple[tuple[str, ...], str, bool, tuple[str, ...] | None]] = []

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

        async def fake_schedule_render_batch(
            view_ids: tuple[str, ...],
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del fast_preview_full_resolution, metadata_mode, mpr_revision
            render_calls.append((view_ids, image_format, fast_preview, target_sids))
            return False

        monkeypatch.setattr(handlers.view_socket_hub, "schedule_render_batch", fake_schedule_render_batch)

        response_task = asyncio.create_task(
            handlers._handle_operation(
                server,  # type: ignore[arg-type]
                "sid-1",
                {"viewId": "v-ax", "opType": "mprOblique", "actionType": "move", "x": 0.5, "y": 0.5},
            )
        )
        response = await asyncio.wait_for(response_task, timeout=0.2)
        assert response == {"ok": True}
        await _wait_for(lambda: len(render_calls) == 1)
        return render_calls

    assert asyncio.run(run()) == [
        (("v-ax", "v-cor", "v-sag"), "jpeg", True, None),
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
        await _wait_for(lambda: calls == [("workspace-a", "start"), ("workspace-a", "end")])
        return calls

    assert asyncio.run(run()) == [
        ("workspace-a", "start"),
        ("workspace-a", "end"),
    ]


def test_mpr_operation_queue_keeps_latest_move_when_worker_is_busy(monkeypatch) -> None:
    async def run() -> list[tuple[str, float | None]]:
        handlers._mpr_operation_queues.clear()
        server = _SocketServerStub()
        calls: list[tuple[str, float | None]] = []
        start_entered = asyncio.Event()
        release_start = asyncio.Event()

        def fake_handle_view_operation(payload, workspace_id=None):
            del workspace_id
            calls.append((str(payload.action_type), payload.x))
            if payload.action_type == "start":
                start_entered.set()
                raise RuntimeError("start should be blocked through async wrapper")
            return OperationRenderOutcome()

        async def fake_process(operation):
            if operation.payload.action_type == "start":
                calls.append(("start", operation.payload.x))
                start_entered.set()
                await release_start.wait()
                return
            calls.append((str(operation.payload.action_type), operation.payload.x))

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="AX"),
        )
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(handlers.viewer_service, "handle_view_operation", fake_handle_view_operation)
        monkeypatch.setattr(handlers, "_process_queued_mpr_operation", fake_process)

        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "start", "x": 0.1, "y": 0.1},
        ) == {"ok": True}
        await _wait_for(start_entered.is_set)
        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "move", "x": 0.2, "y": 0.2},
        ) == {"ok": True}
        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "move", "x": 0.8, "y": 0.8},
        ) == {"ok": True}
        release_start.set()
        await _wait_for(lambda: calls == [("start", 0.1), ("move", 0.8)])
        return calls

    assert asyncio.run(run()) == [("start", 0.1), ("move", 0.8)]


def test_mpr_operation_queue_end_drops_pending_move(monkeypatch) -> None:
    async def run() -> list[tuple[str, float | None]]:
        handlers._mpr_operation_queues.clear()
        server = _SocketServerStub()
        calls: list[tuple[str, float | None]] = []
        start_entered = asyncio.Event()
        release_start = asyncio.Event()

        async def fake_process(operation):
            calls.append((str(operation.payload.action_type), operation.payload.x))
            if operation.payload.action_type == "start":
                start_entered.set()
                await release_start.wait()

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="AX"),
        )
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(handlers, "_process_queued_mpr_operation", fake_process)

        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "mprOblique", "actionType": "start", "line": "horizontal", "x": 0.1, "y": 0.1},
        ) == {"ok": True}
        await _wait_for(start_entered.is_set)
        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "mprOblique", "actionType": "move", "line": "horizontal", "x": 0.2, "y": 0.2},
        ) == {"ok": True}
        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "mprOblique", "actionType": "end", "line": "horizontal", "x": 0.9, "y": 0.9},
        ) == {"ok": True}
        release_start.set()
        await _wait_for(lambda: calls == [("start", 0.1), ("end", 0.9)])
        return calls

    assert asyncio.run(run()) == [("start", 0.1), ("end", 0.9)]


def test_fusion_registration_queue_drops_pending_move_before_end(monkeypatch) -> None:
    async def run() -> list[tuple[str, float | None]]:
        handlers._mpr_operation_queues.clear()
        server = _SocketServerStub()
        calls: list[tuple[str, float | None]] = []
        start_entered = asyncio.Event()
        release_start = asyncio.Event()

        async def fake_process(operation):
            calls.append((str(operation.payload.action_type), operation.payload.x))
            if operation.payload.action_type == "start":
                start_entered.set()
                await release_start.wait()

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="FusionOverlayAxial"),
        )
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(handlers, "_process_queued_mpr_operation", fake_process)

        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {
                "viewId": "fusion-overlay",
                "opType": "fusionRegistration",
                "actionType": "start",
                "subOpType": "translate",
                "x": 0.0,
                "y": 0.0,
            },
        ) == {"ok": True}
        await _wait_for(start_entered.is_set)
        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {
                "viewId": "fusion-overlay",
                "opType": "fusionRegistration",
                "actionType": "move",
                "subOpType": "translate",
                "x": 0.4,
                "y": 0.2,
            },
        ) == {"ok": True}
        assert await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {
                "viewId": "fusion-overlay",
                "opType": "fusionRegistration",
                "actionType": "end",
                "subOpType": "translate",
                "x": 0.9,
                "y": 0.4,
            },
        ) == {"ok": True}
        release_start.set()
        await _wait_for(lambda: calls == [("start", 0.0), ("end", 0.9)])
        return calls

    assert asyncio.run(run()) == [("start", 0.0), ("end", 0.9)]


def test_queued_operation_runs_view_operation_off_event_loop(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, object, str | None]]]:
        server = _SocketServerStub()
        used_to_thread = False

        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="FusionOverlayAxial"),
        )
        monkeypatch.setattr(
            handlers.viewer_service,
            "handle_view_operation",
            lambda payload, workspace_id=None: OperationRenderOutcome(),
        )

        async def fake_to_thread(func, *args):
            nonlocal used_to_thread
            used_to_thread = True
            return func(*args)

        monkeypatch.setattr(handlers.asyncio, "to_thread", fake_to_thread)
        await handlers._process_queued_mpr_operation(
            handlers._QueuedMprOperation(
                payload=handlers.ViewOperationRequest(
                    viewId="fusion-overlay",
                    opType="fusionRegistration",
                    actionType="move",
                    subOpType="translate",
                    x=1.0,
                    y=0.0,
                ),
                server=server,  # type: ignore[arg-type]
                sid="sid-1",
                workspace_id="workspace-a",
            )
        )
        return used_to_thread, server.events

    used_to_thread, events = asyncio.run(run())
    assert used_to_thread is True
    assert events == []


def test_handle_operation_returns_revision_and_schedules_preview_options(monkeypatch) -> None:
    async def run() -> tuple[dict[str, object], list[tuple[int | None, bool]], list[tuple[str, object, str | None]]]:
        server = _SocketServerStub()
        scheduled_options: list[tuple[tuple[str, ...], int | None, bool]] = []

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
                mpr_revision=7,
                broadcast_view_ids=("v-cor",),
                broadcast_image_format="png",
                broadcast_fast_preview=True,
                broadcast_fast_preview_full_resolution=False,
                broadcast_metadata_mode="mpr-crosshair-preview",
                mpr_state_view_ids=("v-cor",),
            ),
        )
        monkeypatch.setattr(
            handlers.viewer_service,
            "build_mpr_state_update_payloads",
            lambda view_ids, workspace_id=None, mpr_revision=None: {
                view_id: {
                    "viewId": view_id,
                    "mprRevision": mpr_revision,
                    "mpr_crosshair": {"centerX": 0.5, "centerY": 0.5},
                }
                for view_id in view_ids
            },
        )
        monkeypatch.setattr(handlers.view_socket_hub, "get_view_sids", lambda view_id: ("sid-2",))

        async def fake_schedule_render_batch(
            view_ids: tuple[str, ...],
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del image_format, fast_preview, metadata_mode, target_sids
            scheduled_options.append((view_ids, mpr_revision, fast_preview_full_resolution))
            return False

        monkeypatch.setattr(handlers.view_socket_hub, "schedule_render_batch", fake_schedule_render_batch)

        response = await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "mprOblique", "actionType": "move", "x": 0.5, "y": 0.5},
        )
        await _wait_for(lambda: len(scheduled_options) == 1)
        return response, scheduled_options, server.events

    response, scheduled_options, events = asyncio.run(run())
    assert response == {"ok": True}
    assert scheduled_options == [(("v-cor",), 7, False)]
    assert events == [
        (
            "mpr_state_update",
            {"viewId": "v-cor", "mprRevision": 7, "mpr_crosshair": {"centerX": 0.5, "centerY": 0.5}},
            "sid-2",
        )
    ]


def test_mpr_crosshair_state_emits_state_and_throttles_preview(monkeypatch) -> None:
    async def run() -> tuple[dict[str, object], list[tuple[tuple[str, ...], str, bool, str, int | None]], list[tuple[str, object, str | None]]]:
        handlers._mpr_crosshair_state_queues.clear()
        handlers._mpr_crosshair_preview_states.clear()
        server = _SocketServerStub()
        scheduled_batches: list[tuple[tuple[str, ...], str, bool, str, int | None]] = []

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
                mpr_revision=12,
                broadcast_view_ids=("v-cor", "v-sag"),
                broadcast_image_format="png",
                broadcast_fast_preview=True,
                broadcast_fast_preview_full_resolution=False,
                broadcast_metadata_mode="mpr-crosshair-preview",
                mpr_state_view_ids=("v-cor", "v-sag"),
            ),
        )
        monkeypatch.setattr(
            handlers.viewer_service,
            "build_mpr_state_update_payloads",
            lambda view_ids, workspace_id=None, mpr_revision=None: {
                view_id: {
                    "viewId": view_id,
                    "mprRevision": mpr_revision,
                }
                for view_id in view_ids
            },
        )
        monkeypatch.setattr(handlers.view_socket_hub, "get_view_sids", lambda view_id: (f"sid-{view_id}",))

        async def fake_schedule_render_batch(
            view_ids: tuple[str, ...],
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del fast_preview_full_resolution, target_sids
            scheduled_batches.append((view_ids, image_format, fast_preview, metadata_mode, mpr_revision))
            return False

        monkeypatch.setattr(handlers.view_socket_hub, "schedule_render_batch", fake_schedule_render_batch)

        response = await handlers._handle_mpr_crosshair_state(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "move", "x": 0.5, "y": 0.5},
        )
        await _wait_for(lambda: len(scheduled_batches) == 1)
        return response, scheduled_batches, server.events

    response, scheduled_batches, events = asyncio.run(run())
    assert response == {"ok": True}
    assert scheduled_batches == [(("v-cor", "v-sag"), "png", True, "mpr-crosshair-preview", 12)]
    assert events == [
        ("mpr_state_update", {"viewId": "v-cor", "mprRevision": 12}, "sid-v-cor"),
        ("mpr_state_update", {"viewId": "v-sag", "mprRevision": 12}, "sid-v-sag"),
    ]


def test_mpr_crosshair_state_queue_keeps_latest_move(monkeypatch) -> None:
    async def run() -> list[tuple[str, float | None]]:
        handlers._mpr_crosshair_state_queues.clear()
        server = _SocketServerStub()
        calls: list[tuple[str, float | None]] = []
        start_entered = asyncio.Event()
        release_start = asyncio.Event()

        async def fake_process(queue_key, operation):
            del queue_key
            calls.append((str(operation.payload.action_type), operation.payload.x))
            if operation.payload.action_type == "start":
                start_entered.set()
                await release_start.wait()

        monkeypatch.setattr(handlers.view_socket_hub, "get_sid_workspace", lambda sid: "workspace-a")
        monkeypatch.setattr(
            handlers.view_registry,
            "get",
            lambda view_id, workspace_id=None: SimpleNamespace(view_id=view_id, view_type="AX"),
        )
        monkeypatch.setattr(handlers.view_socket_hub, "bind_view", lambda sid, view_id: None)
        monkeypatch.setattr(handlers, "_process_queued_mpr_crosshair_state_operation", fake_process)

        assert await handlers._handle_mpr_crosshair_state(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "start", "x": 0.1, "y": 0.1},
        ) == {"ok": True}
        await _wait_for(start_entered.is_set)
        assert await handlers._handle_mpr_crosshair_state(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "move", "x": 0.2, "y": 0.2},
        ) == {"ok": True}
        assert await handlers._handle_mpr_crosshair_state(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "crosshair", "actionType": "move", "x": 0.8, "y": 0.8},
        ) == {"ok": True}
        release_start.set()
        await _wait_for(lambda: calls == [("start", 0.1), ("move", 0.8)])
        return calls

    assert asyncio.run(run()) == [("start", 0.1), ("move", 0.8)]


def test_mpr_crosshair_preview_generation_skips_replaced_request(monkeypatch) -> None:
    async def run() -> list[tuple[tuple[str, ...], int | None]]:
        handlers._mpr_crosshair_preview_states.clear()
        server = _SocketServerStub()
        scheduled_batches: list[tuple[tuple[str, ...], int | None]] = []
        queue_key = "mpr-op:workspace-a:g"
        loop = asyncio.get_running_loop()
        handlers._mpr_crosshair_preview_states[queue_key] = handlers._MprCrosshairPreviewState(
            last_dispatch_at=loop.time(),
        )

        async def fake_schedule_render_batch(
            view_ids: tuple[str, ...],
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del image_format, fast_preview, fast_preview_full_resolution, metadata_mode, target_sids
            scheduled_batches.append((view_ids, mpr_revision))
            return False

        monkeypatch.setattr(handlers, "MPR_CROSSHAIR_PREVIEW_INTERVAL_SECONDS", 0.02)
        monkeypatch.setattr(handlers.view_socket_hub, "schedule_render_batch", fake_schedule_render_batch)

        handlers._schedule_mpr_crosshair_preview(
            queue_key,
            handlers._MprCrosshairPreviewRequest(
                server=server,  # type: ignore[arg-type]
                sid="sid-1",
                view_ids=("v-old",),
                image_format="png",
                fast_preview=True,
                fast_preview_full_resolution=False,
                metadata_mode="mpr-crosshair-preview",
                mpr_revision=1,
            ),
        )
        await asyncio.sleep(0)
        handlers._schedule_mpr_crosshair_preview(
            queue_key,
            handlers._MprCrosshairPreviewRequest(
                server=server,  # type: ignore[arg-type]
                sid="sid-1",
                view_ids=("v-new",),
                image_format="png",
                fast_preview=True,
                fast_preview_full_resolution=False,
                metadata_mode="mpr-crosshair-preview",
                mpr_revision=2,
            ),
        )
        await _wait_for(lambda: len(scheduled_batches) == 1)
        return scheduled_batches

    assert asyncio.run(run()) == [(("v-new",), 2)]


def test_handle_operation_routes_mpr_deferred_preview_through_batch_scheduler(monkeypatch) -> None:
    async def run() -> tuple[dict[str, object], list[tuple[tuple[str, ...], tuple[str, ...] | None, int | None]]]:
        server = _SocketServerStub()
        scheduled_batches: list[tuple[tuple[str, ...], tuple[str, ...] | None, int | None]] = []

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
                mpr_revision=9,
                deferred_view_ids=("v-ax",),
                deferred_image_format="jpeg",
                deferred_fast_preview=True,
            ),
        )

        async def fake_schedule_render_batch(
            view_ids: tuple[str, ...],
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del image_format, fast_preview, fast_preview_full_resolution, metadata_mode
            scheduled_batches.append((view_ids, target_sids, mpr_revision))
            return False

        monkeypatch.setattr(handlers.view_socket_hub, "schedule_render_batch", fake_schedule_render_batch)

        response = await handlers._handle_operation(
            server,  # type: ignore[arg-type]
            "sid-1",
            {"viewId": "v-ax", "opType": "pan", "actionType": "move", "x": 2, "y": 3},
        )
        await _wait_for(lambda: len(scheduled_batches) == 1)
        return response, scheduled_batches

    response, scheduled_batches = asyncio.run(run())
    assert response == {"ok": True}
    assert scheduled_batches == [(("v-ax",), ("sid-1",), 9)]


def test_background_render_error_is_reported_to_socket(monkeypatch) -> None:
    async def run() -> list[tuple[str, object, str | None]]:
        server = _SocketServerStub()

        async def fake_emit_render_for_view(
            view_id: str,
            *,
            image_format: str = "png",
            fast_preview: bool = False,
            fast_preview_full_resolution: bool = False,
            metadata_mode: str = "full",
            target_sids: tuple[str, ...] | None = None,
            mpr_revision: int | None = None,
        ) -> bool:
            del view_id, image_format, fast_preview, fast_preview_full_resolution, metadata_mode, target_sids, mpr_revision
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
