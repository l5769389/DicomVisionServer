import asyncio
from contextlib import suppress
from types import SimpleNamespace
from time import perf_counter
from PIL import Image
import pytest

from app.sockets import runtime as socket_runtime
from app.sockets.runtime import RenderRequest, ViewSocketHub


class _SocketServerStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object], str | None]] = []

    async def emit(self, event: str, payload: dict[str, object], to: str | None = None) -> None:
        self.events.append((event, payload, to))


def test_duplicate_view_bind_is_idempotent() -> None:
    hub = ViewSocketHub()

    hub.bind_view("sid-1", "view-1")
    hub.bind_view("sid-1", "view-1")

    assert hub.get_view_sids("view-1") == ("sid-1",)
    assert hub._sid_views["sid-1"] == {"view-1"}


def test_disconnect_then_reconnect_moves_view_subscription_to_new_sid() -> None:
    hub = ViewSocketHub()
    hub.bind_sid_workspace("sid-before-reconnect", "workspace-a")
    hub.bind_view("sid-before-reconnect", "view-1")

    hub.unbind_sid("sid-before-reconnect")

    assert hub.get_view_sids("view-1") == ()
    assert hub.get_sid_workspace("sid-before-reconnect") != "workspace-a"

    hub.bind_sid_workspace("sid-after-reconnect", "workspace-a")
    hub.bind_view("sid-after-reconnect", "view-1")
    hub.bind_view("sid-after-reconnect", "view-1")

    assert hub.get_view_sids("view-1") == ("sid-after-reconnect",)
    assert hub.get_sid_workspace("sid-after-reconnect") == "workspace-a"


def test_closed_view_cannot_be_rebound_by_late_reconnect() -> None:
    hub = ViewSocketHub()
    hub.bind_view("sid-before-close", "view-1")

    hub.close_view("view-1")
    hub.bind_view("sid-after-reconnect", "view-1")

    assert hub.is_view_closed("view-1") is True
    assert hub.get_view_sids("view-1") == ()
    assert "sid-after-reconnect" not in hub._sid_views


def test_merge_render_request_keeps_full_quality_when_preview_arrives_later() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-2",)),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids == ("sid-1", "sid-2")


def test_merge_render_request_replaces_stale_final_with_newer_preview() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",), mpr_revision=5),
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",), mpr_revision=6),
    )

    assert merged.image_format == "jpeg"
    assert merged.fast_preview is True
    assert merged.mpr_revision == 6


def test_merge_render_request_promotes_pending_preview_to_full_quality() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",)),
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1", "sid-2")),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids == ("sid-1", "sid-2")


def test_merge_render_request_treats_webp_as_full_quality() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",)),
        RenderRequest(image_format="webp", fast_preview=False, target_sids=("sid-1",)),
    )

    assert merged.image_format == "webp"
    assert merged.fast_preview is False


def test_merge_render_request_keeps_broadcast_target_when_either_request_broadcasts() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=None),
        RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
    )

    assert merged.image_format == "png"
    assert merged.fast_preview is False
    assert merged.target_sids is None


def test_merge_render_request_keeps_latest_mpr_revision() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",), mpr_revision=3),
        RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",), mpr_revision=5),
    )

    assert merged.mpr_revision == 5


def test_merge_render_request_preserves_full_resolution_preview_flag() -> None:
    merged = ViewSocketHub._merge_render_request(
        RenderRequest(image_format="jpeg", fast_preview=True, fast_preview_full_resolution=False),
        RenderRequest(image_format="jpeg", fast_preview=True, fast_preview_full_resolution=True),
    )

    assert merged.image_format == "jpeg"
    assert merged.fast_preview is True
    assert merged.fast_preview_full_resolution is True


def test_emit_progress_message_targets_bound_sids() -> None:
    async def run() -> list[tuple[str, dict[str, object], str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]

        await hub._emit_progress_message(
            "view-1",
            ("sid-1", "sid-2"),
            {"phase": "preprocess", "progressPercent": 42, "message": "正在应用 3D 裁剪..."},
        )
        return server.events

    assert asyncio.run(run()) == [
        ("view_progress", {"viewId": "view-1", "phase": "preprocess", "progressPercent": 42, "message": "正在应用 3D 裁剪..."}, "sid-1"),
        ("view_progress", {"viewId": "view-1", "phase": "preprocess", "progressPercent": 42, "message": "正在应用 3D 裁剪..."}, "sid-2"),
    ]


def test_fast_preview_render_skips_progress_messages(monkeypatch) -> None:
    class _Meta:
        def model_dump(self, *, by_alias: bool = False) -> dict[str, object]:
            del by_alias
            return {"viewId": "view-1", "imageFormat": "png"}

    def fake_render_view_by_id(*args, **kwargs):
        assert kwargs["progress_callback"] is None
        return SimpleNamespace(meta=_Meta(), image_bytes=b"image")

    async def run() -> list[tuple[str, dict[str, object], str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub.bind_view("sid-1", "view-1")
        monkeypatch.setattr(socket_runtime.viewer_service, "render_view_by_id", fake_render_view_by_id)

        emitted = await hub._emit_render_message(
            "view-1",
            RenderRequest(image_format="png", fast_preview=True, target_sids=("sid-1",)),
        )

        assert emitted is True
        return server.events

    assert asyncio.run(run()) == [
        (
            "image_update",
            (
                {
                    "viewId": "view-1",
                    "imageFormat": "png",
                    "fastPreview": True,
                    "fastPreviewFullResolution": False,
                    "metadataMode": "full",
                    "renderIntent": "geometry-preview",
                },
                b"image",
            ),
            "sid-1",
        ),
    ]


def test_3d_webrtc_preview_skips_webp_encoding_and_emits_metadata(monkeypatch) -> None:
    class _Meta:
        def model_dump(self, *, by_alias: bool = False) -> dict[str, object]:
            del by_alias
            return {"viewId": "view-3d", "imageFormat": "webp", "render3dMode": "volume"}

    def fake_render_view_by_id(*args, **kwargs):
        assert kwargs["raw_3d_output"] is True
        return SimpleNamespace(
            meta=_Meta(),
            image_bytes=b"",
            raw_image=Image.new("RGB", (8, 8), "red"),
            performance_timings={},
        )

    async def run() -> list[tuple[str, dict[str, object], str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub.bind_view("sid-1", "view-3d")
        monkeypatch.setattr(socket_runtime.viewer_service, "render_view_by_id", fake_render_view_by_id)
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "get_active_sids",
            lambda _view_id, sids: sids,
        )
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "publish",
            lambda _sid, _view_id, _image: 0.2,
        )

        emitted = await hub._emit_render_message(
            "view-3d",
            RenderRequest(image_format="webp", fast_preview=True, target_sids=("sid-1",)),
        )
        assert emitted is True
        return server.events

    events = asyncio.run(run())
    assert not any(event == "image_update" for event, _payload, _sid in events)
    metadata_events = [entry for entry in events if entry[0] == "image_update_metadata"]
    assert metadata_events == [
        (
            "image_update_metadata",
            {
                "viewId": "view-3d",
                "imageFormat": "webp",
                "render3dMode": "volume",
                "fastPreview": True,
                "fastPreviewFullResolution": False,
                "metadataMode": "full",
                "renderIntent": "geometry-preview",
                "imageTransport": "webrtc",
            },
            "sid-1",
        )
    ]


def test_3d_webrtc_final_uses_lossless_webp_still_instead_of_video(monkeypatch) -> None:
    class _Meta:
        def model_dump(self, *, by_alias: bool = False) -> dict[str, object]:
            del by_alias
            return {"viewId": "view-3d", "imageFormat": "webp", "render3dMode": "volume"}

    def fake_render_view_by_id(*args, **kwargs):
        assert kwargs["raw_3d_output"] is False
        return SimpleNamespace(
            meta=_Meta(),
            image_bytes=b"lossless-webp-final",
            raw_image=Image.new("RGB", (8, 8), "red"),
            performance_timings={},
        )

    published: list[tuple[str, str]] = []
    keyframe_requests: list[tuple[str, str, int]] = []

    async def run() -> list[tuple[str, object, str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub.bind_view("sid-1", "view-3d")
        monkeypatch.setattr(socket_runtime.viewer_service, "render_view_by_id", fake_render_view_by_id)
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "get_active_sids",
            lambda _view_id, sids: sids,
        )
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "publish",
            lambda sid, view_id, _image: published.append((sid, view_id)),
        )
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "request_keyframe",
            lambda sid, view_id, *, burst_frames=2: keyframe_requests.append(
                (sid, view_id, burst_frames)
            ),
        )

        emitted = await hub._emit_render_message(
            "view-3d",
            RenderRequest(image_format="webp", fast_preview=False, target_sids=("sid-1",)),
        )
        assert emitted is True
        return server.events

    events = asyncio.run(run())
    assert published == []
    assert keyframe_requests == [("sid-1", "view-3d", 2)]
    assert not any(event == "image_update_metadata" for event, _payload, _sid in events)
    image_events = [entry for entry in events if entry[0] == "image_update"]
    assert image_events == [
        (
            "image_update",
            (
                {
                    "viewId": "view-3d",
                    "imageFormat": "webp",
                    "render3dMode": "volume",
                    "fastPreview": False,
                    "fastPreviewFullResolution": False,
                    "metadataMode": "full",
                    "renderIntent": "full",
                    "imageTransport": "webp-final",
                },
                b"lossless-webp-final",
            ),
            "sid-1",
        )
    ]


def test_3d_mixed_transports_render_once_and_emit_each_transport(monkeypatch) -> None:
    class _Meta:
        def model_dump(self, *, by_alias: bool = False) -> dict[str, object]:
            del by_alias
            return {"viewId": "view-3d", "imageFormat": "webp", "render3dMode": "volume"}

    def fake_render_view_by_id(*args, **kwargs):
        assert kwargs["raw_3d_output"] is False
        return SimpleNamespace(
            meta=_Meta(),
            image_bytes=b"webp-frame",
            raw_image=Image.new("RGB", (8, 8), "red"),
            performance_timings={},
        )

    async def run() -> list[tuple[str, object, str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(socket_runtime.viewer_service, "render_view_by_id", fake_render_view_by_id)
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "get_active_sids",
            lambda _view_id, _sids: ("sid-webrtc",),
        )
        monkeypatch.setattr(
            socket_runtime.webrtc_3d_transport_manager,
            "publish",
            lambda _sid, _view_id, _image: 0.1,
        )

        emitted = await hub._emit_render_message(
            "view-3d",
            RenderRequest(image_format="webp", fast_preview=True, target_sids=("sid-webp", "sid-webrtc")),
        )
        assert emitted is True
        return server.events

    events = asyncio.run(run())

    assert any(event == "image_update" and sid == "sid-webp" for event, _payload, sid in events)
    assert not any(event == "image_update" and sid == "sid-webrtc" for event, _payload, sid in events)
    assert any(event == "image_update_metadata" and sid == "sid-webrtc" for event, _payload, sid in events)


def test_preview_metadata_modes_drop_heavy_fields() -> None:
    meta = SimpleNamespace(
        model_dump=lambda **kwargs: {
            "viewId": "view-1",
            "imageFormat": "png",
            "cornerInfo": {"topLeft": ["A"]},
            "orientation": {"top": "A"},
            "scaleBar": {"visible": True},
            "measurements": [{"measurementId": "m"}],
            "annotations": [{"annotationId": "a"}],
            "mprSegmentationOverlay": {"regions": []},
        }
    )

    stack_pixel_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(image_format="png", fast_preview=True, metadata_mode="stack-pixel-preview", render_revision=12),
    )
    stack_geometry_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(image_format="png", fast_preview=True, metadata_mode="stack-geometry-preview"),
    )
    stack_zoom_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(
            image_format="png",
            fast_preview=True,
            fast_preview_full_resolution=True,
            metadata_mode="stack-zoom-preview",
        ),
    )
    mpr_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(image_format="png", fast_preview=True, metadata_mode="mpr-pan-zoom-preview"),
    )
    mpr_zoom_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(
            image_format="png",
            fast_preview=True,
            fast_preview_full_resolution=True,
            metadata_mode="mpr-zoom-preview",
        ),
    )
    mpr_crosshair_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(image_format="jpeg", fast_preview=True, metadata_mode="mpr-crosshair-preview"),
    )
    interaction_payload = ViewSocketHub._build_image_update_payload(
        meta,
        RenderRequest(image_format="jpeg", fast_preview=True, interaction_id="drag-1"),
    )

    assert "measurements" not in stack_pixel_payload
    assert "annotations" not in stack_pixel_payload
    assert stack_pixel_payload["fastPreview"] is True
    assert stack_pixel_payload["fastPreviewFullResolution"] is False
    assert stack_pixel_payload["metadataMode"] == "stack-pixel-preview"
    assert stack_pixel_payload["renderIntent"] == "pixel-only"
    assert stack_pixel_payload["renderRevision"] == 12
    assert "cornerInfo" in stack_pixel_payload
    assert "orientation" in stack_pixel_payload
    assert stack_geometry_payload["measurements"] == [{"measurementId": "m"}]
    assert stack_geometry_payload["annotations"] == [{"annotationId": "a"}]
    assert stack_geometry_payload["renderIntent"] == "geometry-preview"
    assert stack_zoom_payload["measurements"] == [{"measurementId": "m"}]
    assert stack_zoom_payload["annotations"] == [{"annotationId": "a"}]
    assert stack_zoom_payload["fastPreviewFullResolution"] is True
    assert stack_zoom_payload["metadataMode"] == "stack-zoom-preview"
    assert stack_zoom_payload["renderIntent"] == "geometry-preview"
    assert mpr_payload["measurements"] == [{"measurementId": "m"}]
    assert mpr_payload["annotations"] == [{"annotationId": "a"}]
    assert mpr_payload["fastPreview"] is True
    assert mpr_payload["fastPreviewFullResolution"] is False
    assert mpr_payload["metadataMode"] == "mpr-pan-zoom-preview"
    assert mpr_payload["renderIntent"] == "geometry-preview"
    assert "cornerInfo" not in mpr_payload
    assert "orientation" not in mpr_payload
    assert mpr_zoom_payload["measurements"] == [{"measurementId": "m"}]
    assert mpr_zoom_payload["annotations"] == [{"annotationId": "a"}]
    assert mpr_zoom_payload["fastPreviewFullResolution"] is True
    assert mpr_zoom_payload["metadataMode"] == "mpr-zoom-preview"
    assert mpr_zoom_payload["renderIntent"] == "geometry-preview"
    assert "cornerInfo" not in mpr_zoom_payload
    assert "orientation" not in mpr_zoom_payload
    assert mpr_crosshair_payload["imageFormat"] == "png"
    assert mpr_crosshair_payload["metadataMode"] == "mpr-crosshair-preview"
    assert mpr_crosshair_payload["renderIntent"] == "geometry-preview"
    assert "cornerInfo" not in mpr_crosshair_payload
    assert "orientation" not in mpr_crosshair_payload
    assert "scaleBar" not in mpr_crosshair_payload
    assert "measurements" not in mpr_crosshair_payload
    assert "annotations" not in mpr_crosshair_payload
    assert "mprSegmentationOverlay" not in mpr_crosshair_payload
    assert interaction_payload["interactionId"] == "drag-1"


def test_render_request_revision_is_assigned_at_schedule_time() -> None:
    hub = ViewSocketHub()

    first = hub.make_render_request("view-1")
    second = hub.make_render_request("view-1")
    other = hub.make_render_request("view-2")

    assert first.render_revision == 1
    assert second.render_revision == 2
    assert other.render_revision == 1


def test_view_preview_after_final_is_suppressed_by_render_revision() -> None:
    hub = ViewSocketHub()
    hub._remember_view_final_revision(
        "view-1",
        RenderRequest(image_format="png", fast_preview=False, render_revision=5),
    )

    assert hub._is_stale_preview_after_final(
        "view:view-1",
        "view-1",
        RenderRequest(image_format="jpeg", fast_preview=True, render_revision=4),
    ) is True
    assert hub._is_stale_preview_after_final(
        "view:view-1",
        "view-1",
        RenderRequest(image_format="jpeg", fast_preview=True, render_revision=6),
    ) is False


def test_final_view_render_discards_pending_preview(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, bool, int | None]], dict[str, dict[str, RenderRequest]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: f"view:{view_id}")
        hub._pending_render_requests["view:view-1"] = {
            "view-1": RenderRequest(image_format="jpeg", fast_preview=True, render_revision=1)
        }
        calls: list[tuple[str, bool, int | None]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.fast_preview, request.render_revision))
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        emitted = await hub.emit_render_for_view(
            "view-1",
            image_format="png",
            fast_preview=False,
            render_revision=2,
        )
        return emitted, calls, hub._pending_render_requests

    emitted, calls, pending = asyncio.run(run())

    assert emitted is True
    assert calls == [("view-1", False, 2)]
    assert pending == {}


def test_new_view_interaction_discards_pending_old_interaction(monkeypatch) -> None:
    hub = ViewSocketHub()
    monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: f"view:{view_id}")
    hub._pending_render_requests["view:view-1"] = {
        "view-1": RenderRequest(image_format="png", fast_preview=False, interaction_id="old-drag")
    }

    hub.mark_view_interaction("view-1", "new-drag")

    assert hub._pending_render_requests == {}
    assert hub._view_active_interaction_ids["view-1"] == "new-drag"


def test_old_interaction_render_is_suppressed_after_new_start(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, object, str | None]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub.bind_view("sid-1", "view-1")
        hub.mark_view_interaction("view-1", "new-drag")

        class _Meta:
            def model_dump(self, *, by_alias: bool = False) -> dict[str, object]:
                del by_alias
                return {"viewId": "view-1", "imageFormat": "jpeg"}

        monkeypatch.setattr(
            socket_runtime.viewer_service,
            "render_view_by_id",
            lambda *args, **kwargs: SimpleNamespace(meta=_Meta(), image_bytes=b"old"),
        )

        emitted = await hub._emit_render_message(
            "view-1",
            RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",), interaction_id="old-drag"),
        )
        return emitted, server.events

    emitted, events = asyncio.run(run())
    assert emitted is False
    assert events == []


def test_delayed_final_render_runs_when_no_new_interaction(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, str | None]]:
        hub = ViewSocketHub()
        calls: list[tuple[str, str, str | None]] = []

        async def fake_emit_render_for_view(view_id: str, **kwargs) -> bool:
            calls.append((view_id, kwargs["image_format"], kwargs.get("interaction_id")))
            return True

        monkeypatch.setattr(hub, "emit_render_for_view", fake_emit_render_for_view)
        task = hub.schedule_delayed_final_render_for_view(
            "view-1",
            delay_seconds=0.01,
            image_format="png",
            interaction_id="drag-1",
        )
        await asyncio.wait_for(task, timeout=1.0)
        return calls

    assert asyncio.run(run()) == [("view-1", "png", "drag-1")]


def test_adaptive_final_delay_completes_target_spacing_after_recent_preview(monkeypatch) -> None:
    hub = ViewSocketHub()
    monkeypatch.setattr(socket_runtime, "perf_counter", lambda: 10.02)
    hub._last_preview_emitted_at["view-1"] = 10.0

    assert hub.adaptive_final_render_delay(
        "view-1",
        target_preview_spacing_seconds=0.05,
        minimum_delay_seconds=0.01,
    ) == pytest.approx(0.03)


def test_adaptive_final_delay_uses_minimum_when_preview_is_old_or_missing(monkeypatch) -> None:
    hub = ViewSocketHub()
    monkeypatch.setattr(socket_runtime, "perf_counter", lambda: 10.2)
    hub._last_preview_emitted_at["view-1"] = 10.0

    assert hub.adaptive_final_render_delay(
        "view-1",
        target_preview_spacing_seconds=0.05,
        minimum_delay_seconds=0.01,
    ) == pytest.approx(0.01)
    assert hub.adaptive_final_render_delay(
        "view-missing",
        target_preview_spacing_seconds=0.05,
        minimum_delay_seconds=0.01,
    ) == pytest.approx(0.01)


def test_delayed_final_render_is_cancelled_by_next_interaction(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, str | None]]:
        hub = ViewSocketHub()
        calls: list[tuple[str, str, str | None]] = []

        async def fake_emit_render_for_view(view_id: str, **kwargs) -> bool:
            calls.append((view_id, kwargs["image_format"], kwargs.get("interaction_id")))
            return True

        monkeypatch.setattr(hub, "emit_render_for_view", fake_emit_render_for_view)
        task = hub.schedule_delayed_final_render_for_view(
            "view-1",
            delay_seconds=0.05,
            image_format="png",
            interaction_id="old-drag",
        )
        hub.mark_view_interaction("view-1", "new-drag")
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        await asyncio.sleep(0.06)
        return calls

    assert asyncio.run(run()) == []


def test_close_view_cancels_pending_and_delayed_render(monkeypatch) -> None:
    async def run() -> tuple[dict[str, dict[str, RenderRequest]], list[tuple[str, str, str | None]], bool]:
        hub = ViewSocketHub()
        calls: list[tuple[str, str, str | None]] = []

        async def fake_emit_render_for_view(view_id: str, **kwargs) -> bool:
            calls.append((view_id, kwargs["image_format"], kwargs.get("interaction_id")))
            return True

        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: f"view:{view_id}")
        monkeypatch.setattr(hub, "emit_render_for_view", fake_emit_render_for_view)
        hub._pending_render_requests["view:view-1"] = {
            "view-1": RenderRequest(image_format="jpeg", fast_preview=True)
        }
        task = hub.schedule_delayed_final_render_for_view(
            "view-1",
            delay_seconds=0.05,
            image_format="png",
            interaction_id="drag-1",
        )

        hub.close_view("view-1")
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        await asyncio.sleep(0.06)
        return hub._pending_render_requests, calls, hub.is_view_closed("view-1")

    pending, calls, is_closed = asyncio.run(run())
    assert pending == {}
    assert calls == []
    assert is_closed is True


def test_close_view_suppresses_in_flight_emit_after_render(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, dict[str, object], str | None]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub.bind_view("sid-1", "view-1")

        class _Meta:
            def model_dump(self, *, by_alias: bool = False) -> dict[str, object]:
                del by_alias
                return {"viewId": "view-1", "imageFormat": "png"}

        def fake_render_view_by_id(*args, **kwargs):
            del args, kwargs
            hub.close_view("view-1")
            return SimpleNamespace(meta=_Meta(), image_bytes=b"late")

        monkeypatch.setattr(hub, "_should_render_on_main_thread", lambda view_id: True)
        monkeypatch.setattr(socket_runtime.viewer_service, "render_view_by_id", fake_render_view_by_id)

        emitted = await hub._emit_render_message(
            "view-1",
            RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",)),
        )
        return emitted, server.events

    emitted, events = asyncio.run(run())
    assert emitted is False
    assert events == [
        ("view_progress", {"viewId": "view-1", "phase": "queued", "progressPercent": 2}, "sid-1")
    ]


def test_non_mpr_preview_worker_keeps_latest_pending_request(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, bool, str]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "view:v")

        first_render_started = asyncio.Event()
        release_first_render = asyncio.Event()
        calls: list[tuple[str, str, bool, str]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.image_format, request.fast_preview, request.metadata_mode))
            if len(calls) == 1:
                first_render_started.set()
                await release_first_render.wait()
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub.schedule_render_batch(
            ("v",),
            image_format="jpeg",
            fast_preview=True,
            metadata_mode="first",
        ) is False
        await asyncio.wait_for(first_render_started.wait(), timeout=1.0)
        assert await hub.schedule_render_batch(
            ("v",),
            image_format="jpeg",
            fast_preview=True,
            metadata_mode="second",
        ) is False
        assert await hub.schedule_render_batch(
            ("v",),
            image_format="png",
            fast_preview=True,
            metadata_mode="latest",
        ) is False

        release_first_render.set()
        worker = hub._preview_worker_tasks.get("view:v")
        if worker is not None:
            await asyncio.wait_for(worker, timeout=1.0)
        return calls

    assert asyncio.run(run()) == [
        ("v", "jpeg", True, "first"),
        ("v", "png", True, "latest"),
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
        assert await hub.emit_render_for_view("v-cor", image_format="png", fast_preview=False) is True
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


def test_mpr_group_final_request_discards_pending_previews(monkeypatch) -> None:
    async def run() -> dict[str, RenderRequest]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        first_render_started = asyncio.Event()
        release_first_render = asyncio.Event()

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            if view_id == "v-ax" and request.fast_preview:
                first_render_started.set()
                await release_first_render.wait()
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        first_task = asyncio.create_task(
            hub.emit_render_for_view("v-ax", image_format="jpeg", fast_preview=True)
        )
        await first_render_started.wait()

        assert await hub.emit_render_for_view("v-cor", image_format="jpeg", fast_preview=True) is False
        assert await hub.emit_render_for_view("v-sag", image_format="jpeg", fast_preview=True) is False
        assert await hub.emit_render_for_view("v-cor", image_format="png", fast_preview=False) is True

        pending = dict(hub._pending_render_requests.get("mpr-group:g", {}))
        release_first_render.set()
        await first_task
        return pending

    pending = asyncio.run(run())
    assert pending == {}


def test_mpr_preview_older_than_current_revision_is_emitted_during_drag(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, object, str | None]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]

        class _Meta:
            mpr_revision: int | None = 3

            def model_dump(self, *, by_alias: bool = False):
                del by_alias
                return {"viewId": "v-cor", "imageFormat": "jpeg", "mprRevision": self.mpr_revision}

        monkeypatch.setattr(
            "app.sockets.runtime.view_registry.get",
            lambda view_id: SimpleNamespace(
                view_id=view_id,
                view_group=SimpleNamespace(group_id="g", group_type="MPR", mpr_revision=3),
            ),
        )
        monkeypatch.setattr(
            "app.sockets.runtime.viewer_service.render_view_by_id",
            lambda *args, **kwargs: SimpleNamespace(meta=_Meta(), image_bytes=b"old-preview"),
        )

        emitted = await hub.emit_render_for_view(
            "v-cor",
            image_format="jpeg",
            fast_preview=True,
            target_sids=("sid-1",),
            mpr_revision=2,
        )
        return emitted, server.events

    emitted, events = asyncio.run(run())
    assert emitted is True
    image_updates = [payload for event_name, payload, _ in events if event_name == "image_update"]
    assert len(image_updates) == 1
    assert image_updates[0][0]["mprRevision"] == 3


def test_mpr_preview_is_not_emitted_when_final_is_waiting(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, object, str | None]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub._pending_render_requests["mpr-group:g"] = {
            "v-cor": RenderRequest(image_format="png", fast_preview=False, target_sids=("sid-1",), mpr_revision=6)
        }

        class _Meta:
            mpr_revision: int | None = 5

            def model_dump(self, *, by_alias: bool = False):
                del by_alias
                return {"viewId": "v-cor", "imageFormat": "jpeg", "mprRevision": self.mpr_revision}

        monkeypatch.setattr(
            "app.sockets.runtime.view_registry.get",
            lambda view_id: SimpleNamespace(
                view_id=view_id,
                view_group=SimpleNamespace(group_id="g", group_type="MPR", mpr_revision=6),
            ),
        )
        monkeypatch.setattr(
            "app.sockets.runtime.viewer_service.render_view_by_id",
            lambda *args, **kwargs: SimpleNamespace(meta=_Meta(), image_bytes=b"old-preview"),
        )

        emitted = await hub._emit_render_message(
            "v-cor",
            RenderRequest(image_format="jpeg", fast_preview=True, target_sids=("sid-1",), mpr_revision=5),
        )
        return emitted, server.events

    emitted, events = asyncio.run(run())
    assert emitted is False
    image_updates = [payload for event_name, payload, _ in events if event_name == "image_update"]
    assert image_updates == []


def test_mpr_final_preempts_locked_preview_and_suppresses_preview_emit(monkeypatch) -> None:
    async def run() -> tuple[list[tuple[str, str]], list[tuple[str, object, str | None]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        preview_started = asyncio.Event()
        release_preview = asyncio.Event()
        render_calls: list[tuple[str, str]] = []

        class _Meta:
            def __init__(self, image_format: str, revision: int):
                self.image_format = image_format
                self.mpr_revision = revision

            def model_dump(self, *, by_alias: bool = False):
                del by_alias
                return {
                    "viewId": "v-cor",
                    "imageFormat": self.image_format,
                    "mprRevision": self.mpr_revision,
                }

        async def fake_to_thread(func, view_id: str, **kwargs):
            del func
            image_format = kwargs["image_format"]
            render_calls.append((view_id, image_format))
            if image_format == "jpeg":
                preview_started.set()
                await release_preview.wait()
                return SimpleNamespace(meta=_Meta("jpeg", 5), image_bytes=b"preview")
            return SimpleNamespace(meta=_Meta("png", 6), image_bytes=b"final")

        monkeypatch.setattr("app.sockets.runtime.asyncio.to_thread", fake_to_thread)

        preview_task = asyncio.create_task(
            hub.emit_render_for_view(
                "v-cor",
                image_format="jpeg",
                fast_preview=True,
                target_sids=("sid-1",),
                mpr_revision=5,
            )
        )
        await preview_started.wait()

        final_result = await hub.emit_render_for_view(
            "v-cor",
            image_format="png",
            fast_preview=False,
            target_sids=("sid-1",),
            mpr_revision=6,
        )
        release_preview.set()
        preview_result = await preview_task

        assert final_result is True
        assert preview_result is False
        return render_calls, server.events

    render_calls, events = asyncio.run(run())
    assert render_calls == [("v-cor", "jpeg"), ("v-cor", "png")]
    image_updates = [payload for event_name, payload, _ in events if event_name == "image_update"]
    assert len(image_updates) == 1
    assert image_updates[0][0]["imageFormat"] == "png"


def test_emit_render_message_sends_extra_image_bytes_as_third_socket_argument(monkeypatch) -> None:
    async def run() -> list[tuple[str, dict[str, object], str | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        hub.bind_view("sid-1", "fusion-overlay")

        class _Meta:
            def model_dump(self, *, by_alias: bool = False):
                del by_alias
                return {
                    "viewId": "fusion-overlay",
                    "imageFormat": "png",
                    "fusionComposite": {"mode": "ctPetLayers", "revision": 1},
                }

        async def fake_to_thread(func, view_id: str, **kwargs):
            del func, view_id, kwargs
            return SimpleNamespace(
                meta=_Meta(),
                image_bytes=b"ct",
                extra_image_bytes={"pet": b"pet"},
            )

        monkeypatch.setattr("app.sockets.runtime.asyncio.to_thread", fake_to_thread)
        await hub.emit_render_for_view(
            "fusion-overlay",
            image_format="png",
            fast_preview=False,
            target_sids=("sid-1",),
        )
        return server.events

    events = asyncio.run(run())
    image_updates = [payload for event_name, payload, _ in events if event_name == "image_update"]
    assert len(image_updates) == 1
    assert image_updates[0][0]["fusionComposite"]["mode"] == "ctPetLayers"
    assert image_updates[0][1] == b"ct"
    assert image_updates[0][2] == {"pet": b"pet"}


def test_mpr_low_resolution_preview_below_final_revision_is_dropped_before_render(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, str]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")
        render_calls: list[tuple[str, str]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            render_calls.append((view_id, request.image_format))
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub.emit_render_for_view(
            "v-cor",
            image_format="png",
            fast_preview=False,
            target_sids=("sid-1",),
            mpr_revision=8,
        ) is True
        preview_result = await hub.emit_render_for_view(
            "v-cor",
            image_format="jpeg",
            fast_preview=True,
            target_sids=("sid-1",),
            mpr_revision=7,
        )
        return preview_result, render_calls

    preview_result, render_calls = asyncio.run(run())
    assert preview_result is False
    assert render_calls == [("v-cor", "png")]


def test_schedule_mpr_preview_at_final_revision_is_rendered(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, int | None, bool]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")
        render_calls: list[tuple[str, str, int | None, bool]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            render_calls.append(
                (view_id, request.image_format, request.mpr_revision, request.fast_preview_full_resolution)
            )
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub.schedule_render_batch(
            ("v-cor",),
            image_format="png",
            fast_preview=False,
            target_sids=("sid-1",),
            mpr_revision=8,
        ) is True
        preview_result = await hub.schedule_render_batch(
            ("v-cor",),
            image_format="jpeg",
            fast_preview=True,
            target_sids=("sid-1",),
            mpr_revision=8,
        )
        worker = hub._mpr_preview_worker_tasks.get("mpr-group:g")
        if worker is not None:
            await asyncio.wait_for(worker, timeout=1.0)
        assert preview_result is False
        return render_calls

    assert asyncio.run(run()) == [
        ("v-cor", "png", 8, False),
        ("v-cor", "jpeg", 8, False),
    ]


def test_mpr_preview_after_final_revision_starts_new_interaction(monkeypatch) -> None:
    async def run() -> tuple[bool, list[tuple[str, str]]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")
        render_calls: list[tuple[str, str]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            render_calls.append((view_id, request.image_format))
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub.emit_render_for_view(
            "v-cor",
            image_format="png",
            fast_preview=False,
            target_sids=("sid-1",),
            mpr_revision=8,
        ) is True
        preview_result = await hub.emit_render_for_view(
            "v-cor",
            image_format="jpeg",
            fast_preview=True,
            target_sids=("sid-1",),
            mpr_revision=9,
        )
        return preview_result, render_calls

    preview_result, render_calls = asyncio.run(run())
    assert preview_result is True
    assert render_calls == [("v-cor", "png"), ("v-cor", "jpeg")]


def test_mpr_group_queue_coalesces_sibling_initial_requests(monkeypatch) -> None:
    async def run() -> tuple[list[str], bool, bool]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        coronal_started = asyncio.Event()
        sagittal_started = asyncio.Event()
        release_renders = asyncio.Event()
        calls: list[str] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            del request
            calls.append(view_id)
            if view_id == "v-cor":
                coronal_started.set()
            if view_id == "v-sag":
                sagittal_started.set()
            await release_renders.wait()
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        coronal_task = asyncio.create_task(
            hub.emit_render_for_view("v-cor", image_format="jpeg", fast_preview=True)
        )
        sagittal_task = asyncio.create_task(
            hub.emit_render_for_view("v-sag", image_format="jpeg", fast_preview=True)
        )

        await asyncio.wait_for(coronal_started.wait(), timeout=1.0)
        await asyncio.wait_for(sagittal_started.wait(), timeout=1.0)
        coronal_done_before_release = coronal_task.done()
        sagittal_done_before_release = sagittal_task.done()
        release_renders.set()
        assert await coronal_task is True
        assert await sagittal_task is False
        return calls, coronal_done_before_release, sagittal_done_before_release

    calls, coronal_done_before_release, sagittal_done_before_release = asyncio.run(run())
    assert calls == ["v-cor", "v-sag"]
    assert coronal_done_before_release is False
    assert sagittal_done_before_release is True


def test_drain_skips_current_preview_when_final_is_pending(monkeypatch) -> None:
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


def test_drain_replaces_current_preview_with_latest_pending_preview(monkeypatch) -> None:
    async def run() -> list[tuple[str, int | None]]:
        hub = ViewSocketHub()
        calls: list[tuple[str, int | None]] = []
        hub._pending_render_requests["mpr-group:g"] = {
            "v-cor": RenderRequest(
                image_format="jpeg",
                fast_preview=True,
                target_sids=("sid-1",),
                mpr_revision=5,
            )
        }

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.mpr_revision))
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub._drain_render_requests(
            "mpr-group:g",
            "v-cor",
            RenderRequest(
                image_format="jpeg",
                fast_preview=True,
                target_sids=("sid-1",),
                mpr_revision=3,
            ),
        ) is True
        return calls

    assert asyncio.run(run()) == [("v-cor", 5)]


def test_schedule_mpr_preview_batch_keeps_only_latest_pending(monkeypatch) -> None:
    async def run() -> list[tuple[str, int | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        first_batch_started_count = 0
        first_batch_started = asyncio.Event()
        release_first_batch = asyncio.Event()
        latest_batch_started = asyncio.Event()
        calls: list[tuple[str, int | None]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            nonlocal first_batch_started_count
            calls.append((view_id, request.mpr_revision))
            if request.mpr_revision == 1:
                first_batch_started_count += 1
                if first_batch_started_count == 2:
                    first_batch_started.set()
                await release_first_batch.wait()
            if request.mpr_revision == 3:
                latest_batch_started.set()
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub.schedule_render_batch(
            ("v-cor", "v-sag"),
            image_format="jpeg",
            fast_preview=True,
            mpr_revision=1,
        ) is False
        await asyncio.wait_for(first_batch_started.wait(), timeout=1.0)

        assert await hub.schedule_render_batch(
            ("v-cor", "v-sag"),
            image_format="jpeg",
            fast_preview=True,
            mpr_revision=2,
        ) is False
        assert await hub.schedule_render_batch(
            ("v-cor", "v-sag"),
            image_format="jpeg",
            fast_preview=True,
            mpr_revision=3,
        ) is False

        release_first_batch.set()
        await asyncio.wait_for(latest_batch_started.wait(), timeout=1.0)
        worker = hub._mpr_preview_worker_tasks.get("mpr-group:g")
        if worker is not None:
            await asyncio.wait_for(worker, timeout=1.0)
        return calls

    assert asyncio.run(run()) == [
        ("v-cor", 1),
        ("v-sag", 1),
        ("v-cor", 3),
        ("v-sag", 3),
    ]


def test_schedule_mpr_preview_worker_does_not_sleep_on_previous_batch_interval(monkeypatch) -> None:
    async def run() -> list[tuple[str, str, int | None]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        calls: list[tuple[str, str, int | None]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.image_format, request.mpr_revision))
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)
        hub._last_mpr_preview_batch_started_at["mpr-group:g"] = perf_counter()

        assert await hub.schedule_render_batch(
            ("v-cor",),
            image_format="jpeg",
            fast_preview=True,
            mpr_revision=5,
        ) is False
        worker = hub._mpr_preview_worker_tasks.get("mpr-group:g")
        if worker is not None:
            await asyncio.wait_for(worker, timeout=1.0)
        return calls

    assert asyncio.run(run()) == [
        ("v-cor", "jpeg", 5),
    ]


def test_schedule_mpr_final_batch_failure_does_not_block_siblings(monkeypatch) -> None:
    async def run() -> list[tuple[str, str]]:
        hub = ViewSocketHub()
        server = _SocketServerStub()
        hub.attach_server(server)  # type: ignore[arg-type]
        monkeypatch.setattr(hub, "_resolve_render_queue_key", lambda view_id: "mpr-group:g")

        calls: list[tuple[str, str]] = []

        async def fake_emit_render_message(view_id: str, request: RenderRequest) -> bool:
            calls.append((view_id, request.image_format))
            if view_id == "v-cor":
                raise RuntimeError("render failed")
            return True

        monkeypatch.setattr(hub, "_emit_render_message", fake_emit_render_message)

        assert await hub.schedule_render_batch(
            ("v-cor", "v-sag"),
            image_format="png",
            fast_preview=False,
        ) is True
        return calls

    assert asyncio.run(run()) == [("v-cor", "png"), ("v-sag", "png")]
