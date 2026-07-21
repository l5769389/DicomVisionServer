import asyncio
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from app.core.config import get_settings
from app.services.webrtc_3d_transport import (
    LatestFrameVideoTrack,
    WebRtc3DSession,
    WebRtc3DTransportManager,
    _configure_codec_defaults,
    get_webrtc_3d_client_config,
)


@pytest.fixture
def webrtc_enabled(monkeypatch):
    monkeypatch.setenv("DICOMVISION_3D_TRANSPORT", "webrtc")
    monkeypatch.setenv("DICOMVISION_WEBRTC_VIDEO_CODEC", "vp8")
    monkeypatch.setenv("DICOMVISION_WEBRTC_VIDEO_BITRATE_BPS", "4000000")
    monkeypatch.setenv("DICOMVISION_WEBRTC_VIDEO_FPS", "60")
    monkeypatch.setenv("DICOMVISION_WEBRTC_INITIAL_BURST_FRAMES", "2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_client_config_uses_startup_transport_and_ice_servers(monkeypatch, webrtc_enabled) -> None:
    monkeypatch.setenv(
        "DICOMVISION_WEBRTC_ICE_SERVERS",
        json.dumps(
            [
                {"urls": "stun:stun.example.test:3478"},
                {
                    "urls": ["turn:turn.example.test:3478?transport=udp"],
                    "username": "viewer",
                    "credential": "secret",
                },
            ]
        ),
    )

    config = get_webrtc_3d_client_config()

    assert config["ok"] is True
    assert config["transport"] == "webrtc"
    assert config["videoCodec"] == "vp8"
    assert config["videoBitrateBps"] == 4_000_000
    assert config["videoFps"] == 60
    assert config["iceServers"] == [
        {"urls": "stun:stun.example.test:3478"},
        {
            "urls": ["turn:turn.example.test:3478?transport=udp"],
            "username": "viewer",
            "credential": "secret",
        },
    ]


@pytest.mark.parametrize(
    ("codec", "module_name"),
    (("vp8", "vpx"), ("h264", "h264")),
)
def test_codec_defaults_apply_configured_bitrate_and_fps(
    monkeypatch,
    codec: str,
    module_name: str,
) -> None:
    codec_module = pytest.importorskip(f"aiortc.codecs.{module_name}")
    monkeypatch.setattr(codec_module, "DEFAULT_BITRATE", 500_000)
    monkeypatch.setattr(codec_module, "MAX_BITRATE", 1_500_000)
    monkeypatch.setattr(codec_module, "MAX_FRAME_RATE", 30)

    _configure_codec_defaults(codec, bitrate_bps=4_000_000, fps=60)

    assert codec_module.DEFAULT_BITRATE == 4_000_000
    assert codec_module.MAX_BITRATE == 4_000_000
    assert codec_module.MAX_FRAME_RATE == 60


def test_latest_frame_track_drops_superseded_frames() -> None:
    async def run():
        track = LatestFrameVideoTrack(fps=60, initial_burst_frames=2)
        track.publish(Image.new("RGB", (4, 4), "red"))
        track.publish(Image.new("RGB", (4, 4), "green"))
        frame = await track.recv()
        track.stop()
        return frame.to_ndarray(format="rgb24")

    pixels = asyncio.run(run())

    assert tuple(pixels[0, 0]) == (0, 128, 0)


def test_latest_frame_track_bursts_only_for_first_render() -> None:
    async def run() -> tuple[int, int, bool]:
        track = LatestFrameVideoTrack(fps=60, initial_burst_frames=2)
        track.publish(Image.new("RGB", (4, 4), "red"))
        await track.recv()
        await track.recv()
        first_remaining = track._remaining_burst_frames
        track.publish(Image.new("RGB", (4, 4), "green"))
        await track.recv()
        next_remaining = track._remaining_burst_frames
        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0)
        is_waiting = not waiting.done()
        waiting.cancel()
        track.stop()
        return first_remaining, next_remaining, is_waiting

    first_remaining, next_remaining, is_waiting = asyncio.run(run())

    assert first_remaining == 0
    assert next_remaining == 0
    assert is_waiting is True


def test_latest_frame_track_resync_burst_uses_latest_pixels() -> None:
    async def run():
        track = LatestFrameVideoTrack(fps=120, initial_burst_frames=1)
        track.publish(Image.new("RGB", (4, 4), "black"))
        await track.recv()

        track.arm_resync_burst(2)
        track.publish(Image.new("RGB", (4, 4), "red"))
        first = await track.recv()
        track.publish(Image.new("RGB", (4, 4), "green"))
        second = await track.recv()
        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0)
        is_waiting = not waiting.done()
        waiting.cancel()
        track.stop()
        return (
            first.to_ndarray(format="rgb24"),
            second.to_ndarray(format="rgb24"),
            is_waiting,
        )

    first_pixels, second_pixels, is_waiting = asyncio.run(run())

    assert tuple(first_pixels[0, 0]) == (255, 0, 0)
    assert tuple(second_pixels[0, 0]) == (0, 128, 0)
    assert is_waiting is True


def test_latest_frame_track_resync_discards_pending_old_burst() -> None:
    async def run():
        track = LatestFrameVideoTrack(fps=120, initial_burst_frames=2)
        track.publish(Image.new("RGB", (4, 4), "red"))
        await track.recv()
        track.arm_resync_burst(2)

        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0)
        old_frame_was_discarded = not waiting.done()
        track.publish(Image.new("RGB", (4, 4), "green"))
        current = await waiting
        track.stop()
        return old_frame_was_discarded, current.to_ndarray(format="rgb24")

    old_frame_was_discarded, current_pixels = asyncio.run(run())

    assert old_frame_was_discarded is True
    assert tuple(current_pixels[0, 0]) == (0, 128, 0)


def test_latest_frame_track_timestamps_follow_actual_frame_arrival() -> None:
    timestamps = iter((10.0, 10.04))

    async def run() -> tuple[int, int]:
        track = LatestFrameVideoTrack(
            fps=60,
            initial_burst_frames=1,
            clock=lambda: next(timestamps),
        )
        track.publish(Image.new("RGB", (4, 4), "red"))
        first = await track.recv()
        track.publish(Image.new("RGB", (4, 4), "green"))
        second = await track.recv()
        track.stop()
        return first.pts, second.pts

    first_pts, second_pts = asyncio.run(run())

    # Forty milliseconds on a 90 kHz RTP clock is 3600 ticks. The old fixed
    # 60 fps clock incorrectly reported only 1500 ticks for the same interval.
    assert second_pts - first_pts == 3600


def test_latest_frame_track_replaces_pixels_while_encoder_is_pacing() -> None:
    async def run():
        track = LatestFrameVideoTrack(fps=30, initial_burst_frames=1)
        track.publish(Image.new("RGB", (4, 4), "black"))
        await track.recv()

        track.publish(Image.new("RGB", (4, 4), "red"))
        pending_frame = asyncio.create_task(track.recv())
        await asyncio.sleep(0.005)
        track.publish(Image.new("RGB", (4, 4), "green"))
        frame = await pending_frame
        track.stop()
        return frame.to_ndarray(format="rgb24")

    pixels = asyncio.run(run())

    assert tuple(pixels[0, 0]) == (0, 128, 0)


def test_transport_is_active_only_after_peer_is_connected(webrtc_enabled) -> None:
    manager = WebRtc3DTransportManager()
    track = LatestFrameVideoTrack(fps=60, initial_burst_frames=2)
    peer = SimpleNamespace(connectionState="connecting")
    manager._sessions[("sid-1", "view-1")] = WebRtc3DSession(
        sid="sid-1",
        view_id="view-1",
        peer=peer,
        track=track,
    )

    assert manager.get_active_sids("view-1", ("sid-1",)) == ()

    peer.connectionState = "connected"

    assert manager.get_active_sids("view-1", ("sid-1",)) == ("sid-1",)
    track.stop()


def test_request_keyframe_arms_resync_burst_and_notifies_sender(webrtc_enabled) -> None:
    class _Sender:
        def __init__(self) -> None:
            self.calls = 0

        def _send_keyframe(self) -> None:
            self.calls += 1

    manager = WebRtc3DTransportManager()
    track = LatestFrameVideoTrack(fps=60, initial_burst_frames=1)
    sender = _Sender()
    manager._sessions[("sid-1", "view-1")] = WebRtc3DSession(
        sid="sid-1",
        view_id="view-1",
        peer=SimpleNamespace(connectionState="connected"),
        track=track,
        sender=sender,
    )

    assert manager.request_keyframe("sid-1", "view-1", burst_frames=2) is True
    assert sender.calls == 1
    assert track._next_publish_burst_frames == 2
    manager.publish("sid-1", "view-1", Image.new("RGB", (4, 4), "green"))
    assert sender.calls == 2
    track.stop()


def test_request_keyframe_falls_back_to_static_handoff_without_sender_support(webrtc_enabled) -> None:
    manager = WebRtc3DTransportManager()
    track = LatestFrameVideoTrack(fps=60, initial_burst_frames=1)
    manager._sessions[("sid-1", "view-1")] = WebRtc3DSession(
        sid="sid-1",
        view_id="view-1",
        peer=SimpleNamespace(connectionState="connected"),
        track=track,
        sender=object(),
    )

    assert manager.request_keyframe("sid-1", "view-1", burst_frames=2) is False
    assert track._next_publish_burst_frames == 2
    track.stop()


def test_webp_startup_mode_never_activates_webrtc_session(monkeypatch) -> None:
    monkeypatch.setenv("DICOMVISION_3D_TRANSPORT", "webp")
    get_settings.cache_clear()
    manager = WebRtc3DTransportManager()
    track = LatestFrameVideoTrack(fps=60, initial_burst_frames=2)
    manager._sessions[("sid-1", "view-1")] = WebRtc3DSession(
        sid="sid-1",
        view_id="view-1",
        peer=SimpleNamespace(connectionState="connected"),
        track=track,
    )

    assert manager.get_active_sids("view-1", ("sid-1",)) == ()
    track.stop()
    get_settings.cache_clear()
