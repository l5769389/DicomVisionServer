import asyncio
import json
from types import SimpleNamespace

from PIL import Image

from app.services.webrtc_3d_transport import (
    LatestFrameVideoTrack,
    WebRtc3DSession,
    WebRtc3DTransportManager,
    get_webrtc_3d_client_config,
)


def test_client_config_uses_configured_ice_servers(monkeypatch) -> None:
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
    assert config["iceServers"] == [
        {"urls": "stun:stun.example.test:3478"},
        {
            "urls": ["turn:turn.example.test:3478?transport=udp"],
            "username": "viewer",
            "credential": "secret",
        },
    ]


def test_latest_frame_track_drops_superseded_frames() -> None:
    async def run():
        track = LatestFrameVideoTrack()
        track.publish(Image.new("RGB", (4, 4), "red"))
        track.publish(Image.new("RGB", (4, 4), "green"))
        frame = await track.recv()
        track.stop()
        return frame.to_ndarray(format="rgb24")

    pixels = asyncio.run(run())

    assert tuple(pixels[0, 0]) == (0, 128, 0)


def test_latest_frame_track_emits_short_burst_then_waits() -> None:
    async def run() -> tuple[int, bool]:
        track = LatestFrameVideoTrack()
        track.publish(Image.new("RGB", (4, 4), "red"))
        await track.recv()
        await track.recv()
        await track.recv()
        remaining = track._remaining_burst_frames
        waiting = asyncio.create_task(track.recv())
        await asyncio.sleep(0)
        is_waiting = not waiting.done()
        waiting.cancel()
        track.stop()
        return remaining, is_waiting

    remaining, is_waiting = asyncio.run(run())

    assert remaining == 0
    assert is_waiting is True


def test_transport_is_active_only_after_peer_is_connected() -> None:
    manager = WebRtc3DTransportManager()
    track = LatestFrameVideoTrack()
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
