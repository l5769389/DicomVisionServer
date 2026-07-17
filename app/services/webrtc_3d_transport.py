from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import fractions
import json
import os
from time import perf_counter
import time
from typing import Any

import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
        RTCRtpSender,
        VideoStreamTrack,
    )
    from av import VideoFrame
except ImportError:  # pragma: no cover - reported through signaling in incomplete installs.
    RTCConfiguration = RTCIceServer = RTCPeerConnection = RTCSessionDescription = None  # type: ignore[assignment]
    RTCRtpSender = VideoStreamTrack = VideoFrame = None  # type: ignore[assignment]

_VideoStreamTrackBase = VideoStreamTrack if VideoStreamTrack is not None else object


WEBRTC_3D_ICE_SERVERS_ENV = "DICOMVISION_WEBRTC_ICE_SERVERS"


def _transport_settings():
    return get_settings()


def is_webrtc_3d_enabled() -> bool:
    return _transport_settings().normalized_three_d_transport == "webrtc"


def _load_ice_server_payloads() -> list[dict[str, object]]:
    raw = os.getenv(WEBRTC_3D_ICE_SERVERS_ENV, "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("invalid %s JSON; using host ICE candidates only", WEBRTC_3D_ICE_SERVERS_ENV)
        return []
    if not isinstance(value, list):
        return []

    payloads: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        urls = item.get("urls")
        if not isinstance(urls, (str, list)):
            continue
        payload: dict[str, object] = {"urls": urls}
        if isinstance(item.get("username"), str):
            payload["username"] = item["username"]
        if isinstance(item.get("credential"), str):
            payload["credential"] = item["credential"]
        payloads.append(payload)
    return payloads


def get_webrtc_3d_client_config() -> dict[str, object]:
    settings = _transport_settings()
    codec = settings.normalized_webrtc_video_codec
    return {
        "ok": True,
        "transport": settings.normalized_three_d_transport,
        "iceServers": _load_ice_server_payloads(),
        "videoCodecs": [codec.upper()],
        "videoCodec": codec,
        "videoBitrateBps": settings.normalized_webrtc_video_bitrate_bps,
        "videoFps": settings.normalized_webrtc_video_fps,
    }


def _build_rtc_configuration():
    if RTCConfiguration is None or RTCIceServer is None:
        raise RuntimeError("WebRTC support is unavailable; install the aiortc server dependency")
    ice_servers = [
        RTCIceServer(
            urls=item["urls"],
            username=str(item.get("username") or ""),
            credential=str(item.get("credential") or ""),
        )
        for item in _load_ice_server_payloads()
    ]
    return RTCConfiguration(iceServers=ice_servers)


def _configure_codec_defaults(codec: str, bitrate_bps: int, fps: int) -> None:
    """Configure aiortc's encoder before it is lazily created by the sender."""
    if codec == "h264":
        from aiortc.codecs import h264

        h264.DEFAULT_BITRATE = bitrate_bps
        h264.MAX_BITRATE = max(h264.MAX_BITRATE, bitrate_bps)
        h264.MAX_FRAME_RATE = fps
        return

    from aiortc.codecs import vpx

    vpx.DEFAULT_BITRATE = bitrate_bps
    vpx.MAX_BITRATE = max(vpx.MAX_BITRATE, bitrate_bps)
    vpx.MAX_FRAME_RATE = fps


class LatestFrameVideoTrack(_VideoStreamTrackBase):  # type: ignore[misc,valid-type]
    """A backpressure-free video source that keeps only the newest rendered frame."""

    kind = "video"

    def __init__(
        self,
        *,
        fps: int | None = None,
        initial_burst_frames: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        settings = _transport_settings()
        self._fps = fps or settings.normalized_webrtc_video_fps
        self._initial_burst_frames = initial_burst_frames or settings.normalized_webrtc_initial_burst_frames
        self._latest_frame: np.ndarray | None = None
        self._frame_ready = asyncio.Event()
        self._remaining_burst_frames = 0
        self._logged_first_frame = False
        self._has_published_frame = False
        self._timestamp_origin: float | None = None
        self._last_frame_at: float | None = None
        self._last_timestamp = -1
        self._clock = clock or time.monotonic

    def publish(self, image: Image.Image) -> None:
        self._latest_frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        # Some decoders need more than one packetized frame to present the
        # first image. Only the first render gets a short startup burst. Every
        # later render is a single latest-state frame, preventing the browser
        # jitter buffer from continuing to play old camera poses after release.
        self._remaining_burst_frames = self._initial_burst_frames if not self._has_published_frame else 1
        self._has_published_frame = True
        self._frame_ready.set()

    async def _next_low_latency_timestamp(self) -> tuple[int, fractions.Fraction]:
        time_base = fractions.Fraction(1, 90_000)
        frame_interval = 1.0 / self._fps
        now = self._clock()
        if self._last_frame_at is not None:
            wait = self._last_frame_at + frame_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = self._clock()

        self._last_frame_at = now
        if self._timestamp_origin is None:
            self._timestamp_origin = now
        # The renderer usually produces fewer frames than the configured
        # encoder ceiling. Derive PTS from actual presentation time so the
        # browser does not receive a 25-30 fps stream labelled as 60 fps and
        # compensate by growing its jitter buffer.
        elapsed_ticks = round((now - self._timestamp_origin) * 90_000)
        self._last_timestamp = max(self._last_timestamp + 1, elapsed_ticks)
        return self._last_timestamp, time_base

    async def recv(self):
        while self._latest_frame is None or self._remaining_burst_frames <= 0:
            self._frame_ready.clear()
            await self._frame_ready.wait()
        pts, time_base = await self._next_low_latency_timestamp()
        # A newer render may arrive while the track is pacing the encoder.
        # Select pixels after that wait so the old camera pose is superseded
        # before it ever enters the encoder or browser jitter buffer.
        pixels = self._latest_frame
        self._remaining_burst_frames -= 1
        if self._remaining_burst_frames <= 0:
            self._frame_ready.clear()
        if not self._logged_first_frame:
            logger.info("3d webrtc encoder consumed first frame shape=%s", pixels.shape)
            self._logged_first_frame = True
        frame = VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


@dataclass
class WebRtc3DSession:
    sid: str
    view_id: str
    peer: Any
    track: LatestFrameVideoTrack


class WebRtc3DTransportManager:
    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], WebRtc3DSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(sid: str, view_id: str) -> tuple[str, str]:
        return sid, view_id

    async def create_answer(self, sid: str, view_id: str, sdp: str, description_type: str) -> dict[str, object]:
        if not is_webrtc_3d_enabled():
            return {"ok": False, "message": "The server is configured for WebP 3D transport"}
        if not view_id or not sdp or description_type != "offer":
            return {"ok": False, "message": "A valid WebRTC offer and viewId are required"}
        if RTCPeerConnection is None or RTCSessionDescription is None:
            return {"ok": False, "message": "WebRTC support is unavailable on this server"}

        await self.close(sid, view_id)
        peer = RTCPeerConnection(configuration=_build_rtc_configuration())
        settings = _transport_settings()
        track = LatestFrameVideoTrack(
            fps=settings.normalized_webrtc_video_fps,
            initial_burst_frames=settings.normalized_webrtc_initial_burst_frames,
        )
        transceiver = peer.addTransceiver(track, direction="sendonly")
        try:
            _configure_codec_defaults(
                settings.normalized_webrtc_video_codec,
                settings.normalized_webrtc_video_bitrate_bps,
                settings.normalized_webrtc_video_fps,
            )
            capabilities = RTCRtpSender.getCapabilities("video")
            # Use exactly the codec selected at server startup. Keeping one
            # codec avoids negotiation-dependent quality changes across clients.
            preferred_mime = f"video/{settings.normalized_webrtc_video_codec}"
            preferred = [codec for codec in capabilities.codecs if codec.mimeType.lower() == preferred_mime]
            if preferred:
                transceiver.setCodecPreferences(preferred)
        except Exception:
            logger.debug("failed to set WebRTC 3D codec preferences", exc_info=True)

        session = WebRtc3DSession(sid=sid, view_id=view_id, peer=peer, track=track)
        async with self._lock:
            self._sessions[self._key(sid, view_id)] = session

        @peer.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            logger.info(
                "3d webrtc connection state sid=%s view_id=%s state=%s",
                sid,
                view_id,
                peer.connectionState,
            )
            if peer.connectionState in {"failed", "closed"}:
                await self.close(sid, view_id, expected_peer=peer)

        @peer.on("iceconnectionstatechange")
        async def on_ice_connection_state_change() -> None:
            logger.info(
                "3d webrtc ICE state sid=%s view_id=%s state=%s",
                sid,
                view_id,
                peer.iceConnectionState,
            )

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=description_type))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            local_description = peer.localDescription
            if local_description is None:
                raise RuntimeError("WebRTC answer was not created")
            return {
                "ok": True,
                "viewId": view_id,
                "sdp": local_description.sdp,
                "type": local_description.type,
            }
        except Exception:
            await self.close(sid, view_id, expected_peer=peer)
            raise

    def get_active_sids(self, view_id: str, candidate_sids: tuple[str, ...]) -> tuple[str, ...]:
        if not is_webrtc_3d_enabled():
            return ()
        active: list[str] = []
        for sid in candidate_sids:
            session = self._sessions.get(self._key(sid, view_id))
            if session is None:
                continue
            # Keep WebP as the visible fallback until ICE and DTLS have
            # actually connected. Treating a newly-created peer as active can
            # swallow the first render before a browser is able to receive it.
            if session.peer.connectionState == "connected":
                active.append(sid)
        return tuple(active)

    def publish(self, sid: str, view_id: str, image: Image.Image) -> float | None:
        session = self._sessions.get(self._key(sid, view_id))
        if session is None or session.peer.connectionState in {"failed", "closed"}:
            return None
        started_at = perf_counter()
        session.track.publish(image)
        return (perf_counter() - started_at) * 1000.0

    async def close(self, sid: str, view_id: str, *, expected_peer: Any | None = None) -> None:
        key = self._key(sid, view_id)
        async with self._lock:
            session = self._sessions.get(key)
            if session is None or (expected_peer is not None and session.peer is not expected_peer):
                return
            self._sessions.pop(key, None)
        session.track.stop()
        await session.peer.close()

    async def close_sid(self, sid: str) -> None:
        keys = [key for key in self._sessions if key[0] == sid]
        await asyncio.gather(*(self.close(*key) for key in keys), return_exceptions=True)

    async def close_view(self, view_id: str) -> None:
        keys = [key for key in self._sessions if key[1] == view_id]
        await asyncio.gather(*(self.close(*key) for key in keys), return_exceptions=True)

    def schedule_close_view(self, view_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.close_view(view_id))


webrtc_3d_transport_manager = WebRtc3DTransportManager()
