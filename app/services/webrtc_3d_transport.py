from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from app.core.logging import get_logger

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
    return {
        "ok": True,
        "iceServers": _load_ice_server_payloads(),
        "videoCodecs": ["VP8", "H264"],
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


class LatestFrameVideoTrack(_VideoStreamTrackBase):  # type: ignore[misc,valid-type]
    """A backpressure-free video source that keeps only the newest rendered frame."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._latest_frame: np.ndarray | None = None
        self._frame_ready = asyncio.Event()
        self._remaining_burst_frames = 0
        self._logged_first_frame = False

    def publish(self, image: Image.Image) -> None:
        self._latest_frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        # A single isolated RTP frame is not reliably presented by every video
        # decoder. Emit a short burst for each render, then become idle again.
        # New renders replace the burst immediately, so stale frames never
        # accumulate while the user is interacting.
        self._remaining_burst_frames = 3
        self._frame_ready.set()

    async def recv(self):
        while self._latest_frame is None or self._remaining_burst_frames <= 0:
            self._frame_ready.clear()
            await self._frame_ready.wait()
        pixels = self._latest_frame
        self._remaining_burst_frames -= 1
        if self._remaining_burst_frames <= 0:
            self._frame_ready.clear()
        if not self._logged_first_frame:
            logger.info("3d webrtc encoder consumed first frame shape=%s", pixels.shape)
            self._logged_first_frame = True
        pts, time_base = await self.next_timestamp()
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
        if not view_id or not sdp or description_type != "offer":
            return {"ok": False, "message": "A valid WebRTC offer and viewId are required"}
        if RTCPeerConnection is None or RTCSessionDescription is None:
            return {"ok": False, "message": "WebRTC support is unavailable on this server"}

        await self.close(sid, view_id)
        peer = RTCPeerConnection(configuration=_build_rtc_configuration())
        track = LatestFrameVideoTrack()
        transceiver = peer.addTransceiver(track, direction="sendonly")
        try:
            capabilities = RTCRtpSender.getCapabilities("video")
            # VP8 is aiortc's most portable software path and works consistently
            # across Chromium, Safari/WebKit and headless benchmark clients.
            # H.264 remains negotiated as a fallback for compatible deployments.
            preferred = [codec for codec in capabilities.codecs if codec.mimeType.lower() == "video/vp8"]
            fallback = [codec for codec in capabilities.codecs if codec.mimeType.lower() == "video/h264"]
            if preferred or fallback:
                transceiver.setCodecPreferences([*preferred, *fallback])
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
