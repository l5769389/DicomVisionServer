from __future__ import annotations

import asyncio
from dataclasses import dataclass

import socketio

from app.core.logging import get_logger
from app.schemas.dicom import (
    FourDPlaybackFpsRequest,
    FourDPlaybackPhaseEvent,
    FourDPlaybackStartRequest,
    FourDPlaybackStateEvent,
    FourDPlaybackStopRequest,
)


logger = get_logger(__name__)


def _normalize_phase_count(value: int) -> int:
    return max(1, int(value))


def _normalize_fps(value: int | None) -> int:
    try:
        numeric_value = int(value or 0)
    except (TypeError, ValueError):
        numeric_value = 0
    return max(1, min(30, numeric_value or 2))


def _normalize_phase_index(value: int, phase_count: int) -> int:
    normalized_phase_count = _normalize_phase_count(phase_count)
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        numeric_value = 0
    return max(0, min(numeric_value, normalized_phase_count - 1))


@dataclass
class FourDPlaybackSession:
    sid: str
    tab_key: str
    phase_count: int
    fps: int
    current_phase_index: int = 0
    is_playing: bool = False
    task: asyncio.Task[None] | None = None


class FourDPlaybackHub:
    def __init__(self) -> None:
        self._server: socketio.AsyncServer | None = None
        self._sessions: dict[tuple[str, str], FourDPlaybackSession] = {}

    def attach_server(self, server: socketio.AsyncServer) -> None:
        self._server = server

    @staticmethod
    def _get_session_key(sid: str, tab_key: str) -> tuple[str, str]:
        return sid, tab_key

    def _get_or_create_session(
        self,
        sid: str,
        tab_key: str,
        *,
        phase_count: int,
        fps: int | None = None,
        phase_index: int | None = None,
    ) -> FourDPlaybackSession:
        session_key = self._get_session_key(sid, tab_key)
        normalized_phase_count = _normalize_phase_count(phase_count)
        session = self._sessions.get(session_key)
        if session is None:
            session = FourDPlaybackSession(
                sid=sid,
                tab_key=tab_key,
                phase_count=normalized_phase_count,
                fps=_normalize_fps(fps),
                current_phase_index=_normalize_phase_index(phase_index or 0, normalized_phase_count),
            )
            self._sessions[session_key] = session
            return session

        session.phase_count = normalized_phase_count
        if fps is not None:
            session.fps = _normalize_fps(fps)
        if phase_index is not None:
            session.current_phase_index = _normalize_phase_index(phase_index, normalized_phase_count)
        else:
            session.current_phase_index = _normalize_phase_index(session.current_phase_index, normalized_phase_count)
        return session

    @staticmethod
    def _cancel_session_task(session: FourDPlaybackSession) -> None:
        task = session.task
        if task is None:
            return
        if not task.done():
            task.cancel()
        session.task = None

    async def _emit_phase_event(self, session: FourDPlaybackSession) -> None:
        if self._server is None:
            return
        payload = FourDPlaybackPhaseEvent(
            tabKey=session.tab_key,
            phaseIndex=session.current_phase_index,
        )
        logger.info(
            "socket four_d_phase_index sid=%s tab_key=%s phase_index=%s",
            session.sid,
            session.tab_key,
            session.current_phase_index,
        )
        await self._server.emit("four_d_phase_index", payload.model_dump(by_alias=True), to=session.sid)

    async def _emit_playback_state_event(
        self,
        sid: str,
        tab_key: str,
        *,
        is_playing: bool,
        fps: int | None = None,
        phase_index: int | None = None,
    ) -> None:
        if self._server is None:
            return
        payload = FourDPlaybackStateEvent(
            tabKey=tab_key,
            isPlaying=is_playing,
            fps=fps,
            phaseIndex=phase_index,
        )
        logger.info(
            "socket four_d_playback_state sid=%s tab_key=%s is_playing=%s fps=%s phase_index=%s",
            sid,
            tab_key,
            is_playing,
            fps,
            phase_index,
        )
        await self._server.emit("four_d_playback_state", payload.model_dump(by_alias=True), to=sid)

    async def _emit_session_playback_state(self, session: FourDPlaybackSession) -> None:
        await self._emit_playback_state_event(
            session.sid,
            session.tab_key,
            is_playing=session.is_playing,
            fps=session.fps,
            phase_index=session.current_phase_index,
        )

    async def _run_playback_loop(self, session_key: tuple[str, str]) -> None:
        try:
            while True:
                session = self._sessions.get(session_key)
                if session is None or not session.is_playing or session.phase_count <= 1:
                    return

                await asyncio.sleep(1.0 / max(session.fps, 1))

                session = self._sessions.get(session_key)
                if session is None or not session.is_playing or session.phase_count <= 1:
                    return

                session.current_phase_index = (session.current_phase_index + 1) % max(session.phase_count, 1)
                await self._emit_phase_event(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("4D playback loop failed sid=%s tab_key=%s", session_key[0], session_key[1])
            session = self._sessions.get(session_key)
            if session is None:
                return
            session.is_playing = False
            session.task = None
            await self._emit_session_playback_state(session)

    async def start(self, sid: str, payload: FourDPlaybackStartRequest) -> None:
        session = self._get_or_create_session(
            sid,
            payload.tab_key,
            phase_count=payload.phase_count,
            fps=payload.fps,
            phase_index=payload.phase_index,
        )

        self._cancel_session_task(session)
        session.is_playing = session.phase_count > 1
        if session.is_playing:
            session.task = asyncio.create_task(self._run_playback_loop(self._get_session_key(sid, payload.tab_key)))

        await self._emit_session_playback_state(session)

    async def stop(self, sid: str, payload: FourDPlaybackStopRequest) -> None:
        session = self._sessions.get(self._get_session_key(sid, payload.tab_key))
        if session is None:
            await self._emit_playback_state_event(sid, payload.tab_key, is_playing=False)
            return

        self._cancel_session_task(session)
        session.is_playing = False
        await self._emit_session_playback_state(session)

    async def update_fps(self, sid: str, payload: FourDPlaybackFpsRequest) -> None:
        session = self._sessions.get(self._get_session_key(sid, payload.tab_key))
        if session is None:
            return

        session.fps = _normalize_fps(payload.fps)
        await self._emit_session_playback_state(session)

    async def unbind_sid(self, sid: str) -> None:
        for session_key, session in list(self._sessions.items()):
            if session.sid != sid:
                continue
            self._cancel_session_task(session)
            self._sessions.pop(session_key, None)


four_d_playback_hub = FourDPlaybackHub()
