import asyncio

from app.schemas.dicom import (
    FourDPlaybackStartRequest,
    FourDPlaybackStopRequest,
)
from app.sockets.four_d_playback import FourDPlaybackHub


class _FakeSocketServer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object], str | None]] = []

    async def emit(self, event: str, payload: dict[str, object], to: str | None = None) -> None:
        self.events.append((event, payload, to))


def test_four_d_playback_start_advances_phase_and_stop_emits_stopped_state() -> None:
    async def scenario() -> None:
        server = _FakeSocketServer()
        hub = FourDPlaybackHub()
        hub.attach_server(server)  # type: ignore[arg-type]

        await hub.start(
            "sid-1",
            FourDPlaybackStartRequest(tabKey="tab-1", phaseIndex=0, phaseCount=3, fps=20),
        )

        for _ in range(20):
            if any(event == "four_d_phase_index" and payload.get("phaseIndex") == 1 for event, payload, _ in server.events):
                break
            await asyncio.sleep(0.01)

        await hub.stop("sid-1", FourDPlaybackStopRequest(tabKey="tab-1"))
        await asyncio.sleep(0)

        assert ("four_d_playback_state", {"tabKey": "tab-1", "isPlaying": True, "fps": 20, "phaseIndex": 0}, "sid-1") in server.events
        assert ("four_d_phase_index", {"tabKey": "tab-1", "phaseIndex": 1}, "sid-1") in server.events

        stopped_state_events = [
          payload
          for event, payload, target in server.events
          if event == "four_d_playback_state" and target == "sid-1" and payload.get("isPlaying") is False
        ]
        assert stopped_state_events
        assert stopped_state_events[-1]["tabKey"] == "tab-1"
        assert stopped_state_events[-1]["fps"] == 20
        assert int(stopped_state_events[-1]["phaseIndex"]) >= 1

        await hub.unbind_sid("sid-1")

    asyncio.run(scenario())
