from __future__ import annotations

from typing import Any, Dict

from sparket.validator.events.event import Event

# Provider events are any that are triggered by updates to game data from the ground truth provider. Currently, the only necessary event is the game ended event.


class GameEnded(Event):
    def __init__(self, *, event_id_str: str, league: str | None = None, ended_at_ts: int | None = None, extra: Dict[str, Any] | None = None):
        payload: Dict[str, Any] = {"event_id": event_id_str}
        if league is not None:
            payload["league"] = league
        if ended_at_ts is not None:
            payload["ended_at_ts"] = ended_at_ts
        if extra:
            payload.update(extra)
        # Deterministic id: type + event_id + minute bucket
        minute_bucket = Event.bucket_minute(ended_at_ts or 0)
        ev_id = Event.make_id("provider.gameEnded", event_id_str, minute_bucket)
        super().__init__(event_id=ev_id, event_type="provider.gameEnded", event_data=payload)
