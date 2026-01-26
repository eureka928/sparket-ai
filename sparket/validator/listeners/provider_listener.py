from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List

import bittensor as bt
from sqlalchemy import text

from sparket.validator.events.provider_events import GameEnded


_SELECT_RECENT_FINISHED = text(
    """
    SELECT event_id
    FROM event
    WHERE status = :status
      AND start_time_utc BETWEEN :cutoff AND :now
    ORDER BY start_time_utc DESC
    LIMIT 1000
    """
)


async def detect_game_ended_and_emit(*, database: Any, window_minutes: int = 120) -> List[GameEnded]:
    """
    Minimal provider listener: find events recently finished and emit GameEnded events.
    Uses a time window on start_time_utc as a coarse filter (placeholder until provider fetchers set explicit end times).
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=window_minutes)
    try:
        rows = await database.read(_SELECT_RECENT_FINISHED, params={
            "status": "finished",
            "cutoff": cutoff,
            "now": now,
        })
    except Exception as e:
        bt.logging.warning({"provider_listener_error": str(e)})
        return []

    events: List[GameEnded] = []
    for r in rows:
        ev_id = str(r["event_id"]) if isinstance(r, dict) and "event_id" in r else str(r[0])
        ge = GameEnded(event_id_str=ev_id, league=None, ended_at_ts=int(now.timestamp()))
        bt.logging.info({"provider_game_ended": ge.to_dict()})
        events.append(ge)
    return events
