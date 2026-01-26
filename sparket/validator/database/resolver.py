from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from sparket.shared.rows import EventRow, MarketRow


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume naive datetimes are UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt


_SELECT_EVENT_BY_SDI0 = text(
    """
    SELECT event_id, start_time_utc
    FROM event
    WHERE (ext_ref->'sportsdataio'->>'GameID') = :game_id
    LIMIT 1
    """
)

_INSERT_EVENT = text(
    """
    INSERT INTO event (
        league_id, home_team_id, away_team_id, venue, start_time_utc, status, ext_ref, created_at
    ) VALUES (
        :league_id, :home_team_id, :away_team_id, :venue, :start_time_utc, :status, :ext_ref, :created_at
    ) RETURNING event_id
    """
)

_SELECT_MARKET = text(
    """
    SELECT market_id
    FROM market
    WHERE event_id = :event_id
      AND kind = :kind
      AND COALESCE(line, CAST(0 AS numeric)) = COALESCE(CAST(:line AS numeric), CAST(0 AS numeric))
      AND COALESCE(points_team_id, CAST(0 AS bigint)) = COALESCE(CAST(:points_team_id AS bigint), CAST(0 AS bigint))
    LIMIT 1
    """
)

_INSERT_MARKET = text(
    """
    INSERT INTO market (
        event_id, kind, line, points_team_id, created_at
    ) VALUES (
        :event_id, :kind, :line, :points_team_id, :created_at
    )
    """
)


async def ensure_event_for_sdio(database: Any, event_row: EventRow) -> tuple[int, datetime]:
    ext = (event_row.get("ext_ref") or {}).get("sportsdataio") or {}
    game_id = str(ext.get("GameID", ""))
    if not game_id:
        raise ValueError("missing SDIO GameID in ext_ref")

    found = await database.read(_SELECT_EVENT_BY_SDI0, params={"game_id": game_id}, mappings=True)
    if found:
        row = found[0]
        return int(row["event_id"]), row["start_time_utc"]

    params = {
        "league_id": event_row.get("league_id"),
        "home_team_id": event_row.get("home_team_id"),
        "away_team_id": event_row.get("away_team_id"),
        "venue": event_row.get("venue"),
        "start_time_utc": _ensure_utc(event_row.get("start_time_utc")),
        "status": event_row.get("status"),
        "ext_ref": json.dumps(event_row.get("ext_ref") or {}),
        "created_at": _ensure_utc(event_row.get("created_at")) or datetime.now(timezone.utc),
    }
    rows = await database.write(_INSERT_EVENT, params=params, return_rows=True, mappings=True)
    ev_id = int(rows[0]["event_id"]) if rows else None
    if ev_id is None:
        found = await database.read(_SELECT_EVENT_BY_SDI0, params={"game_id": game_id}, mappings=True)
        if not found:
            raise RuntimeError("failed to upsert event")
        row = found[0]
        return int(row["event_id"]), row["start_time_utc"]
    return ev_id, params["start_time_utc"]


async def ensure_market(database: Any, market_row: MarketRow, *, event_id: int) -> int:
    kind = market_row.get("kind")
    if hasattr(kind, "name"):
        kind = kind.name
    elif isinstance(kind, str):
        kind = kind.upper()
    raw_line = market_row.get("line")
    line = Decimal(str(raw_line)) if raw_line is not None else None
    params = {
        "event_id": event_id,
        "kind": kind,
        "line": line,
        "points_team_id": market_row.get("points_team_id"),
        "created_at": _ensure_utc(market_row.get("created_at")) or datetime.now(timezone.utc),
    }
    # Check if market exists first
    found = await database.read(_SELECT_MARKET, params=params, mappings=True)
    if found:
        return int(found[0]["market_id"])
    # Try insert - may fail on duplicate key if race condition
    try:
        await database.write(_INSERT_MARKET, params=params)
    except Exception:
        pass  # Ignore duplicate key errors, we'll select again
    # Re-select to get the market_id (whether newly inserted or existing)
    found = await database.read(_SELECT_MARKET, params=params, mappings=True)
    if not found:
        raise RuntimeError("failed to upsert market")
    return int(found[0]["market_id"])

