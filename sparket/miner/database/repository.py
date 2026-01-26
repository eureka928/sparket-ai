from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping

from sqlalchemy import select, insert, update
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from .dbm import DBM
from .schema.validator_endpoint import ValidatorEndpoint
from .schema.game_data import Event


async def upsert_validator_endpoint(
    dbm: DBM,
    *,
    hotkey: str,
    host: str | None,
    port: int | None,
    url: str | None,
    token: str | None,
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    stmt = sqlite_upsert(ValidatorEndpoint).values(
        hotkey=hotkey,
        host=host,
        port=port,
        url=url,
        token=token,
        last_seen=now,
        created_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[ValidatorEndpoint.hotkey],  # type: ignore[arg-type]
        set_={
            "host": host,
            "port": port,
            "url": url,
            "token": token,
            "last_seen": now,
            "updated_at": now,
        },
    )
    async with dbm.session() as session:
        async with session.begin():
            await session.execute(stmt)


async def list_validator_endpoints(dbm: DBM) -> list[Mapping[str, Any]]:
    stmt = select(ValidatorEndpoint).order_by(ValidatorEndpoint.updated_at.desc())
    async with dbm.session() as session:
        rows = await session.execute(stmt)
        return [row[0] for row in rows.all()]


async def get_past_events(dbm: DBM, hours_ago: int = 48) -> List[Dict[str, Any]]:
    """Get events that started in the past (potentially finished).
    
    Args:
        dbm: Database manager
        hours_ago: How far back to look (default: 48 hours)
    
    Returns:
        List of event dicts with event_id, home_team, away_team, sport, start_time_utc
    """
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=hours_ago)
    
    stmt = (
        select(Event)
        .where(Event.start_time_utc < now)
        .where(Event.start_time_utc > cutoff)
        .order_by(Event.start_time_utc.desc())
    )
    
    async with dbm.session() as session:
        rows = await session.execute(stmt)
        results = []
        for row in rows.all():
            event = row[0]
            results.append({
                "event_id": event.event_id,
                "home_team": event.home_team,
                "away_team": event.away_team,
                "sport": event.sport or "NFL",
                "start_time_utc": event.start_time_utc,
            })
        return results

