from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bittensor as bt
from sqlalchemy import text

from sparket.validator.handlers.ingest.provider_ingest import upsert_provider_closing
from sparket.providers.providers import get_provider_id


_SELECT_EVENTS_IN_HOT_WINDOW = text(
    """
    SELECT event_id, start_time_utc
    FROM event
    WHERE start_time_utc BETWEEN :now AND :hot_until
    ORDER BY start_time_utc ASC
    LIMIT 1000
    """
)


async def run_provider_ingest_if_due(
    *,
    validator: Any,
    database: Any,
    ingestor: Any | None = None,
) -> None:
    """
    Step-based scheduler for provider ingest and closing snapshots.
    Hot windows increase polling frequency near event start.
    """
    try:
        core_cfg = getattr(validator, "app_config", None)
        timers = getattr(getattr(core_cfg, "core", None), "timers", None)
        now = datetime.now(timezone.utc)

        fetch_steps = 50
        hot_fetch_steps = 10
        hot_window_minutes = 60
        if timers is not None:
            try:
                fetch_steps = max(1, int(timers.provider_fetch_steps))
                hot_fetch_steps = max(1, int(timers.provider_hot_fetch_steps))
                hot_window_minutes = max(1, int(timers.provider_hot_window_minutes))
            except Exception:
                pass

        hot_until = now + timedelta(minutes=hot_window_minutes)
        in_hot = False
        try:
            hot_events = await database.read(
                _SELECT_EVENTS_IN_HOT_WINDOW,
                params={"now": now, "hot_until": hot_until},
                mappings=True,
            )
            in_hot = bool(hot_events)
        except Exception as exc:
            bt.logging.warning({"provider_scheduler_hot_window_error": str(exc)})

        interval = max(1, hot_fetch_steps if in_hot else fetch_steps)

        if ingestor is not None:
            try:
                await ingestor.run_once(now=now)
            except Exception as exc:
                bt.logging.warning({"provider_scheduler_ingestor_error": str(exc)})

        if validator.step % interval != 0:
            return

        provider_id = get_provider_id("SDIO")
        if not provider_id:
            return
        try:
            upserts = await upsert_provider_closing(
                database=database,
                provider_id=provider_id,
                close_ts=now,
            )
            if upserts:
                bt.logging.info({"provider_closing_upserts": upserts})
        except Exception as exc:
            bt.logging.warning({"provider_closing_error": str(exc)})
    except Exception as exc:
        bt.logging.debug({"provider_scheduler_unhandled": str(exc)})
