from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sparket.validator.handlers.ingest import provider_scheduler as sched
from sparket.validator.handlers.ingest.provider_scheduler import run_provider_ingest_if_due


class StubDB:
    def __init__(self, hot: bool = False):
        self.hot = hot
        self.read_calls: list[dict] = []

    async def read(self, query, params=None, mappings=False):
        self.read_calls.append({"query": str(query), "params": params, "mappings": mappings})
        if "FROM event" in str(query) and self.hot:
            return [{"event_id": 1, "start_time_utc": datetime.now(timezone.utc)}]
        return []

    async def write(self, *_, **__):
        return []


class StubIngestor:
    def __init__(self):
        self.calls: list[datetime] = []

    async def run_once(self, *, now):
        self.calls.append(now)


def _validator(step: int, *, fetch_steps=50, hot_steps=10, hot_window=60):
    timers = SimpleNamespace(
        provider_fetch_steps=fetch_steps,
        provider_hot_fetch_steps=hot_steps,
        provider_hot_window_minutes=hot_window,
    )
    core = SimpleNamespace(timers=timers)
    app_config = SimpleNamespace(core=core)
    return SimpleNamespace(step=step, app_config=app_config)


def test_scheduler_runs_ingestor_each_call(monkeypatch):
    asyncio.run(_scheduler_runs_ingestor_each_call(monkeypatch))


async def _scheduler_runs_ingestor_each_call(monkeypatch):
    ingestor = StubIngestor()
    db = StubDB(hot=False)
    validator = _validator(step=1)

    async def fake_upsert_provider_closing(**_):
        pytest.fail("closing should not run when interval not reached")

    monkeypatch.setattr(
        "sparket.validator.handlers.ingest.provider_scheduler.upsert_provider_closing",
        fake_upsert_provider_closing,
    )
    monkeypatch.setattr(
        "sparket.validator.handlers.ingest.provider_scheduler.get_provider_id",
        lambda _: 1,
    )

    await run_provider_ingest_if_due(validator=validator, database=db, ingestor=ingestor)
    assert len(ingestor.calls) == 1


def test_scheduler_triggers_closing_on_interval(monkeypatch):
    asyncio.run(_scheduler_triggers_closing_on_interval(monkeypatch))


async def _scheduler_triggers_closing_on_interval(monkeypatch):
    ingestor = StubIngestor()
    db = StubDB(hot=True)
    validator = _validator(step=5, fetch_steps=50, hot_steps=5, hot_window=60)

    close_calls: list[dict] = []

    async def fake_upsert_provider_closing(*, database, provider_id, close_ts):
        close_calls.append({"provider_id": provider_id, "close_ts": close_ts})
        return 3

    monkeypatch.setattr(
        "sparket.validator.handlers.ingest.provider_scheduler.upsert_provider_closing",
        fake_upsert_provider_closing,
    )
    monkeypatch.setattr(
        "sparket.validator.handlers.ingest.provider_scheduler.get_provider_id",
        lambda _: 99,
    )

    await run_provider_ingest_if_due(validator=validator, database=db, ingestor=ingestor)
    assert len(close_calls) == 1
    # close_ts should be timezone-aware (UTC)
    assert close_calls[0]["close_ts"].tzinfo is not None


@pytest.mark.asyncio
async def test_hot_window_params_are_tz_aware(monkeypatch):
    """Verify that datetime params passed to DB queries are timezone-aware."""
    db = MagicMock()
    db.read = AsyncMock(return_value=[])
    
    class _Timers:
        provider_fetch_steps = 50
        provider_hot_fetch_steps = 10
        provider_hot_window_minutes = 60
    
    class _Core:
        timers = _Timers()
    
    class _Config:
        core = _Core()
    
    class _Validator:
        app_config = _Config()
        step = 0
    
    monkeypatch.setattr(sched, "get_provider_id", lambda _: None)
    
    await sched.run_provider_ingest_if_due(
        validator=_Validator(),
        database=db,
        ingestor=None,
    )
    
    assert db.read.call_count == 1
    params = db.read.call_args.kwargs["params"]
    assert params["now"].tzinfo is not None
    assert params["hot_until"].tzinfo is not None

