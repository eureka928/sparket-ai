from unittest.mock import AsyncMock, MagicMock

import pytest

from sparket.validator.handlers.maintenance.cleanup import run_cleanup_if_due


class _Timers:
    cleanup_steps = 1


class _Core:
    timers = _Timers()


class _Config:
    core = _Core()


class _Validator:
    app_config = _Config()
    step = 1


@pytest.mark.asyncio
async def test_cleanup_runs_when_due():
    db = MagicMock()
    db.write = AsyncMock(return_value=0)
    result = await run_cleanup_if_due(validator=_Validator(), database=db)
    assert isinstance(result, dict)
    assert db.write.call_count == 12


@pytest.mark.asyncio
async def test_cleanup_skips_when_not_due():
    db = MagicMock()
    db.write = AsyncMock(return_value=0)
    validator = _Validator()
    validator.step = 2
    validator.app_config.core.timers.cleanup_steps = 3
    result = await run_cleanup_if_due(validator=validator, database=db)
    assert result == {}
    db.write.assert_not_called()
