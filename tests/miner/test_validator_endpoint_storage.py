import asyncio
import os
import tempfile

import pytest

from sparket.miner.config.config import Config as MinerConfig
from sparket.miner.database import DBM, initialize
from sparket.miner.database.repository import (
    list_validator_endpoints,
    upsert_validator_endpoint,
)


@pytest.mark.asyncio
async def test_upsert_validator_endpoint_persists_and_updates():
    config = MinerConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        initialize(config, tmpdir)
        dbm = DBM(config, tmpdir)

        await upsert_validator_endpoint(
            dbm,
            hotkey="5TestHotkey",
            host="127.0.0.1",
            port=8093,
            url=None,
            token="abc",
        )

        rows = await list_validator_endpoints(dbm)
        assert len(rows) == 1
        row = rows[0]
        assert row.hotkey == "5TestHotkey"
        assert row.host == "127.0.0.1"
        assert row.port == 8093
        assert row.token == "abc"

        await upsert_validator_endpoint(
            dbm,
            hotkey="5TestHotkey",
            host="10.0.0.1",
            port=9000,
            url="https://example.com",
            token="xyz",
        )

        rows = await list_validator_endpoints(dbm)
        assert len(rows) == 1
        row = rows[0]
        assert row.host == "10.0.0.1"
        assert row.port == 9000
        assert row.url == "https://example.com"
        assert row.token == "xyz"

        await dbm.dispose()

