"""Tests for validator database resolver - event and market upsert logic.

Tests cover:
- ensure_event_for_sdio: create, find existing, edge cases
- ensure_market: create, find existing, enum handling
- _ensure_utc: timezone handling
- Error conditions and race condition handling
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from sparket.validator.database.resolver import (
    _ensure_utc,
    ensure_event_for_sdio,
    ensure_market,
)
from sparket.shared.enums import MarketKind


class MockDatabase:
    """Mock database for testing resolver functions."""
    
    def __init__(self):
        self.events: Dict[str, Dict] = {}  # game_id -> event data
        self.markets: Dict[str, Dict] = {}  # composite key -> market data
        self.next_event_id = 1
        self.next_market_id = 1000
        self.read_calls: List[Dict] = []
        self.write_calls: List[Dict] = []
        
        # For simulating race conditions
        self.fail_first_write = False
        self.write_fail_count = 0
    
    async def read(self, query, params=None, mappings=False):
        self.read_calls.append({"query": str(query), "params": params})
        query_str = str(query)
        
        if "FROM event" in query_str and "ext_ref" in query_str:
            # Looking up event by SDIO GameID
            game_id = params.get("game_id")
            if game_id in self.events:
                return [self.events[game_id]]
            return []
        
        if "FROM market" in query_str:
            # Looking up market by composite key
            key = self._market_key(params)
            if key in self.markets:
                return [self.markets[key]]
            return []
        
        return []
    
    async def write(self, query, params=None, return_rows=False, mappings=False):
        self.write_calls.append({"query": str(query), "params": params})
        query_str = str(query)
        
        # Simulate race condition
        if self.fail_first_write and self.write_fail_count == 0:
            self.write_fail_count += 1
            return []
        
        if "INSERT INTO event" in query_str:
            event_id = self.next_event_id
            self.next_event_id += 1
            
            ext_ref = params.get("ext_ref", "{}")
            if isinstance(ext_ref, str):
                import json
                ext_ref = json.loads(ext_ref)
            
            game_id = str(ext_ref.get("sportsdataio", {}).get("GameID", ""))
            
            self.events[game_id] = {
                "event_id": event_id,
                "start_time_utc": params.get("start_time_utc"),
            }
            
            if return_rows:
                return [{"event_id": event_id}]
            return 1
        
        if "INSERT INTO market" in query_str:
            market_id = self.next_market_id
            self.next_market_id += 1
            
            key = self._market_key(params)
            self.markets[key] = {"market_id": market_id}
            
            return 1
        
        return 1
    
    def _market_key(self, params: Dict) -> str:
        return f"{params.get('event_id')}:{params.get('kind')}:{params.get('line')}:{params.get('points_team_id')}"


class TestEnsureUtc:
    """Tests for _ensure_utc helper."""
    
    def test_none_input(self):
        """None should return None."""
        assert _ensure_utc(None) is None
    
    def test_naive_datetime(self):
        """Naive datetime should get UTC timezone."""
        naive = datetime(2025, 1, 15, 12, 0, 0)
        result = _ensure_utc(naive)
        
        assert result.tzinfo == timezone.utc
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 12
    
    def test_aware_datetime_utc(self):
        """UTC-aware datetime should pass through."""
        aware = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _ensure_utc(aware)
        
        assert result == aware
        assert result.tzinfo == timezone.utc
    
    def test_aware_datetime_other_tz(self):
        """Non-UTC aware datetime should be preserved."""
        eastern = timezone(timedelta(hours=-5))
        aware = datetime(2025, 1, 15, 12, 0, 0, tzinfo=eastern)
        result = _ensure_utc(aware)
        
        # Should preserve the timezone
        assert result.tzinfo == eastern


class TestEnsureEventForSdio:
    """Tests for ensure_event_for_sdio."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    async def test_creates_new_event(self, db):
        """Should create event when not found."""
        now = datetime.now(timezone.utc)
        event_row = {
            "league_id": 1,
            "home_team_id": 100,
            "away_team_id": 101,
            "venue": "Madison Square Garden",
            "start_time_utc": now + timedelta(days=1),
            "status": "scheduled",
            "ext_ref": {"sportsdataio": {"GameID": 12345}},
            "created_at": now,
        }
        
        event_id, start_time = await ensure_event_for_sdio(db, event_row)
        
        assert event_id == 1
        assert start_time == event_row["start_time_utc"]
        assert len(db.write_calls) == 1
        assert "INSERT INTO event" in db.write_calls[0]["query"]
    
    async def test_finds_existing_event(self, db):
        """Should return existing event when found."""
        now = datetime.now(timezone.utc)
        
        # Pre-populate the database
        db.events["12345"] = {
            "event_id": 42,
            "start_time_utc": now + timedelta(days=1),
        }
        
        event_row = {
            "league_id": 1,
            "ext_ref": {"sportsdataio": {"GameID": 12345}},
            "start_time_utc": now + timedelta(days=1),
        }
        
        event_id, start_time = await ensure_event_for_sdio(db, event_row)
        
        assert event_id == 42
        assert len(db.write_calls) == 0  # No write needed
    
    async def test_missing_game_id_raises(self, db):
        """Should raise ValueError when GameID is missing."""
        event_row = {
            "league_id": 1,
            "ext_ref": {},  # No sportsdataio
        }
        
        with pytest.raises(ValueError, match="missing SDIO GameID"):
            await ensure_event_for_sdio(db, event_row)
    
    async def test_empty_ext_ref_raises(self, db):
        """Should raise ValueError when ext_ref is empty."""
        event_row = {
            "league_id": 1,
            "ext_ref": None,
        }
        
        with pytest.raises(ValueError, match="missing SDIO GameID"):
            await ensure_event_for_sdio(db, event_row)
    
    async def test_handles_race_condition(self, db):
        """Should handle race condition when insert fails but event exists."""
        now = datetime.now(timezone.utc)
        
        # Simulate race: first write fails, but event appears on re-read
        db.fail_first_write = True
        
        event_row = {
            "league_id": 1,
            "ext_ref": {"sportsdataio": {"GameID": 99999}},
            "start_time_utc": now + timedelta(days=1),
        }
        
        # Add event as if another process inserted it
        db.events["99999"] = {
            "event_id": 77,
            "start_time_utc": now + timedelta(days=1),
        }
        
        event_id, start_time = await ensure_event_for_sdio(db, event_row)
        
        assert event_id == 77
    
    async def test_handles_naive_datetime(self, db):
        """Should handle naive datetimes by assuming UTC."""
        naive_time = datetime(2025, 6, 15, 18, 0, 0)  # No timezone
        
        event_row = {
            "league_id": 1,
            "ext_ref": {"sportsdataio": {"GameID": 11111}},
            "start_time_utc": naive_time,
        }
        
        event_id, start_time = await ensure_event_for_sdio(db, event_row)
        
        # Should have converted to UTC
        write_params = db.write_calls[0]["params"]
        assert write_params["start_time_utc"].tzinfo == timezone.utc
    
    async def test_defaults_created_at_to_now(self, db):
        """Should default created_at to current time if not provided."""
        now = datetime.now(timezone.utc)
        
        event_row = {
            "league_id": 1,
            "ext_ref": {"sportsdataio": {"GameID": 22222}},
            "start_time_utc": now + timedelta(days=1),
            # No created_at
        }
        
        await ensure_event_for_sdio(db, event_row)
        
        write_params = db.write_calls[0]["params"]
        assert write_params["created_at"] is not None
        assert write_params["created_at"].tzinfo == timezone.utc


class TestEnsureMarket:
    """Tests for ensure_market."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    async def test_creates_new_market(self, db):
        """Should create market when not found."""
        market_row = {
            "event_id": 1,
            "kind": MarketKind.MONEYLINE,
            "line": None,
            "points_team_id": None,
        }
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 1000
        assert len(db.write_calls) == 1
    
    async def test_finds_existing_market(self, db):
        """Should return existing market when found."""
        # Pre-populate
        key = "1:MONEYLINE:None:None"
        db.markets[key] = {"market_id": 555}
        
        market_row = {
            "kind": MarketKind.MONEYLINE,
            "line": None,
            "points_team_id": None,
        }
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 555
        assert len(db.write_calls) == 0
    
    async def test_handles_enum_kind(self, db):
        """Should convert MarketKind enum to string."""
        market_row = {
            "kind": MarketKind.SPREAD,
            "line": -3.5,
            "points_team_id": 100,
        }
        
        await ensure_market(db, market_row, event_id=1)
        
        write_params = db.write_calls[0]["params"]
        assert write_params["kind"] == "SPREAD"
    
    async def test_handles_string_kind_lowercase(self, db):
        """Should uppercase string kind."""
        market_row = {
            "kind": "moneyline",
            "line": None,
        }
        
        await ensure_market(db, market_row, event_id=1)
        
        write_params = db.write_calls[0]["params"]
        assert write_params["kind"] == "MONEYLINE"
    
    async def test_handles_string_kind_uppercase(self, db):
        """Should preserve uppercase string kind."""
        market_row = {
            "kind": "TOTAL",
            "line": 220.5,
        }
        
        await ensure_market(db, market_row, event_id=1)
        
        write_params = db.write_calls[0]["params"]
        assert write_params["kind"] == "TOTAL"
    
    async def test_spread_market_with_line(self, db):
        """Should handle spread market with line and points_team_id."""
        market_row = {
            "kind": MarketKind.SPREAD,
            "line": -6.5,
            "points_team_id": 42,
        }
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 1000
        write_params = db.write_calls[0]["params"]
        assert write_params["line"] == -6.5
        assert write_params["points_team_id"] == 42
    
    async def test_total_market_with_line(self, db):
        """Should handle total market with line."""
        market_row = {
            "kind": MarketKind.TOTAL,
            "line": 215.5,
            "points_team_id": None,
        }
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 1000
        write_params = db.write_calls[0]["params"]
        assert write_params["line"] == 215.5
        assert write_params["points_team_id"] is None
    
    async def test_defaults_created_at(self, db):
        """Should default created_at to now if not provided."""
        market_row = {
            "kind": MarketKind.MONEYLINE,
        }
        
        await ensure_market(db, market_row, event_id=1)
        
        write_params = db.write_calls[0]["params"]
        assert write_params["created_at"] is not None
        assert write_params["created_at"].tzinfo == timezone.utc
    
    async def test_different_lines_create_different_markets(self, db):
        """Different lines should create separate markets."""
        # First market
        market_row_1 = {"kind": MarketKind.SPREAD, "line": -3.5}
        market_id_1 = await ensure_market(db, market_row_1, event_id=1)
        
        # Second market with different line
        market_row_2 = {"kind": MarketKind.SPREAD, "line": -7.0}
        market_id_2 = await ensure_market(db, market_row_2, event_id=1)
        
        assert market_id_1 != market_id_2
        assert len(db.write_calls) == 2

