"""Tests for validator/database/resolver.py - Event and market resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, AsyncMock

import pytest

from sparket.validator.database.resolver import (
    _ensure_utc,
    ensure_event_for_sdio,
    ensure_market,
)
from sparket.shared.enums import MarketKind


class TestEnsureUtc:
    """Tests for _ensure_utc helper function."""
    
    def test_none_returns_none(self):
        """None input returns None."""
        assert _ensure_utc(None) is None
    
    def test_naive_datetime_becomes_utc(self):
        """Naive datetime gets UTC timezone added."""
        dt = datetime(2025, 12, 8, 14, 30, 0)
        result = _ensure_utc(dt)
        
        assert result.tzinfo == timezone.utc
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 8
        assert result.hour == 14
    
    def test_aware_datetime_unchanged(self):
        """Already timezone-aware datetime is unchanged."""
        dt = datetime(2025, 12, 8, 14, 30, 0, tzinfo=timezone.utc)
        result = _ensure_utc(dt)
        
        assert result == dt
        assert result.tzinfo == timezone.utc


class MockDatabase:
    """Mock database for testing resolvers."""
    
    def __init__(self):
        self.events: Dict[str, Dict] = {}  # game_id -> event
        self.markets: Dict[str, Dict] = {}  # (event_id, kind, line, team) -> market
        self.next_event_id = 1
        self.next_market_id = 1
        self.writes: List[Dict] = []
    
    async def read(self, query, params=None, mappings=False):
        query_str = str(query)
        
        if "FROM event" in query_str and "ext_ref" in query_str:
            # Lookup by SDIO GameID
            game_id = params.get("game_id")
            if game_id in self.events:
                return [self.events[game_id]]
            return []
        
        if "FROM market" in query_str:
            # Lookup market
            key = self._market_key(params)
            if key in self.markets:
                return [self.markets[key]]
            return []
        
        return []
    
    async def write(self, query, params=None, return_rows=False, mappings=False):
        query_str = str(query)
        self.writes.append({"query": query_str, "params": params})
        
        if "INSERT INTO event" in query_str:
            event_id = self.next_event_id
            self.next_event_id += 1
            game_id = params.get("ext_ref")
            if game_id:
                import json
                ext_ref = json.loads(game_id) if isinstance(game_id, str) else game_id
                sdio_id = ext_ref.get("sportsdataio", {}).get("GameID")
                if sdio_id:
                    self.events[str(sdio_id)] = {
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
            if return_rows:
                return [{"market_id": market_id}]
            return 1
        
        return 1
    
    def _market_key(self, params):
        return (
            params.get("event_id"),
            params.get("kind"),
            params.get("line") or 0,
            params.get("points_team_id") or 0,
        )


class TestEnsureEventForSdio:
    """Tests for ensure_event_for_sdio function."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    async def test_finds_existing_event(self, db):
        """Returns existing event if found by GameID."""
        # Pre-populate database
        start_time = datetime(2025, 12, 15, 19, 0, 0, tzinfo=timezone.utc)
        db.events["12345"] = {"event_id": 42, "start_time_utc": start_time}
        
        event_row = {
            "league_id": 1,
            "ext_ref": {"sportsdataio": {"GameID": 12345}},
            "start_time_utc": start_time,
        }
        
        event_id, returned_time = await ensure_event_for_sdio(db, event_row)
        
        assert event_id == 42
        assert returned_time == start_time
        assert len(db.writes) == 0  # No insert needed
    
    async def test_creates_new_event(self, db):
        """Creates new event if not found."""
        start_time = datetime(2025, 12, 15, 19, 0, 0, tzinfo=timezone.utc)
        
        event_row = {
            "league_id": 1,
            "home_team_id": 100,
            "away_team_id": 101,
            "venue": "Stadium X",
            "start_time_utc": start_time,
            "status": "scheduled",
            "ext_ref": {"sportsdataio": {"GameID": 99999}},
        }
        
        event_id, returned_time = await ensure_event_for_sdio(db, event_row)
        
        assert event_id == 1  # First auto-generated ID
        assert len(db.writes) == 1
        
        params = db.writes[0]["params"]
        assert params["league_id"] == 1
        assert params["home_team_id"] == 100
        assert params["away_team_id"] == 101
        assert params["venue"] == "Stadium X"
        assert params["status"] == "scheduled"
    
    async def test_raises_on_missing_game_id(self, db):
        """Raises ValueError if GameID missing."""
        event_row = {
            "league_id": 1,
            "ext_ref": {},  # No sportsdataio
        }
        
        with pytest.raises(ValueError, match="missing SDIO GameID"):
            await ensure_event_for_sdio(db, event_row)
    
    async def test_raises_on_empty_ext_ref(self, db):
        """Raises ValueError if ext_ref empty."""
        event_row = {
            "league_id": 1,
            "ext_ref": None,
        }
        
        with pytest.raises(ValueError, match="missing SDIO GameID"):
            await ensure_event_for_sdio(db, event_row)
    
    async def test_handles_naive_datetime(self, db):
        """Converts naive datetime to UTC."""
        start_time = datetime(2025, 12, 15, 19, 0, 0)  # Naive
        
        event_row = {
            "league_id": 1,
            "start_time_utc": start_time,
            "ext_ref": {"sportsdataio": {"GameID": 11111}},
        }
        
        event_id, returned_time = await ensure_event_for_sdio(db, event_row)
        
        # Check the write had UTC-aware timestamp
        params = db.writes[0]["params"]
        assert params["start_time_utc"].tzinfo == timezone.utc


class TestEnsureMarket:
    """Tests for ensure_market function."""
    
    @pytest.fixture
    def db(self):
        return MockDatabase()
    
    async def test_finds_existing_market(self, db):
        """Returns existing market if found."""
        # Pre-populate
        db.markets[(1, "MONEYLINE", 0, 0)] = {"market_id": 42}
        
        market_row = {
            "event_id": 1,
            "kind": MarketKind.MONEYLINE,
            "line": None,
            "points_team_id": None,
        }
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 42
        assert len(db.writes) == 0
    
    async def test_creates_new_market(self, db):
        """Creates new market if not found."""
        market_row = {
            "kind": MarketKind.SPREAD,
            "line": -3.5,
            "points_team_id": 100,
        }
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 1  # First auto-generated
        assert len(db.writes) == 1
        
        params = db.writes[0]["params"]
        assert params["event_id"] == 1
        assert params["kind"] == "SPREAD"
        assert params["line"] == -3.5
        assert params["points_team_id"] == 100
    
    async def test_handles_kind_as_string(self, db):
        """Handles kind as string."""
        market_row = {
            "kind": "moneyline",  # lowercase string
            "line": None,
        }
        
        await ensure_market(db, market_row, event_id=1)
        
        params = db.writes[0]["params"]
        assert params["kind"] == "MONEYLINE"  # Uppercased
    
    async def test_handles_kind_enum_with_name(self, db):
        """Handles kind enum with .name attribute."""
        market_row = {
            "kind": MarketKind.TOTAL,
            "line": 210.5,
        }
        
        await ensure_market(db, market_row, event_id=1)
        
        params = db.writes[0]["params"]
        assert params["kind"] == "TOTAL"
    
    async def test_different_lines_different_markets(self, db):
        """Different lines create different markets."""
        market_row1 = {"kind": MarketKind.SPREAD, "line": -3.5}
        market_row2 = {"kind": MarketKind.SPREAD, "line": -7.0}
        
        id1 = await ensure_market(db, market_row1, event_id=1)
        id2 = await ensure_market(db, market_row2, event_id=1)
        
        assert id1 != id2
        assert len(db.writes) == 2
    
    async def test_null_line_treated_as_zero(self, db):
        """Null line treated as 0 for deduplication."""
        # Pre-populate with line=None (stored as 0)
        db.markets[(1, "MONEYLINE", 0, 0)] = {"market_id": 99}
        
        market_row = {"kind": "MONEYLINE", "line": None}
        
        market_id = await ensure_market(db, market_row, event_id=1)
        
        assert market_id == 99  # Found existing
        assert len(db.writes) == 0
    
    async def test_adds_created_at_if_missing(self, db):
        """Adds created_at timestamp if not provided."""
        market_row = {"kind": MarketKind.MONEYLINE}
        
        await ensure_market(db, market_row, event_id=1)
        
        params = db.writes[0]["params"]
        assert "created_at" in params
        assert params["created_at"] is not None
        assert params["created_at"].tzinfo == timezone.utc

