"""Tests for providers/sportsdataio/catalog.py - Team catalog helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sparket.providers.sportsdataio.catalog import (
    sport_rows,
    team_rows_from_catalog,
    build_team_index_by_sdio,
    resolve_game_team_ids,
)
from sparket.providers.sportsdataio.types import Team, Game


class TestSportRows:
    """Tests for sport_rows function."""
    
    def test_returns_canonical_sports(self):
        """Returns list of canonical sport definitions."""
        rows = sport_rows()
        
        assert len(rows) >= 5
        codes = {r["code"] for r in rows}
        assert "nfl" in codes
        assert "nba" in codes
        assert "mlb" in codes
        assert "nhl" in codes
        assert "soccer" in codes
    
    def test_each_row_has_code_and_name(self):
        """Each sport row has code and name."""
        rows = sport_rows()
        
        for row in rows:
            assert "code" in row
            assert "name" in row
            assert row["code"] is not None
            assert row["name"] is not None


class TestTeamRowsFromCatalog:
    """Tests for team_rows_from_catalog function."""
    
    def _make_team(self, team_id: int, key: str, city: str = None, name: str = None, **kwargs):
        """Create a Team object for testing."""
        return Team(
            team_id=team_id,
            key=key,
            city=city,
            name=name,
            conference=kwargs.get("conference"),
            division=kwargs.get("division"),
        )
    
    def test_transforms_team_to_row(self):
        """Transforms Team to row with correct fields."""
        team = self._make_team(
            team_id=123,
            key="NYY",
            city="New York",
            name="Yankees",
            conference="AL",
            division="East",
        )
        
        rows = team_rows_from_catalog([team], league_id=1)
        
        assert len(rows) == 1
        row = rows[0]
        assert row["league_id"] == 1
        assert row["name"] == "New York Yankees"
        assert row["abbrev"] == "NYY"
        assert row["ext_ref"]["sportsdataio"]["TeamID"] == 123
        assert row["ext_ref"]["sportsdataio"]["Key"] == "NYY"
        assert row["ext_ref"]["sportsdataio"]["Conference"] == "AL"
        assert row["ext_ref"]["sportsdataio"]["Division"] == "East"
    
    def test_uses_key_when_name_missing(self):
        """Uses key as name when name is missing."""
        team = self._make_team(team_id=123, key="NYY", city=None, name=None)
        
        rows = team_rows_from_catalog([team], league_id=1)
        
        assert rows[0]["name"] == "NYY"
    
    def test_uses_name_only_when_city_missing(self):
        """Uses just name when city is missing."""
        team = self._make_team(team_id=123, key="NYY", city=None, name="Yankees")
        
        rows = team_rows_from_catalog([team], league_id=1)
        
        assert rows[0]["name"] == "Yankees"
    
    def test_handles_multiple_teams(self):
        """Handles multiple teams."""
        teams = [
            self._make_team(team_id=1, key="NYY", city="New York", name="Yankees"),
            self._make_team(team_id=2, key="BOS", city="Boston", name="Red Sox"),
        ]
        
        rows = team_rows_from_catalog(teams, league_id=5)
        
        assert len(rows) == 2
        assert all(r["league_id"] == 5 for r in rows)
    
    def test_handles_empty_input(self):
        """Handles empty team list."""
        rows = team_rows_from_catalog([], league_id=1)
        assert rows == []


class TestBuildTeamIndexBySdio:
    """Tests for build_team_index_by_sdio function."""
    
    def test_builds_index_from_rows(self):
        """Builds key -> team_id index from rows."""
        rows = [
            {"team_id": 100, "ext_ref": {"sportsdataio": {"Key": "NYY"}}},
            {"team_id": 200, "ext_ref": {"sportsdataio": {"Key": "BOS"}}},
        ]
        
        index = build_team_index_by_sdio(rows)
        
        assert index["NYY"] == 100
        assert index["BOS"] == 200
    
    def test_skips_rows_without_key(self):
        """Skips rows without sportsdataio Key."""
        rows = [
            {"team_id": 100, "ext_ref": {"sportsdataio": {"Key": "NYY"}}},
            {"team_id": 200, "ext_ref": {"sportsdataio": {}}},  # No Key
            {"team_id": 300, "ext_ref": {}},  # No sportsdataio
            {"team_id": 400},  # No ext_ref
        ]
        
        index = build_team_index_by_sdio(rows)
        
        assert len(index) == 1
        assert "NYY" in index
    
    def test_handles_empty_input(self):
        """Handles empty row list."""
        index = build_team_index_by_sdio([])
        assert index == {}
    
    def test_handles_none_ext_ref(self):
        """Handles None ext_ref gracefully."""
        rows = [
            {"team_id": 100, "ext_ref": None},
        ]
        
        index = build_team_index_by_sdio(rows)
        assert index == {}


class TestResolveGameTeamIds:
    """Tests for resolve_game_team_ids function."""
    
    def _make_game(self, home_team: str, away_team: str):
        """Create a minimal Game object for testing."""
        return Game(
            game_id=1,
            season=2025,
            season_type=1,
            home_team=home_team,
            away_team=away_team,
        )
    
    def test_resolves_both_teams(self):
        """Resolves both home and away team IDs."""
        game = self._make_game("NYY", "BOS")
        index = {"NYY": 100, "BOS": 200}
        
        home_id, away_id = resolve_game_team_ids(game, index)
        
        assert home_id == 100
        assert away_id == 200
    
    def test_returns_none_for_unknown_team(self):
        """Returns None for unknown team key."""
        game = self._make_game("NYY", "UNKNOWN")
        index = {"NYY": 100}
        
        home_id, away_id = resolve_game_team_ids(game, index)
        
        assert home_id == 100
        assert away_id is None
    
    def test_returns_none_for_both_unknown(self):
        """Returns None for both when neither team known."""
        game = self._make_game("UNKNOWN1", "UNKNOWN2")
        index = {"NYY": 100}
        
        home_id, away_id = resolve_game_team_ids(game, index)
        
        assert home_id is None
        assert away_id is None
    
    def test_handles_empty_index(self):
        """Handles empty team index."""
        game = self._make_game("NYY", "BOS")
        
        home_id, away_id = resolve_game_team_ids(game, {})
        
        assert home_id is None
        assert away_id is None

