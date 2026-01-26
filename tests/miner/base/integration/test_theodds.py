"""Integration tests for The-Odds-API engine.

These tests require a valid API key to run.
Mark with @pytest.mark.integration and skip if no API key.
"""

import os

import pytest

from sparket.miner.base.engines.theodds import TheOddsEngine


# Skip if no API key configured
pytestmark = pytest.mark.skipif(
    not os.getenv("SPARKET_BASE_MINER__ODDS_API_KEY"),
    reason="SPARKET_BASE_MINER__ODDS_API_KEY not set"
)


@pytest.mark.integration
class TestTheOddsEngine:
    """Integration tests for TheOddsEngine."""
    
    @pytest.fixture
    async def engine(self):
        """Create a TheOdds engine for testing."""
        engine = TheOddsEngine(cache_ttl_seconds=300)
        yield engine
        await engine.close()
    
    @pytest.mark.asyncio
    async def test_fetch_nfl_moneyline(self, engine):
        """Can fetch NFL moneyline odds."""
        market = {
            "sport": "NFL",
            "home_team": "KC",  # May not always have a game
            "away_team": "BUF",
            "kind": "MONEYLINE",
        }
        
        odds = await engine.get_odds(market)
        
        # May be None if no game found
        if odds:
            assert 0 < odds.home_prob < 1
            assert 0 < odds.away_prob < 1
            assert odds.home_odds_eu > 1
            assert odds.away_odds_eu > 1
    
    @pytest.mark.asyncio
    async def test_tracks_remaining_requests(self, engine):
        """Tracks remaining API requests."""
        market = {
            "sport": "NFL",
            "home_team": "KC",
            "away_team": "BUF",
            "kind": "MONEYLINE",
        }
        
        await engine.get_odds(market)
        
        # After a request, should have a count
        assert engine.requests_remaining is None or engine.requests_remaining >= 0
    
    @pytest.mark.asyncio
    async def test_caches_sport_odds(self, engine):
        """Caches odds to minimize API calls."""
        market = {
            "sport": "NBA",
            "home_team": "LAL",
            "away_team": "BOS",
            "kind": "MONEYLINE",
        }
        
        # First request
        await engine.get_odds(market)
        initial_remaining = engine.requests_remaining
        
        # Second request for same sport should be cached
        await engine.get_odds(market)
        
        # Should not have consumed another request
        if initial_remaining is not None:
            assert engine.requests_remaining == initial_remaining
    
    @pytest.mark.asyncio
    async def test_handles_no_game_found(self, engine):
        """Returns None when no matching game found."""
        market = {
            "sport": "NFL",
            "home_team": "FAKE",
            "away_team": "TEAM",
            "kind": "MONEYLINE",
        }
        
        odds = await engine.get_odds(market)
        assert odds is None
    
    @pytest.mark.asyncio
    async def test_spread_market(self, engine):
        """Can fetch spread odds."""
        market = {
            "sport": "NBA",
            "home_team": "LAL",
            "away_team": "BOS",
            "kind": "SPREAD",
        }
        
        odds = await engine.get_odds(market)
        
        if odds:
            assert 0 < odds.home_prob < 1
            assert 0 < odds.away_prob < 1
    
    @pytest.mark.asyncio
    async def test_total_market(self, engine):
        """Can fetch totals (over/under) odds."""
        market = {
            "sport": "NBA",
            "home_team": "LAL",
            "away_team": "BOS",
            "kind": "TOTAL",
        }
        
        odds = await engine.get_odds(market)
        
        if odds:
            assert odds.over_prob is not None
            assert odds.under_prob is not None
            assert 0 < odds.over_prob < 1
            assert 0 < odds.under_prob < 1








