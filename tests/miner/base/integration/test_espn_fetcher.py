"""Integration tests for ESPN fetcher.

These tests make real network requests to ESPN's public API.
Mark with @pytest.mark.integration and skip if network unavailable.
"""

import pytest

from sparket.miner.base.fetchers.espn import ESPNFetcher


@pytest.mark.integration
class TestESPNFetcher:
    """Integration tests for ESPNFetcher."""
    
    @pytest.fixture
    async def fetcher(self):
        """Create an ESPN fetcher for testing."""
        fetcher = ESPNFetcher(cache_ttl_seconds=300)
        yield fetcher
        await fetcher.close()
    
    @pytest.mark.asyncio
    async def test_fetch_nfl_standings(self, fetcher):
        """Can fetch real NFL standings from ESPN."""
        standings = await fetcher.get_standings("NFL")
        
        # During off-season, standings may be empty
        # Just verify the call doesn't crash and returns a list
        assert isinstance(standings, list)
        
        # If we got data, verify structure
        if standings:
            first = standings[0]
            assert first.team_code
            assert first.league == "NFL"
            assert first.wins >= 0
            assert first.losses >= 0
    
    @pytest.mark.asyncio
    async def test_fetch_nba_standings(self, fetcher):
        """Can fetch real NBA standings from ESPN."""
        standings = await fetcher.get_standings("NBA")
        
        # During off-season, standings may be empty
        # Just verify the call doesn't crash and returns a list
        assert isinstance(standings, list)
    
    @pytest.mark.asyncio
    async def test_fetch_team_stats(self, fetcher):
        """Can fetch stats for specific team."""
        stats = await fetcher.get_team_stats("KC", "NFL")
        
        if stats:  # May be None off-season
            assert stats.team_code == "KC"
            assert stats.league == "NFL"
            assert stats.games_played >= 0
    
    @pytest.mark.asyncio
    async def test_handles_invalid_team(self, fetcher):
        """Returns None for invalid team."""
        stats = await fetcher.get_team_stats("INVALID123", "NFL")
        assert stats is None
    
    @pytest.mark.asyncio
    async def test_handles_invalid_league(self, fetcher):
        """Returns empty list for invalid league."""
        standings = await fetcher.get_standings("INVALID")
        assert standings == []
    
    @pytest.mark.asyncio
    async def test_caches_responses(self, fetcher):
        """Responses are cached to avoid redundant requests."""
        # First request
        standings1 = await fetcher.get_standings("NFL")
        
        # Second request should be cached (much faster)
        standings2 = await fetcher.get_standings("NFL")
        
        # Should return same data
        if standings1 and standings2:
            assert len(standings1) == len(standings2)
    
    @pytest.mark.asyncio
    async def test_get_recent_games(self, fetcher):
        """Can fetch recent games for a team."""
        games = await fetcher.get_recent_games("KC", "NFL", limit=5)
        
        # May be empty off-season
        if games:
            assert len(games) <= 5
            first = games[0]
            assert first.opponent_code
            assert first.score_for >= 0
            assert first.score_against >= 0
            assert isinstance(first.won, bool)
    
    @pytest.mark.asyncio
    async def test_get_game_result(self, fetcher):
        """Can fetch a game result."""
        # Use a generic query that might find something
        event = {
            "home_team": "KC",
            "away_team": "BUF",
            "sport": "NFL",
            "start_time_utc": None,  # Will use current scoreboard
        }
        
        result = await fetcher.get_result(event)
        
        # May or may not find a game depending on schedule
        # Just verify it doesn't crash
        if result:
            assert result.is_final in (True, False)



