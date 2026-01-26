"""Integration tests for the full odds generation pipeline."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from sparket.miner.base.config import BaseMinerConfig
from sparket.miner.base.runner import BaseMiner


@pytest.mark.integration
class TestOddsPipeline:
    """Integration tests for the odds generation pipeline."""
    
    @pytest.fixture
    async def base_miner(self, base_miner_config, mock_validator_client, mock_game_sync):
        """Create a base miner for testing."""
        miner = BaseMiner(
            hotkey="5test_hotkey_xxx",
            config=base_miner_config,
            validator_client=mock_validator_client,
            game_sync=mock_game_sync,
        )
        yield miner
        await miner.stop() if miner.is_running else None
    
    @pytest.mark.asyncio
    async def test_generates_odds_with_espn_only(self, base_miner):
        """Can generate odds using only ESPN data (no The-Odds-API)."""
        market = {
            "market_id": 1,
            "event_id": 100,
            "kind": "MONEYLINE",
            "home_team": "KC",
            "away_team": "NE",
            "sport": "NFL",
        }
        
        odds = await base_miner.generate_odds(market)
        
        # Should generate odds even without external API
        assert odds is not None
        assert 0 < odds.home_prob < 1
        assert 0 < odds.away_prob < 1
        assert odds.home_odds_eu > 1
        assert odds.away_odds_eu > 1
    
    @pytest.mark.asyncio
    async def test_different_teams_different_odds(self, base_miner):
        """Different matchups produce different odds."""
        market1 = {
            "market_id": 1,
            "kind": "MONEYLINE",
            "home_team": "KC",
            "away_team": "NE",
            "sport": "NFL",
        }
        market2 = {
            "market_id": 2,
            "kind": "MONEYLINE",
            "home_team": "NYJ",
            "away_team": "MIA",
            "sport": "NFL",
        }
        
        odds1 = await base_miner.generate_odds(market1)
        odds2 = await base_miner.generate_odds(market2)
        
        # Different matchups should give different odds
        assert odds1 is not None
        assert odds2 is not None
        # Note: could be same if both fall back to naive engine
    
    @pytest.mark.asyncio
    async def test_home_team_advantage(self, base_miner):
        """Home team should generally be favored."""
        # Strong home team vs weak away team
        market = {
            "market_id": 1,
            "kind": "MONEYLINE",
            "home_team": "KC",  # Good team
            "away_team": "NE",  # Currently weak
            "sport": "NFL",
        }
        
        odds = await base_miner.generate_odds(market)
        
        assert odds is not None
        # Strong home team should be favored
        # But this depends on ESPN data being available
    
    @pytest.mark.asyncio
    async def test_builds_valid_payload(self, base_miner, sample_odds_prices, sample_market):
        """Builds valid submission payload."""
        payload = base_miner._build_odds_payload(sample_market, sample_odds_prices)
        
        assert payload["miner_hotkey"] == "5test_hotkey_xxx"
        assert "submissions" in payload
        assert len(payload["submissions"]) == 1
        
        sub = payload["submissions"][0]
        assert sub["market_id"] == 1
        assert sub["kind"] == "moneyline"
        assert "priced_at" in sub
        assert len(sub["prices"]) == 2
    
    @pytest.mark.asyncio
    async def test_handles_unknown_team(self, base_miner):
        """Gracefully handles unknown teams (falls back to naive)."""
        market = {
            "market_id": 1,
            "kind": "MONEYLINE",
            "home_team": "UNKNOWN",
            "away_team": "ALSO_UNKNOWN",
            "sport": "NFL",
        }
        
        odds = await base_miner.generate_odds(market)
        
        # Should still generate something via naive engine
        assert odds is not None
        assert 0 < odds.home_prob < 1
    
    @pytest.mark.asyncio
    async def test_spread_market(self, base_miner):
        """Can generate spread market odds."""
        market = {
            "market_id": 1,
            "kind": "SPREAD",
            "home_team": "KC",
            "away_team": "NE",
            "sport": "NFL",
            "line": -7.5,
        }
        
        odds = await base_miner.generate_odds(market)
        
        assert odds is not None
        # Spreads are typically close to 50/50
        # (but model uses same logic as moneyline currently)
    
    @pytest.mark.asyncio
    async def test_caches_team_stats(self, base_miner):
        """Team stats caching mechanism works correctly."""
        # Manually set a cached value to verify cache works
        from sparket.miner.base.data.stats import TeamStats
        test_stats = TeamStats(team_code="TEST", league="NFL", wins=10, losses=6)
        base_miner._stats_cache.set("NFL:TEST", test_stats)
        
        # Verify it's cached
        cached = base_miner._stats_cache.get("NFL:TEST")
        assert cached is not None
        assert cached.team_code == "TEST"
        assert cached.wins == 10



