"""Unit tests for NaiveEngine."""

import pytest

from sparket.miner.base.engines.naive import NaiveEngine, HOME_ADVANTAGE


class TestNaiveEngine:
    """Tests for the naive odds engine."""
    
    @pytest.fixture
    def engine(self) -> NaiveEngine:
        """Create a naive engine for testing."""
        return NaiveEngine()
    
    def test_generates_valid_moneyline_odds(self, engine: NaiveEngine):
        """Naive engine produces valid moneyline odds."""
        odds = engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        
        assert odds is not None
        assert 0 < odds.home_prob < 1
        assert 0 < odds.away_prob < 1
        assert odds.home_odds_eu > 1
        assert odds.away_odds_eu > 1
    
    def test_home_advantage_applied(self, engine: NaiveEngine):
        """Home team should be slightly favored."""
        odds = engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        
        assert odds.home_prob > odds.away_prob, (
            f"Home ({odds.home_prob}) should be favored over away ({odds.away_prob})"
        )
    
    def test_eu_odds_reflect_probability(self, engine: NaiveEngine):
        """EU odds should be approximately 1/probability (with vig)."""
        odds = engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        
        # With vig, odds should be slightly worse than fair
        # 1/prob gives fair odds; actual odds should be lower
        fair_home_odds = 1 / odds.home_prob
        assert odds.home_odds_eu < fair_home_odds, (
            f"Actual odds ({odds.home_odds_eu}) should be less than fair ({fair_home_odds})"
        )
    
    def test_spread_odds_near_fifty_fifty(self, engine: NaiveEngine):
        """Spread odds should be near 50/50."""
        odds = engine.get_odds_sync({"kind": "SPREAD", "sport": "NFL", "line": -3.5})
        
        assert odds is not None
        # Spreads are designed to be ~50/50
        assert 0.45 < odds.home_prob < 0.55, f"Spread should be ~50/50, got {odds.home_prob}"
    
    def test_total_odds(self, engine: NaiveEngine):
        """Total odds should have over/under at ~50/50."""
        odds = engine.get_odds_sync({"kind": "TOTAL", "sport": "NFL"})
        
        assert odds is not None
        assert odds.over_prob is not None
        assert odds.under_prob is not None
        assert 0.45 < odds.over_prob < 0.55
        assert 0.45 < odds.under_prob < 0.55
    
    def test_different_sports_different_advantage(self):
        """Different sports have different home advantages."""
        engine = NaiveEngine()
        
        nfl_odds = engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        nba_odds = engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NBA"})
        
        # NBA has higher home advantage than NFL
        assert HOME_ADVANTAGE["NBA"] > HOME_ADVANTAGE["NFL"]
        # So NBA home prob should be higher
        assert nba_odds.home_prob > nfl_odds.home_prob
    
    def test_custom_vig(self):
        """Custom vig affects odds."""
        low_vig_engine = NaiveEngine(vig=0.02)
        high_vig_engine = NaiveEngine(vig=0.08)
        
        low_odds = low_vig_engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        high_odds = high_vig_engine.get_odds_sync({"kind": "MONEYLINE", "sport": "NFL"})
        
        # Higher vig = lower odds (worse for bettor)
        assert low_odds.home_odds_eu > high_odds.home_odds_eu
    
    @pytest.mark.asyncio
    async def test_async_interface(self, engine: NaiveEngine):
        """Async interface works correctly."""
        odds = await engine.get_odds({"kind": "MONEYLINE", "sport": "NFL"})
        
        assert odds is not None
        assert 0 < odds.home_prob < 1








