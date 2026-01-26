"""Unit tests for odds blending."""

import pytest

from sparket.miner.base.engines.interface import OddsPrices
from sparket.miner.base.model.blend import (
    blend_odds,
    blend_odds_prices,
    adjust_for_confidence,
)


class TestBlendOdds:
    """Tests for blend_odds function."""
    
    def test_pure_market_weight(self):
        """100% market weight returns market odds."""
        result = blend_odds(market_prob=0.6, model_prob=0.4, market_weight=1.0)
        assert result == 0.6
    
    def test_pure_model_weight(self):
        """0% market weight returns model odds."""
        result = blend_odds(market_prob=0.6, model_prob=0.4, market_weight=0.0)
        assert result == 0.4
    
    def test_fifty_fifty_blend(self):
        """50% weight averages the two."""
        result = blend_odds(market_prob=0.6, model_prob=0.4, market_weight=0.5)
        assert result == 0.5
    
    def test_no_market_uses_model(self):
        """When market is None, use model only."""
        result = blend_odds(market_prob=None, model_prob=0.65, market_weight=0.8)
        assert result == 0.65
    
    def test_typical_blend(self):
        """Typical 60% market weight blend."""
        result = blend_odds(market_prob=0.55, model_prob=0.60, market_weight=0.6)
        # 0.55 * 0.6 + 0.60 * 0.4 = 0.33 + 0.24 = 0.57
        assert abs(result - 0.57) < 0.01
    
    def test_stays_in_valid_range(self):
        """Blended probs should always be in (0, 1)."""
        # Edge case: extreme values
        result = blend_odds(market_prob=0.99, model_prob=0.98, market_weight=0.5)
        assert 0 < result < 1
        
        result = blend_odds(market_prob=0.01, model_prob=0.02, market_weight=0.5)
        assert 0 < result < 1


class TestBlendOddsPrices:
    """Tests for blend_odds_prices function."""
    
    def test_no_market_uses_model(self):
        """Without market data, use model with vig."""
        model = OddsPrices(
            home_prob=0.6,
            away_prob=0.4,
            home_odds_eu=1.67,
            away_odds_eu=2.50,
        )
        
        result = blend_odds_prices(market=None, model=model, vig=0.045)
        
        assert result.home_prob == 0.6
        assert result.away_prob == 0.4
        # Odds should have vig applied
        assert result.home_odds_eu < 1.67  # Lower due to vig
    
    def test_with_market_blends(self):
        """With market data, blend the two."""
        market = OddsPrices(
            home_prob=0.55,
            away_prob=0.45,
            home_odds_eu=1.82,
            away_odds_eu=2.22,
        )
        model = OddsPrices(
            home_prob=0.65,
            away_prob=0.35,
            home_odds_eu=1.54,
            away_odds_eu=2.86,
        )
        
        result = blend_odds_prices(
            market=market, 
            model=model, 
            market_weight=0.6,
            vig=0.045,
        )
        
        # Blended prob: 0.55 * 0.6 + 0.65 * 0.4 = 0.59
        assert abs(result.home_prob - 0.59) < 0.01
        assert abs(result.away_prob - 0.41) < 0.01
    
    def test_probabilities_sum_to_one(self):
        """Blended probs should always sum to 1."""
        model = OddsPrices(home_prob=0.6, away_prob=0.4, home_odds_eu=1.67, away_odds_eu=2.50)
        market = OddsPrices(home_prob=0.55, away_prob=0.45, home_odds_eu=1.82, away_odds_eu=2.22)
        
        result = blend_odds_prices(market=market, model=model)
        total = result.home_prob + result.away_prob
        assert abs(total - 1.0) < 0.001


class TestAdjustForConfidence:
    """Tests for confidence adjustment."""
    
    def test_full_confidence(self):
        """100% confidence returns base prob."""
        result = adjust_for_confidence(0.7, confidence=1.0, prior=0.5)
        assert result == 0.7
    
    def test_zero_confidence(self):
        """0% confidence returns prior."""
        result = adjust_for_confidence(0.7, confidence=0.0, prior=0.5)
        assert result == 0.5
    
    def test_partial_confidence(self):
        """Partial confidence regresses toward prior."""
        result = adjust_for_confidence(0.7, confidence=0.5, prior=0.5)
        # 0.7 * 0.5 + 0.5 * 0.5 = 0.35 + 0.25 = 0.6
        assert result == 0.6
    
    def test_custom_prior(self):
        """Works with custom prior."""
        result = adjust_for_confidence(0.8, confidence=0.5, prior=0.6)
        # 0.8 * 0.5 + 0.6 * 0.5 = 0.4 + 0.3 = 0.7
        assert result == 0.7








