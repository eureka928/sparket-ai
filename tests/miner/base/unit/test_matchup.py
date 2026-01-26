"""Unit tests for matchup probability calculation."""

import pytest

from sparket.miner.base.model.matchup import (
    strength_to_probability,
    odds_to_probability,
    probability_to_odds,
)


class TestMatchupProbability:
    """Tests for Log5 matchup probability."""
    
    def test_equal_teams_fifty_fifty(self):
        """Equal strength teams should be ~50/50."""
        home_prob, away_prob = strength_to_probability(0.5, 0.5)
        assert abs(home_prob - 0.5) < 0.01, f"Equal teams should be 50/50, got {home_prob}"
        assert abs(away_prob - 0.5) < 0.01
    
    def test_stronger_team_favored(self):
        """Stronger team should have higher probability."""
        home_prob, away_prob = strength_to_probability(0.7, 0.3)
        assert home_prob > 0.7, f"Strong team should be heavily favored, got {home_prob}"
        assert away_prob < 0.3
    
    def test_probabilities_sum_to_one(self):
        """Probabilities must sum to 1."""
        test_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        for h in test_values:
            for a in test_values:
                home_prob, away_prob = strength_to_probability(h, a)
                total = home_prob + away_prob
                assert abs(total - 1.0) < 0.001, f"Probs should sum to 1, got {total}"
    
    def test_symmetry(self):
        """Swapping teams should give inverse probabilities."""
        p1, _ = strength_to_probability(0.7, 0.4)
        _, p2 = strength_to_probability(0.4, 0.7)
        assert abs(p1 - p2) < 0.01, f"Symmetric matchup should give same prob: {p1} vs {p2}"
    
    def test_edge_cases(self):
        """Handle edge cases gracefully."""
        # Very lopsided matchup
        home_prob, away_prob = strength_to_probability(0.9, 0.1)
        assert home_prob > 0.9
        assert 0 < away_prob < 0.1
        
        # Values at bounds
        home_prob, away_prob = strength_to_probability(0.01, 0.99)
        assert 0 < home_prob < 1
        assert 0 < away_prob < 1


class TestOddsConversion:
    """Tests for odds-to-probability conversion."""
    
    def test_even_odds(self):
        """Even odds (1.91 each) should be ~50/50."""
        home_prob, away_prob = odds_to_probability(1.91, 1.91)
        assert abs(home_prob - 0.5) < 0.05
        assert abs(away_prob - 0.5) < 0.05
    
    def test_favorite_conversion(self):
        """Favorite at 1.50 should have ~63% probability."""
        home_prob, away_prob = odds_to_probability(1.50, 2.75)
        # 1/1.50 = 0.667, 1/2.75 = 0.364, sum = 1.031, normalized ~64%
        assert home_prob > 0.6
        assert away_prob < 0.4
    
    def test_removes_vig(self):
        """Conversion should remove the vig (normalize to 1.0)."""
        # Odds with vig: 1.91 / 1.91 = implied 52.4% + 52.4% = 104.8%
        home_prob, away_prob = odds_to_probability(1.91, 1.91)
        total = home_prob + away_prob
        assert abs(total - 1.0) < 0.01, f"Should normalize to 1.0, got {total}"


class TestProbabilityToOdds:
    """Tests for probability-to-odds conversion."""
    
    def test_fifty_percent(self):
        """50% probability with standard vig should be ~1.91."""
        odds = probability_to_odds(0.5, vig=0.045)
        # implied prob = 0.5 + 0.0225 = 0.5225
        # odds = 1/0.5225 = 1.91
        assert 1.85 < odds < 1.95, f"50% should be ~1.91, got {odds}"
    
    def test_favorite_odds(self):
        """High probability should give low odds."""
        odds = probability_to_odds(0.7, vig=0.045)
        assert odds < 1.5, f"70% favorite should have odds < 1.5, got {odds}"
    
    def test_underdog_odds(self):
        """Low probability should give high odds."""
        odds = probability_to_odds(0.3, vig=0.045)
        assert odds > 3.0, f"30% underdog should have odds > 3.0, got {odds}"
    
    def test_vig_impact(self):
        """Higher vig should result in lower odds."""
        low_vig = probability_to_odds(0.5, vig=0.02)
        high_vig = probability_to_odds(0.5, vig=0.08)
        assert low_vig > high_vig, f"Higher vig should give lower odds"








