"""Unit tests for team strength calculation."""

import pytest

from sparket.miner.base.data.stats import TeamStats
from sparket.miner.base.model.strength import calculate_team_strength


class TestTeamStrength:
    """Tests for calculate_team_strength function."""
    
    def test_perfect_team_high_strength(self):
        """16-0 team should have high strength."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=16,
            losses=0,
            games_played=16,
        )
        strength = calculate_team_strength(stats)
        assert strength > 0.65, f"Perfect team should have strength > 0.65, got {strength}"
    
    def test_terrible_team_low_strength(self):
        """0-16 team should have low strength."""
        stats = TeamStats(
            team_code="CLE",
            league="NFL",
            wins=0,
            losses=16,
            games_played=16,
        )
        strength = calculate_team_strength(stats)
        assert strength < 0.35, f"Winless team should have strength < 0.35, got {strength}"
    
    def test_average_team_near_half(self):
        """8-8 team should be near 0.5."""
        stats = TeamStats(
            team_code="DAL",
            league="NFL",
            wins=8,
            losses=8,
            games_played=16,
        )
        strength = calculate_team_strength(stats)
        assert 0.45 < strength < 0.55, f"Average team should be near 0.5, got {strength}"
    
    def test_home_boost_applied(self):
        """Home team gets boost from home record."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=8,
            losses=8,
            home_wins=6,
            home_losses=2,
            away_wins=2,
            away_losses=6,
        )
        home_strength = calculate_team_strength(stats, at_home=True)
        away_strength = calculate_team_strength(stats, at_home=False)
        assert home_strength > away_strength, (
            f"Home strength ({home_strength}) should exceed away ({away_strength})"
        )
    
    def test_recent_form_matters(self):
        """Hot streak should boost strength."""
        base_stats = {
            "team_code": "TEST",
            "league": "NFL",
            "wins": 8,
            "losses": 8,
            "games_played": 16,
        }
        cold = TeamStats(**base_stats, last_5_wins=1, last_5_losses=4)
        hot = TeamStats(**base_stats, last_5_wins=4, last_5_losses=1)
        
        cold_strength = calculate_team_strength(cold)
        hot_strength = calculate_team_strength(hot)
        
        assert hot_strength > cold_strength, (
            f"Hot team ({hot_strength}) should be stronger than cold team ({cold_strength})"
        )
    
    def test_strength_bounded(self):
        """Strength should always be within valid bounds."""
        # Extreme case: perfect team
        perfect = TeamStats(team_code="A", league="NFL", wins=16, losses=0)
        assert 0.25 <= calculate_team_strength(perfect) <= 0.75
        
        # Extreme case: winless team
        winless = TeamStats(team_code="B", league="NFL", wins=0, losses=16)
        assert 0.25 <= calculate_team_strength(winless) <= 0.75
        
        # No games played
        new_team = TeamStats(team_code="C", league="NFL", wins=0, losses=0)
        assert 0.25 <= calculate_team_strength(new_team) <= 0.75
    
    def test_point_differential_matters(self):
        """Point differential should affect strength."""
        # Same record, different point differential
        base = {"team_code": "TEST", "league": "NFL", "wins": 8, "losses": 8}
        
        good_pd = TeamStats(**base, points_for=400, points_against=320)
        bad_pd = TeamStats(**base, points_for=320, points_against=400)
        
        good_strength = calculate_team_strength(good_pd)
        bad_strength = calculate_team_strength(bad_pd)
        
        assert good_strength > bad_strength, (
            f"Positive point diff ({good_strength}) should beat negative ({bad_strength})"
        )
    
    def test_custom_weights(self):
        """Custom weights should be applied."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=12,
            losses=4,
            last_5_wins=2,  # Recent form is bad
            last_5_losses=3,
        )
        
        # Default weights
        default_strength = calculate_team_strength(stats)
        
        # Weights that emphasize recent form
        form_weights = {
            "season": 0.20,
            "home_away": 0.10,
            "recent_form": 0.60,
            "advanced": 0.10,
        }
        form_strength = calculate_team_strength(stats, weights=form_weights)
        
        # With bad recent form weighted heavily, strength should be lower
        assert form_strength < default_strength, (
            f"Form-weighted ({form_strength}) should be lower than default ({default_strength})"
        )








