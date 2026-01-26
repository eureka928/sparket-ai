"""Unit tests for TeamStats dataclass."""

import pytest

from sparket.miner.base.data.stats import TeamStats


class TestTeamStats:
    """Tests for TeamStats dataclass."""
    
    def test_basic_creation(self):
        """Create basic team stats."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=14,
            losses=2,
        )
        
        assert stats.team_code == "KC"
        assert stats.league == "NFL"
        assert stats.wins == 14
        assert stats.losses == 2
    
    def test_games_played_calculated(self):
        """Games played is auto-calculated from W/L."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=10,
            losses=5,
            ties=1,
        )
        
        assert stats.games_played == 16
    
    def test_games_played_override(self):
        """Explicit games_played takes precedence."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=10,
            losses=5,
            games_played=20,  # Override
        )
        
        assert stats.games_played == 20
    
    def test_win_rate(self):
        """Win rate property calculation."""
        stats = TeamStats(team_code="KC", league="NFL", wins=12, losses=4)
        assert stats.win_rate == 0.75
        
        # Zero games should return 0.5
        empty = TeamStats(team_code="NEW", league="NFL")
        assert empty.win_rate == 0.5
    
    def test_home_win_rate(self):
        """Home win rate property."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=12,
            losses=4,
            home_wins=7,
            home_losses=1,
        )
        
        assert stats.home_win_rate == 0.875  # 7/8
        
        # Falls back to overall if no home games
        no_home = TeamStats(team_code="KC", league="NFL", wins=12, losses=4)
        assert no_home.home_win_rate == 0.75
    
    def test_away_win_rate(self):
        """Away win rate property."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=12,
            losses=4,
            away_wins=5,
            away_losses=3,
        )
        
        assert stats.away_win_rate == 0.625  # 5/8
    
    def test_recent_form(self):
        """Recent form property."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=12,
            losses=4,
            last_5_wins=4,
            last_5_losses=1,
        )
        
        assert stats.recent_form == 0.8  # 4/5
        
        # Falls back to win rate if no recent games
        no_recent = TeamStats(team_code="KC", league="NFL", wins=12, losses=4)
        assert no_recent.recent_form == 0.75
    
    def test_point_differential(self):
        """Point differential property."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=16,
            losses=0,
            points_for=480,
            points_against=320,
        )
        
        assert stats.point_differential == 10.0  # (480-320)/16
        
        # None if points not provided
        no_points = TeamStats(team_code="KC", league="NFL", wins=16, losses=0)
        assert no_points.point_differential is None
    
    def test_extra_dict(self):
        """Extra dict for custom data."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=14,
            losses=2,
        )
        
        # Add custom data
        stats.extra["elo_rating"] = 1650
        stats.extra["custom_metric"] = 0.85
        
        assert stats.extra["elo_rating"] == 1650
        assert stats.extra["custom_metric"] == 0.85
    
    def test_optional_advanced_fields(self):
        """Optional advanced fields."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            wins=14,
            losses=2,
            elo_rating=1700,
            strength_of_schedule=0.55,
        )
        
        assert stats.elo_rating == 1700
        assert stats.strength_of_schedule == 0.55
    
    def test_home_games_property(self):
        """Home games count property."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            home_wins=7,
            home_losses=1,
        )
        
        assert stats.home_games == 8
    
    def test_away_games_property(self):
        """Away games count property."""
        stats = TeamStats(
            team_code="KC",
            league="NFL",
            away_wins=6,
            away_losses=2,
        )
        
        assert stats.away_games == 8








