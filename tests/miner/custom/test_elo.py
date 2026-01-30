"""Tests for the Elo engine."""

import pytest
from sparket.miner.custom.models.engines.elo import (
    EloEngine,
    EloConfig,
    EloRatingStore,
    TeamRating,
)


class TestEloConfig:
    """Tests for EloConfig."""

    def test_default_k_factors(self):
        """Test default K-factors are set for all major sports."""
        config = EloConfig()
        assert config.get_k_factor("NFL") == 20.0
        assert config.get_k_factor("NBA") == 12.0
        assert config.get_k_factor("MLB") == 4.0
        assert config.get_k_factor("NHL") == 10.0

    def test_unknown_sport_defaults_to_nfl(self):
        """Unknown sports should default to NFL K-factor."""
        config = EloConfig()
        assert config.get_k_factor("UNKNOWN") == 20.0

    def test_home_advantage_by_sport(self):
        """Test sport-specific home advantage."""
        config = EloConfig()
        # NBA has highest home advantage
        assert config.get_home_advantage("NBA") > config.get_home_advantage("MLB")


class TestEloRatingStore:
    """Tests for EloRatingStore."""

    def test_get_creates_new_rating(self):
        """Getting a non-existent team creates a new rating."""
        store = EloRatingStore()
        rating = store.get("KC", "NFL")

        assert rating.team_code == "KC"
        assert rating.league == "NFL"
        assert rating.rating == 1500.0
        assert rating.games_played == 0

    def test_get_returns_existing_rating(self):
        """Getting an existing team returns the same rating."""
        store = EloRatingStore()
        rating1 = store.get("KC", "NFL")
        rating1.rating = 1600.0
        store.update(rating1)

        rating2 = store.get("KC", "NFL")
        assert rating2.rating == 1600.0

    def test_season_decay(self):
        """Season decay regresses ratings toward 1500."""
        store = EloRatingStore()
        rating = store.get("KC", "NFL")
        rating.rating = 1700.0
        store.update(rating)

        store.apply_season_decay(decay=0.5)

        # Should be 1500 + 0.5 * (1700 - 1500) = 1600
        updated = store.get("KC", "NFL")
        assert updated.rating == 1600.0


class TestEloEngine:
    """Tests for EloEngine."""

    def test_equal_teams_return_50_percent(self):
        """Equal-rated teams should have roughly 50% win probability each."""
        engine = EloEngine()

        # Both teams at default 1500
        market = {
            "market_id": 1,
            "kind": "MONEYLINE",
            "home_team": "A",
            "away_team": "B",
            "sport": "NFL",
        }

        odds = engine.get_odds_sync(market)

        # Home team gets small advantage, so slightly > 50%
        assert 0.50 < odds.home_prob < 0.60
        assert odds.home_prob + odds.away_prob == pytest.approx(1.0, abs=0.01)

    def test_higher_rated_team_is_favorite(self):
        """Higher-rated team should be the favorite."""
        engine = EloEngine()

        # Set one team higher
        engine.set_team_rating("KC", "NFL", 1700.0)
        engine.set_team_rating("BUF", "NFL", 1300.0)

        market = {
            "market_id": 1,
            "kind": "MONEYLINE",
            "home_team": "KC",
            "away_team": "BUF",
            "sport": "NFL",
        }

        odds = engine.get_odds_sync(market)

        # KC should be heavy favorite
        assert odds.home_prob > 0.80
        assert odds.away_prob < 0.20

    def test_odds_have_vig(self):
        """Odds should include vigorish."""
        engine = EloEngine(vig=0.045)

        market = {
            "market_id": 1,
            "kind": "MONEYLINE",
            "home_team": "A",
            "away_team": "B",
            "sport": "NFL",
        }

        odds = engine.get_odds_sync(market)

        # Implied probabilities from odds should sum > 1 (vig)
        implied_sum = (1 / odds.home_odds_eu) + (1 / odds.away_odds_eu)
        assert implied_sum > 1.0

    def test_update_ratings_after_game(self):
        """Ratings should update after game results."""
        engine = EloEngine()

        # Get initial ratings
        initial_home = engine.get_team_rating("KC", "NFL")
        initial_away = engine.get_team_rating("BUF", "NFL")

        # KC wins by 10
        engine.update_ratings("KC", "BUF", "NFL", home_score=27, away_score=17)

        # Winner should gain, loser should lose
        assert engine.get_team_rating("KC", "NFL") > initial_home
        assert engine.get_team_rating("BUF", "NFL") < initial_away

    def test_log5_probability(self):
        """Log5 formula should work correctly."""
        engine = EloEngine()

        # Team with 70% win rate vs team with 30%
        prob = engine.log5_probability(0.7, 0.3)

        # Strong team should have ~84% chance
        assert 0.80 < prob < 0.90

    def test_elo_to_win_rate(self):
        """Elo rating should convert to win rate correctly."""
        engine = EloEngine()

        # 1500 = average = 50%
        assert engine.elo_to_win_rate(1500) == pytest.approx(0.5, abs=0.01)

        # Higher rating = higher win rate
        assert engine.elo_to_win_rate(1700) > 0.5
        assert engine.elo_to_win_rate(1300) < 0.5
