"""Poisson model for TOTAL (over/under) market predictions.

The Poisson distribution models the probability of scoring events (goals, points)
in a game. This is useful for predicting over/under totals.

Key concepts:
- Lambda (λ) = expected scoring rate for a team
- P(k goals) = (λ^k × e^(-λ)) / k!
- Total = sum of both teams' scores

Usage:
    poisson = PoissonEngine()
    prediction = poisson.predict_total(
        home_team="KC",
        away_team="BUF",
        sport="NFL",
        line=45.5,
    )
    # Returns: TotalPrediction(over_prob=0.52, under_prob=0.48, ...)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import bittensor as bt


# Default scoring rates by sport (points/goals per game)
# These are league averages - will be adjusted per team
DEFAULT_SCORING_RATES: Dict[str, Dict[str, float]] = {
    "NFL": {
        "avg_total": 45.0,  # Average total points per game
        "home_boost": 1.5,  # Home team scores ~1.5 more points
        "std_dev": 10.0,    # Standard deviation of totals
    },
    "NBA": {
        "avg_total": 225.0,
        "home_boost": 3.0,
        "std_dev": 20.0,
    },
    "MLB": {
        "avg_total": 8.5,
        "home_boost": 0.3,
        "std_dev": 3.0,
    },
    "NHL": {
        "avg_total": 6.0,
        "home_boost": 0.2,
        "std_dev": 2.0,
    },
}

# Team-specific offensive/defensive adjustments
# Positive = scores more than average, Negative = scores less
# These can be learned from historical data
DEFAULT_TEAM_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # NFL examples (points above/below average)
    "KC": {"offense": 3.0, "defense": -1.0},   # Chiefs: high scoring, good defense
    "BUF": {"offense": 2.5, "defense": -0.5},  # Bills: high scoring
    "SF": {"offense": 1.5, "defense": -2.0},   # 49ers: balanced, great defense
    "DET": {"offense": 2.0, "defense": 1.5},   # Lions: high scoring, weak defense
    "BAL": {"offense": 1.0, "defense": -2.5},  # Ravens: great defense
    "MIA": {"offense": 2.5, "defense": 0.5},   # Dolphins: explosive offense
    "DAL": {"offense": 0.5, "defense": 0.0},   # Cowboys: average
    "PHI": {"offense": 1.0, "defense": -1.5},  # Eagles: good defense
    # NBA examples (points above/below average per team)
    "BOS": {"offense": 5.0, "defense": -3.0},  # Celtics: elite both ways
    "LAL": {"offense": 2.0, "defense": 1.0},   # Lakers: above average offense
    "MIL": {"offense": 3.0, "defense": -1.0},  # Bucks: high scoring
    "PHX": {"offense": 1.5, "defense": 0.5},   # Suns: above average
    "DEN": {"offense": 2.5, "defense": -0.5},  # Nuggets: good both ways
    "GSW": {"offense": 1.0, "defense": 0.0},   # Warriors: average now
}


@dataclass
class TotalPrediction:
    """Prediction for a TOTAL (over/under) market."""
    over_prob: float
    under_prob: float
    expected_total: float
    home_expected: float
    away_expected: float
    line: float

    def __post_init__(self) -> None:
        # Ensure probabilities sum to 1
        total = self.over_prob + self.under_prob
        if total > 0:
            self.over_prob = self.over_prob / total
            self.under_prob = self.under_prob / total


@dataclass
class TeamScoringProfile:
    """Scoring profile for a team."""
    team: str
    sport: str
    offense_adj: float = 0.0  # Points above/below league average
    defense_adj: float = 0.0  # Points allowed above/below average
    games_played: int = 0
    total_points_for: int = 0
    total_points_against: int = 0

    @property
    def avg_points_for(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.total_points_for / self.games_played

    @property
    def avg_points_against(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.total_points_against / self.games_played


class PoissonEngine:
    """Poisson-based model for TOTAL market predictions.

    Uses team offensive/defensive adjustments to predict expected scoring,
    then calculates over/under probabilities using Poisson distribution.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        sport_params: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Initialize Poisson engine.

        Args:
            data_path: Path to save/load team profiles
            sport_params: Override default sport parameters
        """
        self._sport_params = sport_params or DEFAULT_SCORING_RATES
        self._team_adjustments: Dict[str, Dict[str, float]] = dict(DEFAULT_TEAM_ADJUSTMENTS)
        self._team_profiles: Dict[str, TeamScoringProfile] = {}
        self._data_path = Path(data_path) if data_path else None

        if self._data_path and self._data_path.exists():
            self._load()

    def predict_total(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        line: float,
    ) -> Optional[TotalPrediction]:
        """Predict over/under probabilities for a total line.

        Args:
            home_team: Home team code (e.g., "KC")
            away_team: Away team code (e.g., "BUF")
            sport: Sport code (e.g., "NFL")
            line: The total line (e.g., 45.5)

        Returns:
            TotalPrediction with over/under probabilities
        """
        sport_params = self._sport_params.get(sport)
        if not sport_params:
            bt.logging.debug({
                "poisson": "unknown_sport",
                "sport": sport,
            })
            return None

        # Get expected scoring for each team (including opponent defense)
        home_expected = self._get_team_expected(home_team, sport, is_home=True, opponent=away_team)
        away_expected = self._get_team_expected(away_team, sport, is_home=False, opponent=home_team)

        expected_total = home_expected + away_expected

        # Calculate over/under probabilities using Poisson
        over_prob = self._calculate_over_probability(
            expected_total=expected_total,
            line=line,
            std_dev=sport_params.get("std_dev", 10.0),
        )
        under_prob = 1.0 - over_prob

        bt.logging.debug({
            "poisson": "prediction",
            "home_team": home_team,
            "away_team": away_team,
            "sport": sport,
            "line": line,
            "expected_total": round(expected_total, 1),
            "over_prob": round(over_prob, 3),
        })

        return TotalPrediction(
            over_prob=over_prob,
            under_prob=under_prob,
            expected_total=expected_total,
            home_expected=home_expected,
            away_expected=away_expected,
            line=line,
        )

    def _get_team_expected(
        self,
        team: str,
        sport: str,
        is_home: bool,
        opponent: str = "",
    ) -> float:
        """Get expected points/goals for a team.

        Args:
            team: Team code
            sport: Sport code
            is_home: Whether team is playing at home
            opponent: Opponent team code (for defense adjustment)

        Returns:
            Expected points/goals for the team
        """
        sport_params = self._sport_params.get(sport, DEFAULT_SCORING_RATES["NFL"])

        # Base expected = half of average total
        base_expected = sport_params["avg_total"] / 2

        # Apply home boost
        if is_home:
            base_expected += sport_params.get("home_boost", 0) / 2
        else:
            base_expected -= sport_params.get("home_boost", 0) / 2

        # Apply team offense adjustment
        team_adj = self._team_adjustments.get(team, {})
        offense_adj = team_adj.get("offense", 0.0)

        # Apply opponent defense adjustment
        # Negative defense_adj = good defense = fewer points allowed
        opp_adj = self._team_adjustments.get(opponent, {})
        opp_defense_adj = opp_adj.get("defense", 0.0)

        return base_expected + offense_adj + opp_defense_adj

    def _calculate_over_probability(
        self,
        expected_total: float,
        line: float,
        std_dev: float,
    ) -> float:
        """Calculate probability of going over the line.

        Uses normal approximation to Poisson for large expected values.
        For NFL/NBA with high scoring, this is accurate.

        Args:
            expected_total: Expected combined score
            line: The over/under line
            std_dev: Standard deviation of totals for this sport

        Returns:
            Probability of going over the line
        """
        # Z-score: how many standard deviations above/below expected
        z = (line - expected_total) / std_dev

        # Convert to probability using normal CDF
        # P(over) = P(total > line) = 1 - Φ(z)
        over_prob = 1.0 - self._normal_cdf(z)

        # Clamp to reasonable range
        return max(0.05, min(0.95, over_prob))

    def _normal_cdf(self, z: float) -> float:
        """Standard normal cumulative distribution function.

        Uses error function approximation.
        """
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _poisson_pmf(self, k: int, lambda_: float) -> float:
        """Poisson probability mass function.

        P(X = k) = (λ^k × e^(-λ)) / k!

        Args:
            k: Number of events
            lambda_: Expected rate

        Returns:
            Probability of exactly k events
        """
        if lambda_ <= 0:
            return 1.0 if k == 0 else 0.0

        return (lambda_ ** k) * math.exp(-lambda_) / math.factorial(k)

    def update_from_result(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        home_score: int,
        away_score: int,
    ) -> None:
        """Update team profiles after a game result.

        Args:
            home_team: Home team code
            away_team: Away team code
            sport: Sport code
            home_score: Home team's score
            away_score: Away team's score
        """
        # Update home team profile
        home_key = f"{sport}:{home_team}"
        if home_key not in self._team_profiles:
            self._team_profiles[home_key] = TeamScoringProfile(
                team=home_team, sport=sport
            )
        home_profile = self._team_profiles[home_key]
        home_profile.games_played += 1
        home_profile.total_points_for += home_score
        home_profile.total_points_against += away_score

        # Update away team profile
        away_key = f"{sport}:{away_team}"
        if away_key not in self._team_profiles:
            self._team_profiles[away_key] = TeamScoringProfile(
                team=away_team, sport=sport
            )
        away_profile = self._team_profiles[away_key]
        away_profile.games_played += 1
        away_profile.total_points_for += away_score
        away_profile.total_points_against += home_score

        # Recalculate adjustments based on actual performance
        self._update_team_adjustment(home_profile)
        self._update_team_adjustment(away_profile)

        # Save updated profiles
        if self._data_path:
            self._save()

    def _update_team_adjustment(self, profile: TeamScoringProfile) -> None:
        """Update team adjustment based on actual performance."""
        if profile.games_played < 3:
            return  # Need minimum sample

        sport_params = self._sport_params.get(profile.sport, DEFAULT_SCORING_RATES["NFL"])
        league_avg = sport_params["avg_total"] / 2

        # Calculate offense adjustment
        offense_adj = profile.avg_points_for - league_avg

        # Calculate defense adjustment (negative = good defense)
        defense_adj = profile.avg_points_against - league_avg

        # Blend with prior (shrinkage toward 0)
        shrinkage = min(1.0, profile.games_played / 20)

        self._team_adjustments[profile.team] = {
            "offense": offense_adj * shrinkage,
            "defense": defense_adj * shrinkage,
        }

    def _save(self) -> None:
        """Save team profiles to disk."""
        if not self._data_path:
            return

        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "team_adjustments": self._team_adjustments,
            "team_profiles": {
                key: {
                    "team": p.team,
                    "sport": p.sport,
                    "games_played": p.games_played,
                    "total_points_for": p.total_points_for,
                    "total_points_against": p.total_points_against,
                }
                for key, p in self._team_profiles.items()
            },
        }

        with open(self._data_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load team profiles from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path) as f:
                data = json.load(f)

            self._team_adjustments.update(data.get("team_adjustments", {}))

            for key, profile_data in data.get("team_profiles", {}).items():
                self._team_profiles[key] = TeamScoringProfile(
                    team=profile_data["team"],
                    sport=profile_data["sport"],
                    games_played=profile_data["games_played"],
                    total_points_for=profile_data["total_points_for"],
                    total_points_against=profile_data["total_points_against"],
                )

            bt.logging.info({
                "poisson": "loaded",
                "teams": len(self._team_profiles),
            })
        except Exception as e:
            bt.logging.warning({
                "poisson": "load_failed",
                "error": str(e),
            })
