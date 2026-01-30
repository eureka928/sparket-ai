"""Enhanced Elo-based odds engine.

Features:
- Sport-specific K-factors and home field advantage
- Margin of victory adjustments
- Log5 probability conversion
- Rating persistence and decay
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sparket.miner.base.engines.interface import OddsEngine, OddsPrices
from sparket.miner.custom.config import EloConfig


@dataclass
class TeamRating:
    """Elo rating for a team."""

    team_code: str
    league: str
    rating: float = 1500.0
    games_played: int = 0
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage."""
        return {
            "team_code": self.team_code,
            "league": self.league,
            "rating": self.rating,
            "games_played": self.games_played,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamRating":
        """Deserialize from dict."""
        return cls(
            team_code=data["team_code"],
            league=data["league"],
            rating=data.get("rating", 1500.0),
            games_played=data.get("games_played", 0),
            last_updated=(
                datetime.fromisoformat(data["last_updated"])
                if data.get("last_updated")
                else None
            ),
        )


class EloRatingStore:
    """Persistent storage for Elo ratings."""

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize the store.

        Args:
            data_path: Path to store ratings JSON. If None, in-memory only.
        """
        self._ratings: Dict[str, TeamRating] = {}
        self._data_path = Path(data_path) if data_path else None

        if self._data_path and self._data_path.exists():
            self._load()

    def get(self, team_code: str, league: str, initial: float = 1500.0) -> TeamRating:
        """Get rating for a team, creating if needed."""
        key = f"{league}:{team_code}"
        if key not in self._ratings:
            self._ratings[key] = TeamRating(
                team_code=team_code,
                league=league,
                rating=initial,
            )
        return self._ratings[key]

    def update(self, rating: TeamRating) -> None:
        """Update a team's rating."""
        key = f"{rating.league}:{rating.team_code}"
        self._ratings[key] = rating
        self._save()

    def _load(self) -> None:
        """Load ratings from disk."""
        if not self._data_path or not self._data_path.exists():
            return
        try:
            with open(self._data_path) as f:
                data = json.load(f)
            self._ratings = {
                k: TeamRating.from_dict(v) for k, v in data.items()
            }
        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh if corrupted

    def _save(self) -> None:
        """Save ratings to disk."""
        if not self._data_path:
            return
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._data_path, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._ratings.items()},
                f,
                indent=2,
            )

    def apply_season_decay(self, decay: float = 0.75) -> None:
        """Apply season decay - regress all ratings toward 1500.

        Called at start of new season to prevent stale ratings.
        """
        for rating in self._ratings.values():
            rating.rating = 1500.0 + decay * (rating.rating - 1500.0)
        self._save()


class EloEngine(OddsEngine):
    """Elo-based odds generation engine.

    Uses Elo ratings with sport-specific parameters to generate
    win probabilities, then converts to EU decimal odds.

    Key features:
    - Dynamic K-factors by sport
    - Home field advantage
    - Margin of victory multipliers
    - Log5 probability conversion

    Usage:
        engine = EloEngine(config=EloConfig())
        odds = await engine.get_odds({
            "market_id": 123,
            "kind": "MONEYLINE",
            "home_team": "KC",
            "away_team": "BUF",
            "sport": "NFL",
        })
    """

    def __init__(
        self,
        config: Optional[EloConfig] = None,
        vig: float = 0.045,
        ratings_path: Optional[str] = None,
    ) -> None:
        """Initialize the Elo engine.

        Args:
            config: Elo configuration (K-factors, home advantage, etc.)
            vig: Vigorish to apply to odds (default 4.5%)
            ratings_path: Path to persist ratings (optional)
        """
        self.config = config or EloConfig()
        self.vig = vig
        self._store = EloRatingStore(data_path=ratings_path)

    async def get_odds(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Generate odds for a market using Elo ratings.

        Args:
            market: Market info with home_team, away_team, sport, kind

        Returns:
            OddsPrices with probabilities and decimal odds
        """
        return self.get_odds_sync(market)

    def get_odds_sync(self, market: Dict[str, Any]) -> Optional[OddsPrices]:
        """Synchronous odds generation."""
        home_team = market.get("home_team", "")
        away_team = market.get("away_team", "")
        sport = market.get("sport", "NFL")
        kind = market.get("kind", "MONEYLINE").upper()

        if not home_team or not away_team:
            return None

        # Get ratings
        home_rating = self._store.get(home_team, sport, self.config.initial_rating)
        away_rating = self._store.get(away_team, sport, self.config.initial_rating)

        # Calculate win probabilities
        home_prob = self._calculate_win_probability(
            home_rating.rating,
            away_rating.rating,
            sport,
            is_home=True,
        )
        away_prob = 1.0 - home_prob

        # Clamp probabilities to validator-accepted range (0.001, 0.999)
        home_prob = max(0.001, min(0.999, home_prob))
        away_prob = max(0.001, min(0.999, away_prob))

        # Convert to odds
        home_odds = self._probability_to_odds(home_prob)
        away_odds = self._probability_to_odds(away_prob)

        # For spreads/totals, use same probabilities (simplified)
        # A more sophisticated approach would adjust for line values
        return OddsPrices(
            home_prob=home_prob,
            away_prob=away_prob,
            home_odds_eu=home_odds,
            away_odds_eu=away_odds,
            over_prob=home_prob if kind == "TOTAL" else None,
            under_prob=away_prob if kind == "TOTAL" else None,
            over_odds_eu=home_odds if kind == "TOTAL" else None,
            under_odds_eu=away_odds if kind == "TOTAL" else None,
        )

    def _calculate_win_probability(
        self,
        rating_a: float,
        rating_b: float,
        sport: str,
        is_home: bool = False,
    ) -> float:
        """Calculate win probability using Elo formula.

        P(A wins) = 1 / (1 + 10^((Rb - Ra + HFA) / 400))

        Where HFA is home field advantage in Elo points.
        """
        # Apply home field advantage
        hfa = self.config.get_home_advantage(sport) if is_home else 0

        # Elo win probability formula
        exponent = (rating_b - rating_a - hfa) / 400.0
        prob = 1.0 / (1.0 + math.pow(10, exponent))

        return prob

    def _probability_to_odds(self, prob: float) -> float:
        """Convert probability to EU decimal odds with vig.

        Odds = 1 / (prob + vig/2)

        Validator bounds: odds_eu in (1.01, 1000], imp_prob in (0.001, 0.999)
        """
        # Add half the vig to each side
        implied_prob = prob + (self.vig / 2)
        # Clamp to validator-accepted range
        implied_prob = max(0.001, min(0.999, implied_prob))
        odds = 1.0 / implied_prob
        # Clamp odds to validator-accepted range (1.01, 1000]
        odds = max(1.01, min(1000.0, odds))
        return round(odds, 2)

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        home_score: int,
        away_score: int,
    ) -> None:
        """Update Elo ratings after a game result.

        Uses standard Elo update formula with optional MOV adjustment:
        New_Ra = Ra + K * MOV_mult * (actual - expected)

        Args:
            home_team: Home team code
            away_team: Away team code
            sport: Sport/league code
            home_score: Final score for home team
            away_score: Final score for away team
        """
        home_rating = self._store.get(home_team, sport, self.config.initial_rating)
        away_rating = self._store.get(away_team, sport, self.config.initial_rating)

        # Get K-factor for sport
        k = self.config.get_k_factor(sport)

        # Calculate expected outcome
        expected_home = self._calculate_win_probability(
            home_rating.rating,
            away_rating.rating,
            sport,
            is_home=True,
        )
        expected_away = 1.0 - expected_home

        # Actual outcome (1 = win, 0.5 = tie, 0 = loss)
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif away_score > home_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5

        # Margin of victory multiplier
        # Pass winner/loser ratings in correct order
        mov = abs(home_score - away_score)
        if home_score > away_score:
            winner_rating = home_rating.rating
            loser_rating = away_rating.rating
        elif away_score > home_score:
            winner_rating = away_rating.rating
            loser_rating = home_rating.rating
        else:
            # Draw - no MOV adjustment
            winner_rating = home_rating.rating
            loser_rating = away_rating.rating
        mov_mult = self._mov_multiplier(mov, winner_rating, loser_rating)

        # Update ratings
        delta_home = k * mov_mult * (actual_home - expected_home)
        delta_away = k * mov_mult * (actual_away - expected_away)

        home_rating.rating += delta_home
        home_rating.games_played += 1
        home_rating.last_updated = datetime.now(timezone.utc)

        away_rating.rating += delta_away
        away_rating.games_played += 1
        away_rating.last_updated = datetime.now(timezone.utc)

        self._store.update(home_rating)
        self._store.update(away_rating)

    def _mov_multiplier(
        self,
        margin: int,
        winner_rating: float,
        loser_rating: float,
    ) -> float:
        """Calculate margin of victory multiplier.

        Uses FiveThirtyEight-style formula to prevent blowout inflation:
        mult = ln(margin + 1) * (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))

        This gives higher weight to close games and accounts for
        expected blowouts (good team vs bad team).
        """
        if margin == 0:
            return 1.0

        if self.config.mov_multiplier == 0:
            return 1.0

        # Log scaling for margin
        log_margin = math.log(margin + 1)

        # Reduce multiplier for expected blowouts
        rating_diff = abs(winner_rating - loser_rating)
        expected_factor = 2.2 / (rating_diff * 0.001 + 2.2)

        mult = log_margin * expected_factor * self.config.mov_multiplier
        return max(0.5, min(2.0, mult))  # Clamp to reasonable range

    def log5_probability(
        self,
        team_a_strength: float,
        team_b_strength: float,
    ) -> float:
        """Calculate win probability using Log5 formula.

        This is Bill James' formula for head-to-head probability:
        P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)

        Where pA and pB are each team's general win rates.

        Args:
            team_a_strength: Team A's general win rate (0-1)
            team_b_strength: Team B's general win rate (0-1)

        Returns:
            Probability that team A beats team B
        """
        pA = max(0.01, min(0.99, team_a_strength))
        pB = max(0.01, min(0.99, team_b_strength))

        numerator = pA - (pA * pB)
        denominator = pA + pB - (2 * pA * pB)

        if denominator == 0:
            return 0.5

        return numerator / denominator

    def elo_to_win_rate(self, rating: float, league_avg: float = 1500.0) -> float:
        """Convert Elo rating to approximate win rate against league.

        This gives an estimate of the team's "strength" as a probability,
        useful for Log5 calculations.

        A 1500 team has 50% win rate against average opposition.
        Each 100 Elo points ≈ 6.4% higher win rate.
        """
        exponent = (rating - league_avg) / 400.0
        expected = 1.0 / (1.0 + math.pow(10, -exponent))
        return expected

    def get_team_rating(self, team_code: str, league: str) -> float:
        """Get current Elo rating for a team."""
        return self._store.get(team_code, league, self.config.initial_rating).rating

    def set_team_rating(
        self,
        team_code: str,
        league: str,
        rating: float,
    ) -> None:
        """Manually set a team's rating (e.g., from external source)."""
        team_rating = self._store.get(team_code, league)
        team_rating.rating = rating
        team_rating.last_updated = datetime.now(timezone.utc)
        self._store.update(team_rating)
