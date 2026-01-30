"""Backtest the custom miner against historical or simulated data.

This module provides utilities to:
1. Generate simulated games with known outcomes
2. Run the custom miner's predictions
3. Evaluate using the ACTUAL validator metrics (CLV, Brier, PSS, SOS)

The validator scores miners on four dimensions:
- EconDim (50%): CLV/CLE - beat closing lines
- InfoDim (30%): SOS + lead ratio - originality and speed
- ForecastDim (10%): Brier/LogLoss - calibrated predictions
- SkillDim (10%): PSS vs closing - beat market baseline

Usage:
    # Quick test with simulated data
    python -m sparket.miner.custom.backtest --games 100

    # Test with specific sport
    python -m sparket.miner.custom.backtest --games 50 --sport NFL

    # Verbose output
    python -m sparket.miner.custom.backtest --games 100 -v

    # With pre-trained Elo ratings
    python -m sparket.miner.custom.backtest --games 200 --warmup 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sparket.miner.custom.config import CustomMinerConfig
from sparket.miner.custom.models.calibration.isotonic import IsotonicCalibrator
from sparket.miner.custom.models.engines.elo import EloEngine
from sparket.miner.custom.models.engines.ensemble import EnsembleEngine
from sparket.miner.custom.models.engines.poisson import PoissonEngine
from sparket.miner.custom.data.seed_elo import seed_elo_ratings, ALL_RATINGS


@dataclass
class MockMarketOdds:
    """Mock market odds for backtest."""
    home_prob: float
    num_books: int = 5
    has_pinnacle: bool = True

# Import actual validator scoring metrics
from sparket.validator.scoring.metrics.clv import compute_clv, CLVResult
from sparket.validator.scoring.metrics.proper_scoring import (
    brier_score,
    log_loss,
    pss,
    outcome_to_vector,
)
import numpy as np


@dataclass
class SimulatedGame:
    """A simulated game with known outcome."""

    game_id: int
    home_team: str
    away_team: str
    sport: str
    true_home_prob: float  # Ground truth probability
    home_score: int
    away_score: int
    start_time: datetime
    closing_odds_home: float  # Market closing odds
    closing_odds_away: float

    @property
    def winner(self) -> str:
        if self.home_score > self.away_score:
            return "HOME"
        elif self.away_score > self.home_score:
            return "AWAY"
        return "DRAW"

    @property
    def actual_outcome(self) -> float:
        """1.0 if home won, 0.0 if away won, 0.5 if draw."""
        if self.home_score > self.away_score:
            return 1.0
        elif self.away_score > self.home_score:
            return 0.0
        return 0.5


@dataclass
class PredictionResult:
    """Result of a single prediction."""

    game_id: int
    predicted_home_prob: float
    actual_outcome: float
    true_prob: float
    closing_odds_home: float

    # Metrics
    brier_score: float = 0.0
    log_loss: float = 0.0
    clv: float = 0.0  # Closing line value
    cle: float = 0.0  # Closing line edge
    pss_brier: float = 0.0  # Probability skill score vs closing


@dataclass
class BacktestResults:
    """Aggregated backtest results."""

    total_games: int
    predictions: List[PredictionResult] = field(default_factory=list)

    # Aggregate metrics
    avg_brier: float = 0.0
    avg_log_loss: float = 0.0
    avg_clv: float = 0.0
    avg_cle: float = 0.0
    avg_pss_brier: float = 0.0
    calibration_slope: float = 0.0
    win_rate: float = 0.0  # % of games where we beat closing line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_games": self.total_games,
            "avg_brier": round(self.avg_brier, 4),
            "avg_log_loss": round(self.avg_log_loss, 4),
            "avg_clv": round(self.avg_clv, 4),
            "avg_cle": round(self.avg_cle, 4),
            "avg_pss_brier": round(self.avg_pss_brier, 4),
            "calibration_slope": round(self.calibration_slope, 3),
            "win_rate_vs_close": round(self.win_rate, 3),
        }


class GameSimulator:
    """Generate simulated games with realistic characteristics."""

    # Team pools by sport
    TEAMS = {
        "NFL": ["KC", "BUF", "SF", "PHI", "DAL", "MIA", "DET", "BAL", "CIN", "JAX",
                "MIN", "SEA", "LAR", "GB", "TB", "NO", "ATL", "CAR", "CHI", "NYG"],
        "NBA": ["BOS", "MIL", "DEN", "PHX", "LAL", "GSW", "MEM", "SAC", "CLE", "NYK",
                "BKN", "PHI", "MIA", "ATL", "CHI", "TOR", "IND", "ORL", "WAS", "DET"],
        "MLB": ["LAD", "ATL", "HOU", "NYY", "SD", "PHI", "SEA", "TB", "TOR", "CLE",
                "STL", "NYM", "MIL", "SF", "MIN", "BAL", "TEX", "CHW", "BOS", "ARI"],
        "NHL": ["BOS", "CAR", "NJ", "TOR", "EDM", "COL", "DAL", "VGK", "LAK", "MIN",
                "NYR", "SEA", "WPG", "TB", "FLA", "PIT", "CGY", "NSH", "STL", "BUF"],
    }

    # Score distributions by sport (mean, std)
    SCORE_DIST = {
        "NFL": (22, 8),
        "NBA": (112, 12),
        "MLB": (4, 2),
        "NHL": (3, 1.5),
    }

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def generate_games(
        self,
        n_games: int,
        sport: str = "NFL",
        start_date: Optional[datetime] = None,
        market_noise: float = 0.08,  # More realistic market inefficiency
        use_elo_truth: bool = True,  # Use Elo ratings for true probability
    ) -> List[SimulatedGame]:
        """Generate n simulated games.

        Args:
            n_games: Number of games to generate
            sport: Sport/league
            start_date: First game date (defaults to 30 days ago)
            market_noise: Std dev of market error (default 8% - realistic)
            use_elo_truth: Base true probability on Elo ratings

        Returns:
            List of SimulatedGame objects with known outcomes
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)

        teams = self.TEAMS.get(sport, self.TEAMS["NFL"])
        score_mean, score_std = self.SCORE_DIST.get(sport, (20, 8))

        # Get Elo ratings if available
        elo_ratings = ALL_RATINGS.get(sport, {})

        games = []
        for i in range(n_games):
            # Pick random teams
            home, away = self.rng.sample(teams, 2)

            # Generate true probability based on Elo difference
            if use_elo_truth and home in elo_ratings and away in elo_ratings:
                home_elo = elo_ratings[home]
                away_elo = elo_ratings[away]
                # Elo win probability formula with home advantage (~50 points)
                elo_diff = home_elo - away_elo + 50
                true_prob = 1 / (1 + 10 ** (-elo_diff / 400))
            else:
                # Fallback to random
                true_prob = self.rng.gauss(0.52, 0.12)

            true_prob = max(0.20, min(0.80, true_prob))

            # Generate scores based on probability
            home_advantage = (true_prob - 0.5) * 10
            home_score = max(0, int(self.rng.gauss(score_mean + home_advantage, score_std)))
            away_score = max(0, int(self.rng.gauss(score_mean - home_advantage, score_std)))

            # Adjust scores to match probability expectation
            if self.rng.random() < true_prob:
                if home_score <= away_score:
                    home_score = away_score + self.rng.randint(1, 7)
            else:
                if away_score <= home_score:
                    away_score = home_score + self.rng.randint(1, 7)

            # Generate closing odds (market has realistic noise)
            # Real markets have ~5-10% error vs true probability
            noise = self.rng.gauss(0, market_noise)
            market_prob = true_prob + noise
            market_prob = max(0.15, min(0.85, market_prob))

            vig = 0.045
            closing_home = round(1 / (market_prob + vig / 2), 2)
            closing_away = round(1 / ((1 - market_prob) + vig / 2), 2)

            games.append(SimulatedGame(
                game_id=i + 1,
                home_team=home,
                away_team=away,
                sport=sport,
                true_home_prob=true_prob,
                home_score=home_score,
                away_score=away_score,
                start_time=start_date + timedelta(days=i // 4, hours=(i % 4) * 3),
                closing_odds_home=closing_home,
                closing_odds_away=closing_away,
            ))

        return games


class Backtester:
    """Run backtest of custom miner against simulated or historical data."""

    def __init__(
        self,
        config: Optional[CustomMinerConfig] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        """Initialize backtester.

        Args:
            config: Miner configuration
            data_dir: Directory for persistent data (Elo ratings)
        """
        self.config = config or CustomMinerConfig()

        # Use temp directory for backtest to not pollute real data
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path("/tmp/sparket_backtest")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize engines (fresh for backtest)
        self._elo = EloEngine(
            config=self.config.elo,
            vig=self.config.vig,
            ratings_path=str(self._data_dir / "backtest_elo.json"),
        )

        self._poisson = PoissonEngine(
            data_path=str(self._data_dir / "backtest_poisson.json"),
        )

        # Ensemble combines Elo + Market + Poisson
        self._ensemble = EnsembleEngine(
            elo_engine=self._elo,
            poisson_engine=self._poisson,
            base_weights=self.config.engine_weights,
            confidence_scaling=True,
        )

        self._calibrator = IsotonicCalibrator(
            min_samples=self.config.calibration.min_samples,
        )

        self._use_ensemble = True  # Can toggle for comparison

    def run(
        self,
        games: List[SimulatedGame],
        update_elo: bool = True,
        verbose: bool = False,
    ) -> BacktestResults:
        """Run backtest on a set of games.

        Args:
            games: List of games to test
            update_elo: Whether to update Elo ratings after each game
            verbose: Print per-game results

        Returns:
            BacktestResults with aggregate metrics
        """
        results = BacktestResults(total_games=len(games))

        for game in games:
            # Get miner prediction
            market = {
                "market_id": game.game_id,
                "kind": "MONEYLINE",
                "home_team": game.home_team,
                "away_team": game.away_team,
                "sport": game.sport,
            }

            # Use ensemble model with simulated market odds
            if self._use_ensemble:
                # Create mock market odds from closing line
                closing_prob = 1.0 / game.closing_odds_home
                # Remove vig to get implied prob
                closing_prob = closing_prob / (closing_prob + 1.0 / game.closing_odds_away)
                mock_market = MockMarketOdds(home_prob=closing_prob)

                ensemble_pred = self._ensemble.predict(market, market_odds=mock_market)
                if ensemble_pred is None:
                    continue
                pred_home = ensemble_pred.home_prob
            else:
                # Elo only (for comparison)
                odds = self._elo.get_odds_sync(market)
                if odds is None:
                    continue
                pred_home = odds.home_prob

            # Apply calibration if fitted
            if self._calibrator.is_fitted:
                pred_home, _ = self._calibrator.calibrate_pair(pred_home, 1 - pred_home)

            # Calculate metrics
            result = self._evaluate_prediction(game, pred_home)
            results.predictions.append(result)

            if verbose:
                win = "✓" if result.clv > 0 else "✗"
                print(f"Game {game.game_id}: {game.home_team} vs {game.away_team} "
                      f"| Pred: {pred_home:.3f} | True: {game.true_home_prob:.3f} "
                      f"| CLV: {result.clv:+.3f} {win}")

            # Update Elo ratings with outcome
            if update_elo:
                self._elo.update_ratings(
                    home_team=game.home_team,
                    away_team=game.away_team,
                    sport=game.sport,
                    home_score=game.home_score,
                    away_score=game.away_score,
                )

            # Update calibration with ensemble prediction (what we actually submitted)
            self._calibrator.add_sample(
                predicted=pred_home,  # Submitted prediction
                actual=game.actual_outcome,
            )

            # Periodically refit calibration
            if len(results.predictions) % 50 == 0 and len(results.predictions) >= self.config.calibration.min_samples:
                self._calibrator.fit()

        # Calculate aggregate metrics
        self._aggregate_results(results)

        return results

    def _evaluate_prediction(
        self,
        game: SimulatedGame,
        pred_home: float,
        submitted_ts: Optional[float] = None,
    ) -> PredictionResult:
        """Evaluate a single prediction using actual validator metrics.

        Uses the same scoring functions as the validator for accuracy.
        """
        actual = game.actual_outcome
        true_prob = game.true_home_prob

        # Convert to probability vectors for validator metrics
        miner_probs = np.array([pred_home, 1 - pred_home], dtype=np.float64)

        # Closing line probabilities (from market odds with vig removed)
        close_implied_home = 1 / game.closing_odds_home
        close_implied_away = 1 / game.closing_odds_away
        close_total = close_implied_home + close_implied_away
        close_prob_home = close_implied_home / close_total  # Remove vig
        closing_probs = np.array([close_prob_home, 1 - close_prob_home], dtype=np.float64)

        # Outcome vector (1 = home won, 0 = away won)
        outcome_idx = 0 if actual > 0.5 else 1
        outcome_vec = outcome_to_vector(outcome_idx, 2)

        # Use ACTUAL validator Brier score
        brier = brier_score(miner_probs, outcome_vec)
        close_brier = brier_score(closing_probs, outcome_vec)

        # Use ACTUAL validator Log Loss
        ll = log_loss(miner_probs, outcome_vec)
        close_ll = log_loss(closing_probs, outcome_vec)

        # Use ACTUAL validator PSS
        pss_brier = pss(brier, close_brier)
        pss_log = pss(ll, close_ll)

        # Use ACTUAL validator CLV/CLE
        # Convert our probability to odds
        miner_odds = 1 / max(0.01, pred_home + self.config.vig / 2)

        event_ts = game.start_time.timestamp()
        sub_ts = submitted_ts or (event_ts - 7 * 24 * 3600)  # Default: 7 days before

        clv_result = compute_clv(
            miner_odds=miner_odds,
            miner_prob=pred_home,
            truth_odds=game.closing_odds_home,
            truth_prob=close_prob_home,
            submitted_ts=sub_ts,
            event_start_ts=event_ts,
        )

        return PredictionResult(
            game_id=game.game_id,
            predicted_home_prob=pred_home,
            actual_outcome=actual,
            true_prob=true_prob,
            closing_odds_home=game.closing_odds_home,
            brier_score=brier,
            log_loss=ll,
            clv=clv_result.clv_prob,  # Use probability-based CLV
            cle=clv_result.cle,
            pss_brier=pss_brier,
        )

    def _aggregate_results(self, results: BacktestResults) -> None:
        """Calculate aggregate metrics."""
        if not results.predictions:
            return

        n = len(results.predictions)

        results.avg_brier = sum(p.brier_score for p in results.predictions) / n
        results.avg_log_loss = sum(p.log_loss for p in results.predictions) / n
        results.avg_clv = sum(p.clv for p in results.predictions) / n
        results.avg_cle = sum(p.cle for p in results.predictions) / n
        results.avg_pss_brier = sum(p.pss_brier for p in results.predictions) / n

        # Win rate vs closing
        wins = sum(1 for p in results.predictions if p.pss_brier > 0)
        results.win_rate = wins / n

        # Calibration slope
        results.calibration_slope = self._calculate_calibration_slope(results.predictions)

    def _calculate_calibration_slope(self, predictions: List[PredictionResult]) -> float:
        """Calculate calibration slope (1.0 = perfectly calibrated)."""
        if not predictions:
            return 1.0

        n = len(predictions)
        sum_x = sum(p.predicted_home_prob for p in predictions)
        sum_y = sum(p.actual_outcome for p in predictions)
        sum_xy = sum(p.predicted_home_prob * p.actual_outcome for p in predictions)
        sum_xx = sum(p.predicted_home_prob ** 2 for p in predictions)

        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            return 1.0

        return (n * sum_xy - sum_x * sum_y) / denom


def main():
    """Run backtest from command line."""
    parser = argparse.ArgumentParser(description="Backtest custom miner")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--sport", default="NFL", help="Sport to simulate (NFL, NBA, MLB, NHL)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup games to train Elo (not scored)")
    parser.add_argument("--seed-elo", action="store_true", help="Pre-seed Elo with real ratings")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-game results")
    parser.add_argument("--output", help="Output JSON file for results")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Custom Miner Backtest (using validator metrics)")
    print(f"{'='*60}")
    print(f"Sport: {args.sport}")
    print(f"Games: {args.games}")
    print(f"Warmup: {args.warmup} (Elo training, not scored)")
    print(f"Seed Elo: {args.seed_elo} (pre-seed with real ratings)")
    print(f"Random Seed: {args.seed or 'random'}")
    print(f"{'='*60}\n")

    # Generate games
    simulator = GameSimulator(seed=args.seed)

    # Create backtester
    backtester = Backtester()

    # Pre-seed Elo with real ratings
    if args.seed_elo:
        print(f"Pre-seeding Elo with real {args.sport} ratings...")
        seed_elo_ratings(backtester._elo, sports=[args.sport])
        print(f"Seeded {len(ALL_RATINGS.get(args.sport, {}))} teams.\n")

    # Warmup phase - train Elo without scoring
    if args.warmup > 0:
        print(f"Warming up Elo ratings with {args.warmup} games...")
        warmup_games = simulator.generate_games(args.warmup, sport=args.sport)
        backtester.run(warmup_games, update_elo=True, verbose=False)
        print(f"Warmup complete. Elo ratings established.\n")

    # Generate test games (after warmup, so different matchups)
    games = simulator.generate_games(args.games, sport=args.sport)

    # Run backtest
    results = backtester.run(games, verbose=args.verbose)

    # Print results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Games:        {results.total_games}")
    print(f"Avg Brier Score:    {results.avg_brier:.4f}  (lower is better, 0.25 = random)")
    print(f"Avg Log Loss:       {results.avg_log_loss:.4f}  (lower is better)")
    print(f"Avg CLV:            {results.avg_clv:+.4f}  (positive = beating market)")
    print(f"Avg CLE:            {results.avg_cle:+.4f}  (expected edge per bet)")
    print(f"Avg PSS vs Close:   {results.avg_pss_brier:+.4f}  (positive = better than market)")
    print(f"Win Rate vs Close:  {results.win_rate:.1%}")
    print(f"Calibration Slope:  {results.calibration_slope:.3f}  (1.0 = perfect)")
    print(f"{'='*60}")

    # Interpretation
    print("\nINTERPRETATION:")

    # CLV/CLE - most important for EconDim (50% of score)
    if results.avg_cle > 0:
        print(f"✓ Positive CLE (+{results.avg_cle:.3f}) = profitable edge (EconDim)")
    else:
        print(f"✗ Negative CLE ({results.avg_cle:.3f}) = losing to market")

    # Win rate
    if results.win_rate > 0.50:
        print(f"✓ Win rate {results.win_rate:.1%} > 50% = beating market majority")
    else:
        print(f"⚠ Win rate {results.win_rate:.1%} < 50% = losing to market majority")

    # Calibration
    if 0.8 <= results.calibration_slope <= 1.2:
        print(f"✓ Calibration {results.calibration_slope:.2f} ≈ 1.0 (good for ForecastDim)")
    elif results.calibration_slope < 0.8:
        print(f"⚠ Calibration {results.calibration_slope:.2f} < 0.8 = underconfident")
    else:
        print(f"⚠ Calibration {results.calibration_slope:.2f} > 1.2 = overconfident")

    # Brier score
    if results.avg_brier < 0.25:
        print(f"✓ Brier {results.avg_brier:.3f} < 0.25 = better than random")
    else:
        print(f"⚠ Brier {results.avg_brier:.3f} ≥ 0.25 = room for improvement")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
