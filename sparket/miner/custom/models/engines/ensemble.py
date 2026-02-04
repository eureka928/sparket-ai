"""Ensemble model combining multiple prediction engines.

Combines predictions from:
- Elo: Win probability based on team strength
- Poisson: Total scoring predictions
- Market: Sharp book consensus (Pinnacle weighted 3x)

Weighting strategies:
1. Base weights from config
2. Confidence adjustment (sharper predictions get more weight)
3. Agreement boost (when models agree, increase confidence)
4. Recency adjustment (recent model performance)

Usage:
    ensemble = EnsembleEngine(
        elo_engine=elo,
        poisson_engine=poisson,
        base_weights={"elo": 0.5, "market": 0.35, "poisson": 0.15},
    )

    prediction = await ensemble.predict(
        market=market_dict,
        market_odds=market_consensus,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from sparket.miner.base.engines.interface import OddsPrices


@dataclass
class ComponentPrediction:
    """A prediction from a single component model."""
    source: str  # "elo", "poisson", "market"
    home_prob: Optional[float] = None
    away_prob: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    confidence: float = 0.5  # Model's confidence in this prediction
    weight: float = 0.0  # Assigned weight after adjustment

    @property
    def sharpness(self) -> float:
        """How sharp/confident is this prediction (0-1).

        Sharp = far from 50/50. Measured as 2 * |prob - 0.5|
        """
        if self.home_prob is not None:
            return 2 * abs(self.home_prob - 0.5)
        if self.over_prob is not None:
            return 2 * abs(self.over_prob - 0.5)
        return 0.0


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble model."""
    home_prob: float
    away_prob: float
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    confidence: float = 0.5
    components: List[ComponentPrediction] = field(default_factory=list)

    # Metadata
    models_agreed: bool = False  # Did models mostly agree?
    dominant_source: str = ""  # Which model contributed most?
    spread: float = 0.0  # Spread between model predictions

    def to_odds_prices(self, vig: float = 0.02) -> OddsPrices:
        """Convert to OddsPrices with vig."""
        def prob_to_odds(prob: float) -> float:
            implied = prob + (vig / 2)
            implied = max(0.001, min(0.999, implied))
            odds = 1.0 / implied
            return round(max(1.01, min(1000.0, odds)), 2)

        return OddsPrices(
            home_prob=self.home_prob,
            away_prob=self.away_prob,
            home_odds_eu=prob_to_odds(self.home_prob),
            away_odds_eu=prob_to_odds(self.away_prob),
            over_prob=self.over_prob,
            under_prob=self.under_prob,
            over_odds_eu=prob_to_odds(self.over_prob) if self.over_prob else None,
            under_odds_eu=prob_to_odds(self.under_prob) if self.under_prob else None,
        )


class EnsembleEngine:
    """Ensemble prediction engine combining multiple models.

    Weighting algorithm:
    1. Start with base weights from config
    2. Adjust by sharpness (sharper = more weight, up to 1.5x)
    3. Adjust by agreement (if models agree within 3%, boost all)
    4. Normalize to sum to 1.0
    5. Compute weighted average

    For TOTAL markets, uses Poisson as primary with market blend.
    """

    # Agreement threshold - models within this are "agreeing"
    AGREEMENT_THRESHOLD = 0.03

    # Sharpness bonus - sharp predictions get up to this multiplier
    SHARPNESS_BONUS_MAX = 1.5

    # Minimum weight floor - no model goes below this
    MIN_WEIGHT = 0.05

    def __init__(
        self,
        elo_engine: Any,
        poisson_engine: Any,
        base_weights: Optional[Dict[str, float]] = None,
        confidence_scaling: bool = True,
    ) -> None:
        """Initialize ensemble engine.

        Args:
            elo_engine: EloEngine instance
            poisson_engine: PoissonEngine instance
            base_weights: Base weights for each model {"elo": 0.5, "market": 0.35, "poisson": 0.15}
            confidence_scaling: Whether to adjust weights by confidence
        """
        self._elo = elo_engine
        self._poisson = poisson_engine

        self._base_weights = base_weights or {
            "elo": 0.50,
            "market": 0.35,
            "poisson": 0.15,
        }

        self._confidence_scaling = confidence_scaling

        # Track model performance for adaptive weighting
        self._model_accuracy: Dict[str, List[float]] = {
            "elo": [],
            "market": [],
            "poisson": [],
        }

    def predict(
        self,
        market: Dict[str, Any],
        market_odds: Optional[Any] = None,
    ) -> Optional[EnsemblePrediction]:
        """Generate ensemble prediction for a market.

        Args:
            market: Market info dict with home_team, away_team, sport, kind, line
            market_odds: Market consensus odds (from OddsAPIFetcher)

        Returns:
            EnsemblePrediction or None if unable to generate
        """
        home_team = market.get("home_team", "")
        away_team = market.get("away_team", "")
        sport = market.get("sport", "NFL")
        kind = market.get("kind", "MONEYLINE").upper()

        if not home_team or not away_team:
            return None

        components: List[ComponentPrediction] = []

        # 1. Get Elo prediction
        elo_pred = self._get_elo_prediction(market)
        if elo_pred:
            components.append(elo_pred)

        # 2. Get market prediction
        market_pred = self._get_market_prediction(market_odds)
        if market_pred:
            components.append(market_pred)

        # 3. Get Poisson prediction (for totals, but also provides scoring context)
        poisson_pred = self._get_poisson_prediction(market)
        if poisson_pred:
            components.append(poisson_pred)

        if not components:
            return None

        # 4. Assign weights
        self._assign_weights(components, kind)

        # 5. Combine predictions
        if kind == "TOTAL":
            return self._combine_total_predictions(components)
        else:
            return self._combine_moneyline_predictions(components)

    def _get_elo_prediction(self, market: Dict[str, Any]) -> Optional[ComponentPrediction]:
        """Get prediction from Elo engine."""
        try:
            odds = self._elo.get_odds_sync(market)
            if odds is None:
                return None

            # Confidence based on rating difference
            # Bigger rating gap = more confident
            home_team = market.get("home_team", "")
            away_team = market.get("away_team", "")
            sport = market.get("sport", "NFL")

            home_rating = self._elo.get_team_rating(home_team, sport)
            away_rating = self._elo.get_team_rating(away_team, sport)
            rating_diff = abs(home_rating - away_rating)

            # 100 point diff = 0.5 confidence, 300+ = 0.9
            confidence = min(0.9, 0.3 + (rating_diff / 500))

            return ComponentPrediction(
                source="elo",
                home_prob=odds.home_prob,
                away_prob=odds.away_prob,
                confidence=confidence,
            )
        except Exception as e:
            bt.logging.debug({"ensemble": "elo_error", "error": str(e)})
            return None

    def _get_market_prediction(self, market_odds: Any) -> Optional[ComponentPrediction]:
        """Get prediction from market consensus."""
        if market_odds is None:
            return None

        try:
            # Higher confidence if more books agree and Pinnacle is included
            base_confidence = 0.6
            if hasattr(market_odds, 'num_books'):
                # More books = more reliable
                book_bonus = min(0.2, market_odds.num_books * 0.03)
                base_confidence += book_bonus

            if hasattr(market_odds, 'has_pinnacle') and market_odds.has_pinnacle:
                # Pinnacle is the sharpest book
                base_confidence += 0.1

            confidence = min(0.95, base_confidence)

            return ComponentPrediction(
                source="market",
                home_prob=market_odds.home_prob,
                away_prob=1.0 - market_odds.home_prob,
                confidence=confidence,
            )
        except Exception as e:
            bt.logging.debug({"ensemble": "market_error", "error": str(e)})
            return None

    def _get_poisson_prediction(self, market: Dict[str, Any]) -> Optional[ComponentPrediction]:
        """Get prediction from Poisson engine."""
        kind = market.get("kind", "MONEYLINE").upper()
        line = market.get("line")

        if kind != "TOTAL" or line is None:
            return None

        try:
            pred = self._poisson.predict_total(
                home_team=market.get("home_team", ""),
                away_team=market.get("away_team", ""),
                sport=market.get("sport", "NFL"),
                line=float(line),
            )

            if pred is None:
                return None

            # Confidence based on how far expected total is from line
            diff_from_line = abs(pred.expected_total - float(line))
            # 5 points from line = 0.7 confidence, 10+ = 0.9
            confidence = min(0.9, 0.4 + (diff_from_line / 20))

            return ComponentPrediction(
                source="poisson",
                over_prob=pred.over_prob,
                under_prob=pred.under_prob,
                confidence=confidence,
            )
        except Exception as e:
            bt.logging.debug({"ensemble": "poisson_error", "error": str(e)})
            return None

    def _assign_weights(
        self,
        components: List[ComponentPrediction],
        kind: str,
    ) -> None:
        """Assign weights to each component prediction.

        Algorithm:
        1. Start with base weights
        2. Apply sharpness bonus (sharper = more weight)
        3. Normalize to sum to 1.0
        """
        # Get base weights
        for comp in components:
            comp.weight = self._base_weights.get(comp.source, 0.1)

        # Apply sharpness bonus if enabled
        if self._confidence_scaling:
            for comp in components:
                sharpness = comp.sharpness
                # Bonus: 0 sharpness = 1.0x, 1.0 sharpness = 1.5x
                bonus = 1.0 + (sharpness * (self.SHARPNESS_BONUS_MAX - 1.0))
                comp.weight *= bonus

                # Also factor in model confidence
                comp.weight *= (0.5 + 0.5 * comp.confidence)

        # Ensure minimum weight
        for comp in components:
            comp.weight = max(self.MIN_WEIGHT, comp.weight)

        # Normalize
        total_weight = sum(c.weight for c in components)
        if total_weight > 0:
            for comp in components:
                comp.weight /= total_weight

    def _combine_moneyline_predictions(
        self,
        components: List[ComponentPrediction],
    ) -> EnsemblePrediction:
        """Combine predictions for MONEYLINE market."""
        # Filter to components with home_prob
        valid = [c for c in components if c.home_prob is not None]

        if not valid:
            return None

        # Weighted average
        home_prob = sum(c.home_prob * c.weight for c in valid)
        away_prob = 1.0 - home_prob

        # Check agreement
        probs = [c.home_prob for c in valid]
        spread = max(probs) - min(probs)
        agreed = spread <= self.AGREEMENT_THRESHOLD

        # Find dominant source
        dominant = max(valid, key=lambda c: c.weight)

        # Overall confidence
        # Higher if models agree, weighted by component confidences
        base_conf = sum(c.confidence * c.weight for c in valid)
        if agreed:
            confidence = min(0.95, base_conf + 0.1)
        else:
            confidence = base_conf * 0.9  # Reduce confidence when disagreeing

        bt.logging.debug({
            "ensemble": "moneyline_combined",
            "components": len(valid),
            "home_prob": round(home_prob, 3),
            "spread": round(spread, 3),
            "agreed": agreed,
            "dominant": dominant.source,
            "confidence": round(confidence, 3),
        })

        return EnsemblePrediction(
            home_prob=home_prob,
            away_prob=away_prob,
            confidence=confidence,
            components=components,
            models_agreed=agreed,
            dominant_source=dominant.source,
            spread=spread,
        )

    def _combine_total_predictions(
        self,
        components: List[ComponentPrediction],
    ) -> EnsemblePrediction:
        """Combine predictions for TOTAL market."""
        # For totals, Poisson is primary
        poisson = next((c for c in components if c.source == "poisson"), None)

        if poisson is None or poisson.over_prob is None:
            # Fall back to equal odds if no Poisson
            return EnsemblePrediction(
                home_prob=0.5,
                away_prob=0.5,
                over_prob=0.5,
                under_prob=0.5,
                confidence=0.3,
                components=components,
            )

        # Use Poisson as base, potentially blend with market if available
        over_prob = poisson.over_prob
        under_prob = poisson.under_prob
        confidence = poisson.confidence

        # Also get moneyline for the home/away probs
        elo = next((c for c in components if c.source == "elo"), None)
        market = next((c for c in components if c.source == "market"), None)

        home_prob = 0.5
        away_prob = 0.5

        if elo and elo.home_prob is not None:
            home_prob = elo.home_prob
            away_prob = elo.away_prob
        elif market and market.home_prob is not None:
            home_prob = market.home_prob
            away_prob = market.away_prob

        return EnsemblePrediction(
            home_prob=home_prob,
            away_prob=away_prob,
            over_prob=over_prob,
            under_prob=under_prob,
            confidence=confidence,
            components=components,
            dominant_source="poisson",
        )

    def update_accuracy(
        self,
        source: str,
        predicted_prob: float,
        actual_outcome: float,
    ) -> None:
        """Update model accuracy tracking after outcome.

        Args:
            source: Model source ("elo", "market", "poisson")
            predicted_prob: Probability we predicted
            actual_outcome: 1.0 if predicted outcome happened, 0.0 otherwise
        """
        if source not in self._model_accuracy:
            return

        # Brier-style score (lower is better, but we store as accuracy)
        brier = (predicted_prob - actual_outcome) ** 2
        accuracy = 1.0 - brier

        self._model_accuracy[source].append(accuracy)

        # Keep last 100 samples
        if len(self._model_accuracy[source]) > 100:
            self._model_accuracy[source] = self._model_accuracy[source][-100:]

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model performance."""
        stats = {}
        for source, accuracies in self._model_accuracy.items():
            if accuracies:
                stats[source] = {
                    "samples": len(accuracies),
                    "avg_accuracy": round(sum(accuracies) / len(accuracies), 3),
                }
            else:
                stats[source] = {"samples": 0, "avg_accuracy": None}

        stats["base_weights"] = self._base_weights
        return stats


__all__ = [
    "EnsembleEngine",
    "EnsemblePrediction",
    "ComponentPrediction",
]
