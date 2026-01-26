"""Weighted consensus probability computation.

This module computes ground truth closing probabilities from multiple
sportsbooks using inverse-variance weighting with bias correction.

Algorithm:
1. For each book, get raw probability for the side
2. Apply bias correction: p_adjusted = p_raw / bias_factor
3. Normalize adjusted probabilities across all sides
4. Weight by inverse variance: w_i = 1 / variance_i
5. Compute weighted average: p_consensus = Σ(w_i * p_i) / Σ(w_i)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sparket.validator.config.scoring_params import ScoringParams, get_scoring_params

from ..determinism import round_decimal, safe_divide, safe_sqrt, sort_by_id
from ..types import ConsensusResult, ValidationError
from .bias import BiasKey, BiasState, make_bias_key


@dataclass
class BookQuote:
    """Quote from a single sportsbook for a specific side."""

    sportsbook_id: int
    sport_id: int
    market_kind: str
    side: str
    prob: Decimal
    odds: Decimal
    timestamp: datetime


@dataclass
class ConsensusInput:
    """Input for consensus computation for a single market side."""

    market_id: int
    side: str
    quotes: List[BookQuote]


class ConsensusComputer:
    """Computes weighted consensus probabilities from sportsbook quotes."""

    def __init__(self, params: ScoringParams | None = None):
        """Initialize the consensus computer.

        Args:
            params: Scoring parameters (uses defaults if None)
        """
        self.params = params or get_scoring_params()
        self.gt_params = self.params.ground_truth

    def compute_consensus(
        self,
        quotes: List[BookQuote],
        bias_states: Dict[BiasKey, BiasState],
    ) -> Optional[ConsensusResult]:
        """Compute consensus probability from book quotes.

        Args:
            quotes: List of quotes from different books for the SAME side
            bias_states: Current bias estimates by key

        Returns:
            ConsensusResult if enough books, None otherwise
        """
        if not quotes:
            return None

        # Filter and adjust quotes
        adjusted_quotes = []
        for quote in quotes:
            key = make_bias_key(quote.sportsbook_id, quote.sport_id, quote.market_kind)
            bias = bias_states.get(key)

            if bias is None:
                # Use default bias (1.0) for unknown books
                bias_factor = Decimal("1.0")
                variance = self.gt_params.min_variance
            else:
                bias_factor = bias.bias_factor
                variance = max(bias.variance, self.gt_params.min_variance)

            # Check if bias is in valid range
            if (
                bias_factor < self.gt_params.min_bias_factor
                or bias_factor > self.gt_params.max_bias_factor
            ):
                continue  # Exclude this book

            # Apply bias correction
            adjusted_prob = quote.prob / bias_factor

            # Clamp to valid probability range
            adjusted_prob = max(Decimal("0.001"), min(adjusted_prob, Decimal("0.999")))

            adjusted_quotes.append(
                {
                    "sportsbook_id": quote.sportsbook_id,
                    "prob": adjusted_prob,
                    "variance": variance,
                }
            )

        # Check minimum books
        if len(adjusted_quotes) < self.gt_params.min_books_for_consensus:
            return None

        # Sort for determinism
        adjusted_quotes = sorted(adjusted_quotes, key=lambda x: x["sportsbook_id"])

        # Compute inverse-variance weighted average
        weighted_sum = Decimal("0")
        weight_sum = Decimal("0")
        probs = []

        for aq in adjusted_quotes:
            weight = Decimal("1") / aq["variance"]
            weighted_sum += weight * aq["prob"]
            weight_sum += weight
            probs.append(aq["prob"])

        prob_consensus = safe_divide(weighted_sum, weight_sum, Decimal("0.5"))

        # Clamp to valid range
        prob_consensus = max(Decimal("0.001"), min(prob_consensus, Decimal("0.999")))

        # Compute statistics
        min_prob = min(probs)
        max_prob = max(probs)

        # Compute std dev
        n = Decimal(str(len(probs)))
        mean = sum(probs) / n
        variance = sum((p - mean) ** 2 for p in probs) / n
        std_dev = safe_sqrt(variance)

        # Compute consensus odds
        odds_consensus = safe_divide(Decimal("1"), prob_consensus, Decimal("100"))

        return ConsensusResult(
            prob_consensus=round_decimal(prob_consensus, 8),
            odds_consensus=round_decimal(odds_consensus, 4),
            contributing_books=len(adjusted_quotes),
            min_prob=round_decimal(min_prob, 8),
            max_prob=round_decimal(max_prob, 8),
            std_dev=round_decimal(std_dev, 8),
        )

    def compute_market_consensus(
        self,
        market_quotes: Dict[str, List[BookQuote]],
        bias_states: Dict[BiasKey, BiasState],
    ) -> Dict[str, ConsensusResult]:
        """Compute consensus for all sides of a market.

        Args:
            market_quotes: Quotes grouped by side
            bias_states: Current bias estimates

        Returns:
            ConsensusResult per side
        """
        results = {}

        # Process sides in deterministic order
        for side in sorted(market_quotes.keys()):
            quotes = market_quotes[side]
            consensus = self.compute_consensus(quotes, bias_states)
            if consensus is not None:
                results[side] = consensus

        # Optionally normalize across sides
        if results and len(results) >= 2:
            results = self._normalize_across_sides(results)

        return results

    def _normalize_across_sides(
        self,
        results: Dict[str, ConsensusResult],
    ) -> Dict[str, ConsensusResult]:
        """Normalize consensus probabilities to sum to 1.0.

        Args:
            results: Consensus results per side

        Returns:
            Normalized results
        """
        # Compute sum of probabilities
        prob_sum = sum(r["prob_consensus"] for r in results.values())

        if prob_sum == Decimal("0"):
            return results

        # Normalize each side
        normalized = {}
        for side, result in results.items():
            norm_prob = result["prob_consensus"] / prob_sum
            norm_prob = max(Decimal("0.001"), min(norm_prob, Decimal("0.999")))
            norm_odds = safe_divide(Decimal("1"), norm_prob, Decimal("100"))

            normalized[side] = ConsensusResult(
                prob_consensus=round_decimal(norm_prob, 8),
                odds_consensus=round_decimal(norm_odds, 4),
                contributing_books=result["contributing_books"],
                min_prob=result["min_prob"],
                max_prob=result["max_prob"],
                std_dev=result["std_dev"],
            )

        return normalized


def compute_simple_average(quotes: List[BookQuote]) -> Optional[Decimal]:
    """Compute simple average probability (no bias adjustment).

    Useful for bootstrapping before bias estimates are available.

    Args:
        quotes: List of quotes from different books

    Returns:
        Average probability, or None if no quotes
    """
    if not quotes:
        return None

    probs = [q.prob for q in quotes]
    return round_decimal(sum(probs) / len(probs), 8)


__all__ = [
    "BookQuote",
    "ConsensusInput",
    "ConsensusComputer",
    "compute_simple_average",
]

