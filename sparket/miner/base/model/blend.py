"""Odds blending - combining market and model predictions.

Blends market odds (from The-Odds-API or other sources) with
model-derived odds (from team strength calculations).
"""

from __future__ import annotations

from typing import Optional

from sparket.miner.base.engines.interface import OddsPrices
from sparket.miner.base.model.matchup import probability_to_odds


def blend_odds(
    market_prob: Optional[float],
    model_prob: float,
    market_weight: float = 0.6,
) -> float:
    """Blend market and model probabilities.
    
    Uses a weighted average to combine:
    - Market probability (from sportsbooks, if available)
    - Model probability (from our team strength model)
    
    Args:
        market_prob: Probability implied by market odds (or None if unavailable)
        model_prob: Probability from our model
        market_weight: How much to trust market (0.0-1.0)
    
    Returns:
        Blended probability
    
    Example:
        # Market says 55%, model says 60%, trust market 60%
        blended = blend_odds(0.55, 0.60, market_weight=0.6)
        # blended = 0.55 * 0.6 + 0.60 * 0.4 = 0.57
    """
    if market_prob is None:
        # No market data - use model only
        return model_prob
    
    # Weighted blend
    model_weight = 1.0 - market_weight
    blended = (market_prob * market_weight) + (model_prob * model_weight)
    
    # Keep in valid range
    return max(0.01, min(0.99, blended))


def blend_odds_prices(
    market: Optional[OddsPrices],
    model: OddsPrices,
    market_weight: float = 0.6,
    vig: float = 0.045,
) -> OddsPrices:
    """Blend full OddsPrices from market and model.
    
    Args:
        market: Market odds (or None if unavailable)
        model: Model odds
        market_weight: How much to trust market
        vig: Vigorish to apply to output odds
    
    Returns:
        Blended OddsPrices with recalculated EU odds
    """
    if market is None:
        # No market - use model with vig applied
        return OddsPrices(
            home_prob=model.home_prob,
            away_prob=model.away_prob,
            home_odds_eu=probability_to_odds(model.home_prob, vig),
            away_odds_eu=probability_to_odds(model.away_prob, vig),
            over_prob=model.over_prob,
            under_prob=model.under_prob,
            over_odds_eu=probability_to_odds(model.over_prob, vig) if model.over_prob else None,
            under_odds_eu=probability_to_odds(model.under_prob, vig) if model.under_prob else None,
        )
    
    # Blend each probability
    home_prob = blend_odds(market.home_prob, model.home_prob, market_weight)
    away_prob = 1.0 - home_prob  # Ensure they sum to 1
    
    # Handle totals if present
    over_prob = None
    under_prob = None
    over_odds = None
    under_odds = None
    
    if model.over_prob is not None:
        market_over = market.over_prob if market else None
        over_prob = blend_odds(market_over, model.over_prob, market_weight)
        under_prob = 1.0 - over_prob
        over_odds = probability_to_odds(over_prob, vig)
        under_odds = probability_to_odds(under_prob, vig)
    
    return OddsPrices(
        home_prob=home_prob,
        away_prob=away_prob,
        home_odds_eu=probability_to_odds(home_prob, vig),
        away_odds_eu=probability_to_odds(away_prob, vig),
        over_prob=over_prob,
        under_prob=under_prob,
        over_odds_eu=over_odds,
        under_odds_eu=under_odds,
    )


def adjust_for_confidence(
    base_prob: float,
    confidence: float,
    prior: float = 0.5,
) -> float:
    """Adjust probability based on confidence level.
    
    Lower confidence = regress toward prior (usually 0.5).
    
    Args:
        base_prob: Original probability estimate
        confidence: Confidence in estimate (0.0-1.0)
        prior: Prior probability to regress toward
    
    Returns:
        Adjusted probability
    
    Example:
        # 70% confident that home team has 65% chance
        adjusted = adjust_for_confidence(0.65, confidence=0.7, prior=0.5)
        # adjusted = 0.65 * 0.7 + 0.5 * 0.3 = 0.605
    """
    return (base_prob * confidence) + (prior * (1.0 - confidence))








