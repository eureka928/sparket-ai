"""Matchup probability calculation using Log5.

Converts team strength ratings into win probabilities using
Bill James' Log5 formula.
"""

from __future__ import annotations

from typing import Tuple


def strength_to_probability(
    home_strength: float,
    away_strength: float,
) -> Tuple[float, float]:
    """Convert team strengths to win probabilities using Log5.
    
    The Log5 formula estimates the probability of team A beating team B
    based on their respective win rates (strengths).
    
    Formula:
        P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
    
    Where pA and pB are the team strengths (0-1).
    
    Args:
        home_strength: Home team strength (0-1)
        away_strength: Away team strength (0-1)
    
    Returns:
        Tuple of (home_win_prob, away_win_prob) that sum to 1.0
    
    Example:
        # Strong home team (70%) vs weak away team (30%)
        home_prob, away_prob = strength_to_probability(0.7, 0.3)
        # home_prob ≈ 0.84, away_prob ≈ 0.16
        
        # Equal teams
        home_prob, away_prob = strength_to_probability(0.5, 0.5)
        # home_prob = 0.5, away_prob = 0.5
    """
    # Validate inputs
    home_strength = _clamp(home_strength, 0.01, 0.99)
    away_strength = _clamp(away_strength, 0.01, 0.99)
    
    # Log5 formula
    pA = home_strength
    pB = away_strength
    
    numerator = pA - (pA * pB)
    denominator = pA + pB - (2 * pA * pB)
    
    if denominator == 0:
        # Edge case: shouldn't happen with clamped inputs
        return 0.5, 0.5
    
    home_prob = numerator / denominator
    away_prob = 1.0 - home_prob
    
    return home_prob, away_prob


def odds_to_probability(home_odds_eu: float, away_odds_eu: float) -> Tuple[float, float]:
    """Convert EU decimal odds to implied probabilities.
    
    Removes the vig (overround) to get "true" probabilities.
    
    Args:
        home_odds_eu: EU decimal odds for home team
        away_odds_eu: EU decimal odds for away team
    
    Returns:
        Tuple of (home_prob, away_prob) that sum to ~1.0
    
    Example:
        # -110 on both sides (1.91 EU odds)
        home_prob, away_prob = odds_to_probability(1.91, 1.91)
        # home_prob ≈ 0.5, away_prob ≈ 0.5
    """
    # Implied probabilities (sum > 1 due to vig)
    implied_home = 1.0 / home_odds_eu
    implied_away = 1.0 / away_odds_eu
    
    # Normalize to remove vig
    total = implied_home + implied_away
    true_home = implied_home / total
    true_away = implied_away / total
    
    return true_home, true_away


def probability_to_odds(prob: float, vig: float = 0.045) -> float:
    """Convert probability to EU decimal odds with vig.
    
    Args:
        prob: True probability (0-1)
        vig: Total vigorish to apply (default: 4.5%)
    
    Returns:
        EU decimal odds
    
    Example:
        odds = probability_to_odds(0.5, vig=0.045)
        # odds ≈ 1.91 (equivalent to -110)
    """
    # Add vig proportionally
    implied_prob = prob + (vig / 2)
    implied_prob = _clamp(implied_prob, 0.01, 0.99)
    
    return round(1.0 / implied_prob, 2)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))








