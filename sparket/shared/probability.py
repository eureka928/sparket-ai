"""Probability utilities shared by ingest and scoring.

Functions convert decimal (EU) odds to implied probabilities and normalize
vectors to unit-sum while reporting overround (sum(raw_probs) - 1).

Safe math and bounds:
- EU odds must be > 1.0; otherwise a ValueError is raised.
- Probabilities must be > 0; zero or negative entries raise ValueError.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def eu_to_implied_prob(odds_eu: float) -> float:
    """Convert decimal (EU) odds to implied probability.

    Raises ValueError if odds_eu <= 1.0.
    """
    o = float(odds_eu)
    if not (o > 1.0):
        raise ValueError("decimal odds must be > 1.0")
    return 1.0 / o


def normalize_vector(probs: Iterable[float]) -> Tuple[List[float], float]:
    """Normalize a sequence of probabilities to unit sum and return overround.

    Overround is defined as sum(raw) - 1.0. Underround yields negative values.
    All inputs must be > 0.
    """
    raw = [float(p) for p in probs]
    if any(p <= 0.0 for p in raw):
        raise ValueError("probabilities must be > 0")
    s = sum(raw)
    if s <= 0.0:
        raise ValueError("sum of probabilities must be > 0")
    overround = s - 1.0
    norm = [p / s for p in raw]
    return norm, overround


def normalize_probs(prob_map: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """Normalize a mapping of label->probability to unit sum and report overround."""
    keys = list(prob_map.keys())
    vals = [prob_map[k] for k in keys]
    norm_vals, overround = normalize_vector(vals)
    return {k: v for k, v in zip(keys, norm_vals)}, overround


def implied_from_eu_vector(odds_eu: Iterable[float]) -> Tuple[List[float], List[float], float]:
    """From a vector of decimal odds, return (raw_implied, normalized, overround)."""
    raw = [eu_to_implied_prob(o) for o in odds_eu]
    norm, over = normalize_vector(raw)
    return raw, norm, over


def implied_from_eu_odds(odds_map: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """From a mapping of label->decimal odds, return (raw_implied, normalized, overround)."""
    keys = list(odds_map.keys())
    odds = [odds_map[k] for k in keys]
    raw_list = [eu_to_implied_prob(o) for o in odds]
    norm_list, over = normalize_vector(raw_list)
    return (
        {k: v for k, v in zip(keys, raw_list)},
        {k: v for k, v in zip(keys, norm_list)},
        over,
    )


__all__ = [
    "eu_to_implied_prob",
    "normalize_vector",
    "normalize_probs",
    "implied_from_eu_vector",
    "implied_from_eu_odds",
]


