"""Aggregation utilities for scoring.

Contains:
- Time decay weighting
- Time-to-close scoring (asymmetric bonus)
- Sample-size shrinkage
- Score normalization
"""

from __future__ import annotations

from .time_weight import (
    compute_time_factor,
    compute_time_factors,
    apply_time_bonus,
    apply_time_bonus_batch,
    # Backwards compatibility
    compute_time_weight,
    compute_time_weights,
    apply_time_weighting,
)

__all__ = [
    "compute_time_factor",
    "compute_time_factors",
    "apply_time_bonus",
    "apply_time_bonus_batch",
    # Backwards compatibility
    "compute_time_weight",
    "compute_time_weights",
    "apply_time_weighting",
]

