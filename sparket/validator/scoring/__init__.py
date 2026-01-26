"""Scoring system for evaluating miner forecasting skill.

This package contains scoring logic for the Sparket subnet:
- Ground truth construction (sportsbook bias, consensus)
- Per-submission metrics (CLV, CLE, Brier, log-loss)
- Batch aggregation jobs (rolling scores, calibration, etc.)
- Determinism utilities for cross-validator consensus
- Worker process management for parallel scoring
"""

from __future__ import annotations

__all__: list[str] = []

