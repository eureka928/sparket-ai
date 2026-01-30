"""Configuration for the custom miner optimized for scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EloConfig:
    """Elo rating system configuration.

    Sport-specific K-factors control how quickly ratings adapt:
    - Higher K = more reactive to recent results
    - Lower K = more stable ratings
    """

    # Starting Elo for new teams
    initial_rating: float = 1500.0

    # Sport-specific K-factors (higher = more volatile)
    k_factors: Dict[str, float] = field(default_factory=lambda: {
        "NFL": 20.0,     # Small sample, each game matters more
        "NBA": 12.0,     # 82 games, more stable
        "MLB": 4.0,      # 162 games, very stable
        "NHL": 10.0,     # 82 games, moderate
        "NCAAF": 25.0,   # Small sample, high volatility
        "NCAAB": 15.0,   # Moderate sample
    })

    # Home field advantage in Elo points by sport
    home_advantage: Dict[str, float] = field(default_factory=lambda: {
        "NFL": 48.0,     # ~2.8% win probability boost
        "NBA": 100.0,    # ~3.2% win probability boost
        "MLB": 24.0,     # ~1.4% win probability boost
        "NHL": 33.0,     # ~1.9% win probability boost
        "NCAAF": 55.0,   # Crowd factor higher
        "NCAAB": 65.0,   # Crowd factor higher
    })

    # Margin of victory multiplier (applies log scaling to MOV)
    mov_multiplier: float = 1.0

    # Autocorrelation decay factor (for regressing to mean over time)
    season_decay: float = 0.75  # Carry over 75% of rating between seasons

    def get_k_factor(self, sport: str) -> float:
        """Get K-factor for a sport, default to NFL if unknown."""
        return self.k_factors.get(sport.upper(), self.k_factors["NFL"])

    def get_home_advantage(self, sport: str) -> float:
        """Get home advantage for a sport, default to NFL if unknown."""
        return self.home_advantage.get(sport.upper(), self.home_advantage["NFL"])


@dataclass
class TimingConfig:
    """Timing strategy configuration for maximizing time credit.

    Time bonus rules:
    - 7+ days before: 100% credit
    - 1 hour before: 10% credit (floor)
    - Early bad predictions: 70% penalty (forgiven)
    - Late bad predictions: 100% penalty
    """

    # Target: submit 7 days before event
    early_submission_days: float = 7.0

    # Refresh interval (6 hours = 4x/day)
    refresh_interval_seconds: int = 6 * 3600

    # Minimum time before event (2 hours floor)
    min_hours_before_event: float = 2.0

    # Don't submit if less than this many hours before
    cutoff_hours: float = 0.5


@dataclass
class CalibrationConfig:
    """Calibration layer configuration."""

    # Enable isotonic calibration
    enabled: bool = True

    # Minimum samples needed before calibration kicks in
    min_samples: int = 100

    # Path to store calibration data
    data_path: Optional[str] = None

    # How often to retrain calibration (in submissions)
    retrain_interval: int = 500


@dataclass
class OriginalityConfig:
    """Originality tracking for InfoDim scoring.

    SOS (Submission Originality Score) rewards predictions
    that are different from other miners.
    """

    # Track last N submissions per market for correlation
    history_size: int = 100

    # Weight for originality vs accuracy tradeoff
    originality_weight: float = 0.15

    # Minimum deviation from consensus to get originality bonus
    min_deviation_threshold: float = 0.02  # 2% difference


@dataclass
class CustomMinerConfig:
    """Main configuration for the custom miner.

    Environment variable overrides use SPARKET_CUSTOM_MINER__ prefix.
    Example: SPARKET_CUSTOM_MINER__VIG=0.04
    """

    # Master switch
    enabled: bool = True

    # Sub-configurations
    elo: EloConfig = field(default_factory=EloConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    originality: OriginalityConfig = field(default_factory=OriginalityConfig)

    # API keys for data sources
    odds_api_key: Optional[str] = None

    # Standard vig to apply (4.5% default)
    vig: float = 0.045

    # Engine weights for ensemble (Phase 3)
    engine_weights: Dict[str, float] = field(default_factory=lambda: {
        "elo": 0.50,      # Elo-based predictions
        "market": 0.35,   # Market consensus
        "poisson": 0.15,  # Poisson model (for totals)
    })

    # Rate limiting
    rate_limit_per_minute: int = 60
    per_market_limit_per_minute: int = 10

    # Outcome checking
    outcome_check_seconds: int = 300  # 5 minutes

    @classmethod
    def from_env(cls) -> "CustomMinerConfig":
        """Load configuration from environment variables.

        Supports:
        - SPARKET_CUSTOM_MINER__ENABLED
        - SPARKET_CUSTOM_MINER__VIG
        - SPARKET_CUSTOM_MINER__ODDS_API_KEY
        - SPARKET_CUSTOM_MINER__TIMING__REFRESH_INTERVAL_SECONDS
        - SPARKET_CUSTOM_MINER__TIMING__EARLY_SUBMISSION_DAYS
        - SPARKET_CUSTOM_MINER__CALIBRATION__ENABLED
        - SPARKET_CUSTOM_MINER__RATE_LIMIT_PER_MINUTE
        """

        def get_bool(key: str, default: bool) -> bool:
            val = os.getenv(f"SPARKET_CUSTOM_MINER__{key}")
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def get_int(key: str, default: int) -> int:
            val = os.getenv(f"SPARKET_CUSTOM_MINER__{key}")
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            val = os.getenv(f"SPARKET_CUSTOM_MINER__{key}")
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def get_str(key: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(f"SPARKET_CUSTOM_MINER__{key}", default)

        # Build timing config
        timing = TimingConfig(
            early_submission_days=get_float(
                "TIMING__EARLY_SUBMISSION_DAYS",
                TimingConfig.early_submission_days
            ),
            refresh_interval_seconds=get_int(
                "TIMING__REFRESH_INTERVAL_SECONDS",
                TimingConfig.refresh_interval_seconds
            ),
            min_hours_before_event=get_float(
                "TIMING__MIN_HOURS_BEFORE_EVENT",
                TimingConfig.min_hours_before_event
            ),
            cutoff_hours=get_float(
                "TIMING__CUTOFF_HOURS",
                TimingConfig.cutoff_hours
            ),
        )

        # Build calibration config
        calibration = CalibrationConfig(
            enabled=get_bool("CALIBRATION__ENABLED", True),
            min_samples=get_int("CALIBRATION__MIN_SAMPLES", 100),
            data_path=get_str("CALIBRATION__DATA_PATH"),
            retrain_interval=get_int("CALIBRATION__RETRAIN_INTERVAL", 500),
        )

        return cls(
            enabled=get_bool("ENABLED", True),
            elo=EloConfig(),  # Use defaults, extend as needed
            timing=timing,
            calibration=calibration,
            originality=OriginalityConfig(),
            odds_api_key=get_str("ODDS_API_KEY"),
            vig=get_float("VIG", 0.045),
            rate_limit_per_minute=get_int("RATE_LIMIT_PER_MINUTE", 60),
            per_market_limit_per_minute=get_int("PER_MARKET_LIMIT_PER_MINUTE", 10),
            outcome_check_seconds=get_int("OUTCOME_CHECK_SECONDS", 300),
        )
