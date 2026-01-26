"""Shared fixtures for base miner tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest

from sparket.miner.base.config import BaseMinerConfig, ModelWeights
from sparket.miner.base.data.stats import TeamStats
from sparket.miner.base.engines.interface import OddsPrices


@pytest.fixture
def sample_team_stats() -> Dict[str, TeamStats]:
    """Standard team stats for testing."""
    return {
        "KC": TeamStats(
            team_code="KC",
            league="NFL",
            wins=14,
            losses=2,
            home_wins=7,
            home_losses=1,
            away_wins=7,
            away_losses=1,
            last_5_wins=4,
            last_5_losses=1,
            points_for=448,
            points_against=304,
        ),
        "NE": TeamStats(
            team_code="NE",
            league="NFL",
            wins=4,
            losses=12,
            home_wins=2,
            home_losses=6,
            away_wins=2,
            away_losses=6,
            last_5_wins=1,
            last_5_losses=4,
            points_for=288,
            points_against=400,
        ),
        "DAL": TeamStats(
            team_code="DAL",
            league="NFL",
            wins=8,
            losses=8,
            home_wins=5,
            home_losses=3,
            away_wins=3,
            away_losses=5,
            last_5_wins=2,
            last_5_losses=3,
        ),
        "NYJ": TeamStats(
            team_code="NYJ",
            league="NFL",
            wins=5,
            losses=11,
            home_wins=3,
            home_losses=5,
            away_wins=2,
            away_losses=6,
            last_5_wins=1,
            last_5_losses=4,
        ),
    }


@pytest.fixture
def base_miner_config() -> BaseMinerConfig:
    """Default base miner configuration for tests."""
    return BaseMinerConfig(
        enabled=True,
        odds_api_key=None,
        odds_refresh_seconds=60,
        outcome_check_seconds=60,
        stats_refresh_seconds=300,
        cache_ttl_seconds=300,
        market_blend_weight=0.6,
        vig=0.045,
        model_weights=ModelWeights(),
    )


@pytest.fixture
def mock_validator_client() -> Mock:
    """Mock validator client for testing."""
    client = Mock()
    client.submit_odds = AsyncMock(return_value=True)
    client.submit_outcome = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_game_sync() -> Mock:
    """Mock game sync for testing."""
    sync = Mock()
    sync.get_active_markets = AsyncMock(return_value=[
        {
            "market_id": 1,
            "event_id": 100,
            "kind": "MONEYLINE",
            "home_team": "KC",
            "away_team": "NE",
            "sport": "NFL",
            "start_time_utc": datetime.now(timezone.utc),
        },
        {
            "market_id": 2,
            "event_id": 101,
            "kind": "MONEYLINE",
            "home_team": "DAL",
            "away_team": "NYJ",
            "sport": "NFL",
            "start_time_utc": datetime.now(timezone.utc),
        },
    ])
    return sync


@pytest.fixture
def sample_market() -> Dict[str, Any]:
    """Sample market for testing."""
    return {
        "market_id": 1,
        "event_id": 100,
        "kind": "MONEYLINE",
        "home_team": "KC",
        "away_team": "NE",
        "sport": "NFL",
        "start_time_utc": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_odds_prices() -> OddsPrices:
    """Sample odds for testing."""
    return OddsPrices(
        home_prob=0.65,
        away_prob=0.35,
        home_odds_eu=1.54,
        away_odds_eu=2.86,
    )








