"""Pytest fixtures for localnet E2E tests."""

from __future__ import annotations

import pytest
import pytest_asyncio

from .config import LocalnetConfig, DEFAULT_CONFIG
from .harness import LocalnetHarness


@pytest.fixture(scope="session")
def localnet_config() -> LocalnetConfig:
    """Provide localnet configuration."""
    return DEFAULT_CONFIG


@pytest_asyncio.fixture
async def harness(localnet_config: LocalnetConfig) -> LocalnetHarness:
    """Provide initialized LocalnetHarness.
    
    Automatically sets up and tears down the harness.
    """
    harness = LocalnetHarness(config=localnet_config)
    await harness.setup()
    
    yield harness
    
    await harness.teardown()


@pytest_asyncio.fixture
async def clean_harness(harness: LocalnetHarness) -> LocalnetHarness:
    """Provide harness with clean database state.
    
    Use this for tests that need isolation from previous state.
    """
    await harness.setup_clean_state()
    return harness


@pytest.fixture
def validator_url(localnet_config: LocalnetConfig) -> str:
    """Provide validator control URL."""
    return localnet_config.validator_control_url


@pytest.fixture
def miner_urls(localnet_config: LocalnetConfig) -> list[str]:
    """Provide list of miner control URLs."""
    return [m.control_url for m in localnet_config.miners]
