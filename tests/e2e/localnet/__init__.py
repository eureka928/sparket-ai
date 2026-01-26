"""Localnet E2E testing infrastructure.

This package provides tools for running comprehensive E2E tests
against a local Bittensor network with multiple miners.

Key components:
- LocalnetHarness: Orchestrates E2E tests with validator + N miners
- TimeController: Simulates time progression for scoring windows
- MinerPool: Manages multiple miner instances
- Scenarios: Pre-built test scenarios (odds competition, adversarial, etc.)
"""

from .harness import LocalnetHarness
from .time_controller import TimeController
from .miner_pool import MinerPool

__all__ = ["LocalnetHarness", "TimeController", "MinerPool"]
