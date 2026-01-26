"""Base class for E2E test scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..harness import LocalnetHarness


@dataclass
class ScenarioResult:
    """Result of a scenario execution."""
    
    scenario_id: str
    success: bool
    assertions_passed: int = 0
    assertions_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.success = False
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
    
    def add_pass(self, msg: str) -> None:
        """Record a passing check."""
        self.assertions_passed += 1
    
    def add_fail(self, msg: str) -> None:
        """Record a failing check."""
        self.assertions_failed += 1
        self.add_error(msg)
    
    def assert_true(self, condition: bool, msg: str) -> bool:
        """Assert a condition is true."""
        if condition:
            self.assertions_passed += 1
            return True
        else:
            self.assertions_failed += 1
            self.add_error(f"Assertion failed: {msg}")
            return False
    
    def assert_equal(self, actual: Any, expected: Any, msg: str) -> bool:
        """Assert two values are equal."""
        return self.assert_true(
            actual == expected,
            f"{msg} (expected={expected}, actual={actual})"
        )
    
    def assert_greater(self, actual: Any, threshold: Any, msg: str) -> bool:
        """Assert a value is greater than threshold."""
        return self.assert_true(
            actual > threshold,
            f"{msg} (actual={actual} > {threshold})"
        )
    
    def assert_in_range(
        self,
        actual: float,
        low: float,
        high: float,
        msg: str
    ) -> bool:
        """Assert a value is within a range."""
        in_range = low <= actual <= high
        return self.assert_true(
            in_range,
            f"{msg} (actual={actual} in [{low}, {high}])"
        )
    
    def complete(self) -> None:
        """Mark scenario as complete."""
        self.completed_at = datetime.now(timezone.utc)
        self.success = self.assertions_failed == 0 and len(self.errors) == 0


class BaseScenario(ABC):
    """Base class for E2E test scenarios.
    
    Scenarios define:
    - setup(): Prepare test data
    - execute(): Run the scenario actions
    - verify(): Assert expected outcomes
    
    The harness handles lifecycle management.
    """
    
    SCENARIO_ID: str = "base"
    
    def __init__(self, harness: LocalnetHarness):
        self.harness = harness
        self.result = ScenarioResult(scenario_id=self.SCENARIO_ID, success=True)
        
        # Convenience accessors
        self.validator = harness.validator
        self.miners = harness.miners
        self.time = harness.time_controller
        self.db = harness.db
    
    @abstractmethod
    async def setup(self) -> None:
        """Prepare test data for the scenario.
        
        Override to create events, seed provider quotes, etc.
        """
        pass
    
    @abstractmethod
    async def execute(self) -> None:
        """Execute the scenario actions.
        
        Override to trigger miner submissions, time progression, etc.
        """
        pass
    
    @abstractmethod
    async def verify(self) -> None:
        """Verify scenario outcomes.
        
        Override to assert expected scores, weights, etc.
        """
        pass
    
    async def run(self) -> ScenarioResult:
        """Execute the complete scenario lifecycle."""
        try:
            await self.setup()
            await self.execute()
            await self.verify()
        except Exception as e:
            self.result.add_error(f"Scenario error: {e}")
        finally:
            self.result.complete()
        
        return self.result
    
    async def create_events(self, n: int, hours_ahead: int = 48) -> List[dict]:
        """Helper to create N test events via validator control API."""
        events = []
        for i in range(n):
            result = await self.validator.create_event(
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                hours_ahead=hours_ahead,
            )
            if result.get("status") == "ok":
                events.append(result.get("event", {}))
        return events
    
    async def seed_ground_truth(
        self,
        markets: List[dict],
        home_probs: List[float] | None = None,
    ) -> dict[int, dict]:
        """Seed ground truth closing for markets.
        
        Returns:
            Dict mapping market_id to ground truth data.
        """
        ground_truth = {}
        
        for i, market in enumerate(markets):
            market_id = market.get("db_market_id") or market.get("market_id")
            if market_id is None:
                continue
            
            # Use provided probs or random
            if home_probs and i < len(home_probs):
                home_prob = home_probs[i]
            else:
                home_prob = 0.3 + (i % 5) * 0.1  # Vary: 0.3, 0.4, 0.5, 0.6, 0.7
            
            away_prob = round(1.0 - home_prob, 4)
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / away_prob, 2)
            
            await self.validator.seed_ground_truth(
                market_id=market_id,
                home_prob=home_prob,
                away_prob=away_prob,
            )
            
            ground_truth[market_id] = {
                "home_prob": home_prob,
                "away_prob": away_prob,
                "home_odds": home_odds,
                "away_odds": away_odds,
            }
        
        return ground_truth
    
    async def trigger_scoring(self) -> dict:
        """Trigger full scoring pipeline via validator."""
        return await self.validator.trigger_scoring()
    
    async def get_weights(self) -> dict:
        """Get computed weights from validator."""
        return await self.validator.get_weights()
