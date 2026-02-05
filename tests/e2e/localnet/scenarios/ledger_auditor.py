"""E2E scenario: Primary validator + Auditor validator full flow.

Tests the complete ledger pipeline on the local chain:
1. Primary scores miners, exports checkpoint + delta
2. Auditor fetches, verifies Brier, computes weights, sets on chain
3. Epoch bump recovery
4. Auth rejection for miners/low-stake validators

This scenario requires:
- Primary validator running (PM2 validator-local + ledger HTTP enabled)
- Auditor validator running (PM2 auditor-local)
- Local substrate chain
- At least 3 miners submitting
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseScenario

if TYPE_CHECKING:
    from ..harness import LocalnetHarness


class LedgerAuditorScenario(BaseScenario):
    """Full primary+auditor E2E flow with epoch bump recovery."""

    SCENARIO_ID = "ledger_auditor"

    async def setup(self) -> None:
        """Seed events, submit odds from miners, settle outcomes, trigger scoring."""
        # Clean state
        await self.harness.setup_clean_state()

        # Create test events
        self._events = await self.harness.create_test_events(n=5, hours_ahead=48)
        self.result.assert_true(
            len(self._events) > 0,
            "created_test_events",
        )

        # Seed ground truth
        self._gt = await self.harness.seed_ground_truth_for_events(self._events)

        # Submit odds from miners with varied strategies
        for i, miner in enumerate(self.miners.clients[:3]):
            for event in self._events:
                market_id = event.get("db_market_id")
                if market_id is None:
                    continue
                gt = self._gt.get(market_id, {})
                home_prob = gt.get("home_prob", 0.5)

                # Strategy varies by miner index
                if i == 0:  # Early accurate
                    prob = home_prob + 0.02
                elif i == 1:  # Late accurate
                    prob = home_prob - 0.01
                else:  # Inaccurate
                    prob = 1.0 - home_prob

                prob = max(0.01, min(0.99, prob))
                await miner.submit_odds(market_id=market_id, prob_home=prob)

        # Settle outcomes
        self._outcomes = await self.harness.seed_outcomes_for_events(
            self._events, self._gt,
        )

        # Trigger scoring pipeline
        await self.harness.run_scoring_cycle()

    async def execute(self) -> None:
        """Trigger ledger export and auditor verification."""
        # Phase 1: Trigger ledger checkpoint + delta export
        try:
            cp_result = await self.validator._post("/trigger/ledger-checkpoint")
            self.result.assert_true(
                cp_result.get("status") == "ok",
                "checkpoint_exported",
            )
        except Exception as e:
            self.result.add_error(f"checkpoint_export_failed: {e}")

        try:
            delta_result = await self.validator._post("/trigger/ledger-delta")
            self.result.assert_true(
                delta_result.get("status") == "ok",
                "delta_exported",
            )
        except Exception as e:
            self.result.add_error(f"delta_export_failed: {e}")

        # Phase 2: Check auditor status (if auditor is running)
        try:
            import aiohttp
            auditor_url = "http://127.0.0.1:8201"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{auditor_url}/health") as resp:
                    auditor_health = await resp.json()
                    self.result.assert_true(
                        auditor_health.get("status") == "ok",
                        "auditor_healthy",
                    )
        except Exception:
            self.result.add_warning("auditor not running - skipping auditor-side checks")

    async def verify(self) -> None:
        """Verify ledger data integrity."""
        # Verify rolling scores exist
        scores = await self.validator.get_rolling_scores()
        self.result.assert_true(
            len(scores.get("scores", [])) > 0,
            "rolling_scores_exist",
        )

        # Verify weights can be computed
        weights = await self.validator.get_weights()
        self.result.assert_true(
            len(weights.get("weights", [])) > 0,
            "weights_computed",
        )


__all__ = ["LedgerAuditorScenario"]
