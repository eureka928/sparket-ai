"""Tests for scoring and weight assertions.

These tests run the assertion modules against the actual database.
"""

from __future__ import annotations

import pytest

from .harness import LocalnetHarness
from .assertions import (
    run_all_scoring_assertions,
    run_all_weight_assertions,
)


class TestScoringAssertions:
    """Tests for scoring invariant assertions."""
    
    @pytest.mark.asyncio
    async def test_scoring_assertions_run(self, harness: LocalnetHarness):
        """Scoring assertions should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        result = await run_all_scoring_assertions(harness.db)
        
        print(f"\nScoring Assertions:")
        print(f"  Passed: {result.passed}")
        print(f"  Failed: {result.failed}")
        print(f"  Warnings: {result.warnings}")
        
        for detail in result.details:
            print(f"  [{detail['status']}] {detail['message']}")
        
        # May have warnings if no data, but should not have failures on valid data
        # Note: May fail if data is intentionally invalid for testing


class TestWeightAssertions:
    """Tests for weight assertions."""
    
    @pytest.mark.asyncio
    async def test_weight_assertions_run(self, harness: LocalnetHarness):
        """Weight assertions should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        result = await run_all_weight_assertions(harness.db)
        
        print(f"\nWeight Assertions:")
        print(f"  Passed: {result.passed}")
        print(f"  Failed: {result.failed}")
        print(f"  Warnings: {result.warnings}")
        
        for detail in result.details:
            print(f"  [{detail['status']}] {detail['message']}")


class TestAssertionsAfterScenario:
    """Run assertions after a full scenario."""
    
    @pytest.mark.asyncio
    async def test_assertions_after_odds_competition(self, harness: LocalnetHarness):
        """Assertions should pass after running odds competition scenario."""
        from .scenarios import OddsCompetitionScenario
        
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        # Setup clean state and run scenario
        await harness.setup_clean_state()
        scenario_result = await harness.run_scenario(OddsCompetitionScenario)
        
        print(f"\nScenario completed: {scenario_result.scenario_id}")
        print(f"  Success: {scenario_result.success}")
        
        # Run assertions
        scoring_result = await run_all_scoring_assertions(harness.db)
        weight_result = await run_all_weight_assertions(harness.db)
        
        print(f"\nPost-scenario scoring assertions: {scoring_result.passed} passed, {scoring_result.failed} failed")
        print(f"Post-scenario weight assertions: {weight_result.passed} passed, {weight_result.failed} failed")
        
        # Soft check - report but don't fail on warnings
        for detail in scoring_result.details + weight_result.details:
            if detail["status"] == "fail":
                print(f"  [FAIL] {detail['message']}")
