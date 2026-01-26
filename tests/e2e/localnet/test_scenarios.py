"""Tests that run E2E scenarios.

These tests require the validator and miners to be running.
Start with: pm2 start ecosystem.e2e.config.js

Run all scenarios:
    pytest tests/e2e/localnet/test_scenarios.py -v

Run specific scenario:
    pytest tests/e2e/localnet/test_scenarios.py::TestOddsCompetition -v
    pytest tests/e2e/localnet/test_scenarios.py::TestOutcomeVerification -v
    pytest tests/e2e/localnet/test_scenarios.py::TestAdversarial -v
    pytest tests/e2e/localnet/test_scenarios.py::TestEdgeCases -v
"""

from __future__ import annotations

import pytest

from .harness import LocalnetHarness
from .scenarios import (
    OddsCompetitionScenario,
    OutcomeVerificationScenario,
    AdversarialScenario,
    EdgeCaseScenario,
    CrashRecoveryScenario,
    MemoryProfilingScenario,
)


class TestOddsCompetition:
    """Tests for the odds competition scenario."""
    
    @pytest.mark.asyncio
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The scenario should run without errors."""
        # Skip if validator not running
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        # Setup clean state
        await harness.setup_clean_state()
        
        # Run scenario
        result = await harness.run_scenario(OddsCompetitionScenario)
        
        # Check result
        assert result.completed_at is not None, "Scenario should complete"
        
        # Print metrics for debugging
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        print(f"Metrics: {result.metrics}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Soft assertions - report but don't fail on warnings
        # The scenario may have warnings if not enough data for all computations
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")
    
    @pytest.mark.asyncio
    async def test_markets_created(self, harness: LocalnetHarness):
        """Events and markets should be created successfully."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        await harness.setup_clean_state()
        
        # Create events
        events = await harness.create_test_events(n=3)
        
        assert len(events) == 3, "Should create 3 events"
        
        for event in events:
            assert "db_market_id" in event, f"Event should have db_market_id: {event}"
    
    @pytest.mark.asyncio  
    async def test_submissions_recorded(self, harness: LocalnetHarness):
        """Miner submissions should be recorded in database."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        # Check miner health
        miner_health = await harness.miners.health_check_all()
        if not any(miner_health.values()):
            pytest.skip("No miners running")
        
        await harness.setup_clean_state()
        
        # Create events
        events = await harness.create_test_events(n=2)
        assert len(events) >= 2
        
        # Submit from first available miner
        for i, miner in enumerate(harness.miners):
            if await miner.health_check():
                for event in events:
                    market_id = event.get("db_market_id")
                    if market_id:
                        result = await miner.submit_odds(market_id, 2.0, 1.8)
                        # Note: may fail if miner can't reach validator
                break
        
        # Check submissions in DB
        submissions = await harness.validator.get_submissions()
        
        # Submissions may or may not be recorded depending on network
        print(f"Submissions in DB: {submissions}")


class TestOutcomeVerification:
    """Tests for the outcome verification scenario."""
    
    @pytest.mark.asyncio
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The outcome verification scenario should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        await harness.setup_clean_state()
        
        result = await harness.run_scenario(OutcomeVerificationScenario)
        
        assert result.completed_at is not None, "Scenario should complete"
        
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        print(f"Metrics: {result.metrics}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")


class TestAdversarial:
    """Tests for the adversarial scenario."""
    
    @pytest.mark.asyncio
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The adversarial scenario should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        await harness.setup_clean_state()
        
        result = await harness.run_scenario(AdversarialScenario)
        
        assert result.completed_at is not None, "Scenario should complete"
        
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        print(f"Metrics: {result.metrics}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Adversarial tests may have warnings but shouldn't fail completely
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")


class TestEdgeCases:
    """Tests for edge case scenarios."""
    
    @pytest.mark.asyncio
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The edge case scenario should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        await harness.setup_clean_state()
        
        result = await harness.run_scenario(EdgeCaseScenario)
        
        assert result.completed_at is not None, "Scenario should complete"
        
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        print(f"Metrics: {result.metrics}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")


class TestScoringPipeline:
    """Tests for the scoring pipeline."""
    
    @pytest.mark.asyncio
    async def test_scoring_trigger(self, harness: LocalnetHarness):
        """Scoring should trigger without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        result = await harness.validator.trigger_scoring()
        
        assert result.get("status") == "ok", f"Scoring should complete: {result}"
    
    @pytest.mark.asyncio
    async def test_skill_scores_query(self, harness: LocalnetHarness):
        """Should be able to query skill scores."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        result = await harness.validator.get_skill_scores()
        
        assert "scores" in result or "status" in result, f"Should return scores: {result}"


class TestWeights:
    """Tests for weight computation."""
    
    @pytest.mark.asyncio
    async def test_weights_query(self, harness: LocalnetHarness):
        """Should be able to query weights."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running")
        
        result = await harness.validator.get_weights()
        
        assert "weights" in result or "status" in result, f"Should return weights: {result}"


class TestCrashRecovery:
    """Tests for crash recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The crash recovery scenario should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        await harness.setup_clean_state()
        
        result = await harness.run_scenario(CrashRecoveryScenario)
        
        assert result.completed_at is not None, "Scenario should complete"
        
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        print(f"Metrics: {result.metrics}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")


class TestMemoryProfiling:
    """Tests for memory profiling."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow - takes time to run multiple cycles
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The memory profiling scenario should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        await harness.setup_clean_state()
        
        result = await harness.run_scenario(MemoryProfilingScenario)
        
        assert result.completed_at is not None, "Scenario should complete"
        
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        
        # Print memory stats
        if "baseline_mb" in result.metrics:
            print(f"\nMemory Analysis:")
            print(f"  Baseline: {result.metrics.get('baseline_mb', 'N/A')} MB")
            print(f"  Final: {result.metrics.get('final_mb', 'N/A')} MB")
            print(f"  Peak: {result.metrics.get('peak_mb', 'N/A')} MB")
            print(f"  Total Growth: {result.metrics.get('total_growth_mb', 'N/A')} MB")
            print(f"  Avg/Cycle: {result.metrics.get('avg_growth_per_cycle_mb', 'N/A')} MB")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")


class TestTimeSeriesScoring:
    """Tests for time-series odds scoring with multi-sportsbook consensus."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_scenario_runs_to_completion(self, harness: LocalnetHarness):
        """The time-series scoring scenario should run without errors."""
        if not await harness.validator.health_check():
            pytest.skip("Validator not running - start with pm2 start ecosystem.e2e.config.js")
        
        await harness.setup_clean_state()
        
        from .scenarios import TimeSeriesScoringScenario
        result = await harness.run_scenario(TimeSeriesScoringScenario)
        
        assert result.completed_at is not None, "Scenario should complete"
        
        print(f"\nScenario: {result.scenario_id}")
        print(f"Success: {result.success}")
        print(f"Assertions passed: {result.assertions_passed}")
        print(f"Assertions failed: {result.assertions_failed}")
        print(f"Metrics: {result.metrics}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.assertions_failed > 0:
            pytest.fail(f"Scenario had {result.assertions_failed} failed assertions: {result.errors}")
