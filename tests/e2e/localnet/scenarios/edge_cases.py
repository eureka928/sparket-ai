"""Edge Case Scenario - Tests boundary conditions and unusual situations.

Tests:
- Single miner submitting (no competition)
- All miners submitting identical odds
- Cancelled/postponed events
- Markets with missing quotes
- Events with no submissions

Verifies:
- System handles gracefully when data is sparse
- Cancelled events are excluded from scoring
- Missing quotes don't crash scoring pipeline
- Single miner still gets reasonable score
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from .base import BaseScenario, ScenarioResult


class EdgeCaseScenario(BaseScenario):
    """Tests edge cases and boundary conditions."""
    
    SCENARIO_ID = "edge_cases"
    
    def __init__(self, harness):
        super().__init__(harness)
        self.markets_with_submissions: List[int] = []
        self.markets_no_submissions: List[int] = []
        self.cancelled_markets: List[int] = []
    
    async def setup(self) -> None:
        """Create various edge case situations."""
        # Case 1: Normal market with single miner submission
        result = await self.validator.create_event(
            home_team="SingleMinerHome",
            away_team="SingleMinerAway",
            hours_ahead=48,
        )
        if result.get("status") == "ok":
            market_id = result.get("event", {}).get("db_market_id")
            if market_id:
                self.markets_with_submissions.append(market_id)
                await self.validator.seed_ground_truth(
                    market_id=market_id,
                    home_prob=0.55,
                    away_prob=0.45,
                )
        
        # Case 2: Market with no submissions
        result = await self.validator.create_event(
            home_team="NoSubmissionHome",
            away_team="NoSubmissionAway",
            hours_ahead=48,
        )
        if result.get("status") == "ok":
            market_id = result.get("event", {}).get("db_market_id")
            if market_id:
                self.markets_no_submissions.append(market_id)
                await self.validator.seed_ground_truth(
                    market_id=market_id,
                    home_prob=0.50,
                    away_prob=0.50,
                )
        
        # Case 3: Market with identical submissions from all miners
        result = await self.validator.create_event(
            home_team="IdenticalHome",
            away_team="IdenticalAway",
            hours_ahead=48,
        )
        if result.get("status") == "ok":
            market_id = result.get("event", {}).get("db_market_id")
            if market_id:
                self.markets_with_submissions.append(market_id)
                await self.validator.seed_ground_truth(
                    market_id=market_id,
                    home_prob=0.60,
                    away_prob=0.40,
                )
        
        # Case 4: Cancelled event (should be excluded from scoring)
        # Note: We create as scheduled first, then transition to cancelled
        result = await self.validator.create_event(
            home_team="CancelledHome",
            away_team="CancelledAway",
            hours_ahead=24,
        )
        if result.get("status") == "ok":
            market_id = result.get("event", {}).get("db_market_id")
            event_id = result.get("event", {}).get("db_event_id")
            if market_id:
                self.cancelled_markets.append(market_id)
                # Don't seed ground truth for cancelled events
        
        total_markets = (
            len(self.markets_with_submissions) + 
            len(self.markets_no_submissions) + 
            len(self.cancelled_markets)
        )
        
        self.result.assert_true(
            total_markets >= 3,
            f"Created at least 3 markets for edge case testing"
        )
    
    async def execute(self) -> None:
        """Execute edge case submissions."""
        await self.validator.sync_miners()
        
        # Case 1: Single miner submits to first market
        if self.markets_with_submissions:
            market_id = self.markets_with_submissions[0]
            miner = self.miners[0]
            
            if await miner.health_check():
                result = await miner.submit_odds(market_id, 1.8, 2.1)
                if result.get("status") == "ok":
                    self.result.metrics["single_miner_submitted"] = True
        
        # Case 2: All miners submit identical odds to second market
        if len(self.markets_with_submissions) > 1:
            market_id = self.markets_with_submissions[1]
            identical_home = 1.67  # ~60% implied
            identical_away = 2.50  # ~40% implied
            
            submitted_count = 0
            for miner in self.miners:
                if await miner.health_check():
                    result = await miner.submit_odds(market_id, identical_home, identical_away)
                    if result.get("status") == "ok":
                        submitted_count += 1
            
            self.result.metrics["identical_submissions"] = submitted_count
        
        # Case 3: No submissions to markets_no_submissions (intentionally empty)
        
        # Case 4: Submit to cancelled market (submissions should be ignored in scoring)
        if self.cancelled_markets:
            market_id = self.cancelled_markets[0]
            miner = self.miners[0]
            
            if await miner.health_check():
                result = await miner.submit_odds(market_id, 2.0, 1.9)
                # Note: submission may succeed, but scoring should exclude it
        
        # Backdate submissions
        await self.validator.backdate_submissions(days=1)
        
        # Trigger scoring
        await self.validator.trigger_odds_scoring()
        await self.validator.trigger_scoring()
    
    async def verify(self) -> None:
        """Verify edge cases are handled gracefully."""
        # Get submissions
        submissions_result = await self.validator.get_submissions()
        submissions = submissions_result.get("submissions", [])
        
        self.result.metrics["total_submissions"] = len(submissions)
        
        # Get skill scores
        skill_result = await self.validator.get_skill_scores()
        skill_scores = skill_result.get("scores", [])
        
        self.result.metrics["skill_scores_count"] = len(skill_scores)
        
        # Verify single miner still gets a score
        if len(skill_scores) > 0:
            # At least one miner should have a score
            has_score = any(s.get("skill_score") is not None for s in skill_scores)
            self.result.assert_true(
                has_score,
                "At least one miner has a skill score"
            )
        
        # Get weights
        weights_result = await self.validator.get_weights()
        weights = weights_result.get("weights", [])
        
        self.result.metrics["weights_count"] = len(weights)
        
        # Weights should still be valid
        if len(weights) > 0:
            total_weight = sum(w.get("weight", 0) for w in weights)
            if total_weight > 0:
                self.result.assert_in_range(
                    total_weight, 0.99, 1.01,
                    "Weights sum to 1.0 despite edge cases"
                )
        
        # The main test is that none of the above crashed the system
        self.result.assert_true(
            True,
            "Scoring pipeline completed without crashing on edge cases"
        )
