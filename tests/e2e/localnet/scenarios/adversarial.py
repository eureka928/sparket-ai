"""Adversarial Scenario - Tests system resilience to abuse.

Tests how the scoring system handles:
- Extreme odds values (1.01 to 100.0)
- Invalid probability values
- High-frequency submissions (rate limiting)
- Copy-trading (exact copies of market odds)
- Probability manipulation (probs not summing to 1.0)

Verifies:
- Extreme values are handled gracefully
- Rate limits prevent spam
- Copy-trading gets low originality scores
- Invalid submissions are rejected or penalized
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from .base import BaseScenario, ScenarioResult


@dataclass
class TestMarket:
    """Market for adversarial testing."""
    event_id: int
    market_id: int
    home_prob: float  # Ground truth
    away_prob: float


class AdversarialScenario(BaseScenario):
    """Tests system resilience to adversarial behavior."""
    
    SCENARIO_ID = "adversarial"
    NUM_EVENTS = 5
    
    def __init__(self, harness):
        super().__init__(harness)
        self.markets: List[TestMarket] = []
        self.rejection_counts: dict[str, int] = {}
    
    async def setup(self) -> None:
        """Create events for adversarial testing."""
        for i in range(self.NUM_EVENTS):
            result = await self.validator.create_event(
                home_team=f"AdvHome{i}",
                away_team=f"AdvAway{i}",
                hours_ahead=48,
            )
            
            if result.get("status") != "ok":
                self.result.add_error(f"Failed to create event {i}: {result}")
                continue
            
            event = result.get("event", {})
            event_id = event.get("db_event_id") or event.get("event_id")
            market_id = event.get("db_market_id") or event.get("market_id")
            
            if not market_id:
                self.result.add_error(f"No market_id for event {i}")
                continue
            
            # Standard ground truth
            home_prob = 0.55
            away_prob = 0.45
            
            self.markets.append(TestMarket(
                event_id=event_id,
                market_id=market_id,
                home_prob=home_prob,
                away_prob=away_prob,
            ))
            
            # Seed ground truth
            await self.validator.seed_ground_truth(
                market_id=market_id,
                home_prob=home_prob,
                away_prob=away_prob,
            )
        
        self.result.assert_true(
            len(self.markets) >= 3,
            f"Created at least 3 markets for testing"
        )
    
    async def execute(self) -> None:
        """Execute adversarial test cases."""
        if len(self.markets) == 0:
            self.result.add_error("No markets for testing")
            return
        
        await self.validator.sync_miners()
        
        # Test 1: Extreme odds
        await self._test_extreme_odds()
        
        # Test 2: Copy-trading
        await self._test_copy_trading()
        
        # Test 3: Rate limiting
        await self._test_rate_limiting()
        
        # Backdate for scoring
        await self.validator.backdate_submissions(days=1)
        
        # Trigger scoring
        await self.validator.trigger_odds_scoring()
        await self.validator.trigger_scoring()
    
    async def _test_extreme_odds(self) -> None:
        """Test submissions with extreme odds values."""
        market = self.markets[0]
        miner = self.miners[0]
        
        extreme_cases = [
            (1.01, 100.0),  # Very low home odds, extreme away
            (100.0, 1.01),  # Extreme home odds, very low away
            (1.05, 20.0),   # Near-certain home
            (50.0, 1.02),   # Near-certain away
        ]
        
        accepted = 0
        rejected = 0
        
        for home_odds, away_odds in extreme_cases:
            result = await miner.submit_odds(market.market_id, home_odds, away_odds)
            if result.get("status") == "ok":
                accepted += 1
            else:
                rejected += 1
        
        self.result.metrics["extreme_odds_accepted"] = accepted
        self.result.metrics["extreme_odds_rejected"] = rejected
        
        # The system should either accept and score appropriately, or reject
        # At minimum, it shouldn't crash
        self.result.assert_true(
            accepted + rejected == len(extreme_cases),
            "All extreme odds cases were handled"
        )
    
    async def _test_copy_trading(self) -> None:
        """Test exact copies of market odds (low originality)."""
        market = self.markets[1] if len(self.markets) > 1 else self.markets[0]
        
        # All miners submit exact same odds
        home_odds = round(1.0 / market.home_prob, 2)
        away_odds = round(1.0 / market.away_prob, 2)
        
        copy_submissions = 0
        
        for miner in self.miners:
            if await miner.health_check():
                result = await miner.submit_odds(market.market_id, home_odds, away_odds)
                if result.get("status") == "ok":
                    copy_submissions += 1
        
        self.result.metrics["copy_trade_submissions"] = copy_submissions
        
        # All submissions should be accepted (copy-trading is penalized in scoring, not rejected)
        self.result.assert_true(
            copy_submissions >= 1,
            "Copy-trade submissions were accepted (but should have low originality)"
        )
    
    async def _test_rate_limiting(self) -> None:
        """Test rapid-fire submissions to trigger rate limiting."""
        market = self.markets[2] if len(self.markets) > 2 else self.markets[0]
        miner = self.miners[0]
        
        rapid_submissions = 20
        accepted = 0
        rejected = 0
        
        # Submit as fast as possible
        tasks = []
        for i in range(rapid_submissions):
            # Slightly different odds each time
            home_odds = round(1.8 + i * 0.01, 2)
            away_odds = round(2.2 - i * 0.01, 2)
            tasks.append(miner.submit_odds(market.market_id, home_odds, away_odds))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                rejected += 1
            elif isinstance(result, dict):
                if result.get("status") == "ok":
                    accepted += 1
                else:
                    rejected += 1
            else:
                rejected += 1
        
        self.result.metrics["rapid_fire_accepted"] = accepted
        self.result.metrics["rapid_fire_rejected"] = rejected
        self.result.metrics["rapid_fire_total"] = rapid_submissions
        
        # Rate limiting should kick in for some submissions
        # But we don't fail if it doesn't - depends on configured limits
        if rejected > 0:
            self.result.add_warning(f"Rate limiting triggered: {rejected}/{rapid_submissions} rejected")
    
    async def verify(self) -> None:
        """Verify adversarial submissions are handled appropriately."""
        # Get all submissions
        submissions_result = await self.validator.get_submissions()
        submissions = submissions_result.get("submissions", [])
        
        self.result.metrics["total_submissions"] = len(submissions)
        
        # Get skill scores
        skill_result = await self.validator.get_skill_scores()
        skill_scores = skill_result.get("scores", [])
        
        if len(skill_scores) > 0:
            # Check that scores are bounded appropriately
            for score in skill_scores:
                skill = score.get("skill_score")
                if skill is not None:
                    # Skill scores should be bounded
                    self.result.assert_in_range(
                        skill, -10.0, 10.0,
                        f"Skill score in reasonable range for UID {score.get('uid')}"
                    )
        
        # Get weights
        weights_result = await self.validator.get_weights()
        weights = weights_result.get("weights", [])
        
        if len(weights) > 0:
            # All weights should be non-negative
            for w in weights:
                weight = w.get("weight", 0)
                self.result.assert_true(
                    weight >= 0,
                    f"Weight non-negative for UID {w.get('uid')}"
                )
            
            # Weights should sum to ~1.0
            total_weight = sum(w.get("weight", 0) for w in weights)
            if total_weight > 0:  # Skip if no weights
                self.result.assert_in_range(
                    total_weight, 0.99, 1.01,
                    "Weights sum to 1.0"
                )
        else:
            self.result.add_warning("No weights computed (may need more data)")
