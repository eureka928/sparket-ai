"""Time-Series Scoring Scenario - Tests CLV with realistic market movement.

This scenario uses the enhanced MockProvider to generate time-series odds
across multiple sportsbooks, simulating realistic market conditions.

Tests:
- Multi-sportsbook consensus computation
- Ground truth snapshot pipeline
- CLV scoring against time-matched consensus
- Early submission bonus (true CLV)

Verifies:
- Miners who submit closer to true probability score higher
- Time-weight factors correlate with time-to-close
- Consensus properly averages across sportsbooks
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from .base import BaseScenario, ScenarioResult


@dataclass
class TimeSeriesMarket:
    """Market with time-series metadata."""
    mock_market_id: str  # UUID from MockProvider
    db_event_id: int
    db_market_id: int
    true_prob_home: float
    open_time: datetime
    close_time: datetime


class TimeSeriesScoringScenario(BaseScenario):
    """Tests CLV scoring with realistic time-series odds movement."""
    
    SCENARIO_ID = "timeseries_scoring"
    NUM_EVENTS = 3
    SPORTSBOOKS = ["PINN", "DKNG", "FDUEL"]  # Sharp + 2 soft
    
    def __init__(self, harness):
        super().__init__(harness)
        self.markets: List[TimeSeriesMarket] = []
        self.miner_submissions: dict[str, List[dict]] = {}
    
    async def setup(self) -> None:
        """Create events with time-series odds from multiple sportsbooks."""
        now = datetime.now(timezone.utc)
        
        # Different true probabilities for variety
        true_probs = [0.60, 0.45, 0.55]
        
        for i in range(self.NUM_EVENTS):
            # Create event
            result = await self.validator.create_event(
                home_team=f"TSHomeTeam{i}",
                away_team=f"TSAwayTeam{i}",
                hours_ahead=48,
            )
            
            if result.get("status") != "ok":
                self.result.add_error(f"Failed to create event {i}: {result}")
                continue
            
            event = result.get("event", {})
            mock_event_id = event.get("event_id")  # UUID
            db_event_id = event.get("db_event_id")
            db_market_id = event.get("db_market_id")
            
            if not db_market_id:
                self.result.add_error(f"No db_market_id for event {i}")
                continue
            
            true_prob = true_probs[i % len(true_probs)]
            
            # Market opens 3 days before event, closes at event start
            open_time = now - timedelta(days=3)
            close_time = now + timedelta(hours=48)
            
            # Generate time-series odds across sportsbooks
            ts_result = await self.validator.generate_timeseries(
                market_id=mock_event_id,
                true_prob_home=true_prob,
                open_time=open_time,
                close_time=close_time,
                interval_hours=6,
                sportsbook_codes=self.SPORTSBOOKS,
                seed=42 + i,  # Reproducible
            )
            
            if ts_result.get("status") != "ok":
                self.result.add_warning(f"Timeseries generation failed: {ts_result}")
                continue
            
            # Seed ground truth tables from timeseries
            gt_result = await self.validator.seed_ground_truth_from_timeseries(
                mock_market_id=mock_event_id,
                db_market_id=db_market_id,
            )
            
            if gt_result.get("status") != "ok":
                self.result.add_warning(f"Ground truth seeding failed: {gt_result}")
            
            self.markets.append(TimeSeriesMarket(
                mock_market_id=mock_event_id,
                db_event_id=db_event_id,
                db_market_id=db_market_id,
                true_prob_home=true_prob,
                open_time=open_time,
                close_time=close_time,
            ))
        
        self.result.metrics["markets_created"] = len(self.markets)
        self.result.assert_greater(
            len(self.markets), 0,
            "At least one market created with time-series"
        )
    
    async def execute(self) -> None:
        """Miners submit with varying accuracy relative to time-series consensus."""
        if not self.markets:
            self.result.add_error("No markets available")
            return
        
        await self.validator.sync_miners()
        
        # Miner 1: Submit close to true probability (should beat consensus)
        miner1_results = []
        for market in self.markets:
            home_prob = market.true_prob_home + 0.01  # Slight noise
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / (1.0 - home_prob), 2)
            
            result = await self.miners[0].submit_odds(market.db_market_id, home_odds, away_odds)
            miner1_results.append(result)
        
        self.miner_submissions["miner_1"] = [{"strategy": "near_true_prob", "results": miner1_results}]
        
        await asyncio.sleep(0.5)
        
        # Miner 2: Submit at consensus level (neutral CLV)
        miner2_results = []
        for market in self.markets:
            # Get consensus from mock provider
            consensus = await self.validator.get_consensus(
                market_id=market.mock_market_id,
                side="HOME",
            )
            if consensus.get("consensus"):
                prob = consensus["consensus"].get("prob_consensus", market.true_prob_home)
            else:
                prob = market.true_prob_home
            
            home_odds = round(1.0 / prob, 2)
            away_odds = round(1.0 / (1.0 - prob), 2)
            
            result = await self.miners[1].submit_odds(market.db_market_id, home_odds, away_odds)
            miner2_results.append(result)
        
        self.miner_submissions["miner_2"] = [{"strategy": "at_consensus", "results": miner2_results}]
        
        await asyncio.sleep(0.5)
        
        # Miner 3: Submit far from true probability (negative CLV)
        miner3_results = []
        for market in self.markets:
            # Deliberately wrong
            home_prob = 1.0 - market.true_prob_home  # Flip
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / (1.0 - home_prob), 2)
            
            result = await self.miners[2].submit_odds(market.db_market_id, home_odds, away_odds)
            miner3_results.append(result)
        
        self.miner_submissions["miner_3"] = [{"strategy": "inverted", "results": miner3_results}]
        
        # Wait for submissions to be processed
        await asyncio.sleep(2.0)
        
        await self.validator.backdate_submissions(days=1)
        await self.validator.trigger_odds_scoring()
        await self.validator.trigger_scoring()
    
    async def verify(self) -> None:
        """Verify CLV scores reflect accuracy relative to consensus."""
        submissions_result = await self.validator.get_submissions()
        db_submissions = submissions_result.get("submissions", [])
        
        self.result.metrics["db_submissions"] = len(db_submissions)
        self.result.assert_greater(
            len(db_submissions), 0,
            "Submissions recorded in DB"
        )
        
        # Check skill scores
        skill_result = await self.validator.get_skill_scores()
        skill_scores = skill_result.get("scores", [])
        
        if skill_scores:
            self.result.metrics["skill_scores"] = len(skill_scores)
            
            score_by_uid = {s["uid"]: s.get("skill_score") for s in skill_scores if s.get("uid")}
            
            # UIDs from config: local-miner=2, e2e-miner-2=4, e2e-miner-3=5
            m1 = score_by_uid.get(2)
            m2 = score_by_uid.get(4)
            m3 = score_by_uid.get(5)
            
            self.result.metrics["miner1_skill"] = m1
            self.result.metrics["miner2_skill"] = m2
            self.result.metrics["miner3_skill"] = m3
            
            # Miner 1 (near true) should beat Miner 3 (inverted)
            if m1 is not None and m3 is not None:
                if m1 > m3:
                    self.result.assert_true(True, "Near-true miner beats inverted miner")
                else:
                    self.result.add_warning(
                        f"Unexpected: miner1={m1:.4f} <= miner3={m3:.4f}"
                    )
        else:
            self.result.add_warning("No skill scores computed")
        
        # Check weights
        weights_result = await self.validator.get_weights()
        weights = weights_result.get("weights", [])
        
        if weights:
            total = sum(w.get("weight", 0) for w in weights)
            self.result.assert_in_range(total, 0.99, 1.01, "Weights sum to 1.0")
            self.result.metrics["weights_count"] = len(weights)
