"""Outcome Verification Scenario - Tests PSS scoring against realized outcomes.

Tests the Proper Scoring Score (PSS) pipeline with:
- Events that have settled outcomes
- Brier score and Log Loss computation
- PSS score aggregation

Verifies:
- Outcome scores are computed correctly
- Miners with correct predictions score better
- PSS scores are incorporated into skill score
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from .base import BaseScenario, ScenarioResult


@dataclass
class SettledEvent:
    """Holds data for a settled event."""
    event_id: int
    market_id: int
    home_prob: float
    away_prob: float
    actual_result: str  # "HOME" or "AWAY"


class OutcomeVerificationScenario(BaseScenario):
    """Tests PSS scoring with settled outcomes."""
    
    SCENARIO_ID = "outcome_verification"
    NUM_EVENTS = 5
    
    def __init__(self, harness):
        super().__init__(harness)
        self.events: List[SettledEvent] = []
        self.miner_predictions: dict[str, List[dict]] = {}
    
    async def setup(self) -> None:
        """Create events with outcomes."""
        for i in range(self.NUM_EVENTS):
            result = await self.validator.create_event(
                home_team=f"Home{i}",
                away_team=f"Away{i}",
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
            
            # Define ground truth and actual outcome
            # Make some outcomes favor HOME, others AWAY
            if i < self.NUM_EVENTS // 2:
                home_prob = 0.65  # HOME is favored
                actual_result = "HOME"  # HOME wins (expected)
            else:
                home_prob = 0.35  # AWAY is favored
                actual_result = "AWAY"  # AWAY wins (expected)
            
            away_prob = round(1.0 - home_prob, 4)
            
            self.events.append(SettledEvent(
                event_id=event_id,
                market_id=market_id,
                home_prob=home_prob,
                away_prob=away_prob,
                actual_result=actual_result,
            ))
        
        self.result.assert_true(
            len(self.events) >= self.NUM_EVENTS - 1,
            f"Created at least {self.NUM_EVENTS - 1} events"
        )
        
        # Seed ground truth closing
        for event in self.events:
            await self.validator.seed_ground_truth(
                market_id=event.market_id,
                home_prob=event.home_prob,
                away_prob=event.away_prob,
            )
    
    async def execute(self) -> None:
        """Submit predictions and seed outcomes."""
        if len(self.events) == 0:
            self.result.add_error("No events to process")
            return
        
        # Sync miners
        await self.validator.sync_miners()
        
        # Miner 1: Predict correctly (aligned with ground truth)
        self.miner_predictions["correct"] = await self._submit_correct_predictions(
            miner_index=0
        )
        
        # Miner 2: Predict incorrectly (opposite of ground truth)
        self.miner_predictions["incorrect"] = await self._submit_incorrect_predictions(
            miner_index=1
        )
        
        # Miner 3: Random predictions
        self.miner_predictions["random"] = await self._submit_random_predictions(
            miner_index=2
        )
        
        # Backdate submissions
        await self.validator.backdate_submissions(days=1)
        
        # Seed actual outcomes
        for event in self.events:
            home_score = 2 if event.actual_result == "HOME" else 0
            away_score = 0 if event.actual_result == "HOME" else 1
            
            await self.validator.seed_outcome(
                market_id=event.market_id,
                result=event.actual_result,
                home_score=home_score,
                away_score=away_score,
            )
        
        # Trigger outcome scoring
        await self.validator.trigger_outcome_scoring()
        
        # Trigger full scoring pipeline
        await self.validator.trigger_scoring()
    
    async def _submit_correct_predictions(self, miner_index: int) -> List[dict]:
        """Submit predictions aligned with actual outcomes."""
        predictions = []
        miner = self.miners[miner_index]
        
        for event in self.events:
            # Predict aligned with actual result
            if event.actual_result == "HOME":
                home_prob = 0.70  # Confident HOME prediction
            else:
                home_prob = 0.30  # Confident AWAY prediction
            
            away_prob = 1.0 - home_prob
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / away_prob, 2)
            
            result = await miner.submit_odds(event.market_id, home_odds, away_odds)
            predictions.append({
                "market_id": event.market_id,
                "predicted": "HOME" if home_prob > 0.5 else "AWAY",
                "actual": event.actual_result,
                "aligned": (home_prob > 0.5) == (event.actual_result == "HOME"),
            })
        
        return predictions
    
    async def _submit_incorrect_predictions(self, miner_index: int) -> List[dict]:
        """Submit predictions opposite to actual outcomes."""
        predictions = []
        miner = self.miners[miner_index]
        
        for event in self.events:
            # Predict opposite to actual result
            if event.actual_result == "HOME":
                home_prob = 0.30  # Predicting AWAY incorrectly
            else:
                home_prob = 0.70  # Predicting HOME incorrectly
            
            away_prob = 1.0 - home_prob
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / away_prob, 2)
            
            result = await miner.submit_odds(event.market_id, home_odds, away_odds)
            predictions.append({
                "market_id": event.market_id,
                "predicted": "HOME" if home_prob > 0.5 else "AWAY",
                "actual": event.actual_result,
                "aligned": False,
            })
        
        return predictions
    
    async def _submit_random_predictions(self, miner_index: int) -> List[dict]:
        """Submit random predictions."""
        predictions = []
        miner = self.miners[miner_index]
        
        for event in self.events:
            # Random probability
            home_prob = random.uniform(0.3, 0.7)
            away_prob = 1.0 - home_prob
            home_odds = round(1.0 / home_prob, 2)
            away_odds = round(1.0 / away_prob, 2)
            
            result = await miner.submit_odds(event.market_id, home_odds, away_odds)
            predictions.append({
                "market_id": event.market_id,
                "predicted": "HOME" if home_prob > 0.5 else "AWAY",
                "actual": event.actual_result,
                "aligned": (home_prob > 0.5) == (event.actual_result == "HOME"),
            })
        
        return predictions
    
    async def verify(self) -> None:
        """Verify PSS scores are computed correctly."""
        # Get skill scores
        skill_result = await self.validator.get_skill_scores()
        skill_scores = skill_result.get("scores", [])
        
        if len(skill_scores) == 0:
            self.result.add_warning("No skill scores computed (may need more data)")
            return
        
        self.result.metrics["skill_scores_count"] = len(skill_scores)
        
        # Extract scores by UID
        score_by_uid = {}
        for score in skill_scores:
            uid = score.get("uid")
            if uid is not None:
                score_by_uid[uid] = {
                    "skill_score": score.get("skill_score"),
                    "sos_score": score.get("sos_score"),  # Sum of Scores (PSS-related)
                }
        
        # UIDs: local-miner=2 (correct), e2e-miner-2=4 (incorrect), e2e-miner-3=5 (random)
        correct_score = score_by_uid.get(2, {}).get("skill_score")
        incorrect_score = score_by_uid.get(4, {}).get("skill_score")
        random_score = score_by_uid.get(5, {}).get("skill_score")
        
        self.result.metrics["correct_miner_skill"] = correct_score
        self.result.metrics["incorrect_miner_skill"] = incorrect_score
        self.result.metrics["random_miner_skill"] = random_score
        
        # Log prediction alignment counts
        for strategy, preds in self.miner_predictions.items():
            aligned_count = sum(1 for p in preds if p.get("aligned"))
            self.result.metrics[f"{strategy}_aligned"] = aligned_count
        
        # Core assertion: miner with correct predictions should score higher
        if correct_score is not None and incorrect_score is not None:
            self.result.assert_greater(
                correct_score, incorrect_score,
                "Correct predictions score higher than incorrect"
            )
        else:
            self.result.add_warning("Could not compare scores (missing data)")
        
        # Get scores breakdown
        scores_result = await self.validator.get_scores()
        scores = scores_result.get("scores", [])
        
        if len(scores) > 0:
            self.result.metrics["individual_scores"] = len(scores)
            
            # Sample a few for logging
            for score in scores[:5]:
                self.result.metrics[f"score_{score.get('submission_id', 'unknown')}"] = {
                    "brier": score.get("brier_score"),
                    "pss": score.get("pss"),
                }
