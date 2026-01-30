"""Originality tracking for maximizing InfoDim score.

InfoDim (30% of total score) = 0.6 * SOS + 0.4 * LeadRatio

Where:
- SOS (Source of Signal): 1 - |correlation| with market. Higher = more original.
- LeadRatio: Fraction of market moves we anticipated (moved first).

Strategy:
1. Track how different our predictions are from market consensus
2. Identify opportunities to "lead" market moves
3. Balance originality with accuracy (being different but wrong hurts)

Usage:
    tracker = OriginalityTracker(data_path="/path/to/originality.json")

    # Before submitting, check if we should differentiate
    decision = tracker.should_differentiate(
        market_id=123,
        our_prob=0.55,
        market_prob=0.52,
        confidence=0.8,
    )

    # After submitting, record for tracking
    tracker.record_submission(market_id=123, our_prob=0.55, market_prob=0.52)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt


@dataclass
class SubmissionRecord:
    """Record of a submission for originality tracking."""
    timestamp: float
    market_id: int
    our_prob: float
    market_prob: float
    difference: float  # our_prob - market_prob

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "market_id": self.market_id,
            "our_prob": self.our_prob,
            "market_prob": self.market_prob,
            "difference": self.difference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubmissionRecord":
        return cls(
            timestamp=data["timestamp"],
            market_id=data["market_id"],
            our_prob=data["our_prob"],
            market_prob=data["market_prob"],
            difference=data["difference"],
        )


@dataclass
class MarketMoveRecord:
    """Record of a market move we anticipated or followed."""
    timestamp: float
    market_id: int
    direction: str  # "home" or "away"
    magnitude: float
    we_led: bool  # Did we move first?
    our_move_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "market_id": self.market_id,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "we_led": self.we_led,
            "our_move_time": self.our_move_time,
        }


@dataclass
class DifferentiationDecision:
    """Decision about whether to differentiate from market."""
    should_differentiate: bool
    reason: str
    suggested_adjustment: float  # How much to adjust our prob
    expected_sos_boost: float  # Expected SOS improvement
    risk_level: str  # "low", "medium", "high"


class OriginalityTracker:
    """Tracks and optimizes originality metrics for InfoDim scoring.

    Key strategies:
    1. Maintain moderate difference from market (SOS)
    2. Anticipate market moves (LeadRatio)
    3. Balance originality with accuracy
    """

    # Thresholds based on validator scoring params
    MOVE_THRESHOLD = 0.02  # 2% = significant move
    OPTIMAL_DIFFERENCE = 0.03  # 3% difference is good for SOS
    MAX_SAFE_DIFFERENCE = 0.08  # Beyond this, risk of being wrong increases
    MIN_DIFFERENCE = 0.01  # Below this, too correlated with market

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_history_days: float = 30.0,
    ) -> None:
        """Initialize originality tracker.

        Args:
            data_path: Path to persist tracking data
            max_history_days: Days of history to retain
        """
        self._data_path = Path(data_path) if data_path else None
        self._max_history_seconds = max_history_days * 24 * 3600

        # Submission history for SOS calculation
        self._submissions: List[SubmissionRecord] = []

        # Market move history for lead ratio
        self._market_moves: List[MarketMoveRecord] = []

        # Per-market tracking of our last submission
        self._last_submission: Dict[int, SubmissionRecord] = {}

        # Running stats
        self._total_submissions = 0
        self._total_leads = 0
        self._total_lags = 0

        if self._data_path and self._data_path.exists():
            self._load()

    def should_differentiate(
        self,
        market_id: int,
        our_prob: float,
        market_prob: float,
        confidence: float = 0.5,
        hours_to_game: float = 24.0,
    ) -> DifferentiationDecision:
        """Decide whether to differentiate our prediction from market.

        Balances:
        - SOS reward for being different
        - Accuracy risk of being different
        - Confidence in our model
        - Time until game (early = more room to differentiate)

        Args:
            market_id: Market identifier
            our_prob: Our model's probability
            market_prob: Market consensus probability
            confidence: Our confidence in our prediction (0-1)
            hours_to_game: Hours until game starts

        Returns:
            DifferentiationDecision with recommendation
        """
        current_diff = our_prob - market_prob
        abs_diff = abs(current_diff)

        # Early games: more room to differentiate and lead
        time_factor = min(1.0, hours_to_game / 48.0)  # Max boost at 48+ hours

        # Confidence affects how much we trust our edge
        confidence_factor = max(0.3, min(1.0, confidence))

        # Check current difference level
        if abs_diff < self.MIN_DIFFERENCE:
            # Too correlated - suggest differentiating
            optimal_diff = self.OPTIMAL_DIFFERENCE * confidence_factor * time_factor
            suggested_adj = optimal_diff if current_diff >= 0 else -optimal_diff

            return DifferentiationDecision(
                should_differentiate=True,
                reason="too_correlated_with_market",
                suggested_adjustment=suggested_adj,
                expected_sos_boost=0.05,  # Modest SOS improvement
                risk_level="low",
            )

        elif abs_diff > self.MAX_SAFE_DIFFERENCE:
            # Very different - check if we should moderate
            if confidence < 0.7:
                # Low confidence, too risky
                moderate_to = self.OPTIMAL_DIFFERENCE * (1 if current_diff > 0 else -1)
                adjustment = moderate_to - current_diff

                return DifferentiationDecision(
                    should_differentiate=False,
                    reason="too_different_low_confidence",
                    suggested_adjustment=adjustment,
                    expected_sos_boost=-0.02,  # Already high SOS
                    risk_level="high",
                )
            else:
                # High confidence, keep our edge
                return DifferentiationDecision(
                    should_differentiate=True,
                    reason="high_confidence_edge",
                    suggested_adjustment=0.0,
                    expected_sos_boost=0.08,  # Good SOS from being different
                    risk_level="medium",
                )

        else:
            # In optimal range
            return DifferentiationDecision(
                should_differentiate=True,
                reason="optimal_difference",
                suggested_adjustment=0.0,
                expected_sos_boost=0.05,
                risk_level="low",
            )

    def record_submission(
        self,
        market_id: int,
        our_prob: float,
        market_prob: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a submission for originality tracking.

        Args:
            market_id: Market identifier
            our_prob: Probability we submitted
            market_prob: Market consensus at submission time
            timestamp: Unix timestamp (defaults to now)
        """
        ts = timestamp or time.time()

        record = SubmissionRecord(
            timestamp=ts,
            market_id=market_id,
            our_prob=our_prob,
            market_prob=market_prob,
            difference=our_prob - market_prob,
        )

        self._submissions.append(record)
        self._last_submission[market_id] = record
        self._total_submissions += 1

        # Check if we led a market move
        self._check_for_lead(market_id, our_prob, market_prob, ts)

        # Prune old data
        self._prune_old_data()

        # Periodic save
        if self._total_submissions % 20 == 0:
            self._save()

    def record_market_move(
        self,
        market_id: int,
        old_prob: float,
        new_prob: float,
        timestamp: Optional[float] = None,
    ) -> Optional[MarketMoveRecord]:
        """Record a significant market move.

        Call this when market consensus changes by >= MOVE_THRESHOLD.

        Args:
            market_id: Market identifier
            old_prob: Previous market probability
            new_prob: New market probability
            timestamp: Unix timestamp

        Returns:
            MarketMoveRecord if move was significant
        """
        ts = timestamp or time.time()
        magnitude = new_prob - old_prob

        if abs(magnitude) < self.MOVE_THRESHOLD:
            return None

        direction = "home" if magnitude > 0 else "away"

        # Check if we anticipated this move
        we_led = False
        our_move_time = None

        last_sub = self._last_submission.get(market_id)
        if last_sub:
            # Did we move in this direction before the market?
            our_direction = "home" if last_sub.difference > 0 else "away"
            if our_direction == direction and abs(last_sub.difference) >= self.MOVE_THRESHOLD:
                we_led = True
                our_move_time = last_sub.timestamp
                self._total_leads += 1
            else:
                self._total_lags += 1

        record = MarketMoveRecord(
            timestamp=ts,
            market_id=market_id,
            direction=direction,
            magnitude=abs(magnitude),
            we_led=we_led,
            our_move_time=our_move_time,
        )

        self._market_moves.append(record)

        if we_led:
            bt.logging.info({
                "originality": "led_market_move",
                "market_id": market_id,
                "direction": direction,
                "magnitude": round(abs(magnitude), 3),
            })

        return record

    def get_sos_estimate(self, window_hours: float = 168.0) -> float:
        """Estimate our current SOS score.

        SOS = 1 - |correlation with market|
        Higher is better (more original).

        Args:
            window_hours: Hours to look back

        Returns:
            Estimated SOS score (0-1)
        """
        cutoff = time.time() - (window_hours * 3600)
        recent = [s for s in self._submissions if s.timestamp >= cutoff]

        if len(recent) < 10:
            return 0.5  # Default for insufficient data

        # Calculate average absolute difference
        avg_abs_diff = sum(abs(s.difference) for s in recent) / len(recent)

        # Convert to SOS estimate
        # SOS ≈ 1 - correlation
        # If avg_diff is ~3%, correlation is ~0.7, SOS is ~0.3
        # If avg_diff is ~5%, correlation is ~0.5, SOS is ~0.5
        # If avg_diff is ~8%, correlation is ~0.2, SOS is ~0.8

        # Approximate mapping (based on typical correlations)
        if avg_abs_diff < 0.01:
            sos = 0.1  # Very correlated
        elif avg_abs_diff < 0.02:
            sos = 0.25
        elif avg_abs_diff < 0.03:
            sos = 0.4
        elif avg_abs_diff < 0.05:
            sos = 0.55
        elif avg_abs_diff < 0.08:
            sos = 0.7
        else:
            sos = 0.85  # Very different

        return sos

    def get_lead_ratio(self, window_hours: float = 168.0) -> float:
        """Get our lead ratio for recent market moves.

        LeadRatio = leads / (leads + lags)
        Higher is better (we anticipate moves).

        Args:
            window_hours: Hours to look back

        Returns:
            Lead ratio (0-1)
        """
        cutoff = time.time() - (window_hours * 3600)
        recent = [m for m in self._market_moves if m.timestamp >= cutoff]

        if not recent:
            return 0.5  # Default

        leads = sum(1 for m in recent if m.we_led)
        total = len(recent)

        return leads / total if total > 0 else 0.5

    def get_info_dim_estimate(self, window_hours: float = 168.0) -> float:
        """Estimate our InfoDim score.

        InfoDim = 0.6 * SOS + 0.4 * LeadRatio

        Args:
            window_hours: Hours to look back

        Returns:
            Estimated InfoDim score (0-1)
        """
        sos = self.get_sos_estimate(window_hours)
        lead_ratio = self.get_lead_ratio(window_hours)

        return 0.6 * sos + 0.4 * lead_ratio

    def _check_for_lead(
        self,
        market_id: int,
        our_prob: float,
        market_prob: float,
        timestamp: float,
    ) -> None:
        """Check if our submission anticipates a market move."""
        diff = our_prob - market_prob

        # If we're significantly different from market, we might be leading
        if abs(diff) >= self.MOVE_THRESHOLD:
            bt.logging.debug({
                "originality": "potential_lead",
                "market_id": market_id,
                "our_prob": round(our_prob, 3),
                "market_prob": round(market_prob, 3),
                "difference": round(diff, 3),
            })

    def _prune_old_data(self) -> None:
        """Remove data older than max_history."""
        cutoff = time.time() - self._max_history_seconds

        self._submissions = [s for s in self._submissions if s.timestamp >= cutoff]
        self._market_moves = [m for m in self._market_moves if m.timestamp >= cutoff]

        # Clean up last_submission for old markets
        old_markets = [
            mid for mid, sub in self._last_submission.items()
            if sub.timestamp < cutoff
        ]
        for mid in old_markets:
            del self._last_submission[mid]

    def stats(self) -> Dict[str, Any]:
        """Get originality tracking statistics."""
        return {
            "total_submissions": self._total_submissions,
            "total_leads": self._total_leads,
            "total_lags": self._total_lags,
            "lead_ratio": self.get_lead_ratio(),
            "sos_estimate": round(self.get_sos_estimate(), 3),
            "info_dim_estimate": round(self.get_info_dim_estimate(), 3),
            "recent_submissions": len(self._submissions),
            "recent_market_moves": len(self._market_moves),
        }

    def _save(self) -> None:
        """Persist tracking data to disk."""
        if not self._data_path:
            return

        try:
            self._data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "submissions": [s.to_dict() for s in self._submissions[-1000:]],
                "market_moves": [m.to_dict() for m in self._market_moves[-500:]],
                "total_submissions": self._total_submissions,
                "total_leads": self._total_leads,
                "total_lags": self._total_lags,
            }

            with open(self._data_path, "w") as f:
                json.dump(data, f)

        except Exception as e:
            bt.logging.warning({
                "originality": "save_failed",
                "error": str(e),
            })

    def _load(self) -> None:
        """Load tracking data from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path) as f:
                data = json.load(f)

            self._submissions = [
                SubmissionRecord.from_dict(s)
                for s in data.get("submissions", [])
            ]

            for m in data.get("market_moves", []):
                self._market_moves.append(MarketMoveRecord(
                    timestamp=m["timestamp"],
                    market_id=m["market_id"],
                    direction=m["direction"],
                    magnitude=m["magnitude"],
                    we_led=m["we_led"],
                    our_move_time=m.get("our_move_time"),
                ))

            self._total_submissions = data.get("total_submissions", len(self._submissions))
            self._total_leads = data.get("total_leads", 0)
            self._total_lags = data.get("total_lags", 0)

            bt.logging.info({
                "originality": "loaded",
                "submissions": len(self._submissions),
                "market_moves": len(self._market_moves),
            })

        except Exception as e:
            bt.logging.warning({
                "originality": "load_failed",
                "error": str(e),
            })


__all__ = [
    "OriginalityTracker",
    "DifferentiationDecision",
    "SubmissionRecord",
    "MarketMoveRecord",
]
