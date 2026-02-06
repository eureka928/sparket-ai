"""Timing strategy for maximizing time credit in scoring.

The scoring system applies time bonuses/penalties:
- 7+ days before event: 100% credit
- 1 hour before: 10% credit (floor)
- Early bad predictions: 70% penalty (forgiven)
- Late bad predictions: 100% penalty

Strategy:
1. Submit as early as possible (7+ days)
2. Adaptive refresh: 5min (≤6h), 30min (≤24h), 1h (≤72h), 6h (>72h)
3. Never submit within 2 hours of event start
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sparket.miner.custom.config import TimingConfig


class SubmissionDecision(Enum):
    """Decision for whether/when to submit."""

    SUBMIT_NOW = "submit_now"  # Submit immediately
    WAIT = "wait"  # Wait before submitting
    SKIP = "skip"  # Don't submit (too late, etc.)
    UPDATE = "update"  # Update existing submission


@dataclass
class TimingDecision:
    """Result of timing strategy evaluation."""

    decision: SubmissionDecision
    reason: str
    time_credit: float  # Expected time credit (0-1)
    next_check_seconds: int  # When to check again
    priority: int  # Higher = more urgent


class TimingStrategy:
    """Strategic timing for odds submissions.

    Optimizes for time credit in the scoring system while
    balancing update frequency and prediction quality.

    Key principles:
    1. Early is better - submit 7+ days before for full credit
    2. Update frequently - every 6 hours to capture new info
    3. Never submit late - avoid the 2-hour penalty zone

    Usage:
        strategy = TimingStrategy(config)

        for market in markets:
            decision = strategy.evaluate(
                market_id=123,
                event_start=event_start_time,
                last_submission=last_submitted_at,
            )

            if decision.decision == SubmissionDecision.SUBMIT_NOW:
                # Submit odds
                ...
    """

    def __init__(self, config: Optional[TimingConfig] = None) -> None:
        """Initialize timing strategy.

        Args:
            config: Timing configuration parameters
        """
        self.config = config or TimingConfig()
        self._submissions: Dict[int, datetime] = {}  # market_id -> last submission time

    def evaluate(
        self,
        market_id: int,
        event_start: datetime,
        last_submission: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> TimingDecision:
        """Evaluate whether to submit for a market.

        Args:
            market_id: Market identifier
            event_start: When the event starts (UTC)
            last_submission: When we last submitted (optional)
            now: Current time (defaults to UTC now)

        Returns:
            TimingDecision with action and metadata
        """
        now = now or datetime.now(timezone.utc)

        # Ensure event_start is timezone-aware
        if event_start.tzinfo is None:
            event_start = event_start.replace(tzinfo=timezone.utc)

        # Time until event
        time_until = event_start - now
        hours_until = time_until.total_seconds() / 3600

        # Calculate expected time credit
        time_credit = self._calculate_time_credit(hours_until)

        # Check if too late (cutoff)
        if hours_until < self.config.cutoff_hours:
            return TimingDecision(
                decision=SubmissionDecision.SKIP,
                reason=f"Too late: only {hours_until:.1f}h until event (cutoff: {self.config.cutoff_hours}h)",
                time_credit=time_credit,
                next_check_seconds=0,
                priority=0,
            )

        # Check if in penalty zone
        if hours_until < self.config.min_hours_before_event:
            return TimingDecision(
                decision=SubmissionDecision.SKIP,
                reason=f"In penalty zone: {hours_until:.1f}h until event",
                time_credit=time_credit,
                next_check_seconds=0,
                priority=0,
            )

        # Check if we've submitted before
        last_sub = last_submission or self._submissions.get(market_id)

        adaptive_refresh = self._adaptive_refresh_seconds(hours_until)

        if last_sub is None:
            # First submission - always do it
            priority = self._calculate_priority(hours_until, time_credit)
            return TimingDecision(
                decision=SubmissionDecision.SUBMIT_NOW,
                reason="Initial submission",
                time_credit=time_credit,
                next_check_seconds=adaptive_refresh,
                priority=priority,
            )

        # Check if enough time has passed since last submission
        since_last = now - last_sub
        refresh_delta = timedelta(seconds=adaptive_refresh)

        if since_last >= refresh_delta:
            priority = self._calculate_priority(hours_until, time_credit)
            return TimingDecision(
                decision=SubmissionDecision.UPDATE,
                reason=f"Refresh interval ({since_last.total_seconds() / 3600:.1f}h since last)",
                time_credit=time_credit,
                next_check_seconds=adaptive_refresh,
                priority=priority,
            )

        # Wait for next refresh
        wait_seconds = int((refresh_delta - since_last).total_seconds())
        return TimingDecision(
            decision=SubmissionDecision.WAIT,
            reason=f"Wait {wait_seconds}s for next refresh",
            time_credit=time_credit,
            next_check_seconds=wait_seconds,
            priority=0,
        )

    def _adaptive_refresh_seconds(self, hours_until: float) -> int:
        """Calculate adaptive refresh interval based on proximity to game.

        Closer games refresh more frequently to capture market movement
        and improve lead ratio and time credit.

        Args:
            hours_until: Hours until event start

        Returns:
            Refresh interval in seconds
        """
        min_refresh = self.config.min_refresh_seconds

        if hours_until <= 6:
            return min_refresh
        elif hours_until <= 24:
            return max(min_refresh, 1800)
        elif hours_until <= 72:
            return max(min_refresh, 3600)
        else:
            return self.config.refresh_interval_seconds

    def _calculate_time_credit(self, hours_until: float) -> float:
        """Calculate expected time credit matching validator's logarithmic formula.

        Validator uses: normalized = (log(minutes) - log(min)) / (log(max) - log(min))
        with min=60min, max=early_submission_days*24*60, floor=0.1
        """
        import math

        if hours_until <= 0:
            return 0.0

        minutes = hours_until * 60.0
        min_minutes = 60.0  # 1 hour floor
        max_minutes = self.config.early_submission_days * 24 * 60  # 7 days default
        floor_factor = 0.1

        if minutes >= max_minutes:
            return 1.0
        if minutes <= min_minutes:
            return floor_factor

        log_min = math.log(min_minutes)
        log_max = math.log(max_minutes)
        log_val = math.log(minutes)
        normalized = (log_val - log_min) / (log_max - log_min)

        return floor_factor + normalized * (1.0 - floor_factor)

    def _calculate_priority(self, hours_until: float, time_credit: float) -> int:
        """Calculate submission priority (0-100).

        Higher priority for:
        - Markets about to lose time credit
        - Markets we haven't submitted to yet
        """
        # Base priority from time credit (losing credit = higher priority)
        if hours_until > self.config.early_submission_days * 24:
            # Very early - lower priority
            return 30
        elif hours_until > 24:
            # 1-7 days out - medium priority
            return 50
        elif hours_until > 6:
            # Same day - higher priority
            return 70
        else:
            # Last few hours - highest priority
            return 90

    def record_submission(self, market_id: int, timestamp: Optional[datetime] = None) -> None:
        """Record that a submission was made.

        Args:
            market_id: Market identifier
            timestamp: When submitted (defaults to now)
        """
        self._submissions[market_id] = timestamp or datetime.now(timezone.utc)

    def get_submission_schedule(
        self,
        markets: List[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> List[Tuple[Dict[str, Any], TimingDecision]]:
        """Get prioritized submission schedule for markets.

        Args:
            markets: List of market dicts with event_start times
            now: Current time

        Returns:
            List of (market, decision) tuples, sorted by priority
        """
        now = now or datetime.now(timezone.utc)
        results = []

        for market in markets:
            market_id = market.get("market_id", 0)
            event_start = market.get("start_time_utc") or market.get("event_start")

            if event_start is None:
                continue

            # Parse datetime if string
            if isinstance(event_start, str):
                event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))

            decision = self.evaluate(
                market_id=market_id,
                event_start=event_start,
                now=now,
            )

            if decision.decision in (SubmissionDecision.SUBMIT_NOW, SubmissionDecision.UPDATE):
                results.append((market, decision))

        # Sort by priority (highest first)
        results.sort(key=lambda x: x[1].priority, reverse=True)

        return results

    def get_next_refresh_time(self, market_id: int) -> Optional[datetime]:
        """Get when this market should next be refreshed.

        Args:
            market_id: Market identifier

        Returns:
            Next refresh time, or None if no prior submission
        """
        last_sub = self._submissions.get(market_id)
        if last_sub is None:
            return None

        return last_sub + timedelta(seconds=self.config.refresh_interval_seconds)

    def clear_submissions(self) -> None:
        """Clear submission history."""
        self._submissions.clear()
