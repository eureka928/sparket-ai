"""Timing strategy for maximizing time credit in scoring.

The scoring system applies time bonuses/penalties:
- 7+ days before event: 100% credit
- 1 hour before: 10% credit (floor)
- Early bad predictions: 70% penalty (forgiven)
- Late bad predictions: 100% penalty

Strategy:
1. Submit as early as possible (7+ days)
2. Update predictions every 6 hours
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

        if last_sub is None:
            # First submission - always do it
            priority = self._calculate_priority(hours_until, time_credit)
            return TimingDecision(
                decision=SubmissionDecision.SUBMIT_NOW,
                reason="Initial submission",
                time_credit=time_credit,
                next_check_seconds=self.config.refresh_interval_seconds,
                priority=priority,
            )

        # Check if enough time has passed since last submission
        since_last = now - last_sub
        refresh_delta = timedelta(seconds=self.config.refresh_interval_seconds)

        if since_last >= refresh_delta:
            priority = self._calculate_priority(hours_until, time_credit)
            return TimingDecision(
                decision=SubmissionDecision.UPDATE,
                reason=f"Refresh interval ({since_last.total_seconds() / 3600:.1f}h since last)",
                time_credit=time_credit,
                next_check_seconds=self.config.refresh_interval_seconds,
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

    def _calculate_time_credit(self, hours_until: float) -> float:
        """Calculate expected time credit based on hours until event.

        Time credit follows a decay curve:
        - 7+ days (168h): 100%
        - 1 day (24h): ~60%
        - 12h: ~40%
        - 2h: ~15%
        - 1h: 10% (floor)
        """
        if hours_until <= 0:
            return 0.0

        # Full credit threshold (7 days)
        full_credit_hours = self.config.early_submission_days * 24

        if hours_until >= full_credit_hours:
            return 1.0

        # Floor at 1 hour
        if hours_until <= 1:
            return 0.1

        # Linear interpolation between floor and full credit
        # Using a slightly concave curve to reward earlier submissions
        ratio = hours_until / full_credit_hours
        # Curve: credit = 0.1 + 0.9 * ratio^0.7
        credit = 0.1 + 0.9 * (ratio ** 0.7)

        return min(1.0, max(0.1, credit))

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
