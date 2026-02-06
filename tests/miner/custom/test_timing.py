"""Tests for timing strategy."""

import pytest
from datetime import datetime, timedelta, timezone

from sparket.miner.custom.strategy.timing import (
    TimingStrategy,
    TimingConfig,
    SubmissionDecision,
)


class TestTimingStrategy:
    """Tests for TimingStrategy."""

    def test_first_submission_always_allowed(self):
        """First submission to a market should always be allowed."""
        strategy = TimingStrategy()
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(days=3)

        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=now,
        )

        assert decision.decision == SubmissionDecision.SUBMIT_NOW
        assert "Initial" in decision.reason

    def test_full_credit_for_early_submission(self):
        """7+ days out should get full time credit."""
        strategy = TimingStrategy()
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(days=8)

        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=now,
        )

        assert decision.time_credit == 1.0

    def test_reduced_credit_for_late_submission(self):
        """Same-day submissions should get reduced credit (log curve)."""
        strategy = TimingStrategy()
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(hours=12)

        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=now,
        )

        # Log curve gives ~0.54 at 12h (between floor 0.1 and full 1.0)
        assert decision.time_credit < 0.7
        assert decision.time_credit > 0.1

    def test_skip_when_too_late(self):
        """Should skip when past cutoff."""
        config = TimingConfig(cutoff_hours=0.5)
        strategy = TimingStrategy(config=config)
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(minutes=20)

        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=now,
        )

        assert decision.decision == SubmissionDecision.SKIP
        assert "Too late" in decision.reason

    def test_skip_penalty_zone(self):
        """Should skip when in penalty zone."""
        config = TimingConfig(min_hours_before_event=2.0, cutoff_hours=0.5)
        strategy = TimingStrategy(config=config)
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(hours=1.5)

        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=now,
        )

        assert decision.decision == SubmissionDecision.SKIP
        assert "penalty zone" in decision.reason

    def test_wait_before_refresh_interval(self):
        """Should wait if refresh interval hasn't passed."""
        config = TimingConfig(refresh_interval_seconds=3600)
        strategy = TimingStrategy(config=config)
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(days=3)

        # First submission
        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=now,
        )
        strategy.record_submission(123, now)

        # 30 minutes later
        later = now + timedelta(minutes=30)
        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=later,
        )

        assert decision.decision == SubmissionDecision.WAIT
        assert decision.next_check_seconds > 0

    def test_update_after_refresh_interval(self):
        """Should allow update after refresh interval."""
        config = TimingConfig(refresh_interval_seconds=3600)
        strategy = TimingStrategy(config=config)
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(days=3)

        # First submission
        strategy.record_submission(123, now)

        # 2 hours later
        later = now + timedelta(hours=2)
        decision = strategy.evaluate(
            market_id=123,
            event_start=event_start,
            now=later,
        )

        assert decision.decision == SubmissionDecision.UPDATE

    def test_get_submission_schedule(self):
        """Should return prioritized schedule of markets to submit."""
        strategy = TimingStrategy()
        now = datetime.now(timezone.utc)

        markets = [
            {
                "market_id": 1,
                "start_time_utc": (now + timedelta(hours=3)).isoformat(),
            },
            {
                "market_id": 2,
                "start_time_utc": (now + timedelta(days=5)).isoformat(),
            },
            {
                "market_id": 3,
                "start_time_utc": (now + timedelta(minutes=10)).isoformat(),
            },
        ]

        schedule = strategy.get_submission_schedule(markets, now=now)

        # Should include market 1 and 2, but not 3 (too late)
        market_ids = [m["market_id"] for m, _ in schedule]
        assert 1 in market_ids
        assert 2 in market_ids
        # Market 3 might be skipped depending on penalty zone config

    def test_priority_increases_as_event_approaches(self):
        """Markets closer to event should have higher priority."""
        strategy = TimingStrategy()
        now = datetime.now(timezone.utc)

        # Event far away
        decision_far = strategy.evaluate(
            market_id=1,
            event_start=now + timedelta(days=10),
            now=now,
        )

        # Event soon
        decision_soon = strategy.evaluate(
            market_id=2,
            event_start=now + timedelta(hours=8),
            now=now,
        )

        assert decision_soon.priority > decision_far.priority
