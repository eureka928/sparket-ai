"""Integration tests for originality and lead-lag job."""

import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from sparket.validator.scoring.jobs.originality_lead_lag import OriginalityLeadLagJob



@pytest.fixture
def mock_db():
    """Create mock database manager."""
    db = MagicMock()
    db.write = AsyncMock()
    db.read = AsyncMock(return_value=[])
    return db


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def job(mock_db, mock_logger):
    """Create job instance."""
    return OriginalityLeadLagJob(mock_db, mock_logger)


def make_miner_quote(
    market_id: int,
    side: str,
    submitted_at: datetime,
    imp_prob: float,
):
    """Create a mock miner quote row."""
    return {
        "market_id": market_id,
        "side": side,
        "submitted_at": submitted_at,
        "imp_prob": imp_prob,
    }


def make_truth_quote(
    market_id: int,
    side: str,
    ts: datetime,
    imp_prob: float,
):
    """Create a mock truth quote row."""
    return {
        "market_id": market_id,
        "side": side,
        "ts": ts,
        "imp_prob": imp_prob,
    }


class TestOriginalityLeadLagJobInit:
    """Tests for job initialization."""

    def test_job_id(self, job):
        """Should have correct JOB_ID."""
        assert job.JOB_ID == "originality_lead_lag_v1"

    def test_has_params(self, job):
        """Should load scoring params."""
        assert job.params is not None


class TestOriginalityLeadLagJobExecute:
    """Tests for execute method."""

    async def test_no_miners(self, mock_db, mock_logger):
        """Should handle no miners gracefully."""
        mock_db.read = AsyncMock(return_value=[])

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job.execute()

        mock_db.write.assert_not_called()

    async def test_processes_miners(self, mock_db, mock_logger):
        """Should process each miner."""
        now = datetime.now(timezone.utc)

        # Create quotes with enough variance
        miner_quotes = []
        truth_quotes = []
        for i in range(50):  # Need enough samples
            t = now - timedelta(hours=i)
            prob = 0.5 + 0.1 * (i % 3 - 1)  # Vary probability
            miner_quotes.append(make_miner_quote(1, "home", t, prob))
            truth_quotes.append(make_truth_quote(1, "home", t, prob * 0.95))

        mock_db.read = AsyncMock(
            side_effect=[
                # Miners
                [{"miner_id": 1, "miner_hotkey": "hotkey1"}],
                # Miner quotes
                miner_quotes,
                # Truth quotes
                truth_quotes,
            ]
        )

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job.execute()

        # Should have written results
        mock_db.write.assert_called()


class TestProcessMiner:
    """Tests for _process_miner method."""

    async def test_no_quotes(self, mock_db, mock_logger):
        """Should handle no miner quotes."""
        mock_db.read = AsyncMock(return_value=[])

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        mock_db.write.assert_not_called()

    async def test_insufficient_samples(self, mock_db, mock_logger):
        """Should skip miners with too few samples."""
        now = datetime.now(timezone.utc)

        mock_db.read = AsyncMock(
            return_value=[
                make_miner_quote(1, "home", now, 0.6),
                make_miner_quote(1, "home", now - timedelta(hours=1), 0.55),
            ]
        )

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=now - timedelta(days=30),
            window_end=now,
        )

        mock_db.write.assert_not_called()

    async def test_no_matching_truth(self, mock_db, mock_logger):
        """Should handle no matching truth quotes."""
        now = datetime.now(timezone.utc)

        miner_quotes = [
            make_miner_quote(1, "home", now - timedelta(hours=i), 0.5 + i * 0.01)
            for i in range(50)
        ]

        mock_db.read = AsyncMock(
            side_effect=[
                miner_quotes,  # Miner quotes
                [],  # No truth quotes
            ]
        )

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=now - timedelta(days=30),
            window_end=now,
        )

        mock_db.write.assert_not_called()

    async def test_computes_sos(self, mock_db, mock_logger):
        """Should compute SOS score."""
        now = datetime.now(timezone.utc)

        miner_quotes = []
        truth_quotes = []
        for i in range(50):
            t = now - timedelta(hours=i)
            # Correlated but not identical
            miner_prob = 0.5 + 0.1 * (i % 5 - 2) / 5
            truth_prob = 0.5 + 0.08 * (i % 5 - 2) / 5
            miner_quotes.append(make_miner_quote(1, "home", t, miner_prob))
            truth_quotes.append(make_truth_quote(1, "home", t, truth_prob))

        mock_db.read = AsyncMock(
            side_effect=[
                miner_quotes,
                truth_quotes,
            ]
        )

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=now - timedelta(days=30),
            window_end=now,
        )

        mock_db.write.assert_called()
        call_args = mock_db.write.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["sos_score"] is not None

    async def test_computes_lead_lag(self, mock_db, mock_logger):
        """Should compute lead-lag metrics."""
        now = datetime.now(timezone.utc)

        miner_quotes = []
        truth_quotes = []
        for i in range(50):
            t = now - timedelta(hours=i)
            # Miner moves before truth
            miner_prob = 0.5 + 0.1 * ((i + 1) % 5 - 2) / 5
            truth_prob = 0.5 + 0.1 * (i % 5 - 2) / 5
            miner_quotes.append(make_miner_quote(1, "home", t, miner_prob))
            truth_quotes.append(make_truth_quote(1, "home", t, truth_prob))

        mock_db.read = AsyncMock(
            side_effect=[
                miner_quotes,
                truth_quotes,
            ]
        )

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=now - timedelta(days=30),
            window_end=now,
        )

        mock_db.write.assert_called()

    async def test_multiple_markets(self, mock_db, mock_logger):
        """Should aggregate across markets."""
        now = datetime.now(timezone.utc)

        miner_quotes = []
        truth_quotes = []
        for market_id in [1, 2]:
            for i in range(30):
                t = now - timedelta(hours=i)
                prob = 0.5 + 0.05 * (i % 3 - 1)
                miner_quotes.append(make_miner_quote(market_id, "home", t, prob))
                truth_quotes.append(make_truth_quote(market_id, "home", t, prob * 0.95))

        mock_db.read = AsyncMock(
            side_effect=[
                miner_quotes,
                truth_quotes,
            ]
        )

        job = OriginalityLeadLagJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=now - timedelta(days=30),
            window_end=now,
        )

        mock_db.write.assert_called()

