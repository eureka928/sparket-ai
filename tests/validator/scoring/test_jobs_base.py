"""Tests for scoring job base class."""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from sparket.validator.scoring.jobs.base import ScoringJob


class ConcreteJob(ScoringJob):
    """Concrete implementation for testing."""

    JOB_ID = "test_job"
    CHECKPOINT_INTERVAL = 5

    def __init__(self, db, logger, execute_fn=None, job_id_override=None):
        super().__init__(db, logger, job_id_override=job_id_override)
        self._execute_fn = execute_fn

    async def execute(self):
        self.items_total = 10
        for i in range(10):
            if self._execute_fn:
                result = self._execute_fn()
                if hasattr(result, '__await__'):
                    await result
            self.items_processed += 1
            await self.checkpoint_if_due()


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


class TestScoringJobInit:
    """Tests for ScoringJob initialization."""

    def test_requires_job_id(self, mock_db, mock_logger):
        """Should raise if JOB_ID not set."""
        class BadJob(ScoringJob):
            JOB_ID = ""

            async def execute(self):
                pass

        with pytest.raises(ValueError, match="JOB_ID must be set"):
            BadJob(mock_db, mock_logger)

    def test_valid_init(self, mock_db, mock_logger):
        """Should initialize with valid JOB_ID."""
        job = ConcreteJob(mock_db, mock_logger)
        assert job.JOB_ID == "test_job"
        assert job.job_id == "test_job"
        assert job.items_processed == 0
        assert job.state == {}

    def test_job_id_override(self, mock_db, mock_logger):
        """Should allow job_id override."""
        job = ConcreteJob(mock_db, mock_logger, job_id_override="override_job")
        assert job.job_id == "override_job"


class TestScoringJobRun:
    """Tests for run method."""

    @pytest.mark.asyncio
    async def test_run_executes_job(self, mock_db, mock_logger):
        """Run should call execute method."""
        execute_called = False

        async def track_execute():
            nonlocal execute_called
            execute_called = True

        job = ConcreteJob(mock_db, mock_logger, track_execute)
        await job.run()

        assert execute_called

    @pytest.mark.asyncio
    async def test_run_updates_status_running(self, mock_db, mock_logger):
        """Run should mark job as running."""
        job = ConcreteJob(mock_db, mock_logger)
        await job.run()

        # Check write was called with running status
        calls = mock_db.write.call_args_list
        assert any("running" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_run_updates_status_completed(self, mock_db, mock_logger):
        """Run should mark job as completed."""
        job = ConcreteJob(mock_db, mock_logger)
        await job.run()

        # Check write was called with completed status
        calls = mock_db.write.call_args_list
        assert any("completed" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_run_handles_failure(self, mock_db, mock_logger):
        """Run should mark job as failed on exception."""
        async def fail():
            raise ValueError("Test error")

        job = ConcreteJob(mock_db, mock_logger, fail)

        with pytest.raises(ValueError):
            await job.run()

        # Check write was called with failed status
        calls = mock_db.write.call_args_list
        assert any("failed" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_run_logs_completion(self, mock_db, mock_logger):
        """Run should log completion info."""
        job = ConcreteJob(mock_db, mock_logger)
        await job.run()

        # Check info was called with completion message
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("completed" in c for c in info_calls)


class TestCheckpoint:
    """Tests for checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_saves_state(self, mock_db, mock_logger):
        """Checkpoint should save current state."""
        job = ConcreteJob(mock_db, mock_logger)
        job.state = {"key": "value"}
        job.items_processed = 50
        job.items_total = 100
        job._started_at = datetime.now(timezone.utc)

        await job.checkpoint()

        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_checkpoint_handles_error(self, mock_db, mock_logger):
        """Checkpoint should handle database errors gracefully."""
        mock_db.write = AsyncMock(side_effect=Exception("DB error"))

        job = ConcreteJob(mock_db, mock_logger)
        job._started_at = datetime.now(timezone.utc)

        # Should not raise
        await job.checkpoint()

        mock_logger.warning.assert_called()


class TestCheckpointIfDue:
    """Tests for checkpoint_if_due method."""

    @pytest.mark.asyncio
    async def test_checkpoints_at_interval(self, mock_db, mock_logger):
        """Should checkpoint at CHECKPOINT_INTERVAL."""
        job = ConcreteJob(mock_db, mock_logger)
        job.CHECKPOINT_INTERVAL = 5
        job._started_at = datetime.now(timezone.utc)

        job.items_processed = 5
        await job.checkpoint_if_due()

        # Should have called checkpoint
        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_no_checkpoint_before_interval(self, mock_db, mock_logger):
        """Should not checkpoint before interval."""
        job = ConcreteJob(mock_db, mock_logger)
        job.CHECKPOINT_INTERVAL = 5
        job._started_at = datetime.now(timezone.utc)

        job.items_processed = 3
        await job.checkpoint_if_due()

        # Should not have called checkpoint
        mock_db.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_checkpoint_at_zero(self, mock_db, mock_logger):
        """Should not checkpoint at zero items."""
        job = ConcreteJob(mock_db, mock_logger)
        job.items_processed = 0

        await job.checkpoint_if_due()

        mock_db.write.assert_not_called()


class TestLoadCheckpoint:
    """Tests for checkpoint restoration."""

    @pytest.mark.asyncio
    async def test_load_restores_state(self, mock_db, mock_logger):
        """Should restore state from checkpoint."""
        mock_db.read = AsyncMock(
            return_value=[
                {
                    "checkpoint_data": json.dumps({"key": "value"}),
                    "items_processed": 50,
                    "started_at": datetime.now(timezone.utc),
                }
            ]
        )

        job = ConcreteJob(mock_db, mock_logger)
        await job._load_checkpoint()

        assert job.state == {"key": "value"}
        assert job.items_processed == 50

    @pytest.mark.asyncio
    async def test_load_handles_no_checkpoint(self, mock_db, mock_logger):
        """Should handle missing checkpoint gracefully."""
        mock_db.read = AsyncMock(return_value=[])

        job = ConcreteJob(mock_db, mock_logger)
        await job._load_checkpoint()

        assert job.state == {}
        assert job.items_processed == 0

    @pytest.mark.asyncio
    async def test_load_handles_dict_checkpoint(self, mock_db, mock_logger):
        """Should handle dict checkpoint data (not JSON string)."""
        mock_db.read = AsyncMock(
            return_value=[
                {
                    "checkpoint_data": {"key": "value"},  # Already a dict
                    "items_processed": 25,
                    "started_at": None,
                }
            ]
        )

        job = ConcreteJob(mock_db, mock_logger)
        await job._load_checkpoint()

        assert job.state == {"key": "value"}

    @pytest.mark.asyncio
    async def test_load_handles_error(self, mock_db, mock_logger):
        """Should handle database error gracefully."""
        mock_db.read = AsyncMock(side_effect=Exception("DB error"))

        job = ConcreteJob(mock_db, mock_logger)
        await job._load_checkpoint()

        # Should not raise, state should be empty
        assert job.state == {}
        mock_logger.warning.assert_called()


class TestUpdateStatus:
    """Tests for status update method."""

    @pytest.mark.asyncio
    async def test_update_running(self, mock_db, mock_logger):
        """Should update to running status."""
        job = ConcreteJob(mock_db, mock_logger)
        await job._update_status("running")

        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_update_completed(self, mock_db, mock_logger):
        """Should update to completed status."""
        job = ConcreteJob(mock_db, mock_logger)
        await job._update_status("completed")

        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_update_failed_with_error(self, mock_db, mock_logger):
        """Should update to failed with error message."""
        job = ConcreteJob(mock_db, mock_logger)
        await job._update_status("failed", error="Test error message")

        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_update_truncates_long_error(self, mock_db, mock_logger):
        """Should truncate very long error messages."""
        job = ConcreteJob(mock_db, mock_logger)
        long_error = "x" * 2000  # Very long error

        await job._update_status("failed", error=long_error)

        # Should have been called (error gets truncated to 1000 chars internally)
        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_update_handles_error(self, mock_db, mock_logger):
        """Should handle database error gracefully."""
        mock_db.write = AsyncMock(side_effect=Exception("DB error"))

        job = ConcreteJob(mock_db, mock_logger)
        await job._update_status("running")

        mock_logger.warning.assert_called()


class TestClearCheckpoint:
    """Tests for checkpoint clearing."""

    @pytest.mark.asyncio
    async def test_clears_checkpoint(self, mock_db, mock_logger):
        """Should clear checkpoint data."""
        job = ConcreteJob(mock_db, mock_logger)
        await job._clear_checkpoint()

        mock_db.write.assert_called()

    @pytest.mark.asyncio
    async def test_handles_error(self, mock_db, mock_logger):
        """Should handle database error gracefully."""
        mock_db.write = AsyncMock(side_effect=Exception("DB error"))

        job = ConcreteJob(mock_db, mock_logger)
        await job._clear_checkpoint()

        mock_logger.warning.assert_called()

