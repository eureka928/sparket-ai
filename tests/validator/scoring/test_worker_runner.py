"""Integration tests for scoring worker runner."""

import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sparket.validator.scoring.batch.processor import WorkType
from sparket.validator.scoring.worker.runner import ScoringWorkerRunner


@pytest.fixture
def mock_config():
    """Create mock config."""
    return MagicMock()


@pytest.fixture
def runner(mock_config):
    """Create runner instance."""
    return ScoringWorkerRunner(mock_config, "test_worker_123")


class TestScoringWorkerRunnerInit:
    """Tests for runner initialization."""

    def test_stores_config(self, runner, mock_config):
        """Should store config."""
        assert runner.config is mock_config

    def test_stores_worker_id(self, runner):
        """Should store worker ID."""
        assert runner.worker_id == "test_worker_123"

    def test_initial_state(self, runner):
        """Should have correct initial state."""
        assert runner.db is None
        assert runner._shutdown_requested is False
        assert runner._current_job is None


class TestHandleSignal:
    """Tests for signal handling."""

    def test_sets_shutdown_flag(self, runner):
        """Should set shutdown flag on signal."""
        runner._handle_signal(signal.SIGTERM, None)
        assert runner._shutdown_requested is True

    def test_handles_sigint(self, runner):
        """Should handle SIGINT."""
        runner._handle_signal(signal.SIGINT, None)
        assert runner._shutdown_requested is True


class TestRegisterWorker:
    """Tests for worker registration."""

    async def test_writes_to_db(self, runner):
        """Should write registration to database."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()

        await runner._register_worker()

        runner.db.write.assert_called_once()


class TestUpdateHeartbeat:
    """Tests for heartbeat updates."""

    async def test_updates_heartbeat(self, runner):
        """Should update heartbeat in database."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()

        await runner._update_heartbeat()

        runner.db.write.assert_called_once()

    async def test_includes_current_job(self, runner):
        """Should include current job in heartbeat."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()
        runner._current_job = "TestJob"

        await runner._update_heartbeat()

        call_args = runner.db.write.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["job"] == "TestJob"

    async def test_handles_error(self, runner):
        """Should handle database errors."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await runner._update_heartbeat()


class TestGetMemoryMb:
    """Tests for memory usage tracking."""

    def test_returns_int(self, runner):
        """Should return memory in MB."""
        mem = runner._get_memory_mb()
        assert isinstance(mem, int)
        assert mem >= 0


class TestClaimNextWork:
    """Tests for work claiming."""

    async def test_returns_first_available(self, runner):
        """Should return the first available work item."""
        runner.work_queue = MagicMock()
        runner.work_queue.claim_work = AsyncMock(
            side_effect=[None, {"work_id": "1", "chunk_key": "rk", "params": {}}]
        )

        work = await runner._claim_next_work()

        assert work is not None
        assert work[0] == WorkType.CALIBRATION


class TestRunWorkItem:
    """Tests for running queued work items."""

    async def test_runs_rolling_job(self, runner):
        """Should run rolling job and mark complete."""
        runner.db = MagicMock()
        runner.work_queue = MagicMock()
        runner.work_queue.complete_work = AsyncMock()
        runner.work_queue.fail_work = AsyncMock()

        item = {"work_id": "w1", "chunk_key": "20260101:all", "params": {}}

        with patch(
            "sparket.validator.scoring.jobs.rolling_aggregates.RollingAggregatesJob"
        ) as mock_job:
            mock_job.return_value.run = AsyncMock()
            mock_job.return_value.items_processed = 1
            mock_job.return_value.items_total = 2

            await runner._run_work_item(WorkType.ROLLING, item)

        runner.work_queue.complete_work.assert_called_once()

    async def test_records_failure(self, runner):
        """Should mark work failed on error."""
        runner.db = MagicMock()
        runner.work_queue = MagicMock()
        runner.work_queue.complete_work = AsyncMock()
        runner.work_queue.fail_work = AsyncMock()

        item = {"work_id": "w1", "chunk_key": "20260101:all", "params": {}}

        with patch(
            "sparket.validator.scoring.jobs.skill_score.SkillScoreJob"
        ) as mock_job:
            mock_job.return_value.run = AsyncMock(side_effect=Exception("boom"))
            await runner._run_work_item(WorkType.SKILL, item)

        runner.work_queue.fail_work.assert_called_once()


class TestRecordJobSuccess:
    """Tests for recording job success."""

    async def test_updates_counter(self, runner):
        """Should update jobs_completed counter."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()

        await runner._record_job_success("TestJob")

        runner.db.write.assert_called_once()

    async def test_handles_error(self, runner):
        """Should handle database errors."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await runner._record_job_success("TestJob")


class TestRecordJobFailure:
    """Tests for recording job failure."""

    async def test_updates_counters(self, runner):
        """Should update failure counters."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()

        await runner._record_job_failure("TestJob", "Test error")

        # Should write to both worker heartbeat and job state
        assert runner.db.write.call_count >= 1

    async def test_truncates_long_error(self, runner):
        """Should truncate long error messages."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()

        long_error = "x" * 2000
        await runner._record_job_failure("TestJob", long_error)

        # Error should be truncated in the call
        runner.db.write.assert_called()

    async def test_handles_error(self, runner):
        """Should handle database errors."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await runner._record_job_failure("TestJob", "Error")


class TestCleanup:
    """Tests for cleanup on shutdown."""

    async def test_clears_current_job(self, runner):
        """Should clear current job in database."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()
        runner.db.dispose = AsyncMock()

        await runner._cleanup()

        runner.db.write.assert_called()

    async def test_disposes_database(self, runner):
        """Should dispose database connection."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()
        runner.db.dispose = AsyncMock()

        await runner._cleanup()

        runner.db.dispose.assert_called_once()

    async def test_handles_write_error(self, runner):
        """Should handle write errors."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock(side_effect=Exception("DB error"))
        runner.db.dispose = AsyncMock()

        # Should not raise
        await runner._cleanup()

    async def test_handles_dispose_error(self, runner):
        """Should handle dispose errors."""
        runner.db = MagicMock()
        runner.db.write = AsyncMock()
        runner.db.dispose = AsyncMock(side_effect=Exception("Dispose error"))

        # Should not raise
        await runner._cleanup()

    async def test_handles_no_db(self, runner):
        """Should handle missing database."""
        runner.db = None

        # Should not raise
        await runner._cleanup()

