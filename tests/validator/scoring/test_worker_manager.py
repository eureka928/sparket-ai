"""Tests for scoring worker manager."""
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sparket.validator.scoring.worker.manager import ScoringWorkerManager



@pytest.fixture
def mock_config():
    """Create mock config."""
    return MagicMock()


@pytest.fixture
def mock_database():
    """Create mock database."""
    db = MagicMock()
    db.read = AsyncMock(return_value=[])
    db.write = AsyncMock()
    return db


@pytest.fixture
def manager(mock_config, mock_database):
    """Create manager instance."""
    return ScoringWorkerManager(mock_config, mock_database, worker_count=2)


class TestScoringWorkerManagerInit:
    """Tests for manager initialization."""

    def test_initial_state(self, manager):
        """Should have correct initial state."""
        assert len(manager.workers) == 2
        assert all(slot.process is None for slot in manager.workers)
        assert all(slot.restart_count == 0 for slot in manager.workers)
        assert not manager._shutdown_requested

    def test_worker_id_unique(self, mock_config, mock_database):
        """Worker IDs should be unique."""
        m1 = ScoringWorkerManager(mock_config, mock_database, worker_count=2)
        ids = [slot.worker_id for slot in m1.workers]
        assert len(set(ids)) == len(ids)
        assert all("worker_" in wid for wid in ids)


class TestStart:
    """Tests for start method."""

    def test_no_start_if_shutdown(self, manager):
        """Should not start if shutdown requested."""
        manager._shutdown_requested = True
        manager.start()
        assert all(slot.process is None for slot in manager.workers)

    def test_no_start_if_already_running(self, manager):
        """Should not start if already running."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        manager.workers[0].process = mock_process
        manager.workers[1].process = mock_process

        manager.start()

        # Process unchanged (not replaced)
        assert manager.workers[0].process is mock_process

    @patch('sparket.validator.scoring.worker.manager.mp.Process')
    def test_starts_process(self, mock_process_cls, manager):
        """Should start worker process."""
        mock_process_a = MagicMock()
        mock_process_a.pid = 12345
        mock_process_b = MagicMock()
        mock_process_b.pid = 54321
        mock_process_cls.side_effect = [mock_process_a, mock_process_b]

        manager.start()

        mock_process_a.start.assert_called_once()
        mock_process_b.start.assert_called_once()


class TestMonitor:
    """Tests for monitor method."""

    def test_no_monitor_if_shutdown(self, manager):
        """Should not monitor if shutdown."""
        manager._shutdown_requested = True
        manager.monitor()  # Should not raise

    def test_no_monitor_if_no_process(self, manager):
        """Should not monitor if no worker started."""
        manager.workers[0].process = None
        manager.monitor()  # Should not raise

    def test_handles_dead_worker(self, manager):
        """Should handle dead worker."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 1
        manager.workers[0].process = mock_process

        with patch.object(manager, '_start_worker'):
            with patch('time.sleep'):  # Skip delay
                manager.monitor()

        assert manager.workers[0].restart_count == 1

    def test_max_restarts(self, manager):
        """Should stop after max restarts."""
        manager.workers[0].restart_count = manager.max_restart_attempts

        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 1
        manager.workers[0].process = mock_process

        manager.monitor()

        # Should not try to restart
        assert manager.workers[0].restart_count == manager.max_restart_attempts


class TestCheckHeartbeat:
    """Tests for check_heartbeat method."""

    async def test_fresh_heartbeat(self, manager, mock_database):
        """Should return True for fresh heartbeat."""
        now = datetime.now(timezone.utc)
        manager.workers[0].process = MagicMock()
        mock_database.read = AsyncMock(
            return_value=[{"worker_id": manager.workers[0].worker_id, "last_heartbeat": now}]
        )

        result = await manager.check_heartbeat()
        assert result is True

    async def test_stale_heartbeat(self, manager, mock_database):
        """Should return False for stale heartbeat."""
        old = datetime.now(timezone.utc) - timedelta(seconds=120)
        manager.workers[0].process = MagicMock()
        mock_database.read = AsyncMock(
            return_value=[{"worker_id": manager.workers[0].worker_id, "last_heartbeat": old}]
        )

        result = await manager.check_heartbeat()
        assert result is False

    async def test_no_heartbeat(self, manager, mock_database):
        """Should return False if no heartbeat record."""
        manager.workers[0].process = MagicMock()
        mock_database.read = AsyncMock(return_value=[])

        result = await manager.check_heartbeat()
        assert result is False

    async def test_null_heartbeat(self, manager, mock_database):
        """Should return False if heartbeat is null."""
        manager.workers[0].process = MagicMock()
        mock_database.read = AsyncMock(
            return_value=[{"worker_id": manager.workers[0].worker_id, "last_heartbeat": None}]
        )

        result = await manager.check_heartbeat()
        assert result is False

    async def test_resets_restart_count(self, manager, mock_database):
        """Should reset restart count on fresh heartbeat."""
        manager.workers[0].restart_count = 5
        now = datetime.now(timezone.utc)
        manager.workers[0].process = MagicMock()
        mock_database.read = AsyncMock(
            return_value=[{"worker_id": manager.workers[0].worker_id, "last_heartbeat": now}]
        )

        await manager.check_heartbeat()
        assert manager.workers[0].restart_count == 0

    async def test_handles_error(self, manager, mock_database):
        """Should handle database errors."""
        mock_database.read = AsyncMock(side_effect=Exception("DB error"))

        result = await manager.check_heartbeat()
        assert result is False


class TestShutdown:
    """Tests for shutdown method."""

    def test_sets_shutdown_flag(self, manager):
        """Should set shutdown flag."""
        manager.shutdown()
        assert manager._shutdown_requested is True

    def test_no_op_if_no_process(self, manager):
        """Should not raise if no worker."""
        manager.workers[0].process = None
        manager.shutdown()  # Should not raise

    def test_no_op_if_already_stopped(self, manager):
        """Should handle already stopped worker."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        manager.workers[0].process = mock_process

        manager.shutdown()
        mock_process.terminate.assert_not_called()

    def test_terminates_process(self, manager):
        """Should terminate running worker."""
        mock_process = MagicMock()
        mock_process.is_alive.side_effect = [True, False]  # Alive, then dead
        mock_process.pid = 12345
        manager.workers[0].process = mock_process

        manager.shutdown()

        mock_process.terminate.assert_called_once()
        mock_process.join.assert_called()

    def test_kills_if_needed(self, manager):
        """Should kill if terminate fails."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True  # Never dies
        mock_process.pid = 12345
        manager.workers[0].process = mock_process

        manager.shutdown()

        mock_process.kill.assert_called_once()

    def test_handles_terminate_error(self, manager):
        """Should handle terminate error."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345
        mock_process.terminate.side_effect = Exception("Error")
        manager.workers[0].process = mock_process

        # Should not raise
        manager.shutdown()


class TestIsRunning:
    """Tests for is_running property."""

    def test_false_if_no_process(self, manager):
        """Should be False if no process."""
        manager.workers[0].process = None
        assert manager.is_running is False

    def test_false_if_dead(self, manager):
        """Should be False if process dead."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        manager.workers[0].process = mock_process

        assert manager.is_running is False

    def test_true_if_alive(self, manager):
        """Should be True if process alive."""
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        manager.workers[0].process = mock_process

        assert manager.is_running is True

