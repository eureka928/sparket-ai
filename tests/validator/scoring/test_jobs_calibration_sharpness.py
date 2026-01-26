"""Integration tests for calibration and sharpness job."""

import json
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sparket.validator.scoring.jobs.calibration_sharpness import (
    CalibrationSharpnessJob,
    SIDE_TO_INDEX,
)



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
    return CalibrationSharpnessJob(mock_db, mock_logger)


def make_prediction(
    submission_id: int,
    side: str = "home",
    imp_prob: float = 0.6,
    outcome_vector: list = None,
):
    """Create a mock prediction row."""
    if outcome_vector is None:
        outcome_vector = [1, 0]  # home wins
    return {
        "submission_id": submission_id,
        "side": side,
        "imp_prob": imp_prob,
        "outcome_vector": json.dumps(outcome_vector),
    }


class TestSideToIndex:
    """Tests for SIDE_TO_INDEX mapping."""

    def test_home(self):
        assert SIDE_TO_INDEX["home"] == 0

    def test_away(self):
        assert SIDE_TO_INDEX["away"] == 1

    def test_draw(self):
        assert SIDE_TO_INDEX["draw"] == 2

    def test_over(self):
        assert SIDE_TO_INDEX["over"] == 0

    def test_under(self):
        assert SIDE_TO_INDEX["under"] == 1


class TestCalibrationSharpnessJobInit:
    """Tests for job initialization."""

    def test_job_id(self, job):
        """Should have correct JOB_ID."""
        assert job.JOB_ID == "calibration_sharpness_v1"

    def test_has_params(self, job):
        """Should load scoring params."""
        assert job.params is not None


class TestCalibrationSharpnessJobExecute:
    """Tests for execute method."""

    async def test_no_miners(self, mock_db, mock_logger):
        """Should handle no miners gracefully."""
        mock_db.read = AsyncMock(return_value=[])

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job.execute()

        # Only one read call (for miners)
        assert mock_db.read.call_count == 1
        mock_db.write.assert_not_called()

    async def test_processes_miners(self, mock_db, mock_logger):
        """Should process each miner with outcomes."""
        # First call returns miners, second returns predictions
        mock_db.read = AsyncMock(
            side_effect=[
                # Miners
                [
                    {"miner_id": 1, "miner_hotkey": "hotkey1"},
                    {"miner_id": 2, "miner_hotkey": "hotkey2"},
                ],
                # Predictions for miner 1
                [
                    make_prediction(1, "home", 0.6, [1, 0]),
                    make_prediction(2, "home", 0.7, [1, 0]),
                    make_prediction(3, "home", 0.55, [0, 1]),
                ],
                # Predictions for miner 2
                [
                    make_prediction(4, "away", 0.4, [0, 1]),
                ],
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job.execute()

        # Should have written 2 results
        assert mock_db.write.call_count == 2

    async def test_handles_no_predictions(self, mock_db, mock_logger):
        """Should handle miners with no predictions."""
        mock_db.read = AsyncMock(
            side_effect=[
                [{"miner_id": 1, "miner_hotkey": "hotkey1"}],
                [],  # No predictions
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job.execute()

        # Should not write (no scores)
        mock_db.write.assert_not_called()


class TestProcessMiner:
    """Tests for _process_miner method."""

    async def test_computes_calibration(self, mock_db, mock_logger):
        """Should compute calibration score."""
        mock_db.read = AsyncMock(
            return_value=[
                make_prediction(1, "home", 0.6, [1, 0]),
                make_prediction(2, "home", 0.6, [1, 0]),
                make_prediction(3, "home", 0.6, [0, 1]),
                make_prediction(4, "home", 0.7, [1, 0]),
                make_prediction(5, "home", 0.7, [1, 0]),
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        # Should write calibration/sharpness
        mock_db.write.assert_called()

    async def test_computes_sharpness(self, mock_db, mock_logger):
        """Should compute sharpness score."""
        mock_db.read = AsyncMock(
            return_value=[
                make_prediction(1, "home", 0.9, [1, 0]),  # Sharp
                make_prediction(2, "home", 0.85, [1, 0]),  # Sharp
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        mock_db.write.assert_called()

    async def test_handles_null_outcome(self, mock_db, mock_logger):
        """Should skip predictions with null outcome."""
        mock_db.read = AsyncMock(
            return_value=[
                {
                    "submission_id": 1,
                    "side": "home",
                    "imp_prob": 0.6,
                    "outcome_vector": None,  # Not settled
                },
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        # No valid predictions, so no write
        mock_db.write.assert_not_called()

    async def test_handles_dict_outcome(self, mock_db, mock_logger):
        """Should handle outcome_vector as dict (not JSON string)."""
        mock_db.read = AsyncMock(
            return_value=[
                {
                    "submission_id": 1,
                    "side": "home",
                    "imp_prob": 0.7,
                    "outcome_vector": [1, 0],  # Already a list, not JSON string
                },
                {
                    "submission_id": 2,
                    "side": "home",
                    "imp_prob": 0.7,
                    "outcome_vector": [1, 0],
                },
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        mock_db.write.assert_called()

    async def test_handles_unknown_side(self, mock_db, mock_logger):
        """Should skip predictions with unknown side."""
        mock_db.read = AsyncMock(
            return_value=[
                make_prediction(1, "unknown_side", 0.6, [1, 0]),
            ]
        )

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        # No valid predictions, so no write
        mock_db.write.assert_not_called()

    async def test_handles_empty_predictions(self, mock_db, mock_logger):
        """Should handle no predictions."""
        mock_db.read = AsyncMock(return_value=[])

        job = CalibrationSharpnessJob(mock_db, mock_logger)
        await job._process_miner(
            miner_id=1,
            miner_hotkey="test",
            window_start=datetime.now(timezone.utc) - timedelta(days=30),
            window_end=datetime.now(timezone.utc),
        )

        mock_db.write.assert_not_called()

