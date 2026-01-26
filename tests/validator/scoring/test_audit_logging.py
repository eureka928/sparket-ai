"""Tests for audit logging utilities."""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sparket.validator.scoring.audit.logging import (
    ScoringAuditLogger,
    get_audit_logger,
)


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def audit_logger(mock_logger):
    """Create audit logger with mock."""
    return ScoringAuditLogger(logger=mock_logger)


class TestScoringAuditLogger:
    """Tests for ScoringAuditLogger class."""

    def test_default_logger(self):
        """Should create default logger if not provided."""
        logger = ScoringAuditLogger()
        assert logger.logger is not None
        assert logger.logger.name == "scoring.audit"

    def test_custom_logger(self, mock_logger):
        """Should use provided logger."""
        logger = ScoringAuditLogger(logger=mock_logger)
        assert logger.logger is mock_logger


class TestLogJobStart:
    """Tests for log_job_start method."""

    def test_logs_info(self, audit_logger, mock_logger):
        """Should log at INFO level."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

        audit_logger.log_job_start(
            job_id="job_123",
            window_start=dt,
            window_end=dt,
            params_hash="a" * 64,
        )

        mock_logger.info.assert_called_once()

    def test_log_structure(self, audit_logger, mock_logger):
        """Should log structured data."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

        audit_logger.log_job_start(
            job_id="job_123",
            window_start=dt,
            window_end=dt,
            params_hash="a" * 64,
        )

        call_args = mock_logger.info.call_args[0][0]
        assert call_args["event"] == "job_start"
        assert call_args["job_id"] == "job_123"
        assert "timestamp" in call_args


class TestLogJobComplete:
    """Tests for log_job_complete method."""

    def test_logs_info(self, audit_logger, mock_logger):
        """Should log at INFO level."""
        audit_logger.log_job_complete(
            job_id="job_123",
            items_processed=100,
            output_hash="b" * 64,
            duration_seconds=5.5,
        )

        mock_logger.info.assert_called_once()

    def test_log_structure(self, audit_logger, mock_logger):
        """Should log structured data."""
        audit_logger.log_job_complete(
            job_id="job_123",
            items_processed=100,
            output_hash="b" * 64,
            duration_seconds=5.5,
        )

        call_args = mock_logger.info.call_args[0][0]
        assert call_args["event"] == "job_complete"
        assert call_args["items_processed"] == 100
        assert call_args["duration_seconds"] == 5.5


class TestLogMinerScore:
    """Tests for log_miner_score method."""

    def test_logs_debug(self, audit_logger, mock_logger):
        """Should log at DEBUG level."""
        audit_logger.log_miner_score(
            miner_id=123,
            miner_hotkey="abc123def456" * 4,
            scores={"skill_score": 0.75},
            score_hash="c" * 64,
        )

        mock_logger.debug.assert_called_once()

    def test_truncates_hotkey(self, audit_logger, mock_logger):
        """Should truncate hotkey for privacy."""
        long_hotkey = "a" * 64

        audit_logger.log_miner_score(
            miner_id=123,
            miner_hotkey=long_hotkey,
            scores={"skill_score": 0.75},
            score_hash="c" * 64,
        )

        call_args = mock_logger.debug.call_args[0][0]
        assert len(call_args["miner_hotkey"]) < len(long_hotkey)
        assert call_args["miner_hotkey"].endswith("...")


class TestLogBiasUpdate:
    """Tests for log_bias_update method."""

    def test_logs_info(self, audit_logger, mock_logger):
        """Should log at INFO level."""
        audit_logger.log_bias_update(
            n_entries=50,
            version=3,
            bias_hash="d" * 64,
        )

        mock_logger.info.assert_called_once()

    def test_log_structure(self, audit_logger, mock_logger):
        """Should log structured data."""
        audit_logger.log_bias_update(
            n_entries=50,
            version=3,
            bias_hash="d" * 64,
        )

        call_args = mock_logger.info.call_args[0][0]
        assert call_args["event"] == "bias_update"
        assert call_args["n_entries"] == 50
        assert call_args["version"] == 3


class TestLogGroundTruthSnapshot:
    """Tests for log_ground_truth_snapshot method."""

    def test_logs_info(self, audit_logger, mock_logger):
        """Should log at INFO level."""
        audit_logger.log_ground_truth_snapshot(
            n_markets=20,
            n_sides=40,
            snapshot_hash="e" * 64,
        )

        mock_logger.info.assert_called_once()

    def test_log_structure(self, audit_logger, mock_logger):
        """Should log structured data."""
        audit_logger.log_ground_truth_snapshot(
            n_markets=20,
            n_sides=40,
            snapshot_hash="e" * 64,
        )

        call_args = mock_logger.info.call_args[0][0]
        assert call_args["event"] == "ground_truth_snapshot"
        assert call_args["n_markets"] == 20
        assert call_args["n_sides"] == 40


class TestLogConsensusCheck:
    """Tests for log_consensus_check method."""

    def test_logs_info(self, audit_logger, mock_logger):
        """Should log at INFO level."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

        audit_logger.log_consensus_check(
            as_of=dt,
            n_miners=100,
            batch_hash="f" * 64,
        )

        mock_logger.info.assert_called_once()

    def test_logs_full_hash(self, audit_logger, mock_logger):
        """Should log full hash for consensus comparison."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        full_hash = "f" * 64

        audit_logger.log_consensus_check(
            as_of=dt,
            n_miners=100,
            batch_hash=full_hash,
        )

        call_args = mock_logger.info.call_args[0][0]
        # Full hash should be logged (not truncated)
        assert call_args["batch_hash"] == full_hash


class TestGetAuditLogger:
    """Tests for get_audit_logger singleton."""

    def test_returns_logger(self):
        """Should return ScoringAuditLogger instance."""
        logger = get_audit_logger()
        assert isinstance(logger, ScoringAuditLogger)

    def test_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2

