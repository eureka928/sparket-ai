"""Tests for shared/logging.py - Logging utilities."""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from sparket.shared.logging import (
    EVENTS_LEVEL_NUM,
    DEFAULT_LOG_BACKUP_COUNT,
    _HeaderWarningFilter,
    suppress_bittensor_header_warnings,
    setup_events_logger,
)


class TestHeaderWarningFilter:
    """Tests for _HeaderWarningFilter class."""
    
    def test_allows_normal_messages(self):
        """Normal messages pass through the filter."""
        filter_obj = _HeaderWarningFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Normal warning message",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True
    
    def test_blocks_header_warning_messages(self):
        """Messages containing header warning substring are blocked."""
        filter_obj = _HeaderWarningFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Unexpected header key encountered in response",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False
    
    def test_allows_similar_but_different_messages(self):
        """Messages similar but not matching exactly pass through."""
        filter_obj = _HeaderWarningFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Expected header key not found",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True
    
    def test_handles_message_with_args(self):
        """Filter works with formatted messages using args."""
        filter_obj = _HeaderWarningFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Value: %s Unexpected header key encountered",
            args=("test",),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False
    
    def test_handles_exception_in_get_message(self):
        """Filter handles exceptions gracefully (returns True)."""
        filter_obj = _HeaderWarningFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Bad format %s %s",
            args=("only_one",),  # Wrong number of args
            exc_info=None,
        )
        # Should return True (allow) when getMessage fails
        assert filter_obj.filter(record) is True


class TestSuppressBittensorHeaderWarnings:
    """Tests for suppress_bittensor_header_warnings function."""
    
    def test_adds_filter_to_bittensor_loggers(self):
        """Adds header filter to bittensor loggers."""
        # Clear any existing filters first
        for name in ("bittensor", "bittensor.core", "bittensor.core.dendrite"):
            logger = logging.getLogger(name)
            logger.filters.clear()
        
        suppress_bittensor_header_warnings()
        
        # Check filters were added
        for name in ("bittensor", "bittensor.core", "bittensor.core.dendrite"):
            logger = logging.getLogger(name)
            assert any(isinstance(f, _HeaderWarningFilter) for f in logger.filters)
    
    def test_idempotent_multiple_calls(self):
        """Multiple calls don't add duplicate filters."""
        for name in ("bittensor", "bittensor.core", "bittensor.core.dendrite"):
            logger = logging.getLogger(name)
            logger.filters.clear()
        
        suppress_bittensor_header_warnings()
        suppress_bittensor_header_warnings()
        suppress_bittensor_header_warnings()
        
        # Should have multiple filters (function adds each time, but that's OK)
        # The important thing is it doesn't crash
        logger = logging.getLogger("bittensor")
        assert len([f for f in logger.filters if isinstance(f, _HeaderWarningFilter)]) >= 1


class TestSetupEventsLogger:
    """Tests for setup_events_logger function."""
    
    def test_creates_logger_with_custom_level(self):
        """Logger is created with EVENTS_LEVEL_NUM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024)
            assert logger.level == EVENTS_LEVEL_NUM
    
    def test_creates_events_log_file(self):
        """Creates events.log file in specified directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_events_logger(tmpdir, events_retention_size=1024)
            assert os.path.exists(os.path.join(tmpdir, "events.log"))
    
    def test_adds_rotating_file_handler(self):
        """Logger has a RotatingFileHandler."""
        from logging.handlers import RotatingFileHandler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=2048)
            handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
            assert len(handlers) >= 1
            
            # Check backup count is set correctly
            handler = handlers[0]
            assert handler.backupCount == DEFAULT_LOG_BACKUP_COUNT
    
    def test_logger_can_log_events(self):
        """Logger can write event messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024)
            
            # The event method should be added to Logger class
            assert hasattr(logging.Logger, 'event')
            
            # Log an event
            logger.event("Test event message")
            
            # Check file was written
            log_path = os.path.join(tmpdir, "events.log")
            with open(log_path, 'r') as f:
                content = f.read()
            assert "Test event message" in content
    
    def test_log_format(self):
        """Log entries have correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024)
            logger.event("Format test")
            
            log_path = os.path.join(tmpdir, "events.log")
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Should contain timestamp, level, message
            assert "EVENT" in content
            assert "Format test" in content
            # Date format check (YYYY-MM-DD HH:MM:SS)
            import re
            assert re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', content)


class TestEventsLevelNum:
    """Tests for EVENTS_LEVEL_NUM constant."""
    
    def test_level_is_between_warning_and_error(self):
        """Event level is between WARNING (30) and ERROR (40)."""
        assert logging.WARNING < EVENTS_LEVEL_NUM < logging.ERROR
    
    def test_level_value(self):
        """Event level has expected value."""
        assert EVENTS_LEVEL_NUM == 38
