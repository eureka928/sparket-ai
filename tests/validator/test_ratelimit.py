"""Tests for validator/utils/ratelimit.py - DDoS protection rate limiter."""

from __future__ import annotations

import time
from threading import Thread
from unittest.mock import patch

import pytest

from sparket.validator.utils.ratelimit import (
    RateLimitConfig,
    SlidingWindow,
    RateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""
    
    def test_default_values(self):
        """Config has sensible defaults."""
        config = RateLimitConfig()
        assert config.per_hotkey_per_second == 5
        assert config.per_hotkey_per_minute == 60
        assert config.global_per_second == 100
        assert config.global_per_minute == 2000
        assert config.cleanup_interval == 300
    
    def test_custom_values(self):
        """Config accepts custom values."""
        config = RateLimitConfig(
            per_hotkey_per_second=10,
            per_hotkey_per_minute=100,
            global_per_second=200,
            global_per_minute=5000,
            cleanup_interval=600,
        )
        assert config.per_hotkey_per_second == 10
        assert config.per_hotkey_per_minute == 100
        assert config.global_per_second == 200
        assert config.global_per_minute == 5000
        assert config.cleanup_interval == 600


class TestSlidingWindow:
    """Tests for SlidingWindow dataclass."""
    
    def test_empty_window(self):
        """Empty window has zero count."""
        window = SlidingWindow()
        assert window.count_in_window(1.0, time.time()) == 0
    
    def test_add_timestamp(self):
        """Adding timestamps increases count."""
        window = SlidingWindow()
        now = time.time()
        window.add(now)
        assert window.count_in_window(1.0, now) == 1
    
    def test_old_timestamps_filtered(self):
        """Old timestamps outside window are filtered."""
        window = SlidingWindow()
        now = time.time()
        window.add(now - 100)  # 100 seconds ago (outside 60s window)
        window.add(now - 0.5)  # 0.5 seconds ago
        window.add(now)        # now
        
        # Test large window first (before mutations from smaller windows)
        # 120 second window should include all 3
        assert window.count_in_window(120.0, now) == 3
        
        # 60 second window should include 2 (100s ago is outside)
        # Note: count_in_window mutates the list, removing old entries
        window2 = SlidingWindow()
        window2.add(now - 100)
        window2.add(now - 0.5)
        window2.add(now)
        assert window2.count_in_window(60.0, now) == 2
        
        # 1 second window should only include recent 2
        window3 = SlidingWindow()
        window3.add(now - 100)
        window3.add(now - 0.5)
        window3.add(now)
        assert window3.count_in_window(1.0, now) == 2
    
    def test_count_updates_window(self):
        """count_in_window removes old timestamps from list."""
        window = SlidingWindow()
        now = time.time()
        window.add(now - 100)  # Very old
        window.add(now - 50)   # Old
        window.add(now)
        
        assert len(window.timestamps) == 3
        window.count_in_window(10.0, now)
        assert len(window.timestamps) == 1  # Only recent one remains


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def setup_method(self):
        """Reset global rate limiter before each test."""
        reset_rate_limiter()
    
    def test_allows_first_request(self):
        """First request is always allowed."""
        limiter = RateLimiter()
        allowed, reason = limiter.check_and_record("hotkey_1")
        assert allowed is True
        assert reason is None
    
    def test_allows_within_limits(self):
        """Requests within limits are allowed."""
        config = RateLimitConfig(per_hotkey_per_second=5)
        limiter = RateLimiter(config)
        
        for _ in range(4):
            allowed, _ = limiter.check_and_record("hotkey_1")
            assert allowed is True
    
    def test_blocks_per_hotkey_per_second(self):
        """Blocks when per-hotkey per-second limit exceeded."""
        config = RateLimitConfig(per_hotkey_per_second=3)
        limiter = RateLimiter(config)
        
        for _ in range(3):
            allowed, _ = limiter.check_and_record("hotkey_1")
            assert allowed is True
        
        # 4th request should be blocked
        allowed, reason = limiter.check_and_record("hotkey_1")
        assert allowed is False
        assert reason == "hotkey_rate_limit_per_second"
    
    def test_blocks_per_hotkey_per_minute(self):
        """Blocks when per-hotkey per-minute limit exceeded."""
        config = RateLimitConfig(per_hotkey_per_second=100, per_hotkey_per_minute=5)
        limiter = RateLimiter(config)
        
        for _ in range(5):
            allowed, _ = limiter.check_and_record("hotkey_1")
            assert allowed is True
        
        allowed, reason = limiter.check_and_record("hotkey_1")
        assert allowed is False
        assert reason == "hotkey_rate_limit_per_minute"
    
    def test_blocks_global_per_second(self):
        """Blocks when global per-second limit exceeded."""
        config = RateLimitConfig(
            per_hotkey_per_second=100,
            per_hotkey_per_minute=1000,
            global_per_second=5,
        )
        limiter = RateLimiter(config)
        
        # Different hotkeys but same global limit
        for i in range(5):
            allowed, _ = limiter.check_and_record(f"hotkey_{i}")
            assert allowed is True
        
        allowed, reason = limiter.check_and_record("hotkey_new")
        assert allowed is False
        assert reason == "global_rate_limit_per_second"
    
    def test_blocks_global_per_minute(self):
        """Blocks when global per-minute limit exceeded."""
        config = RateLimitConfig(
            per_hotkey_per_second=100,
            per_hotkey_per_minute=1000,
            global_per_second=100,
            global_per_minute=5,
        )
        limiter = RateLimiter(config)
        
        for i in range(5):
            allowed, _ = limiter.check_and_record(f"hotkey_{i}")
            assert allowed is True
        
        allowed, reason = limiter.check_and_record("hotkey_new")
        assert allowed is False
        assert reason == "global_rate_limit_per_minute"
    
    def test_different_hotkeys_separate_limits(self):
        """Different hotkeys have separate per-hotkey limits."""
        config = RateLimitConfig(per_hotkey_per_second=2, global_per_second=100)
        limiter = RateLimiter(config)
        
        # 2 requests for hotkey_1
        limiter.check_and_record("hotkey_1")
        limiter.check_and_record("hotkey_1")
        
        # hotkey_1 should be blocked
        allowed, reason = limiter.check_and_record("hotkey_1")
        assert allowed is False
        
        # hotkey_2 should still be allowed
        allowed, reason = limiter.check_and_record("hotkey_2")
        assert allowed is True
    
    def test_limit_resets_after_window(self):
        """Limits reset after time window passes."""
        config = RateLimitConfig(per_hotkey_per_second=2)
        limiter = RateLimiter(config)
        
        limiter.check_and_record("hotkey_1")
        limiter.check_and_record("hotkey_1")
        
        allowed, _ = limiter.check_and_record("hotkey_1")
        assert allowed is False
        
        # Wait for second to pass
        time.sleep(1.1)
        
        allowed, _ = limiter.check_and_record("hotkey_1")
        assert allowed is True
    
    def test_cleanup_removes_stale_entries(self):
        """Cleanup removes stale hotkey entries."""
        config = RateLimitConfig(cleanup_interval=0)  # Immediate cleanup
        limiter = RateLimiter(config)
        
        # Add some requests
        limiter.check_and_record("hotkey_old")
        
        # Force cleanup by waiting and making another request
        time.sleep(0.1)
        
        # Manually trigger cleanup with old timestamps
        limiter._per_hotkey["hotkey_old"].timestamps = [time.time() - 300]
        limiter._cleanup(time.time())
        
        assert "hotkey_old" not in limiter._per_hotkey
    
    def test_get_stats(self):
        """get_stats returns correct statistics."""
        limiter = RateLimiter()
        
        limiter.check_and_record("hotkey_1")
        limiter.check_and_record("hotkey_2")
        limiter.check_and_record("hotkey_1")
        
        stats = limiter.get_stats()
        assert stats["tracked_hotkeys"] == 2
        assert stats["global_per_second"] == 3
        assert stats["global_per_minute"] == 3
    
    def test_thread_safety(self):
        """Rate limiter is thread-safe."""
        config = RateLimitConfig(
            per_hotkey_per_second=1000,
            per_hotkey_per_minute=10000,
            global_per_second=1000,
            global_per_minute=10000,
        )
        limiter = RateLimiter(config)
        results = []
        
        def make_requests():
            for i in range(100):
                allowed, _ = limiter.check_and_record(f"hotkey_{i % 10}")
                results.append(allowed)
        
        threads = [Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should be allowed (high limits)
        assert all(results)


class TestGlobalRateLimiter:
    """Tests for get_rate_limiter and reset_rate_limiter."""
    
    def setup_method(self):
        """Reset global rate limiter before each test."""
        reset_rate_limiter()
    
    def test_get_rate_limiter_creates_singleton(self):
        """get_rate_limiter creates and returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2
    
    def test_get_rate_limiter_with_config(self):
        """get_rate_limiter accepts initial config."""
        config = RateLimitConfig(per_hotkey_per_second=10)
        limiter = get_rate_limiter(config)
        assert limiter.config.per_hotkey_per_second == 10
    
    def test_reset_rate_limiter_clears_singleton(self):
        """reset_rate_limiter clears the singleton."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2
    
    def test_reset_clears_state(self):
        """Reset clears all tracked state."""
        limiter = get_rate_limiter()
        limiter.check_and_record("hotkey_1")
        
        reset_rate_limiter()
        
        limiter = get_rate_limiter()
        stats = limiter.get_stats()
        assert stats["tracked_hotkeys"] == 0
        assert stats["global_per_second"] == 0
