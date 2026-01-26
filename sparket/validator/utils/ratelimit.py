"""Server-level rate limiting for DDoS protection.

Implements a sliding window rate limiter with:
- Per-hotkey limits (individual miner throttling)
- Global limits (total request throttling)
- Configurable windows and thresholds
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Tuple


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    # Per-hotkey limits
    per_hotkey_per_second: int = 5
    per_hotkey_per_minute: int = 60
    
    # Global limits (all miners combined)
    global_per_second: int = 100
    global_per_minute: int = 2000
    
    # Cleanup interval (seconds) - remove stale entries
    cleanup_interval: int = 300


@dataclass
class SlidingWindow:
    """Sliding window counter for rate limiting."""
    timestamps: list = field(default_factory=list)
    
    def add(self, ts: float) -> None:
        self.timestamps.append(ts)
    
    def count_in_window(self, window_seconds: float, now: float) -> int:
        cutoff = now - window_seconds
        # Remove old timestamps
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        return len(self.timestamps)


class RateLimiter:
    """Thread-safe sliding window rate limiter.
    
    Tracks request rates per hotkey and globally to prevent DDoS.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._lock = Lock()
        self._per_hotkey: Dict[str, SlidingWindow] = defaultdict(SlidingWindow)
        self._global = SlidingWindow()
        self._last_cleanup = time.time()
    
    def check_and_record(self, hotkey: str) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed and record it.
        
        Returns:
            (allowed, reason) - reason is None if allowed, else describes limit hit
        """
        now = time.time()
        
        with self._lock:
            # Periodic cleanup of stale entries
            if now - self._last_cleanup > self.config.cleanup_interval:
                self._cleanup(now)
                self._last_cleanup = now
            
            # Check global limits first (faster rejection under heavy load)
            global_per_sec = self._global.count_in_window(1.0, now)
            if global_per_sec >= self.config.global_per_second:
                return False, "global_rate_limit_per_second"
            
            global_per_min = self._global.count_in_window(60.0, now)
            if global_per_min >= self.config.global_per_minute:
                return False, "global_rate_limit_per_minute"
            
            # Check per-hotkey limits
            window = self._per_hotkey[hotkey]
            
            per_sec = window.count_in_window(1.0, now)
            if per_sec >= self.config.per_hotkey_per_second:
                return False, "hotkey_rate_limit_per_second"
            
            per_min = window.count_in_window(60.0, now)
            if per_min >= self.config.per_hotkey_per_minute:
                return False, "hotkey_rate_limit_per_minute"
            
            # Record the request
            window.add(now)
            self._global.add(now)
            
            return True, None
    
    def _cleanup(self, now: float) -> None:
        """Remove stale hotkey entries."""
        cutoff = now - 120.0  # Keep 2 minutes of history
        stale_keys = []
        
        for hotkey, window in self._per_hotkey.items():
            window.timestamps = [t for t in window.timestamps if t > cutoff]
            if not window.timestamps:
                stale_keys.append(hotkey)
        
        for key in stale_keys:
            del self._per_hotkey[key]
    
    def get_stats(self) -> Dict[str, int]:
        """Get current rate limiter stats."""
        now = time.time()
        with self._lock:
            return {
                "tracked_hotkeys": len(self._per_hotkey),
                "global_per_second": self._global.count_in_window(1.0, now),
                "global_per_minute": self._global.count_in_window(60.0, now),
            }


# Singleton instance for the validator
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the rate limiter (for testing)."""
    global _rate_limiter
    _rate_limiter = None


__all__ = ["RateLimitConfig", "RateLimiter", "get_rate_limiter", "reset_rate_limiter"]

