from __future__ import annotations

import time


class TokenBucket:
    def __init__(self, rate_per_minute: int, *, time_fn=time.time) -> None:
        self.capacity = max(1, int(rate_per_minute))
        self.tokens = float(self.capacity)
        self.refill_rate_per_sec = float(rate_per_minute) / 60.0
        self.last_refill = time_fn()
        self._now = time_fn

    def allow(self, n: int = 1) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def _refill(self) -> None:
        now = self._now()
        elapsed = max(0.0, now - self.last_refill)
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_sec)


