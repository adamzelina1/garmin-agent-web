"""In-memory attempt limiting for the auth endpoints.

Signup and re-linking each trigger a real Garmin login from this server's IP,
and Garmin answers a burst of those with an IP-level block that stops every
account's sync — so they are capped per caller and server-wide. Password login
is capped per caller to slow guessing.

Counters live in process memory (the server is single-process, like the pending
MFA sessions in ``auth``), so a restart resets them.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding window: at most ``limit`` recorded attempts per ``window_s``."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if not hits:
            del self._hits[key]
        return hits

    def retry_after(self, key: str) -> int:
        """Seconds until ``key`` may try again; 0 when it is under the limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._prune(key, now)
            if len(hits) < self.limit:
                return 0
            return max(1, int(self.window_s - (now - hits[0])) + 1)

    def record(self, key: str) -> None:
        """Count one attempt against ``key``."""
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            self._hits[key].append(now)
