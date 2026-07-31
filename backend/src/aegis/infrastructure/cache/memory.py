"""In-process cache and rate limiter.

For unit tests and for a single-process development run without Redis. Semantics match the
Redis adapters exactly — including expiry, ``add`` returning ``False`` on an existing key,
and rejected requests not consuming budget — because a fake that is easier to satisfy than
the real thing turns green tests into false confidence.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from aegis.domain.ports.infrastructure import RateLimitResult


@dataclass(slots=True)
class _Entry:
    value: bytes
    expires_at: float | None


class InMemoryCache:
    """Dict with expiry. Not shared between processes, and not meant to be."""

    def __init__(self) -> None:
        self._data: dict[str, _Entry] = {}

    async def get(self, key: str) -> bytes | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= time.monotonic():
            del self._data[key]
            return None
        return entry.value

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self._data[key] = _Entry(value, _deadline(ttl_seconds))

    async def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                found[key] = value
        return found

    async def set_many(self, items: dict[str, bytes], *, ttl_seconds: int | None = None) -> None:
        for key, value in items.items():
            await self.set(key, value, ttl_seconds=ttl_seconds)

    async def delete(self, *keys: str) -> int:
        return sum(1 for key in keys if self._data.pop(key, None) is not None)

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        current = await self.get(key)
        value = int(current) + 1 if current is not None else 1
        # Preserve the original deadline, matching `EXPIRE ... NX` in the Redis adapter.
        existing = self._data.get(key)
        deadline = existing.expires_at if existing is not None else _deadline(ttl_seconds)
        self._data[key] = _Entry(str(value).encode(), deadline)
        return value

    async def add(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        if await self.get(key) is not None:
            return False
        await self.set(key, value, ttl_seconds=ttl_seconds)
        return True

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self._data.clear()


class InMemoryRateLimiter:
    """Sliding-window log, one deque per key."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    async def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        return self._evaluate(key, limit=limit, window_seconds=window_seconds, consume=True)

    async def peek(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        return self._evaluate(key, limit=limit, window_seconds=window_seconds, consume=False)

    async def reset(self, key: str) -> None:
        self._hits.pop(key, None)

    def _evaluate(
        self, key: str, *, limit: int, window_seconds: int, consume: bool
    ) -> RateLimitResult:
        now = time.monotonic()
        window = self._hits.setdefault(key, deque())
        while window and window[0] <= now - window_seconds:
            window.popleft()

        allowed = len(window) < limit
        if allowed and consume:
            window.append(now)

        oldest = window[0] if window else now
        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, limit - len(window)),
            limit=limit,
            reset_after_seconds=max(1, int(window_seconds - (now - oldest))),
        )


def _deadline(ttl_seconds: int | None) -> float | None:
    return None if ttl_seconds is None else time.monotonic() + ttl_seconds
