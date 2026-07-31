"""Distributed rate limiting.

A sliding-window log in a sorted set, evaluated by a Lua script so the read, the eviction,
and the write are one atomic operation. Doing it in three round trips from Python is a race
that lets N concurrent requests all observe the same under-limit count and all proceed — and
"the limit was 10 but 40 got through" is exactly the case the limiter exists to prevent.

A fixed-window counter would be cheaper, and is what most quick implementations use. It also
permits twice the limit across a window boundary: ten requests at 11:59:59 and ten more at
12:00:00 both pass a per-minute limit of ten. For login throttling that is the difference
between a control and the appearance of one.

Rejected requests do not consume budget. Otherwise a client retrying in a loop keeps its own
window permanently full and extends its lockout indefinitely, which turns a limit into a
self-inflicted denial of service.
"""

from __future__ import annotations

import secrets
import time
from typing import TYPE_CHECKING, Any

from redis.exceptions import RedisError

from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import RateLimitResult

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

# KEYS[1] = window key
# ARGV = now_ms, window_ms, limit, consume, member
_SLIDING_WINDOW = """
local key      = KEYS[1]
local now      = tonumber(ARGV[1])
local window   = tonumber(ARGV[2])
local limit    = tonumber(ARGV[3])
local consume  = tonumber(ARGV[4])
local member   = ARGV[5]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local used = redis.call('ZCARD', key)
local allowed = 0

if used < limit then
  allowed = 1
  if consume == 1 then
    redis.call('ZADD', key, now, member)
    used = used + 1
  end
end

-- Expire the key one window after its newest entry: the set is empty of relevant data by
-- then, and letting it live forever would leak a key per distinct principal.
redis.call('PEXPIRE', key, window)

local reset = window
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
if oldest[2] then
  reset = window - (now - tonumber(oldest[2]))
end

return {allowed, used, reset}
"""


class RedisRateLimiter:
    """Sliding-window limiter over Redis.

    ``fail_open`` decides behaviour when Redis is unreachable. It defaults to open, which is
    a deliberate and narrow judgement: the destructive case for a limiter outage is brute
    force against ``/auth/login``, and that path is *also* protected by a database-backed
    failed-login lockout (``max_failed_logins``). Failing closed instead would make Redis a
    hard dependency of every request, converting a cache outage into a full outage.
    """

    def __init__(self, client: Redis, *, namespace: str = "aegis", fail_open: bool = True) -> None:
        self._redis = client
        self._prefix = f"{namespace}:"
        self._fail_open = fail_open
        self._script: Any = client.register_script(_SLIDING_WINDOW)

    async def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        return await self._evaluate(key, limit=limit, window_seconds=window_seconds, consume=True)

    async def peek(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        return await self._evaluate(key, limit=limit, window_seconds=window_seconds, consume=False)

    async def reset(self, key: str) -> None:
        """Clear a window. Called after a successful login so one typo is not punished."""
        try:
            await self._redis.delete(f"{self._prefix}{key}")
        except RedisError as exc:
            logger.warning("ratelimit.reset_failed", error=str(exc))

    async def _evaluate(
        self, key: str, *, limit: int, window_seconds: int, consume: bool
    ) -> RateLimitResult:
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        # A unique member per request: two requests in the same millisecond would otherwise
        # collide on score alone and the second would be silently free.
        member = f"{now_ms}-{secrets.token_hex(6)}"

        try:
            raw = await self._script(
                keys=[f"{self._prefix}{key}"],
                args=[now_ms, window_ms, limit, 1 if consume else 0, member],
            )
        except (RedisError, OSError) as exc:
            logger.error("ratelimit.unavailable", error=str(exc), fail_open=self._fail_open)
            return RateLimitResult(
                allowed=self._fail_open,
                remaining=limit if self._fail_open else 0,
                limit=limit,
                reset_after_seconds=window_seconds,
            )

        allowed, used, reset_ms = int(raw[0]), int(raw[1]), int(raw[2])
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=max(0, limit - used),
            limit=limit,
            reset_after_seconds=max(1, (reset_ms + 999) // 1000),
        )
