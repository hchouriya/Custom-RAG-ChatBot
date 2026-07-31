"""Cache, rate limiting, and idempotency adapters.

``build_cache`` and ``build_rate_limiter`` share one Redis client. Two clients would mean two
connection pools competing for the same server limit, and a health check that can pass for
one while the other is broken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.infrastructure.cache.idempotency import (
    Claim,
    IdempotencyStore,
    require_single_flight,
    single_flight,
)
from aegis.infrastructure.cache.limiter import RedisRateLimiter
from aegis.infrastructure.cache.memory import InMemoryCache, InMemoryRateLimiter
from aegis.infrastructure.cache.redis_cache import RedisCache, build_redis

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from aegis.core.config import Settings
    from aegis.domain.ports.infrastructure import Cache, RateLimiter

__all__ = [
    "Claim",
    "IdempotencyStore",
    "InMemoryCache",
    "InMemoryRateLimiter",
    "RedisCache",
    "RedisRateLimiter",
    "build_cache",
    "build_rate_limiter",
    "build_redis",
    "require_single_flight",
    "single_flight",
]


def build_cache(settings: Settings, *, client: Redis | None = None) -> Cache:
    return RedisCache(
        client or build_redis(settings), namespace=f"{settings.cache_namespace}:{settings.app_env}"
    )


def build_rate_limiter(settings: Settings, *, client: Redis | None = None) -> RateLimiter:
    return RedisRateLimiter(
        client or build_redis(settings), namespace=f"{settings.cache_namespace}:{settings.app_env}"
    )
