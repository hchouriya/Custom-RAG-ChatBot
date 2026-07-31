"""Redis cache adapter.

Values are opaque bytes. No pickle, no JSON-by-default, no automatic serialization: the two
things cached at volume are float32 embedding vectors and rendered payloads, both of which
have their own encodings, and a helpful serializer here would silently double the memory of
the first one.

Every key is namespaced. One Redis often serves several environments during development, and
a namespace is what makes ``FLUSHDB`` unnecessary — and unnecessary is the only safe state
for ``FLUSHDB`` to be in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from aegis.core.errors import ServiceUnavailableError
from aegis.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aegis.core.config import Settings

logger = get_logger(__name__)


def build_redis(settings: Settings, *, db: int | None = None) -> Redis:
    """Client with a bounded pool and health checks.

    ``health_check_interval`` matters behind a load balancer or a cloud Redis with an idle
    timeout: without it the first command after an idle period fails on a connection the
    server closed minutes ago, and the failure surfaces as a random request error.
    """
    url = settings.redis_url if db is None else _with_db(settings.redis_url, db)
    client: Redis = Redis.from_url(
        url,
        max_connections=settings.redis_max_connections,
        socket_timeout=5,
        socket_connect_timeout=2,
        health_check_interval=30,
        retry_on_timeout=True,
        decode_responses=False,
    )
    return client


def _with_db(url: str, db: int) -> str:
    base, _, _ = url.rpartition("/")
    return f"{base or url}/{db}"


class RedisCache:
    """``Cache`` over Redis.

    Read failures are the caller's to interpret — ``CachedEmbedder`` swallows them, a
    distributed lock must not — so this adapter raises and lets the policy live where the
    consequence is understood.
    """

    def __init__(self, client: Redis, *, namespace: str = "aegis") -> None:
        self._redis = client
        self._prefix = f"{namespace}:"

    async def get(self, key: str) -> bytes | None:
        try:
            value = await self._redis.get(self._k(key))
        except RedisError as exc:
            raise self._wrap(exc, "get") from exc
        return bytes(value) if value is not None else None

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        try:
            await self._redis.set(self._k(key), value, ex=ttl_seconds)
        except RedisError as exc:
            raise self._wrap(exc, "set") from exc

    async def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """One round trip for a whole batch.

        An embedding batch of 64 chunks is 64 round trips as individual ``GET``s — about
        30 ms of pure latency on a local Redis and considerably more across a network,
        which is the same order as the work being avoided.
        """
        if not keys:
            return {}
        try:
            values = await self._redis.mget([self._k(k) for k in keys])
        except RedisError as exc:
            raise self._wrap(exc, "get_many") from exc
        return {
            key: bytes(value) for key, value in zip(keys, values, strict=True) if value is not None
        }

    async def set_many(self, items: dict[str, bytes], *, ttl_seconds: int | None = None) -> None:
        if not items:
            return
        try:
            pipe = self._redis.pipeline(transaction=False)
            for key, value in items.items():
                pipe.set(self._k(key), value, ex=ttl_seconds)
            await pipe.execute()
        except RedisError as exc:
            raise self._wrap(exc, "set_many") from exc

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        try:
            return int(await self._redis.delete(*(self._k(k) for k in keys)))
        except RedisError as exc:
            raise self._wrap(exc, "delete") from exc

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        """Increment, setting the TTL only on creation.

        Refreshing the expiry on every increment would turn a fixed window into one that
        never closes for an active caller — the precise failure mode that makes a
        home-grown counter useless as a limit.
        """
        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.incr(self._k(key))
            if ttl_seconds is not None:
                pipe.expire(self._k(key), ttl_seconds, nx=True)
            results = await pipe.execute()
        except RedisError as exc:
            raise self._wrap(exc, "incr") from exc
        return int(results[0])

    async def add(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        """``SET NX``: first writer wins.

        The primitive behind idempotency keys, TOTP replay protection, and single-flight
        locks. All three are correctness properties, so the TTL is mandatory: a lock whose
        holder crashed must expire rather than wedge the operation forever.
        """
        try:
            created = await self._redis.set(self._k(key), value, ex=ttl_seconds, nx=True)
        except RedisError as exc:
            raise self._wrap(exc, "add") from exc
        return bool(created)

    async def health(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except (RedisError, OSError) as exc:
            logger.warning("cache.unhealthy", error=str(exc))
            return False

    async def close(self) -> None:
        await self._redis.aclose()

    def script(self, source: str) -> Any:
        """Register a Lua script, for adapters that need atomicity across keys."""
        return self._redis.register_script(source)

    def namespaced(self, key: str) -> str:
        return self._k(key)

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _wrap(self, exc: RedisError, operation: str) -> ServiceUnavailableError:
        logger.warning("cache.operation_failed", operation=operation, error=str(exc))
        return ServiceUnavailableError("Cache is unavailable.")
