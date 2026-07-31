"""Idempotency keys and single-flight locks, on top of the ``Cache`` port.

Both are the same primitive — ``SET NX`` with a TTL — used for two different problems:

* **Idempotency.** A client retries a ``POST`` after a timeout. Without a claim, the retry
  creates a second document, a second ticket, or a second charge. With one, the retry is
  recognised and the first response is replayed.
* **Single flight.** Twenty replicas start at once and all try to create the same Qdrant
  collection, or ten requests miss the cache for the same expensive answer. One should do
  the work; the rest should wait for it rather than duplicate it.

Written against the port, not against Redis, so the in-memory cache makes both testable.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aegis.core.errors import ConflictError
from aegis.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aegis.domain.ports.infrastructure import Cache

logger = get_logger(__name__)

IN_FLIGHT = b"\x00in-flight"
DEFAULT_TTL = 86_400
"""24 hours. Long enough to cover any client retry policy worth honouring, short enough that
a replayed key cannot resurrect a response describing state that has since changed."""


@dataclass(frozen=True, slots=True)
class Claim:
    """The outcome of trying to claim an idempotency key."""

    acquired: bool
    stored: dict[str, Any] | None = None

    @property
    def in_flight(self) -> bool:
        """Someone else holds the key and has not finished.

        Distinct from a completed replay: the correct response is 409, not the stored one,
        because there is no stored one yet.
        """
        return not self.acquired and self.stored is None


class IdempotencyStore:
    """Claims a key, then records the response so a retry can replay it."""

    def __init__(self, cache: Cache, *, ttl_seconds: int = DEFAULT_TTL) -> None:
        self._cache = cache
        self._ttl = ttl_seconds

    async def claim(self, scope: str, key: str) -> Claim:
        """Attempt to take ownership of ``key`` within ``scope``.

        The scope is the route or operation. Without it, one client's idempotency key would
        collide across endpoints, and the second endpoint would replay the first's response.
        """
        cache_key = _key(scope, key)
        if await self._cache.add(cache_key, IN_FLIGHT, ttl_seconds=self._ttl):
            return Claim(acquired=True)

        existing = await self._cache.get(cache_key)
        if existing is None or existing == IN_FLIGHT:
            return Claim(acquired=False)
        try:
            return Claim(acquired=False, stored=json.loads(existing))
        except ValueError:
            logger.warning("idempotency.corrupt_entry", scope=scope)
            return Claim(acquired=False)

    async def complete(self, scope: str, key: str, response: dict[str, Any]) -> None:
        """Record the response for replay."""
        await self._cache.set(
            _key(scope, key),
            json.dumps(response, separators=(",", ":"), default=str).encode(),
            ttl_seconds=self._ttl,
        )

    async def release(self, scope: str, key: str) -> None:
        """Drop a claim whose operation failed.

        Failures must not be idempotent: if the first attempt errored, the retry deserves a
        real second attempt rather than a replay of the error.
        """
        await self._cache.delete(_key(scope, key))


@asynccontextmanager
async def single_flight(
    cache: Cache, key: str, *, ttl_seconds: int = 30, owner: str = ""
) -> AsyncIterator[bool]:
    """Hold an advisory lock for the block, yielding whether it was acquired.

    Advisory, not a guarantee: a holder that outlives the TTL loses the lock while still
    working. That is acceptable for the uses here — creating a collection is idempotent,
    recomputing a cache entry is wasteful but correct — and it is why this is not used for
    anything where double execution would be a data error. Those go through a database
    constraint instead.
    """
    acquired = await cache.add(f"lock:{key}", (owner or "-").encode(), ttl_seconds=ttl_seconds)
    try:
        yield acquired
    finally:
        if acquired:
            await cache.delete(f"lock:{key}")


async def require_single_flight(cache: Cache, key: str, *, ttl_seconds: int = 30) -> None:
    """Raise ``ConflictError`` instead of yielding ``False``.

    For operations a user triggers explicitly — a reindex, an export — where "already
    running" is information they should receive rather than a duplicate job they should get.
    """
    if not await cache.add(f"lock:{key}", IN_FLIGHT, ttl_seconds=ttl_seconds):
        raise ConflictError("This operation is already in progress.")


def _key(scope: str, key: str) -> str:
    return f"idem:{scope}:{key}"
