"""Cache, rate limiter, and idempotency semantics.

The in-memory implementations are the subject here because they are what the rest of the test
suite runs against: if they are more permissive than Redis, every test that depends on them is
measuring the wrong thing. The Redis adapters are covered for the parts that do not need a
server — key namespacing, and what happens when the server is gone.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from aegis.core.config import Settings
from aegis.core.errors import ConflictError, ServiceUnavailableError
from aegis.core.ratelimit import LimitBucket, RateLimitPolicies
from aegis.infrastructure.cache import (
    IdempotencyStore,
    InMemoryCache,
    InMemoryRateLimiter,
    RedisCache,
    RedisRateLimiter,
    require_single_flight,
    single_flight,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis


def as_redis(fake: FakeRedis) -> Redis:
    """The adapters are typed against the real client; the fake implements the slice used."""
    return cast("Redis", fake)


class FakeRedis:
    """Just enough Redis to check what the adapter sends, or to fail on command."""

    def __init__(self, *, fail: bool = False) -> None:
        self.data: dict[str, bytes] = {}
        self.calls: list[tuple[str, Any]] = []
        self._fail = fail

    def _check(self) -> None:
        if self._fail:
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("simulated outage")

    async def get(self, key: str) -> bytes | None:
        self._check()
        self.calls.append(("get", key))
        return self.data.get(key)

    async def set(
        self, key: str, value: bytes, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        self._check()
        self.calls.append(("set", key))
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        self._check()
        self.calls.append(("mget", tuple(keys)))
        return [self.data.get(k) for k in keys]

    async def delete(self, *keys: str) -> int:
        self._check()
        return sum(1 for k in keys if self.data.pop(k, None) is not None)

    async def ping(self) -> bool:
        self._check()
        return True

    async def aclose(self) -> None:
        return None

    def register_script(self, source: str) -> Any:
        async def run(keys: list[str], args: list[Any]) -> list[int]:
            self._check()
            return [1, 1, int(args[1])]

        return run

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._queued: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def set(self, *args: Any, **kwargs: Any) -> None:
        self._queued.append(("set", args, kwargs))

    def incr(self, *args: Any, **kwargs: Any) -> None:
        self._queued.append(("incr", args, kwargs))

    def expire(self, *args: Any, **kwargs: Any) -> None:
        self._queued.append(("expire", args, kwargs))

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for name, args, kwargs in self._queued:
            if name == "set":
                results.append(await self._redis.set(*args, **kwargs))
            elif name == "incr":
                current = int(self._redis.data.get(args[0], b"0"))
                self._redis.data[args[0]] = str(current + 1).encode()
                results.append(current + 1)
            else:
                results.append(True)
        return results


class TestInMemoryCache:
    async def test_round_trip(self) -> None:
        cache = InMemoryCache()
        await cache.set("k", b"v")
        assert await cache.get("k") == b"v"

    async def test_absent_key_is_none(self) -> None:
        assert await InMemoryCache().get("nope") is None

    async def test_expiry_is_honoured(self) -> None:
        cache = InMemoryCache()
        await cache.set("k", b"v", ttl_seconds=0)
        await asyncio.sleep(0.01)
        assert await cache.get("k") is None

    async def test_batch_read_skips_misses(self) -> None:
        cache = InMemoryCache()
        await cache.set("a", b"1")
        assert await cache.get_many(["a", "b"]) == {"a": b"1"}

    async def test_add_is_first_writer_wins(self) -> None:
        cache = InMemoryCache()
        assert await cache.add("k", b"first", ttl_seconds=60) is True
        assert await cache.add("k", b"second", ttl_seconds=60) is False
        assert await cache.get("k") == b"first"

    async def test_incr_does_not_extend_the_original_deadline(self) -> None:
        """A window that refreshes on every hit never closes, which makes it not a window."""
        cache = InMemoryCache()
        assert await cache.incr("c", ttl_seconds=1) == 1
        assert await cache.incr("c", ttl_seconds=3600) == 2
        await asyncio.sleep(1.05)
        assert await cache.get("c") is None

    async def test_delete_counts_only_what_existed(self) -> None:
        cache = InMemoryCache()
        await cache.set("a", b"1")
        assert await cache.delete("a", "b") == 1


class TestRedisCache:
    async def test_keys_are_namespaced(self) -> None:
        redis = FakeRedis()
        cache = RedisCache(as_redis(redis), namespace="aegis:test")
        await cache.set("emb:x", b"v")
        assert "aegis:test:emb:x" in redis.data

    async def test_batch_read_is_one_round_trip(self) -> None:
        redis = FakeRedis()
        cache = RedisCache(as_redis(redis), namespace="ns")
        await cache.set_many({"a": b"1", "b": b"2"})
        found = await cache.get_many(["a", "b", "c"])
        assert found == {"a": b"1", "b": b"2"}
        assert [c for c in redis.calls if c[0] == "mget"] == [("mget", ("ns:a", "ns:b", "ns:c"))]

    async def test_empty_batch_does_not_talk_to_redis(self) -> None:
        redis = FakeRedis()
        cache = RedisCache(as_redis(redis))
        assert await cache.get_many([]) == {}
        await cache.set_many({})
        assert redis.calls == []

    async def test_an_outage_surfaces_as_service_unavailable(self) -> None:
        cache = RedisCache(as_redis(FakeRedis(fail=True)))
        with pytest.raises(ServiceUnavailableError):
            await cache.get("k")

    async def test_health_reports_false_instead_of_raising(self) -> None:
        assert await RedisCache(as_redis(FakeRedis(fail=True))).health() is False
        assert await RedisCache(as_redis(FakeRedis())).health() is True


class TestRateLimiter:
    async def test_requests_are_allowed_up_to_the_limit(self) -> None:
        limiter = InMemoryRateLimiter()
        results = [await limiter.check("k", limit=3, window_seconds=60) for _ in range(4)]
        assert [r.allowed for r in results] == [True, True, True, False]
        assert results[2].remaining == 0

    async def test_a_rejected_request_does_not_consume_budget(self) -> None:
        """Otherwise a client retrying in a loop extends its own lockout forever."""
        limiter = InMemoryRateLimiter()
        for _ in range(2):
            await limiter.check("k", limit=2, window_seconds=60)
        first_rejection = await limiter.check("k", limit=2, window_seconds=60)
        second_rejection = await limiter.check("k", limit=2, window_seconds=60)
        assert not first_rejection.allowed
        assert second_rejection.reset_after_seconds <= first_rejection.reset_after_seconds

    async def test_peek_does_not_consume(self) -> None:
        limiter = InMemoryRateLimiter()
        for _ in range(3):
            assert (await limiter.peek("k", limit=1, window_seconds=60)).allowed
        assert (await limiter.check("k", limit=1, window_seconds=60)).allowed

    async def test_the_window_slides(self) -> None:
        limiter = InMemoryRateLimiter()
        assert (await limiter.check("k", limit=1, window_seconds=1)).allowed
        assert not (await limiter.check("k", limit=1, window_seconds=1)).allowed
        await asyncio.sleep(1.05)
        assert (await limiter.check("k", limit=1, window_seconds=1)).allowed

    async def test_reset_clears_a_window(self) -> None:
        limiter = InMemoryRateLimiter()
        await limiter.check("k", limit=1, window_seconds=60)
        await limiter.reset("k")
        assert (await limiter.check("k", limit=1, window_seconds=60)).allowed

    async def test_scopes_are_independent(self) -> None:
        limiter = InMemoryRateLimiter()
        await limiter.check("rl:login:i:1.2.3.4", limit=1, window_seconds=60)
        assert (await limiter.check("rl:login:p:someone", limit=1, window_seconds=60)).allowed

    async def test_redis_limiter_fails_open_but_says_so(self) -> None:
        limiter = RedisRateLimiter(as_redis(FakeRedis(fail=True)), fail_open=True)
        result = await limiter.check("k", limit=5, window_seconds=60)
        assert result.allowed
        assert result.remaining == 5

    async def test_redis_limiter_can_fail_closed(self) -> None:
        limiter = RedisRateLimiter(as_redis(FakeRedis(fail=True)), fail_open=False)
        result = await limiter.check("k", limit=5, window_seconds=60)
        assert not result.allowed
        assert result.remaining == 0

    async def test_reset_after_is_never_zero(self) -> None:
        """A zero would invite an immediate hot retry loop against ``Retry-After``."""
        limiter = RedisRateLimiter(as_redis(FakeRedis()))
        result = await limiter.check("k", limit=5, window_seconds=60)
        assert result.reset_after_seconds >= 1


class TestPolicyIntegration:
    async def test_login_is_limited_per_ip_and_per_principal(self, settings: Settings) -> None:
        policies = RateLimitPolicies(settings)
        policy = policies.resolve(LimitBucket.LOGIN)
        scopes = policies.scopes(policy, principal="user@example.com", ip="203.0.113.7")
        assert len(scopes) == 2

        limiter = InMemoryRateLimiter()
        # Every scope must pass, so exhausting the IP alone is enough to reject.
        for _ in range(policy.limit):
            await limiter.check(scopes[1], limit=policy.limit, window_seconds=policy.window_seconds)
        outcomes = [
            await limiter.check(s, limit=policy.limit, window_seconds=policy.window_seconds)
            for s in scopes
        ]
        assert not all(o.allowed for o in outcomes)

    async def test_chat_limits_scale_with_role(self, settings: Settings) -> None:
        policies = RateLimitPolicies(settings)
        guest = policies.resolve(LimitBucket.CHAT, "guest")
        admin = policies.resolve(LimitBucket.CHAT, "admin")
        assert guest.limit < admin.limit


class TestIdempotency:
    async def test_the_first_claim_wins_and_the_retry_replays(self) -> None:
        store = IdempotencyStore(InMemoryCache())
        first = await store.claim("documents.create", "key-1")
        assert first.acquired

        second = await store.claim("documents.create", "key-1")
        assert not second.acquired
        assert second.in_flight

        await store.complete("documents.create", "key-1", {"id": "abc"})
        replay = await store.claim("documents.create", "key-1")
        assert not replay.acquired
        assert replay.stored == {"id": "abc"}
        assert not replay.in_flight

    async def test_the_same_key_in_two_scopes_does_not_collide(self) -> None:
        store = IdempotencyStore(InMemoryCache())
        assert (await store.claim("documents.create", "k")).acquired
        assert (await store.claim("tickets.create", "k")).acquired

    async def test_a_released_claim_can_be_retried_for_real(self) -> None:
        """A failed first attempt must not make the failure itself idempotent."""
        store = IdempotencyStore(InMemoryCache())
        await store.claim("upload", "k")
        await store.release("upload", "k")
        assert (await store.claim("upload", "k")).acquired

    async def test_a_corrupt_entry_degrades_to_in_flight(self) -> None:
        cache = InMemoryCache()
        await cache.set("idem:scope:k", b"not json at all", ttl_seconds=60)
        claim = await IdempotencyStore(cache).claim("scope", "k")
        assert not claim.acquired
        assert claim.stored is None


class TestSingleFlight:
    async def test_only_one_holder_at_a_time(self) -> None:
        cache = InMemoryCache()
        async with single_flight(cache, "reindex:1") as first:
            assert first
            async with single_flight(cache, "reindex:1") as second:
                assert not second

    async def test_the_lock_is_released_on_the_way_out(self) -> None:
        cache = InMemoryCache()
        async with single_flight(cache, "reindex:1") as acquired:
            assert acquired
        async with single_flight(cache, "reindex:1") as again:
            assert again

    async def test_the_lock_is_released_even_when_the_body_raises(self) -> None:
        cache = InMemoryCache()
        with pytest.raises(RuntimeError):
            async with single_flight(cache, "k"):
                raise RuntimeError("boom")
        async with single_flight(cache, "k") as acquired:
            assert acquired

    async def test_require_variant_raises_for_a_user_triggered_operation(self) -> None:
        cache = InMemoryCache()
        await require_single_flight(cache, "export:1")
        with pytest.raises(ConflictError, match="already in progress"):
            await require_single_flight(cache, "export:1")
