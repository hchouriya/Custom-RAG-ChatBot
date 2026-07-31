"""In-process job queue for tests and for a single-process development run.

Two modes, and the distinction matters. By default it *records* enqueues: the contract a
service is responsible for is "the right job was queued with the right arguments", and
whether the worker then succeeds belongs to the worker's own tests. Given a handler map it
*runs* them inline instead, which is what makes an end-to-end upload-to-searchable test
possible without a worker container.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import EnqueuedJob

logger = get_logger(__name__)

Handler = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class RecordedJob:
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    idempotency_key: str | None = None
    delay: timedelta | None = None


class InMemoryJobQueue:
    """Records enqueues, and optionally executes them inline."""

    def __init__(self, handlers: dict[str, Handler] | None = None) -> None:
        self.jobs: list[RecordedJob] = []
        self._handlers = handlers or {}
        self._claimed: set[str] = set()

    async def enqueue(
        self,
        name: str,
        *args: Any,
        idempotency_key: str | None = None,
        delay: timedelta | None = None,
        **kwargs: Any,
    ) -> EnqueuedJob | None:
        if idempotency_key is not None:
            if idempotency_key in self._claimed:
                return None
            self._claimed.add(idempotency_key)

        self.jobs.append(RecordedJob(name, args, dict(kwargs), idempotency_key, delay))
        handler = self._handlers.get(name)
        if handler is not None:
            # Inline execution deliberately does not catch: a job that raises should fail the
            # test that queued it, rather than being swallowed the way a real worker would.
            await handler(*args, **kwargs)
        return EnqueuedJob(
            job_id=idempotency_key or f"{name}-{len(self.jobs)}", queued_at=datetime.now(UTC)
        )

    async def queue_depth(self, queue: str = "default") -> int:
        return len(self.jobs)

    async def oldest_job_age_seconds(self, queue: str = "default") -> float | None:
        return 0.0 if self.jobs else None

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    # ── Test helpers ────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return [job.name for job in self.jobs]

    def find(self, name: str) -> list[RecordedJob]:
        return [job for job in self.jobs if job.name == name]

    def clear(self) -> None:
        self.jobs.clear()
        self._claimed.clear()


@dataclass(slots=True)
class NullJobQueue:
    """Drops everything. For CLI commands that must not schedule background work."""

    async def enqueue(
        self,
        name: str,
        *args: Any,
        idempotency_key: str | None = None,
        delay: timedelta | None = None,
        **kwargs: Any,
    ) -> EnqueuedJob | None:
        logger.debug("queue.discarded", job=name)
        return None

    async def queue_depth(self, queue: str = "default") -> int:
        return 0

    async def oldest_job_age_seconds(self, queue: str = "default") -> float | None:
        return None

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None
