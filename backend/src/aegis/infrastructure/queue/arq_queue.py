"""arq job queue adapter.

arq rather than Celery: it is async-native, so a job that spends its time awaiting a parser,
an embedding provider, and a vector store does not need a thread per job, and the whole
worker is a few hundred lines rather than a framework. Redis is already a dependency, so the
queue adds no new infrastructure.

Deduplication is the interesting part. arq's ``_job_id`` is a uniqueness key: enqueueing twice
with the same id returns ``None`` the second time while the first is still queued or running.
Ingestion keys jobs on the content checksum, so re-uploading identical bytes — the single most
common accidental duplicate — becomes a no-op at the queue rather than a second full pass
through parsing, chunking, and embedding.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from redis.exceptions import RedisError

from aegis.core.errors import ServiceUnavailableError
from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import EnqueuedJob

if TYPE_CHECKING:
    from aegis.core.config import Settings

logger = get_logger(__name__)

DEFAULT_QUEUE = "aegis:queue"
"""arq's default is ``arq:queue``. Naming it ours means a Redis shared with another arq
application cannot silently hand our jobs to its workers."""


class ArqJobQueue:
    """``JobQueue`` over arq.

    The pool is created lazily on first enqueue. The API process must be able to boot with
    Redis down — a worker outage should degrade ingestion to "queued later", not prevent the
    service from starting and serving reads.
    """

    def __init__(self, redis_settings: RedisSettings, *, queue_name: str = DEFAULT_QUEUE) -> None:
        self._redis_settings = redis_settings
        self._queue_name = queue_name
        self._pool: ArqRedis | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ArqJobQueue:
        return cls(RedisSettings.from_dsn(settings.redis_queue_url))

    async def _get_pool(self) -> ArqRedis:
        if self._pool is None:
            try:
                self._pool = await create_pool(
                    self._redis_settings, default_queue_name=self._queue_name
                )
            except (RedisError, OSError) as exc:
                logger.error("queue.connect_failed", error=str(exc))
                raise ServiceUnavailableError("Job queue is unavailable.") from exc
        return self._pool

    async def enqueue(
        self,
        name: str,
        *args: Any,
        idempotency_key: str | None = None,
        delay: timedelta | None = None,
        **kwargs: Any,
    ) -> EnqueuedJob | None:
        """Enqueue a job, or return ``None`` when ``idempotency_key`` is already queued."""
        pool = await self._get_pool()
        try:
            job = await pool.enqueue_job(
                name,
                *args,
                _job_id=idempotency_key,
                _defer_by=delay,
                _queue_name=self._queue_name,
                **kwargs,
            )
        except (RedisError, OSError) as exc:
            logger.error("queue.enqueue_failed", job=name, error=str(exc))
            raise ServiceUnavailableError("Could not queue background work.") from exc

        if job is None:
            logger.info("queue.duplicate_suppressed", job=name, idempotency_key=idempotency_key)
            return None
        logger.info("queue.enqueued", job=name, job_id=job.job_id)
        return EnqueuedJob(job_id=job.job_id, queued_at=datetime.now(UTC))

    async def queue_depth(self, queue: str = DEFAULT_QUEUE) -> int:
        """Jobs waiting to be picked up.

        The signal that matters for ingestion: latency looks fine while the queue grows,
        right up until documents are hours stale. It is a gauge on the dashboard and an
        alert threshold, not a debugging convenience.
        """
        pool = await self._get_pool()
        try:
            return int(await pool.zcard(queue or self._queue_name))
        except (RedisError, OSError) as exc:
            logger.warning("queue.depth_failed", error=str(exc))
            return -1

    async def oldest_job_age_seconds(self, queue: str = DEFAULT_QUEUE) -> float | None:
        """Age of the head of the queue, or ``None`` when empty.

        Depth alone does not distinguish a burst that is being worked through from a stalled
        worker: both show a hundred jobs. Age separates them.
        """
        pool = await self._get_pool()
        try:
            head = await pool.zrange(queue or self._queue_name, 0, 0, withscores=True)
        except (RedisError, OSError) as exc:
            logger.warning("queue.age_failed", error=str(exc))
            return None
        if not head:
            return None
        # arq scores a job with the epoch milliseconds at which it becomes eligible, so a
        # deferred job that is not due yet has a negative age. Clamp it: "due in 30 s" is
        # not a backlog.
        scheduled_at = float(head[0][1]) / 1000.0
        return max(0.0, datetime.now(UTC).timestamp() - scheduled_at)

    async def health(self) -> bool:
        try:
            pool = await self._get_pool()
            return bool(await pool.ping())
        except (ServiceUnavailableError, RedisError, OSError) as exc:
            logger.warning("queue.unhealthy", error=str(exc))
            return False

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None
