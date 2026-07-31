"""arq worker settings for ingestion, reindex, and purge jobs.

The worker builds the same :class:`~aegis.core.container.Container` as the API and runs
:class:`~aegis.services.ingestion.IngestionService` inside a session. Job names match
:class:`~aegis.domain.enums.JobName` so a renamed function cannot silently orphan the queue.
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import UUID

from arq.connections import RedisSettings

from aegis.core.config import get_settings
from aegis.core.container import Container
from aegis.core.logging import configure_logging, get_logger
from aegis.domain.enums import JobName
from aegis.infrastructure.queue.arq_queue import DEFAULT_QUEUE
from aegis.services.ingestion import is_retryable

logger = get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings)
    container = Container.build(settings)
    await container.startup()
    ctx["container"] = container
    logger.info("worker.started")


async def shutdown(ctx: dict[str, Any]) -> None:
    container: Container | None = ctx.get("container")
    if container is not None:
        await container.shutdown()
    logger.info("worker.stopped")


async def ingest(ctx: dict[str, Any], version_id: str, **_: Any) -> dict[str, Any]:
    """Ingest one document version end-to-end."""
    container: Container = ctx["container"]
    async with container.session() as (_session, repos):
        service = container.ingestion_service(repos)
        report = await service.ingest(UUID(version_id), worker_id=str(ctx.get("job_id", "worker")))
        return {
            "version_id": str(report.version_id),
            "status": report.status.value,
            "chunks": report.chunks,
            "error": report.error,
        }


async def reindex(
    ctx: dict[str, Any],
    document_id: str,
    *,
    force_reparse: bool = False,
    acl_only: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Re-embed, re-parse, or patch ACL payloads for one document."""
    container: Container = ctx["container"]
    async with container.session() as (_session, repos):
        service = container.ingestion_service(repos)
        if acl_only:
            patched = await service.refresh_acl(UUID(document_id))
            return {"document_id": document_id, "points_patched": patched}
        report = await service.reindex_document(
            UUID(document_id), force_reparse=force_reparse
        )
        return {
            "document_id": document_id,
            "version_id": str(report.version_id),
            "status": report.status.value,
            "chunks": report.chunks,
        }


async def purge(ctx: dict[str, Any], document_id: str, **_: Any) -> dict[str, Any]:
    """Remove a document's vectors from the index."""
    container: Container = ctx["container"]
    async with container.session() as (_session, repos):
        service = container.ingestion_service(repos)
        removed = await service.purge_document(UUID(document_id))
        return {"document_id": document_id, "vectors_removed": removed}


class WorkerSettings:
    """arq entrypoint: ``arq aegis.worker.WorkerSettings``."""

    functions: ClassVar[list[Any]] = [ingest, reindex, purge]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_queue_url)
    queue_name = DEFAULT_QUEUE
    max_jobs = 10
    job_timeout = 1800
    keep_result = 3600


JOB_INGEST = JobName.INGEST.value
JOB_REINDEX = JobName.REINDEX.value
JOB_PURGE = JobName.PURGE.value

__all__ = [
    "JOB_INGEST",
    "JOB_PURGE",
    "JOB_REINDEX",
    "WorkerSettings",
    "ingest",
    "is_retryable",
    "purge",
    "reindex",
]
