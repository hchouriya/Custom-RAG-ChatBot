"""Job queue adapters and their factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.infrastructure.queue.memory import InMemoryJobQueue, NullJobQueue, RecordedJob

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.domain.ports.infrastructure import JobQueue

__all__ = [
    "InMemoryJobQueue",
    "NullJobQueue",
    "RecordedJob",
    "build_job_queue",
]


def build_job_queue(settings: Settings) -> JobQueue:
    """Real queue everywhere except the test environment.

    Tests get the recording queue by default so that a service under test does not need a
    Redis, while an integration test can still construct ``ArqJobQueue`` explicitly.
    """
    if settings.app_env == "test":
        return InMemoryJobQueue()

    from aegis.infrastructure.queue.arq_queue import ArqJobQueue

    return ArqJobQueue.from_settings(settings)
