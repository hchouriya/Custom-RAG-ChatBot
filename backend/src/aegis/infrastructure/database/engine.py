"""Async engine, session factory, and the unit of work.

Pool sizing is the real concurrency ceiling of this service, not the number of asyncio tasks.
With ``pool_size=20`` and ``max_overflow=10`` each process can hold 30 connections; four API
workers plus three ingestion workers is already ~210 connections, which is past a default
PostgreSQL ``max_connections`` of 100. That is why PgBouncer in transaction mode is a
documented requirement beyond a handful of replicas rather than an optimization.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self

from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from aegis.core.config import Settings
from aegis.core.errors import ConflictError
from aegis.core.logging import get_logger

logger = get_logger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine.

    ``pool_pre_ping`` costs one cheap round trip per checkout and removes the entire class of
    "server closed the connection unexpectedly" errors that appear after a database failover
    or an idle timeout in a managed service.

    Tests use ``NullPool`` so that each event loop gets fresh connections; a pooled connection
    created on one loop and reused on another is a well-known source of hangs that reproduce
    only in CI.
    """
    is_test = settings.app_env == "test"
    kwargs: dict[str, Any] = {
        "echo": settings.database_echo,
        "future": True,
        "pool_pre_ping": True,
        "connect_args": {
            "timeout": settings.database_pool_timeout,
            "command_timeout": 60,
            "server_settings": {
                "application_name": f"aegis-{settings.app_env}",
                # Bound the damage a runaway analytics query can do to the pool.
                "statement_timeout": "60000",
                "idle_in_transaction_session_timeout": "30000",
                "jit": "off",
            },
        },
    }
    if is_test:
        kwargs["poolclass"] = NullPool
    else:
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        kwargs["pool_timeout"] = settings.database_pool_timeout
        kwargs["pool_recycle"] = 1800

    return create_async_engine(settings.database_url, **kwargs)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory.

    ``expire_on_commit=False`` so that entities read before a commit remain usable after it.
    With the default, every attribute access after commit triggers a lazy refresh — which in
    async SQLAlchemy raises ``MissingGreenlet`` rather than doing anything useful.
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )


class SqlAlchemyUnitOfWork:
    """Transaction boundary over one :class:`AsyncSession`.

    Commits on clean exit, rolls back on any exception, and translates unique-constraint
    violations into a domain :class:`ConflictError` so services never import a database
    exception type.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._depth = 0

    @property
    def session(self) -> AsyncSession:
        return self._session

    async def __aenter__(self) -> Self:
        # Re-entrant: a service calling another service inside a transaction should join it
        # rather than opening a nested one that could commit half the work.
        self._depth += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._depth -= 1
        if self._depth > 0:
            return
        if exc_type is not None:
            await self.rollback()
            return
        await self.commit()

    async def commit(self) -> None:
        try:
            await self._session.commit()
        except IntegrityError as err:
            await self._session.rollback()
            raise _translate_integrity_error(err) from err
        except DBAPIError:
            await self._session.rollback()
            raise

    async def rollback(self) -> None:
        await self._session.rollback()

    async def flush(self) -> None:
        try:
            await self._session.flush()
        except IntegrityError as err:
            raise _translate_integrity_error(err) from err


def _translate_integrity_error(err: IntegrityError) -> ConflictError:
    """Turn a constraint violation into a message a caller can act on.

    Constraint names are mapped explicitly rather than echoed, because a raw PostgreSQL error
    string leaks schema details and reads as a crash to whoever receives it.
    """
    detail = str(getattr(err.orig, "args", ("",))[0] if err.orig else err)
    known = {
        "uq_users_email": "A user with this email already exists.",
        "uq_collections_slug": "A collection with this slug already exists.",
        "uq_tags_name": "This tag already exists.",
        "uq_document_versions_document_id_version_no": "That version number is already taken.",
        "uq_document_versions_document_id_checksum_sha256": (
            "This exact file is already stored as a version of this document."
        ),
        "uq_ingest_jobs_idempotency_key": "This job has already been queued.",
        "uq_chunks_version_id_ordinal": "Duplicate chunk ordinal for this version.",
        "ux_prompt_active": "Another version of this prompt is already active.",
        "uq_api_keys_prefix": "Key prefix collision; retry.",
        "uq_feedback_message_id_user_id": "You have already rated this answer.",
    }
    for name, message in known.items():
        if name in detail:
            return ConflictError(message)
    logger.warning("db.integrity_error.unmapped", detail=detail[:400])
    return ConflictError("The request conflicts with existing data.")


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[AsyncSession, SqlAlchemyUnitOfWork]]:
    """Open a session and its unit of work, closing both on exit.

    Used by workers and CLI commands. The API gets its session from a FastAPI dependency so
    that one request maps to exactly one session and one transaction.
    """
    async with factory() as session:
        uow = SqlAlchemyUnitOfWork(session)
        try:
            yield session, uow
        except Exception:
            await session.rollback()
            raise


async def check_database(engine: AsyncEngine) -> bool:
    """Readiness probe. Deliberately trivial: connectivity, not correctness."""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("db.health_check_failed", error=str(exc)[:200])
        return False


async def current_migration_revision(engine: AsyncEngine) -> str | None:
    """Applied Alembic revision, reported by ``/health/deep`` and ``/version``.

    Comparing this against the revision baked into the image is how a half-deployed
    environment is detected before someone spends an hour debugging a missing column.
    """
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
            row = result.first()
            return str(row[0]) if row else None
    except Exception:
        return None
