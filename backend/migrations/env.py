"""Alembic environment.

Two decisions worth stating:

1. **The URL comes from Settings, not alembic.ini.** Migrations then cannot possibly run
   against a different database than the application reads, and there is no second copy of
   credentials to keep in sync.
2. **Autogenerate is advisory here, not authoritative.** ``ltree`` columns, partitioned
   tables, generated ``tsvector`` columns, triggers, and the enum types are all things
   Alembic either cannot see or renders wrongly, so the initial revision is hand-written and
   the comparison hooks below exist to stop autogenerate from proposing to "fix" them.
"""

from __future__ import annotations

import asyncio
import re
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from aegis.core.config import get_settings
from aegis.infrastructure.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Objects the application does not own and must never drop.
SKIP_TABLES = {"spatial_ref_sys"}
# Partition children of query_traces / audit_logs: `..._y2026m07` and `..._default`. They
# exist in the database but not in the metadata, by design.
PARTITION_CHILD = re.compile(r"_(y\d{4}m\d{2}|default)$")


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Filter autogenerate.

    Monthly partitions of ``query_traces`` and ``audit_logs`` appear in the database but not
    in the metadata. Without this filter every autogenerate run would helpfully propose to
    drop last month's telemetry.
    """
    if type_ == "table":
        if name in SKIP_TABLES:
            return False
        if reflected and name and PARTITION_CHILD.search(name):
            return False
    return True


def process_revision_directives(context_: Any, revision: Any, directives: list[Any]) -> None:
    """Drop empty revisions instead of committing a no-op file."""
    if getattr(config.cmd_opts, "autogenerate", False) and directives:
        script = directives[0]
        if script.upgrade_ops is not None and script.upgrade_ops.is_empty():
            directives[:] = []


def _configure(connection: Connection | None = None, **extra: Any) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        process_revision_directives=process_revision_directives,
        compare_type=True,
        compare_server_default=True,
        # One transaction for the whole upgrade: a migration that fails halfway leaves the
        # schema untouched rather than in a state nobody has ever tested against.
        transaction_per_migration=False,
        render_as_batch=False,
        **extra,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout for review, used by ``make migrate-sql``.

    Production changes are reviewed as SQL before they are applied; ``alembic upgrade head``
    against a live database is not the first time anyone sees the statements.
    """
    _configure(
        url=settings.database_url.replace("+asyncpg", ""),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
