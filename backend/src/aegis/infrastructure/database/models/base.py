"""Declarative base, shared column types, and mixins.

Two things here are load-bearing beyond convenience.

**Enum mapping uses ``values_callable``.** Without it SQLAlchemy sends Python enum *names*
(``INTERNAL_EMPLOYEE``) where PostgreSQL expects the *labels* it was created with
(``internal_employee``). The failure appears as a cryptic invalid-input-value error at
runtime rather than at import, so the helper below exists to make the correct form the only
form available.

**``create_type=False`` on every enum.** Types are created once by the initial migration.
Letting the ORM emit ``CREATE TYPE`` means two API replicas booting concurrently race each
other, and one loses with a duplicate-object error.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy import DateTime, Enum, LargeBinary, MetaData, Text, func, text
from sqlalchemy.dialects.postgresql import CITEXT as PgCitext
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator, UserDefinedType

# Explicit, deterministic constraint names. Without a naming convention, Alembic
# autogenerate produces unnamed constraints that later revisions cannot drop by name.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map: ClassVar[dict[Any, Any]] = {
        datetime: DateTime(timezone=True),
        bytes: LargeBinary,
    }

    def __repr__(self) -> str:
        identifier = getattr(self, "id", None)
        return f"<{type(self).__name__} {identifier}>"


class LTREE(UserDefinedType[str]):
    """PostgreSQL ``ltree``, for department hierarchies.

    ACL resolution runs on every query, and ``ltree`` with a GiST index turns "is this
    document inside my department subtree" into one indexed operator instead of a recursive
    CTE. SQLAlchemy has no built-in mapping, so this thin wrapper provides one.
    """

    cache_ok = True

    def get_col_spec(self, **_kw: Any) -> str:
        return "LTREE"

    def bind_processor(self, dialect: Any) -> Any:
        def process(value: str | None) -> str | None:
            return value

        return process

    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        def process(value: Any) -> str | None:
            return str(value) if value is not None else None

        return process


class CITEXT(TypeDecorator[str]):
    """Case-insensitive text, used for email and slug columns.

    Case-insensitive *uniqueness* is the point: ``Alice@x.com`` and ``alice@x.com`` must be
    the same account, and enforcing that with ``lower()`` expression indexes everywhere is
    easy to forget in exactly one place. Falls back to ``TEXT`` on non-PostgreSQL dialects
    so that metadata introspection in tooling does not fail.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PgCitext())
        return dialect.type_descriptor(Text())


def pg_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Map a ``StrEnum`` to an existing PostgreSQL enum type, by value."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=True,
        create_type=False,
        validate_strings=True,
        values_callable=lambda e: [member.value for member in e],
    )


def uuid_pk() -> Mapped[UUID]:
    """Primary key column. Ids are generated in Python (UUIDv7, see ``core.ids``).

    Application-side generation lets a worker build a whole object graph — document,
    version, chunks, vector point ids — before touching the database, which a
    database-generated default would force into a round trip per row.
    """
    return mapped_column(PgUUID(as_uuid=True), primary_key=True)


def fk_uuid(target: str, *, nullable: bool = False, ondelete: str = "CASCADE") -> Mapped[UUID]:
    from sqlalchemy import ForeignKey

    return mapped_column(
        PgUUID(as_uuid=True), ForeignKey(target, ondelete=ondelete), nullable=nullable
    )


class TimestampMixin:
    """``created_at`` / ``updated_at``, maintained by the database.

    Server-side defaults rather than Python defaults so that a direct SQL fix, a migration
    backfill, or a bulk insert cannot leave them null.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def utcnow() -> datetime:
    """Timezone-aware now. ``datetime.utcnow`` is naive and banned by the ``DTZ`` lint rule."""
    return datetime.now(UTC)


NOW = text("now()")
