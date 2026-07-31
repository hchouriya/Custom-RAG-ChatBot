"""Collections, documents, versions, ACL grants, chunks, and ingestion jobs."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, BIGINT, JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aegis.domain.enums import (
    ChunkType,
    IngestStatus,
    JobStatus,
    Mode,
    PrincipalType,
    Role,
    Visibility,
)
from aegis.infrastructure.database.models.base import (
    CITEXT,
    LTREE,
    Base,
    CreatedAtMixin,
    TimestampMixin,
    fk_uuid,
    pg_enum,
    uuid_pk,
)


class CollectionModel(Base, TimestampMixin):
    """A vector namespace with a fixed embedding model.

    ``embedding_model``/``embedding_dim`` are immutable once chunks exist: mixing dimensions
    in one namespace is unrecoverable, so a model upgrade creates a new collection and
    backfills (docs/architecture/09 §4).
    """

    __tablename__ = "collections"

    id: Mapped[UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(CITEXT(), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[Mode] = mapped_column(pg_enum(Mode, "assistant_mode"), nullable=False)
    default_visibility: Mapped[Visibility] = mapped_column(
        pg_enum(Visibility, "visibility"), nullable=False, server_default=text("'internal'")
    )
    embedding_provider: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_backend: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'qdrant'")
    )
    vector_namespace: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_strategy: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'adaptive'")
    )
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("800"))
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("120"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class TagModel(Base):
    __tablename__ = "tags"

    id: Mapped[UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(CITEXT(), nullable=False, unique=True)
    color: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentTagModel(Base):
    __tablename__ = "document_tags"
    # Reverse lookup for "documents with tag X"; the composite primary key only serves the
    # forward direction.
    __table_args__ = (Index("ix_document_tags_tag_id", "tag_id"),)

    document_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )


class DocumentModel(Base, TimestampMixin):
    """Stable document identity. Content lives in :class:`DocumentVersionModel`.

    ``visibility_level`` is denormalized from ``visibility`` by a trigger, and mirrored again
    into ``chunks`` and the vector payload. Filtering must happen *inside* the search rather
    than as a join afterwards, which is the whole reason for the duplication; the trigger and
    the nightly reconciler are what keep the copies honest.
    """

    __tablename__ = "documents"
    __table_args__ = (
        Index(
            "ix_documents_retrievable",
            "collection_id",
            "visibility_level",
            postgresql_where=text(
                "deleted_at IS NULL AND NOT is_archived AND active_version_id IS NOT NULL"
            ),
        ),
        Index("ix_documents_dept", "department_path", postgresql_using="gist"),
        Index(
            "ix_documents_title_trgm",
            "title",
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"},
        ),
        Index("ix_documents_owner", "owner_id"),
    )

    id: Mapped[UUID] = uuid_pk()
    collection_id: Mapped[UUID] = fk_uuid("collections.id", ondelete="RESTRICT")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'upload'"))
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    visibility: Mapped[Visibility] = mapped_column(
        pg_enum(Visibility, "visibility"), nullable=False
    )
    visibility_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    department_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )
    department_path: Mapped[str | None] = mapped_column(LTREE, nullable=True)
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # FK to document_versions is added by the migration after that table exists.
    active_version_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    expires_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    collection: Mapped[CollectionModel] = relationship(lazy="noload")
    tags: Mapped[list[TagModel]] = relationship(secondary="document_tags", lazy="selectin")


class DocumentVersionModel(Base):
    """Immutable content. A replacement inserts a new row; nothing is ever mutated in place."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "version_no", name="uq_document_versions_document_id_version_no"
        ),
        # Re-uploading identical bytes is a no-op rather than a duplicate version.
        UniqueConstraint(
            "document_id",
            "checksum_sha256",
            name="uq_document_versions_document_id_checksum_sha256",
        ),
        Index("ix_versions_status", "status"),
        Index("ix_versions_document", "document_id"),
    )

    id: Mapped[UUID] = uuid_pk()
    document_id: Mapped[UUID] = fk_uuid("documents.id")
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    # Sniffed from magic bytes, never the client-declared Content-Type.
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BIGINT, nullable=False)
    checksum_sha256: Mapped[bytes] = mapped_column(nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    used_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    parser: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_strategy: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    status: Mapped[IngestStatus] = mapped_column(
        pg_enum(IngestStatus, "ingest_status"), nullable=False, server_default=text("'pending'")
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    injection_flags: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    change_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentAclModel(Base, CreatedAtMixin):
    """Explicit read grant, overriding level and department for one document."""

    __tablename__ = "document_acl"
    __table_args__ = (
        CheckConstraint(
            "(principal_type = 'role' AND principal_role IS NOT NULL AND principal_id IS NULL) "
            "OR (principal_type <> 'role' AND principal_id IS NOT NULL AND principal_role IS NULL)",
            name="principal_shape",
        ),
        Index("ix_acl_document", "document_id"),
        Index("ix_acl_user", "principal_id", postgresql_where=text("principal_type = 'user'")),
    )

    id: Mapped[UUID] = uuid_pk()
    document_id: Mapped[UUID] = fk_uuid("documents.id")
    principal_type: Mapped[PrincipalType] = mapped_column(
        pg_enum(PrincipalType, "principal_type"), nullable=False
    )
    principal_role: Mapped[Role | None] = mapped_column(pg_enum(Role, "user_role"), nullable=True)
    principal_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    include_subtree: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    granted_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ChunkModel(Base, CreatedAtMixin):
    """Retrievable unit of text, with the locators a citation needs.

    Text is stored here and not only in the vector payload for three reasons: BM25 needs the
    ``tsvector``; the citation drawer needs exact offsets to highlight; and a reindex after a
    model upgrade must be able to read chunks without re-parsing originals.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "ordinal", name="uq_chunks_version_id_ordinal"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_version", "version_id"),
        Index("ix_chunks_doc", "document_id"),
        Index("ix_chunks_hash", "content_hash"),
        Index("ix_chunks_unindexed", "collection_id", postgresql_where=text("indexed_at IS NULL")),
        Index("ix_chunks_keywords", "keywords", postgresql_using="gin"),
        Index("ix_chunks_acl", "collection_id", "visibility_level"),
        Index("ix_chunks_dept", "department_path", postgresql_using="gist"),
    )

    id: Mapped[UUID] = uuid_pk()
    document_id: Mapped[UUID] = fk_uuid("documents.id")
    version_id: Mapped[UUID] = fk_uuid("document_versions.id")
    collection_id: Mapped[UUID] = fk_uuid("collections.id", ondelete="RESTRICT")
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[bytes] = mapped_column(nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[ChunkType] = mapped_column(
        pg_enum(ChunkType, "chunk_type"), nullable=False, server_default=text("'text'")
    )
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heading_path: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    context_header: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Denormalized from documents by trigger; the vector payload carries the same values.
    visibility_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    department_path: Mapped[str | None] = mapped_column(LTREE, nullable=True)
    vector_point_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    injection_flag: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))", persisted=True),
        nullable=True,
    )


class IngestJobModel(Base):
    __tablename__ = "ingest_jobs"
    __table_args__ = (
        Index("ix_jobs_status", "status", "queued_at"),
        Index("ix_jobs_version", "version_id"),
        UniqueConstraint("idempotency_key", name="uq_ingest_jobs_idempotency_key"),
    )

    id: Mapped[UUID] = uuid_pk()
    version_id: Mapped[UUID] = fk_uuid("document_versions.id")
    job_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        pg_enum(JobStatus, "job_status"), nullable=False, server_default=text("'queued'")
    )
    stage: Mapped[IngestStatus | None] = mapped_column(
        pg_enum(IngestStatus, "ingest_status"), nullable=True
    )
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("3")
    )
    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(Text, nullable=True)


class IndexDiscrepancyModel(Base):
    """Divergence between PostgreSQL and the vector store, written by the reconciler.

    The count is also a health signal: it should sit at zero, and a spike means the reindex
    path is broken rather than that a document is unusual.
    """

    __tablename__ = "index_discrepancies"
    __table_args__ = (
        Index("ix_discrepancies_open", "detected_at", postgresql_where=text("repaired_at IS NULL")),
    )

    id: Mapped[UUID] = uuid_pk()
    collection_id: Mapped[UUID] = fk_uuid("collections.id")
    chunk_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    repaired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


__all__ = [
    "ChunkModel",
    "CollectionModel",
    "DocumentAclModel",
    "DocumentModel",
    "DocumentTagModel",
    "DocumentVersionModel",
    "IndexDiscrepancyModel",
    "IngestJobModel",
    "TagModel",
]
