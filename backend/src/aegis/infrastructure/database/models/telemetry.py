"""Query traces, audit logs, prompt templates, settings, and evaluation records.

``query_traces`` and ``audit_logs`` are range-partitioned by month in the migration. The
models below describe them as ordinary tables — partitioning is a physical property that
SQLAlchemy does not express and that queries do not need to know about. Both carry a
composite primary key including ``created_at`` because PostgreSQL requires the partition
key to be part of any unique constraint.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, INET, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from aegis.domain.enums import AnswerStatus, EvalMetric, Mode, Role
from aegis.infrastructure.database.models.base import Base, CreatedAtMixin, pg_enum, uuid_pk


class QueryTraceModel(Base):
    """One row per question, whether it succeeded, refused, or failed.

    This is the product-analytics and quality-forensics record, distinct from an OTel span:
    observability backends retain days and are optimized for spans, while "every question
    last quarter whose top score was under 0.4, by department" is a SQL workload.
    """

    __tablename__ = "query_traces"
    __table_args__ = (
        Index("ix_traces_created", text("created_at DESC")),
        Index("ix_traces_status", "answer_status", text("created_at DESC")),
        Index("ix_traces_user", "user_id", text("created_at DESC")),
        Index("ix_traces_message", "message_id"),
        {"postgresql_partition_by": "RANGE (created_at)"},
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=text("now()")
    )
    message_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    conversation_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    user_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    role: Mapped[Role | None] = mapped_column(pg_enum(Role, "user_role"), nullable=True)
    mode: Mapped[Mode] = mapped_column(pg_enum(Mode, "assistant_mode"), nullable=False)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    rewritten_queries: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # The exact ACL filter that was applied. Recording it is what makes "why could this user
    # see that?" answerable after the fact instead of a reconstruction exercise.
    filter_applied: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    candidates: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    context_chunk_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]")
    )
    context_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage_latency_ms: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    # Promoted out of `candidates` so dashboards never have to open the jsonb.
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_top5_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_status: Mapped[AnswerStatus | None] = mapped_column(
        pg_enum(AnswerStatus, "answer_status"), nullable=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    guardrail_flags: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    error_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLogModel(Base):
    """Append-only record of every mutation and every authorization denial.

    The application database role is granted ``INSERT`` and ``SELECT`` only. An audit table
    the application can rewrite is not an audit table.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_actor", "actor_id", text("created_at DESC")),
        Index("ix_audit_resource", "resource_type", "resource_id", text("created_at DESC")),
        Index("ix_audit_action", "action", text("created_at DESC")),
        {"postgresql_partition_by": "RANGE (created_at)"},
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=text("now()")
    )
    actor_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    actor_role: Mapped[Role | None] = mapped_column(pg_enum(Role, "user_role"), nullable=True)
    actor_ip: Mapped[str | None] = mapped_column(INET, nullable=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[str] = mapped_column(Text, nullable=False)
    resource_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    outcome: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'success'"))
    before_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)


class PromptTemplateModel(Base, CreatedAtMixin):
    """Versioned prompts with exactly one active row per ``(key, mode)``.

    Prompt edits change answer quality more than most code changes and are the least
    reviewed change in a RAG system. Versioning them here means an eval run can be pinned to
    a prompt version and a regression rolled back without a deploy.
    """

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint("key", "version", name="uq_prompt_templates_key_version"),
        Index(
            "ux_prompt_active",
            "key",
            text("coalesce(mode::text, 'internal')"),
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[UUID] = uuid_pk()
    key: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    mode: Mapped[Mode | None] = mapped_column(pg_enum(Mode, "assistant_mode"), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class SettingModel(Base):
    """Runtime override for a tunable. A row here beats the environment default."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)
    updated_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )


class EvalDatasetModel(Base, CreatedAtMixin):
    __tablename__ = "eval_datasets"

    id: Mapped[UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    mode: Mapped[Mode] = mapped_column(pg_enum(Mode, "assistant_mode"), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvalCaseModel(Base):
    __tablename__ = "eval_cases"
    __table_args__ = (Index("ix_eval_cases_dataset", "dataset_id"),)

    id: Mapped[UUID] = uuid_pk()
    dataset_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("eval_datasets.id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_document_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]")
    )
    expected_pages: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, server_default=text("'{}'::integer[]")
    )
    # The same question is evaluated under different principals, which is how the permission
    # boundary is tested as a quality property and not only as a security one.
    as_role: Mapped[Role] = mapped_column(
        pg_enum(Role, "user_role"), nullable=False, server_default=text("'internal_employee'")
    )
    must_refuse: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )


class EvalRunModel(Base):
    __tablename__ = "eval_runs"

    id: Mapped[UUID] = uuid_pk()
    dataset_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("eval_datasets.id", ondelete="CASCADE"), nullable=False
    )
    git_sha: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Without a config snapshot, comparing two runs is meaningless: a metric moved and you
    # cannot tell whether the cause was code, configuration, or a prompt edit.
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'running'"))
    summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class EvalResultModel(Base):
    __tablename__ = "eval_results"
    __table_args__ = (
        UniqueConstraint("run_id", "case_id", "metric", name="uq_eval_results_run_case_metric"),
    )

    id: Mapped[UUID] = uuid_pk()
    run_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("eval_cases.id", ondelete="CASCADE"), nullable=False
    )
    metric: Mapped[EvalMetric] = mapped_column(pg_enum(EvalMetric, "eval_metric"), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunk_ids: Mapped[list[UUID] | None] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=True
    )
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


__all__ = [
    "AuditLogModel",
    "EvalCaseModel",
    "EvalDatasetModel",
    "EvalResultModel",
    "EvalRunModel",
    "PromptTemplateModel",
    "QueryTraceModel",
    "SettingModel",
]
