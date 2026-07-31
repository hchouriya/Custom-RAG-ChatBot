"""Conversations, messages, citations, feedback, tickets, and the CRM outbox.

Written in Phase 2 with the rest of the schema so there is one initial migration rather
than a trickle of them; the chat endpoints that use these tables arrive in Phase 3.
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
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from aegis.domain.enums import (
    AnswerStatus,
    FeedbackRating,
    MessageRole,
    Mode,
    TicketPriority,
    TicketStatus,
)
from aegis.infrastructure.database.models.base import (
    CITEXT,
    Base,
    CreatedAtMixin,
    TimestampMixin,
    fk_uuid,
    pg_enum,
    uuid_pk,
)


class ConversationModel(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index(
            "ix_conv_user",
            "user_id",
            text("last_message_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index(
            "ix_conv_guest",
            "guest_session_id",
            postgresql_where=text("guest_session_id IS NOT NULL"),
        ),
    )

    id: Mapped[UUID] = uuid_pk()
    # Nullable for guests, who have a session but no account.
    user_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    guest_session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[Mode] = mapped_column(pg_enum(Mode, "assistant_mode"), nullable=False)
    collection_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]")
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_message_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), nullable=True
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    is_pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MessageModel(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conv", "conversation_id", "created_at"),)

    id: Mapped[UUID] = uuid_pk()
    conversation_id: Mapped[UUID] = fk_uuid("conversations.id")
    # Regeneration creates a sibling branch rather than overwriting, so history stays honest
    # about what the user actually saw.
    parent_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    role: Mapped[MessageRole] = mapped_column(pg_enum(MessageRole, "message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[AnswerStatus | None] = mapped_column(
        pg_enum(AnswerStatus, "answer_status"), nullable=True
    )
    refusal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_grounded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MessageCitationModel(Base):
    """A validated pointer from an answer marker to the exact text that supported it.

    ``ON DELETE RESTRICT`` on the chunk, version, and document is deliberate and
    load-bearing: it makes it *impossible* to delete content that a stored answer cites. A
    cascade here would silently corrupt audit history, which is the one thing a citation
    exists to protect.
    """

    __tablename__ = "message_citations"
    __table_args__ = (
        UniqueConstraint("message_id", "marker", name="uq_message_citations_message_id_marker"),
        Index("ix_citations_chunk", "chunk_id"),
        Index("ix_citations_doc", "document_id"),
    )

    id: Mapped[UUID] = uuid_pk()
    message_id: Mapped[UUID] = fk_uuid("messages.id")
    chunk_id: Mapped[UUID] = fk_uuid("chunks.id", ondelete="RESTRICT")
    document_id: Mapped[UUID] = fk_uuid("documents.id", ondelete="RESTRICT")
    version_id: Mapped[UUID] = fk_uuid("document_versions.id", ondelete="RESTRICT")
    marker: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    quote_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_dense: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_sparse: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_fused: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_rerank: Mapped[float | None] = mapped_column(Float, nullable=True)
    # False = retrieved and offered to the model, but not cited. Kept because "what did the
    # model ignore" is the question that explains a bad answer.
    was_used: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class FeedbackModel(Base, CreatedAtMixin):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uq_feedback_message_id_user_id"),
    )

    id: Mapped[UUID] = uuid_pk()
    message_id: Mapped[UUID] = fk_uuid("messages.id")
    user_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        pg_enum(FeedbackRating, "feedback_rating"), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)


class SupportTicketModel(Base, TimestampMixin):
    __tablename__ = "support_tickets"
    __table_args__ = (Index("ix_tickets_status", "status", "priority", text("created_at DESC")),)

    id: Mapped[UUID] = uuid_pk()
    conversation_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    message_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    requester_name: Mapped[str] = mapped_column(Text, nullable=False)
    requester_email: Mapped[str] = mapped_column(CITEXT(), nullable=False)
    requester_phone: Mapped[str | None] = mapped_column(Text, nullable=True)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # Snapshot, so the ticket survives deletion of the conversation it came from.
    transcript: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    priority: Mapped[TicketPriority] = mapped_column(
        pg_enum(TicketPriority, "ticket_priority"), nullable=False, server_default=text("'normal'")
    )
    status: Mapped[TicketStatus] = mapped_column(
        pg_enum(TicketStatus, "ticket_status"), nullable=False, server_default=text("'open'")
    )
    assigned_to: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    crm_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    crm_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OutboxEventModel(Base, CreatedAtMixin):
    """Transactional outbox.

    The ticket row and its future CRM delivery commit together. Calling a CRM inside the
    request transaction gives either lost tickets (call fails after commit) or phantom
    tickets (commit fails after the call); the outbox removes both.
    """

    __tablename__ = "outbox_events"
    __table_args__ = (
        Index("ix_outbox_pending", "next_retry_at", postgresql_where=text("published_at IS NULL")),
    )

    id: Mapped[UUID] = uuid_pk()
    aggregate_type: Mapped[str] = mapped_column(Text, nullable=False)
    aggregate_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("0"))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


__all__ = [
    "ConversationModel",
    "FeedbackModel",
    "MessageCitationModel",
    "MessageModel",
    "OutboxEventModel",
    "SupportTicketModel",
]
