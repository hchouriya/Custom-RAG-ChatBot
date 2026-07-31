"""Pydantic request/response models aligned with docs/architecture/04-api-design.md."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from aegis.domain.enums import (
    AnswerStatus,
    FeedbackRating,
    IngestStatus,
    MessageRole,
    Mode,
    Role,
    Visibility,
)


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


# ── Auth ────────────────────────────────────────────────────────────────────


class LoginRequest(APIModel):
    email: EmailStr
    password: str
    mode: Mode | None = None
    totp_code: str | None = None


class RefreshRequest(APIModel):
    refresh_token: str
    mode: Mode | None = None


class LogoutRequest(APIModel):
    refresh_token: str | None = None


class PasswordChangeRequest(APIModel):
    current_password: str
    new_password: str


class TokenResponse(APIModel):
    access_token: str
    access_expires_at: datetime
    refresh_token: str | None = None
    refresh_expires_at: datetime | None = None
    token_type: str = "bearer"  # noqa: S105 - OAuth token_type claim, not a secret
    mode: Mode
    is_guest: bool = False


class MeResponse(APIModel):
    id: UUID | None = None
    email: str | None = None
    full_name: str | None = None
    role: Role
    mode: Mode
    permissions: list[str]
    allowed_modes: list[Mode]
    must_change_password: bool = False
    is_guest: bool = False
    department_id: UUID | None = None


# ── Chat ────────────────────────────────────────────────────────────────────


class ConversationCreate(APIModel):
    collection_ids: list[UUID] = Field(default_factory=list)
    title: str | None = None


class ConversationUpdate(APIModel):
    title: str | None = None
    is_pinned: bool | None = None
    is_archived: bool | None = None


class ConversationOut(APIModel):
    id: UUID
    mode: Mode
    title: str | None = None
    collection_ids: list[UUID] = Field(default_factory=list)
    message_count: int = 0
    is_pinned: bool = False
    is_archived: bool = False
    created_at: datetime | None = None
    last_message_at: datetime | None = None


class MessageOut(APIModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    parent_id: UUID | None = None
    status: AnswerStatus | None = None
    model: str | None = None
    confidence: float | None = None
    is_grounded: bool | None = None
    created_at: datetime | None = None


class CitationOut(APIModel):
    marker: int
    document_id: UUID
    document_title: str | None = None
    version_id: UUID | None = None
    version_no: int | None = None
    page: int | None = None
    section: str | None = None
    quote: str | None = None
    score_rerank: float | None = None
    was_used: bool = True


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)
    citations: dict[str, list[CitationOut]] = Field(default_factory=dict)


class ChatFilters(APIModel):
    tags: list[str] | None = None
    document_ids: list[UUID] | None = None


class ChatOptions(APIModel):
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=0.3)
    max_citations: int = Field(default=8, ge=1, le=20)


class MessageCreate(APIModel):
    content: str
    stream: bool = True
    collection_ids: list[UUID] = Field(default_factory=list)
    filters: ChatFilters | None = None
    options: ChatOptions | None = None


class FeedbackRequest(APIModel):
    rating: FeedbackRating
    reason: str | None = None
    comment: str | None = None


class SuggestionsResponse(APIModel):
    suggestions: list[str]


# ── Documents ───────────────────────────────────────────────────────────────


class UploadTicketRequest(APIModel):
    filename: str
    size_bytes: int = Field(gt=0)
    declared_mime: str
    collection_id: UUID


class UploadTicketResponse(APIModel):
    upload_id: UUID
    url: str
    fields: dict[str, str] = Field(default_factory=dict)
    expires_at: datetime
    max_bytes: int


class DocumentRegisterRequest(APIModel):
    upload_id: UUID
    title: str
    visibility: Visibility
    description: str | None = None
    department_id: UUID | None = None
    tags: list[str] = Field(default_factory=list)
    language: str | None = None
    effective_from: date | None = None
    expires_at: date | None = None
    change_note: str | None = None
    document_id: UUID | None = None


class DocumentUpdateRequest(APIModel):
    title: str | None = None
    description: str | None = None
    visibility: Visibility | None = None
    department_id: UUID | None = None
    tags: list[str] | None = None
    language: str | None = None
    effective_from: date | None = None
    expires_at: date | None = None
    is_archived: bool | None = None


class DocumentOut(APIModel):
    id: UUID
    collection_id: UUID
    title: str
    description: str | None = None
    visibility: Visibility
    department_id: UUID | None = None
    language: str | None = None
    owner_id: UUID | None = None
    active_version_id: UUID | None = None
    tags: list[str] = Field(default_factory=list)
    effective_from: date | None = None
    expires_at: date | None = None
    is_archived: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentVersionOut(APIModel):
    id: UUID
    document_id: UUID
    version_no: int
    original_filename: str
    mime_type: str
    size_bytes: int
    status: IngestStatus
    page_count: int | None = None
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime | None = None
    indexed_at: datetime | None = None


class RegisteredDocumentResponse(APIModel):
    document_id: UUID
    version_id: UUID
    status: IngestStatus
    job_id: UUID | None = None
    poll: str


class AclGrantIn(APIModel):
    principal_type: str
    principal_role: Role | None = None
    principal_id: UUID | None = None
    include_subtree: bool = True
    expires_at: datetime | None = None


class AclReplaceRequest(APIModel):
    grants: list[AclGrantIn]


class DownloadUrlResponse(APIModel):
    url: str


class ReindexRequest(APIModel):
    force_reparse: bool = False


# ── Collections ─────────────────────────────────────────────────────────────


class CollectionCreate(APIModel):
    name: str
    slug: str
    description: str | None = None
    mode: Mode
    default_visibility: Visibility = Visibility.INTERNAL


class CollectionOut(APIModel):
    id: UUID
    name: str
    slug: str
    description: str | None = None
    mode: Mode
    default_visibility: Visibility
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    vector_namespace: str
    is_active: bool = True
    created_at: datetime | None = None


# ── Pagination / health ─────────────────────────────────────────────────────


class PageOut(APIModel):
    items: list[Any]
    next_cursor: str | None = None
    has_more: bool = False
    total_estimate: int | None = None


class HealthLiveResponse(APIModel):
    status: str = "ok"


class HealthReadyResponse(APIModel):
    status: str
    checks: dict[str, bool]
