"""Domain entities.

Pydantic models with ``from_attributes=True``, so a repository maps a SQLAlchemy row with
``User.model_validate(row)`` and no hand-written field assignment. That keeps the
dependency rule (services never see an ORM object) affordable — the usual objection to
ports and adapters is the mapping boilerplate, and this removes most of it.

Entities carry behaviour where the behaviour is about the entity itself
(``User.is_locked``, ``DocumentVersion.is_searchable``). Anything that needs two
aggregates or a policy decision belongs in ``domain.policies`` or a service.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from aegis.domain.enums import (
    AnswerStatus,
    ChunkType,
    IngestStatus,
    JobStatus,
    MessageRole,
    Mode,
    Permission,
    PrincipalType,
    Role,
    Visibility,
    VisibilityLevel,
)


class Entity(BaseModel):
    """Base for persisted entities."""

    model_config = ConfigDict(from_attributes=True, frozen=False, extra="forbid")


# ── Identity ────────────────────────────────────────────────────────────────


class Department(Entity):
    id: UUID
    name: str
    slug: str
    parent_id: UUID | None = None
    path: str
    created_at: datetime | None = None

    def contains(self, other_path: str) -> bool:
        return other_path == self.path or other_path.startswith(self.path + ".")


class User(Entity):
    id: UUID
    email: str
    full_name: str
    role: Role
    department_id: UUID | None = None
    department_path: str | None = None
    is_active: bool = True
    must_change_password: bool = False
    has_mfa: bool = False
    permission_epoch: int = 0
    failed_logins: int = 0
    locked_until: datetime | None = None
    last_login_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def is_locked(self) -> bool:
        return self.locked_until is not None and self.locked_until > datetime.now(UTC)

    @property
    def can_authenticate(self) -> bool:
        return self.is_active and not self.is_locked

    @property
    def default_mode(self) -> Mode:
        return Mode.INTERNAL if self.role.is_internal else Mode.CUSTOMER

    @property
    def allowed_modes(self) -> tuple[Mode, ...]:
        """Internal principals may use either assistant; customers only theirs.

        The reverse direction is never allowed, which is why this is a property of the
        role rather than a request parameter.
        """
        return (Mode.INTERNAL, Mode.CUSTOMER) if self.role.is_internal else (Mode.CUSTOMER,)


class UserCredentials(Entity):
    """Secret-bearing projection of a user, loaded only by the auth service.

    Separate from :class:`User` so that a password hash or TOTP secret cannot reach a
    response model by accident — the type that leaves the auth boundary simply does not
    have the fields.
    """

    id: UUID
    email: str
    password_hash: str | None = None
    mfa_secret: str | None = None
    recovery_code_hashes: list[str] = Field(default_factory=list)


class ApiKey(Entity):
    id: UUID
    name: str
    prefix: str
    role: Role
    scopes: list[str] = Field(default_factory=list)
    created_by: UUID | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def is_valid(self) -> bool:
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > datetime.now(UTC)


class RolePermissions(Entity):
    role: Role
    permissions: set[Permission]


# ── Collections and documents ───────────────────────────────────────────────


class Collection(Entity):
    id: UUID
    name: str
    slug: str
    description: str | None = None
    mode: Mode
    default_visibility: Visibility
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    vector_backend: str = "qdrant"
    vector_namespace: str
    chunk_strategy: str = "adaptive"
    chunk_size: int = 800
    chunk_overlap: int = 120
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentAclGrant(Entity):
    id: UUID
    document_id: UUID
    principal_type: PrincipalType
    principal_role: Role | None = None
    principal_id: UUID | None = None
    include_subtree: bool = True
    expires_at: datetime | None = None
    granted_by: UUID | None = None
    created_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.expires_at is None or self.expires_at > datetime.now(UTC)


class DocumentVersion(Entity):
    id: UUID
    document_id: UUID
    version_no: int
    storage_uri: str
    original_filename: str
    mime_type: str
    size_bytes: int
    checksum_sha256: bytes
    page_count: int | None = None
    extracted_chars: int | None = None
    used_ocr: bool = False
    parser: str | None = None
    chunk_strategy: str | None = None
    embedding_model: str | None = None
    chunk_count: int = 0
    token_count: int = 0
    status: IngestStatus = IngestStatus.PENDING
    error_message: str | None = None
    injection_flags: int = 0
    change_note: str | None = None
    created_by: UUID | None = None
    created_at: datetime | None = None
    indexed_at: datetime | None = None
    superseded_at: datetime | None = None

    @property
    def is_searchable(self) -> bool:
        return self.status.is_searchable

    @property
    def checksum_hex(self) -> str:
        return self.checksum_sha256.hex()


class Document(Entity):
    id: UUID
    collection_id: UUID
    title: str
    description: str | None = None
    source_type: str = "upload"
    source_ref: str | None = None
    visibility: Visibility
    department_id: UUID | None = None
    department_path: str | None = None
    language: str | None = None
    owner_id: UUID | None = None
    active_version_id: UUID | None = None
    # Calendar dates, not instants: "this policy applies from 1 April" is a business fact
    # that must not shift with the reader's timezone.
    effective_from: date | None = None
    expires_at: date | None = None
    is_archived: bool = False
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None

    @property
    def visibility_level(self) -> VisibilityLevel:
        return self.visibility.level

    @property
    def is_retrievable(self) -> bool:
        """Whether this document should contribute chunks to a search.

        Archived, deleted, expired, and never-indexed documents are all excluded, and the
        same predicate is mirrored in the vector payload so the filter can apply it.
        """
        if self.deleted_at is not None or self.is_archived or self.active_version_id is None:
            return False
        today = datetime.now(UTC).date()
        if self.expires_at is not None and self.expires_at <= today:
            return False
        return not (self.effective_from is not None and self.effective_from > today)


class Chunk(Entity):
    id: UUID
    document_id: UUID
    version_id: UUID
    collection_id: UUID
    ordinal: int
    content: str
    content_hash: bytes
    token_count: int
    chunk_type: ChunkType = ChunkType.TEXT
    page_from: int | None = None
    page_to: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: dict[str, Any] | None = None
    context_header: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    language: str | None = None
    visibility_level: int = VisibilityLevel.INTERNAL
    department_path: str | None = None
    vector_point_id: UUID | None = None
    embedding_model: str | None = None
    indexed_at: datetime | None = None
    injection_flag: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None

    @property
    def embed_text(self) -> str:
        """Text actually sent to the embedding model.

        The contextual header is prepended here and nowhere else, so the embedded text and
        the displayed text cannot drift apart. The header is excluded from the quote shown
        in a citation — a user should see their document, not our indexing scaffolding.
        """
        return (
            f"{self.context_header}\n---\n{self.content}" if self.context_header else self.content
        )


class IngestJob(Entity):
    id: UUID
    version_id: UUID
    job_type: str
    status: JobStatus = JobStatus.QUEUED
    stage: IngestStatus | None = None
    attempts: int = 0
    max_attempts: int = 3
    idempotency_key: str | None = None
    error_message: str | None = None
    error_class: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    queued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worker_id: str | None = None

    @property
    def can_retry(self) -> bool:
        return self.status is JobStatus.FAILED and self.attempts < self.max_attempts


class UploadTicket(Entity):
    """Server-issued permission to place one object in storage.

    Exists so the API can validate intent (extension, size, quota, collection access)
    before any bytes move, while the bytes themselves go browser → storage directly and
    never occupy an API process.
    """

    upload_id: UUID
    collection_id: UUID
    storage_key: str
    url: str
    fields: dict[str, str] = Field(default_factory=dict)
    max_bytes: int
    expires_at: datetime
    declared_mime: str
    original_filename: str


# ── Conversations ───────────────────────────────────────────────────────────


class Conversation(Entity):
    id: UUID
    mode: Mode
    user_id: UUID | None = None
    guest_session_id: str | None = None
    collection_ids: list[UUID] = Field(default_factory=list)
    title: str | None = None
    summary: str | None = None
    summary_upto_message_id: UUID | None = None
    message_count: int = 0
    total_tokens: int = 0
    is_pinned: bool = False
    is_archived: bool = False
    created_at: datetime | None = None
    last_message_at: datetime | None = None
    deleted_at: datetime | None = None

    def belongs_to(self, *, user_id: UUID | None, guest_session_id: str | None) -> bool:
        """Ownership check for every conversation-scoped endpoint.

        Conversations are private to their principal with no sharing model, so this is a
        simple equality — and being explicit about it here keeps the check from being
        re-derived, slightly differently, in each router.
        """
        if self.user_id is not None:
            return user_id is not None and self.user_id == user_id
        return guest_session_id is not None and self.guest_session_id == guest_session_id


class Message(Entity):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    parent_id: UUID | None = None
    status: AnswerStatus | None = None
    refusal_reason: str | None = None
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    ttft_ms: int | None = None
    confidence: float | None = None
    is_grounded: bool | None = None
    finish_reason: str | None = None
    created_at: datetime | None = None
    edited_at: datetime | None = None


class MessageCitation(Entity):
    id: UUID
    message_id: UUID
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    marker: int
    rank: int
    quote: str | None = None
    quote_start: int | None = None
    quote_end: int | None = None
    page: int | None = None
    score_dense: float | None = None
    score_sparse: float | None = None
    score_fused: float | None = None
    score_rerank: float | None = None
    was_used: bool = True


class AuditEntry(Entity):
    id: UUID
    actor_id: UUID | None = None
    actor_role: Role | None = None
    actor_ip: str | None = None
    action: str
    resource_type: str
    resource_id: UUID | None = None
    outcome: str = "success"
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    request_id: str | None = None
    user_agent: str | None = None
    created_at: datetime | None = None


class IndexDiscrepancy(Entity):
    """Detected divergence between PostgreSQL and the vector store."""

    id: UUID
    collection_id: UUID
    chunk_id: UUID | None = None
    kind: str
    details: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime | None = None
    repaired_at: datetime | None = None


class EmailAddress(BaseModel):
    """Validated email, for the few places that accept one from outside."""

    value: EmailStr
