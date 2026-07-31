"""Repository ports.

One protocol per aggregate. Repositories speak in domain entities, never ORM rows, which is
what keeps ``services`` free of SQLAlchemy and makes the in-memory fakes in
``tests/fakes`` a complete substitute rather than an approximation.

Transaction control lives in :class:`UnitOfWork`, not in individual repositories. A
repository that commits on its own makes multi-step operations impossible to make atomic —
and the operations here genuinely need it: creating a document *and* its first version *and*
enqueuing its ingest job either all happen or none do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable
from uuid import UUID

from aegis.domain.entities import (
    ApiKey,
    AuditEntry,
    Chunk,
    Collection,
    Conversation,
    Department,
    Document,
    DocumentAclGrant,
    DocumentVersion,
    IndexDiscrepancy,
    IngestJob,
    Message,
    MessageCitation,
    User,
    UserCredentials,
)
from aegis.domain.enums import (
    IngestStatus,
    JobStatus,
    Mode,
    Permission,
    Role,
    Visibility,
)
from aegis.domain.policies.permissions import PermissionOverride


@runtime_checkable
class UnitOfWork(Protocol):
    """Transaction boundary.

    Used as an async context manager: the block commits on clean exit and rolls back on any
    exception. Nothing else in the codebase calls ``commit``.
    """

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def flush(self) -> None:
        """Push pending changes to the database without committing.

        Needed when a later step in the same transaction depends on a generated value or on
        a constraint being checked early.
        """
        ...


# ── Identity ────────────────────────────────────────────────────────────────


@runtime_checkable
class UserRepository(Protocol):
    async def get(self, user_id: UUID) -> User | None: ...

    async def get_by_email(self, email: str) -> User | None: ...

    async def get_credentials(self, user_id: UUID) -> UserCredentials | None:
        """Load the secret-bearing projection.

        Separate from :meth:`get` so a password hash cannot reach a response model by
        accident: the type returned by ``get`` simply has no such field.
        """
        ...

    async def create(
        self,
        *,
        email: str,
        full_name: str,
        role: Role,
        password_hash: str | None,
        department_id: UUID | None = None,
        must_change_password: bool = False,
    ) -> User: ...

    async def update(self, user_id: UUID, **fields: Any) -> User: ...

    async def set_password(self, user_id: UUID, password_hash: str) -> None: ...

    async def set_mfa(
        self, user_id: UUID, *, secret: str | None, recovery_code_hashes: list[str] | None = None
    ) -> None: ...

    async def consume_recovery_code(self, user_id: UUID, code_hash: str) -> bool:
        """Atomically spend a recovery code. Returns whether it was valid and unused."""
        ...

    async def record_login_success(self, user_id: UUID, *, at: datetime) -> None: ...

    async def record_login_failure(
        self, user_id: UUID, *, max_failures: int, lockout_minutes: int
    ) -> User:
        """Increment the failure counter and lock the account when it reaches the limit.

        Done in the repository so the read-modify-write is a single atomic statement;
        doing it in a service would let concurrent attempts race past the threshold.
        """
        ...

    async def bump_permission_epoch(self, user_id: UUID) -> int:
        """Invalidate every issued access token for this user.

        Called on role change, department change, deactivation, and password change. The
        epoch is what makes a revocation effective in about a second instead of after the
        full token TTL.
        """
        ...

    async def list_users(
        self,
        *,
        role: Role | None = None,
        department_id: UUID | None = None,
        query: str | None = None,
        include_inactive: bool = False,
        limit: int = 50,
        cursor_created_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[User]: ...

    async def count_by_role(self) -> dict[Role, int]: ...


@runtime_checkable
class DepartmentRepository(Protocol):
    async def get(self, department_id: UUID) -> Department | None: ...

    async def get_by_slug(self, slug: str) -> Department | None: ...

    async def list_all(self) -> list[Department]: ...

    async def create(self, *, name: str, slug: str, parent_id: UUID | None) -> Department: ...

    async def subtree_paths(self, department_id: UUID) -> list[str]: ...


@runtime_checkable
class RefreshTokenRepository(Protocol):
    """Rotating refresh tokens with reuse detection.

    Only digests are stored, so a database compromise yields no usable session.
    """

    async def create(
        self,
        *,
        user_id: UUID,
        jti: UUID,
        token_hash: bytes,
        family_id: UUID,
        expires_at: datetime,
        user_agent: str | None,
        ip: str | None,
    ) -> None: ...

    async def find_active(self, token_hash: bytes) -> dict[str, Any] | None:
        """Return ``{user_id, jti, family_id, expires_at}`` for an unrevoked token."""
        ...

    async def was_used(self, token_hash: bytes) -> UUID | None:
        """Return the family id if this token was already rotated away.

        Presenting a spent token means it leaked, which is why the response is to revoke the
        whole family rather than just reject the request.
        """
        ...

    async def revoke(self, jti: UUID) -> None: ...

    async def revoke_family(self, family_id: UUID) -> int: ...

    async def revoke_all_for_user(
        self, user_id: UUID, *, except_jti: UUID | None = None
    ) -> int: ...

    async def purge_expired(self, *, before: datetime) -> int: ...


@runtime_checkable
class ApiKeyRepository(Protocol):
    async def create(
        self,
        *,
        name: str,
        prefix: str,
        key_hash: bytes,
        role: Role,
        scopes: list[str],
        created_by: UUID | None,
        expires_at: datetime | None,
    ) -> ApiKey: ...

    async def find_by_hash(self, key_hash: bytes) -> ApiKey | None: ...

    async def touch(self, key_id: UUID, *, at: datetime) -> None: ...

    async def revoke(self, key_id: UUID) -> None: ...

    async def list_all(self, *, include_revoked: bool = False) -> list[ApiKey]: ...


@runtime_checkable
class PermissionRepository(Protocol):
    async def load_matrix(self) -> dict[Role, frozenset[Permission]]:
        """Role → permissions. Empty when unseeded, which the resolver treats as a signal
        to fall back to the domain defaults rather than as "nobody may do anything"."""
        ...

    async def replace_role_permissions(self, role: Role, permissions: set[Permission]) -> None: ...

    async def overrides_for(self, user_id: UUID) -> list[PermissionOverride]: ...

    async def set_override(
        self,
        *,
        user_id: UUID,
        permission: Permission,
        allow: bool,
        granted_by: UUID | None,
        expires_at: datetime | None,
    ) -> None: ...

    async def clear_override(self, *, user_id: UUID, permission: Permission) -> None: ...

    async def seed_defaults(self, matrix: dict[Role, frozenset[Permission]]) -> None: ...


# ── Collections and documents ───────────────────────────────────────────────


@runtime_checkable
class CollectionRepository(Protocol):
    async def get(self, collection_id: UUID) -> Collection | None: ...

    async def get_by_slug(self, slug: str) -> Collection | None: ...

    async def list_for_mode(
        self, mode: Mode, *, include_inactive: bool = False
    ) -> list[Collection]: ...

    async def list_all(self, *, include_inactive: bool = False) -> list[Collection]: ...

    async def create(self, collection: Collection) -> Collection: ...

    async def update(self, collection_id: UUID, **fields: Any) -> Collection: ...

    async def document_count(self, collection_id: UUID) -> int: ...

    async def chunk_count(self, collection_id: UUID) -> int: ...


@dataclass(slots=True)
class DocumentQuery:
    """Filters for the admin document list.

    A dataclass rather than fifteen keyword arguments so the signature stays readable and
    so the API layer can build it from query parameters in one place.
    """

    collection_id: UUID | None = None
    visibility: Visibility | None = None
    department_id: UUID | None = None
    department_path: str | None = None
    owner_id: UUID | None = None
    tags: tuple[str, ...] = ()
    status: IngestStatus | None = None
    search: str | None = None
    include_archived: bool = False
    include_deleted: bool = False
    created_after: datetime | None = None
    created_before: datetime | None = None
    max_visibility_level: int | None = None
    """Applied to the *admin list* as well as to retrieval.

    An internal employee browsing the document table must not see the titles of
    confidential documents they cannot read — a title is content.
    """


@runtime_checkable
class DocumentRepository(Protocol):
    async def get(self, document_id: UUID, *, include_deleted: bool = False) -> Document | None: ...

    async def get_many(self, document_ids: Sequence[UUID]) -> list[Document]: ...

    async def create(self, document: Document) -> Document: ...

    async def update(self, document_id: UUID, **fields: Any) -> Document: ...

    async def soft_delete(self, document_id: UUID, *, at: datetime) -> None: ...

    async def set_active_version(self, document_id: UUID, version_id: UUID) -> None:
        """Atomically publish a version.

        The single statement that makes replacement zero-downtime: until it runs, the
        previous version keeps serving, and a failed ingest never reaches it.
        """
        ...

    async def list_documents(
        self,
        query: DocumentQuery,
        *,
        limit: int,
        sort_field: str = "created_at",
        descending: bool = True,
        cursor_value: Any = None,
        cursor_id: UUID | None = None,
    ) -> list[Document]: ...

    async def estimate_count(self, query: DocumentQuery) -> int:
        """Planner estimate, not ``COUNT(*)`` — see ``core.pagination``."""
        ...

    async def set_tags(self, document_id: UUID, tags: Sequence[str]) -> None: ...

    async def find_by_checksum(self, collection_id: UUID, checksum: bytes) -> Document | None: ...


@runtime_checkable
class DocumentVersionRepository(Protocol):
    async def get(self, version_id: UUID) -> DocumentVersion | None: ...

    async def list_for_document(self, document_id: UUID) -> list[DocumentVersion]: ...

    async def create(self, version: DocumentVersion) -> DocumentVersion: ...

    async def update_status(
        self,
        version_id: UUID,
        *,
        status: IngestStatus,
        error_message: str | None = None,
        indexed_at: datetime | None = None,
    ) -> None: ...

    async def update_stats(
        self,
        version_id: UUID,
        *,
        page_count: int | None = None,
        extracted_chars: int | None = None,
        chunk_count: int | None = None,
        token_count: int | None = None,
        used_ocr: bool | None = None,
        parser: str | None = None,
        chunk_strategy: str | None = None,
        embedding_model: str | None = None,
        injection_flags: int | None = None,
        checksum_sha256: bytes | None = None,
    ) -> None:
        """Record what a stage discovered.

        ``checksum_sha256`` is written by the worker rather than at registration: the ETag of
        a multipart upload is not a SHA-256, so the only trustworthy digest is the one
        computed from the bytes we actually parsed.
        """
        ...

    async def mark_superseded(self, version_id: UUID, *, at: datetime) -> None: ...

    async def next_version_no(self, document_id: UUID) -> int: ...

    async def find_by_checksum(
        self, document_id: UUID, checksum: bytes
    ) -> DocumentVersion | None: ...

    async def find_stale_pending(self, *, older_than: datetime) -> list[DocumentVersion]:
        """Versions stuck in a non-terminal state.

        Read at worker startup to re-enqueue work stranded by a lost queue message. This is
        what makes Redis-grade durability acceptable: the authoritative record of "needs
        indexing" is a row here, not a message.
        """
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    async def bulk_create(self, chunks: list[Chunk]) -> int: ...

    async def get(self, chunk_id: UUID) -> Chunk | None: ...

    async def get_many(self, chunk_ids: Sequence[UUID]) -> list[Chunk]: ...

    async def list_for_version(
        self, version_id: UUID, *, limit: int = 100, offset: int = 0
    ) -> list[Chunk]: ...

    async def list_for_reindex(
        self, collection_id: UUID, *, batch_size: int = 500, after_ordinal: int = -1
    ) -> list[Chunk]:
        """Stream chunks for re-embedding.

        Reads stored text rather than re-parsing originals, which is what makes an
        embedding-model migration hours instead of days — and possible at all without the
        original files still being present.
        """
        ...

    async def mark_indexed(
        self, chunk_ids: Sequence[UUID], *, at: datetime, model: str, point_ids: dict[UUID, UUID]
    ) -> None: ...

    async def delete_for_version(self, version_id: UUID) -> int: ...

    async def count_for_version(self, version_id: UUID) -> int: ...

    async def keyword_search(
        self,
        query: str,
        *,
        limit: int,
        collection_ids: Sequence[UUID] = (),
        max_visibility_level: int = 0,
        department_path: str | None = None,
        granted_document_ids: Sequence[UUID] = (),
    ) -> list[tuple[UUID, float]]:
        """BM25 over ``tsvector``, used by the pgvector backend and as a sparse fallback.

        The ACL parameters are required rather than optional: a keyword search helper with
        an optional permission filter is a leak waiting for a caller in a hurry.
        """
        ...

    async def verify_readable(
        self,
        chunk_ids: Sequence[UUID],
        *,
        max_visibility_level: int,
        department_path: str | None,
        granted_document_ids: Sequence[UUID],
        mode: Mode,
        now: datetime,
    ) -> set[UUID]:
        """Post-retrieval ACL re-verification — enforcement layer 2.

        PostgreSQL is authoritative; the vector payload is a replica that can be stale
        between an ACL change and a reindex. Returns the subset that is genuinely readable.
        """
        ...


@runtime_checkable
class AclRepository(Protocol):
    async def list_for_document(self, document_id: UUID) -> list[DocumentAclGrant]: ...

    async def replace_for_document(
        self, document_id: UUID, grants: list[DocumentAclGrant]
    ) -> None: ...

    async def granted_document_ids(
        self,
        *,
        user_id: UUID | None,
        role: Role,
        department_paths: Sequence[str],
        limit: int,
    ) -> list[UUID]:
        """Documents reachable by explicit grant for this principal.

        Bounded by ``limit``; the caller raises rather than truncating, because a silently
        truncated ACL produces answers that look correct while omitting documents the user
        is entitled to.
        """
        ...

    async def effective_reader_count(self, document_id: UUID) -> int:
        """How many users can read this document.

        Surfaced in the admin UI, where it turns an abstract policy into a checkable fact —
        the only way ACL mistakes get caught before an audit does.
        """
        ...


@runtime_checkable
class IngestJobRepository(Protocol):
    async def create(self, job: IngestJob) -> IngestJob: ...

    async def get(self, job_id: UUID) -> IngestJob | None: ...

    async def find_by_idempotency_key(self, key: str) -> IngestJob | None: ...

    async def start(self, job_id: UUID, *, worker_id: str, at: datetime) -> None: ...

    async def update_stage(
        self, job_id: UUID, *, stage: IngestStatus, metrics: dict[str, Any]
    ) -> None: ...

    async def finish(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        at: datetime,
        error_message: str | None = None,
        error_class: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None: ...

    async def list_recent(
        self, *, status: JobStatus | None = None, limit: int = 50
    ) -> list[IngestJob]: ...

    async def queue_stats(self) -> dict[str, int]: ...


@runtime_checkable
class DiscrepancyRepository(Protocol):
    async def record(self, discrepancy: IndexDiscrepancy) -> None: ...

    async def list_open(self, *, limit: int = 100) -> list[IndexDiscrepancy]: ...

    async def mark_repaired(self, discrepancy_id: UUID, *, at: datetime) -> None: ...

    async def counts_by_kind(self) -> dict[str, int]: ...


# ── Conversations ───────────────────────────────────────────────────────────


@runtime_checkable
class ConversationRepository(Protocol):
    """Conversations, their messages, and the citations attached to them.

    One repository for the whole aggregate rather than three. A message is never created
    outside a conversation and a citation is never created outside a message, so splitting
    them would produce three objects that must be used together in one transaction and could
    be used apart by mistake.
    """

    async def create(self, conversation: Conversation) -> Conversation: ...

    async def get(self, conversation_id: UUID) -> Conversation | None: ...

    async def update(self, conversation_id: UUID, **fields: Any) -> Conversation: ...

    async def soft_delete(self, conversation_id: UUID, *, at: datetime) -> None: ...

    async def list_for_principal(
        self,
        *,
        user_id: UUID | None,
        guest_session_id: str | None,
        search: str | None = None,
        archived: bool | None = False,
        pinned: bool | None = None,
        limit: int = 30,
        cursor_last_message_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[Conversation]: ...

    async def add_message(self, message: Message) -> Message: ...

    async def update_message(self, message_id: UUID, **fields: Any) -> Message: ...

    async def get_message(self, message_id: UUID) -> Message | None: ...

    async def list_messages(
        self, conversation_id: UUID, *, limit: int = 50, before: datetime | None = None
    ) -> list[Message]: ...

    async def recent_turns(self, conversation_id: UUID, *, limit: int = 6) -> list[Message]:
        """The last few messages, oldest first, for prompt history.

        Separate from :meth:`list_messages` because the prompt path wants a small window in
        chronological order while the UI wants a page in reverse — and doing both with one
        method means one of the two callers reverses a list it did not have to.
        """
        ...

    async def add_citations(self, citations: list[MessageCitation]) -> int: ...

    async def citations_for(
        self, message_ids: Sequence[UUID]
    ) -> dict[UUID, list[MessageCitation]]: ...

    async def record_turn(
        self, conversation_id: UUID, *, at: datetime, tokens: int, title: str | None = None
    ) -> None:
        """Update the conversation's counters after a turn completes.

        A single statement rather than read-modify-write: two tabs answering in the same
        conversation would otherwise lose a message count increment.
        """
        ...

    async def set_feedback(
        self,
        *,
        message_id: UUID,
        user_id: UUID | None,
        rating: str,
        reason: str | None,
        comment: str | None,
    ) -> None: ...


# ── Cross-cutting ───────────────────────────────────────────────────────────


@runtime_checkable
class AuditRepository(Protocol):
    """Append-only. The database role holds ``INSERT``/``SELECT`` and nothing else —
    an audit log the application can rewrite is not an audit log."""

    async def append(self, entry: AuditEntry) -> None: ...

    async def list_entries(
        self,
        *,
        actor_id: UUID | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        cursor_created_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[AuditEntry]: ...


@dataclass(slots=True)
class SettingRecord:
    key: str
    value: Any
    updated_at: datetime | None = None
    updated_by: UUID | None = None


@runtime_checkable
class SettingsRepository(Protocol):
    """Runtime overrides for tunables.

    Lets an operator retune a confidence threshold during an incident without a redeploy.
    The env value is the boot default; a row here wins, and the admin UI shows which layer
    supplied each effective value.
    """

    async def get(self, key: str) -> Any | None: ...

    async def get_all(self) -> dict[str, Any]: ...

    async def set(self, key: str, value: Any, *, updated_by: UUID | None) -> None: ...

    async def delete(self, key: str) -> None: ...


@dataclass(slots=True)
class Repositories:
    """Everything a service might need, resolved once per request.

    A single injected bundle rather than eight constructor parameters per service. The
    alternative — passing individual repositories — reads better in isolation but turns
    every new dependency into a signature change across the call chain.
    """

    uow: UnitOfWork
    users: UserRepository
    departments: DepartmentRepository
    refresh_tokens: RefreshTokenRepository
    api_keys: ApiKeyRepository
    permissions: PermissionRepository
    collections: CollectionRepository
    documents: DocumentRepository
    versions: DocumentVersionRepository
    chunks: ChunkRepository
    acl: AclRepository
    conversations: ConversationRepository
    jobs: IngestJobRepository
    discrepancies: DiscrepancyRepository
    audit: AuditRepository
    settings: SettingsRepository
    extras: dict[str, Any] = field(default_factory=dict)
