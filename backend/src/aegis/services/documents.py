"""Document lifecycle: upload, registration, metadata, versions, ACL, deletion.

The upload handshake is three steps — reserve, PUT to storage, register — for one reason
worth stating: bytes never transit the API process. A 200 MB scanned PDF through the
application would occupy a worker for the duration of the transfer, cap document size at the
proxy timeout, and make upload throughput a function of API replica count. Presigned PUT
makes it the storage service's problem, which is what storage services are for.

Two invariants are enforced here rather than trusted:

**The declared MIME type is a hint, never a parser selection.** It is used to reject obvious
nonsense before issuing a URL. What actually decides the parser is a magic-byte sniff of the
stored object, because ``declared_mime`` is attacker-controlled.

**A version becomes active only after it is indexed.** Replacement writes a new version and
flips ``active_version_id`` in a single statement once ingestion succeeds. Until then the
previous version keeps serving, so a failed ingest is invisible to readers instead of being an
outage on that document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aegis.core.errors import (
    AuthorizationError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from aegis.core.ids import new_id
from aegis.core.logging import get_logger
from aegis.domain.entities import (
    Document,
    DocumentAclGrant,
    DocumentVersion,
    IngestJob,
    UploadTicket,
)
from aegis.domain.enums import IngestStatus, JobName, JobStatus, Mode, Role, Visibility
from aegis.domain.policies.acl import can_administer_document, can_assign_visibility
from aegis.infrastructure.storage.keys import sanitize_filename, upload_key
from aegis.rag.parsing.sniff import EXTENSION_MIME, LEGACY_OFFICE, SUPPORTED

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.domain.entities import Collection
    from aegis.domain.ports.infrastructure import Cache, JobQueue, ObjectStore
    from aegis.domain.ports.repositories import DocumentQuery, Repositories
    from aegis.services.principal import Principal

logger = get_logger(__name__)

# Extensions we can actually parse. Checked before a URL is issued so a user learns their
# ``.pages`` file is unsupported before uploading 40 MB of it.
ALLOWED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".xlsx",
        ".xls",
        ".csv",
        ".tsv",
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".json",
    }
)
UPLOAD_TICKET_TTL = timedelta(minutes=30)


@dataclass(frozen=True, slots=True)
class RegisteredDocument:
    document: Document
    version: DocumentVersion
    job_id: UUID | None
    duplicate_of: UUID | None = None


class DocumentService:
    def __init__(
        self,
        repos: Repositories,
        *,
        settings: Settings,
        storage: ObjectStore,
        queue: JobQueue,
        cache: Cache,
    ) -> None:
        self._repos = repos
        self._settings = settings
        self._storage = storage
        self._queue = queue
        self._cache = cache

    # ── upload ──────────────────────────────────────────────────────────────

    async def create_upload_ticket(
        self,
        principal: Principal,
        *,
        filename: str,
        size_bytes: int,
        declared_mime: str,
        collection_id: UUID,
    ) -> UploadTicket:
        """Validate intent, then issue a presigned PUT."""
        collection = await self._collection(collection_id)
        if not collection.is_active:
            raise ConflictError("This collection is not accepting uploads.")

        safe_name = sanitize_filename(filename)
        extension = f".{safe_name.rsplit('.', 1)[-1].lower()}" if "." in safe_name else ""
        expected = EXTENSION_MIME.get(extension)
        if expected in LEGACY_OFFICE:
            raise UnsupportedMediaTypeError(
                f"Legacy Office files ({extension}) cannot be parsed. "
                f"Save as .docx, .xlsx, or .pptx."
            )
        if expected is None or expected not in SUPPORTED:
            raise UnsupportedMediaTypeError(
                f"{extension or 'Files without an extension'} is not supported. "
                f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        if size_bytes <= 0:
            raise ValidationError("size_bytes must be positive")
        if size_bytes > self._settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"Maximum upload size is {self._settings.max_upload_bytes // (1024 * 1024)} MB."
            )
        if declared_mime and declared_mime != expected:
            # Recorded, not rejected: browsers routinely send the wrong type for Markdown and
            # CSV, and the parser is chosen by sniffing the bytes anyway.
            logger.info(
                "upload.declared_mime_mismatch",
                declared=declared_mime,
                expected=expected,
                ext=extension,
            )

        upload_id = new_id()
        key = upload_key(upload_id=upload_id, filename=safe_name)
        presigned = await self._storage.presign_upload(
            key,
            content_type=declared_mime or "application/octet-stream",
            max_bytes=size_bytes,
            ttl=UPLOAD_TICKET_TTL,
        )
        # The ticket is what proves, at registration time, that this key was authorised for
        # this collection by this principal. Without it, `POST /documents` would accept any
        # key a caller could name.
        await self._cache.set(
            f"upload:{upload_id}",
            self._encode_ticket(
                collection_id=collection_id,
                key=key,
                filename=safe_name,
                declared_mime=declared_mime,
                actor_id=principal.user_id,
            ),
            ttl_seconds=int(UPLOAD_TICKET_TTL.total_seconds()),
        )
        return UploadTicket(
            upload_id=upload_id,
            collection_id=collection_id,
            storage_key=key,
            url=presigned.url,
            fields=presigned.fields,
            max_bytes=presigned.max_bytes,
            expires_at=presigned.expires_at,
            declared_mime=declared_mime,
            original_filename=safe_name,
        )

    async def register(
        self,
        principal: Principal,
        *,
        upload_id: UUID,
        title: str,
        visibility: Visibility,
        description: str | None = None,
        department_id: UUID | None = None,
        tags: list[str] | None = None,
        language: str | None = None,
        effective_from: date | None = None,
        expires_at: date | None = None,
        change_note: str | None = None,
        document_id: UUID | None = None,
    ) -> RegisteredDocument:
        """Create a document (or a new version of one) from an uploaded object.

        Returns 202-shaped data: nothing is searchable yet. Ingestion is queued, and the
        version's status is what the client polls.
        """
        ticket = await self._consume_ticket(upload_id, principal)
        collection = await self._collection(UUID(ticket["collection_id"]))

        if not can_assign_visibility(principal.role, visibility):
            raise AuthorizationError(
                f"Role {principal.role.value} cannot assign {visibility.value} visibility."
            )
        if visibility.requires_department_match and department_id is None:
            # Confidential is scoped by department; without one the ACL branch that grants
            # access can never match, so the document would be invisible to everyone.
            raise ValidationError(
                "Confidential documents must name a department.",
                errors=[{"field": "department_id", "message": "required for confidential"}],
            )
        if expires_at and effective_from and expires_at <= effective_from:
            raise ValidationError("expires_at must be after effective_from")

        metadata = await self._storage.head(ticket["key"])
        if metadata is None:
            raise ConflictError(
                "The uploaded file was not found in storage. Upload the file, then register it."
            )
        if metadata.size_bytes > self._settings.max_upload_bytes:
            raise PayloadTooLargeError("The uploaded file is larger than the configured maximum.")

        department_path = await self._department_path(department_id)

        if document_id is not None:
            document = await self._owned(document_id, principal)
            version_no = await self._repos.versions.next_version_no(document_id)
        else:
            document = Document(
                id=new_id(),
                collection_id=collection.id,
                title=title.strip(),
                description=description,
                visibility=visibility,
                department_id=department_id,
                department_path=department_path,
                language=language or self._settings.default_language,
                owner_id=principal.user_id,
                effective_from=effective_from,
                expires_at=expires_at,
                tags=[t.strip().lower() for t in (tags or []) if t.strip()],
            )
            document = await self._repos.documents.create(document)
            version_no = 1

        version = DocumentVersion(
            id=new_id(),
            document_id=document.id,
            version_no=version_no,
            storage_uri=ticket["key"],
            original_filename=ticket["filename"],
            mime_type=metadata.content_type
            or ticket.get("declared_mime")
            or "application/octet-stream",
            size_bytes=metadata.size_bytes,
            # The real checksum is computed by the worker when it reads the bytes; the ETag
            # is not a SHA-256 for multipart uploads, so trusting it would break dedup.
            checksum_sha256=b"\x00" * 32,
            status=IngestStatus.PENDING,
            change_note=change_note,
            created_by=principal.user_id,
        )
        version = await self._repos.versions.create(version)

        job = await self._repos.jobs.create(
            IngestJob(
                id=new_id(),
                version_id=version.id,
                job_type=JobName.INGEST.value,
                status=JobStatus.QUEUED,
                stage=IngestStatus.PENDING,
                max_attempts=self._settings.ingest_max_attempts,
                idempotency_key=f"ingest:{version.id}",
                payload={"document_id": str(document.id), "version_id": str(version.id)},
            )
        )
        await self._repos.uow.commit()

        # Enqueued after commit: a job that starts before its row is visible finds nothing to
        # do and fails, and the row is the authoritative record of work anyway.
        await self._queue.enqueue(
            JobName.INGEST.value,
            str(version.id),
            idempotency_key=f"ingest:{version.id}",
        )
        logger.info(
            "document.registered",
            document_id=str(document.id),
            version_id=str(version.id),
            version_no=version_no,
            collection=collection.slug,
            visibility=visibility.value,
        )
        return RegisteredDocument(document=document, version=version, job_id=job.id)

    # ── reads ───────────────────────────────────────────────────────────────

    async def list_documents(
        self,
        principal: Principal,
        query: DocumentQuery,
        *,
        limit: int,
        cursor_value: Any = None,
        cursor_id: UUID | None = None,
    ) -> tuple[list[Document], int]:
        """List documents the principal may see.

        The visibility ceiling is applied to the *listing*, not only to retrieval: a title is
        content, and "Q3 Layoff Plan" appearing in a list an employee cannot open is still a
        disclosure.
        """
        query.max_visibility_level = int(principal.ctx.ceiling)
        if principal.role is Role.INTERNAL_EMPLOYEE:
            # An employee's admin list is their own department's documents. They can still
            # *retrieve* from anything the ceiling allows; browsing the corpus is a wider
            # capability than answering questions from it.
            query.department_path = query.department_path or principal.ctx.department_path
        documents = await self._repos.documents.list_documents(
            query, limit=limit, cursor_value=cursor_value, cursor_id=cursor_id
        )
        return documents, await self._repos.documents.estimate_count(query)

    async def get(self, document_id: UUID, principal: Principal) -> Document:
        """Fetch one document, or 404 if the principal may not see it.

        404 rather than 403 by design: distinguishing "does not exist" from "exists but is
        not yours" hands out the existence of confidential documents.
        """
        document = await self._repos.documents.get(document_id)
        if document is None or not self._readable(document, principal):
            raise NotFoundError("Document", document_id)
        return document

    async def versions(self, document_id: UUID, principal: Principal) -> list[DocumentVersion]:
        await self.get(document_id, principal)
        return await self._repos.versions.list_for_document(document_id)

    async def chunks(
        self, document_id: UUID, principal: Principal, *, limit: int = 50, offset: int = 0
    ) -> list[Any]:
        document = await self.get(document_id, principal)
        if document.active_version_id is None:
            return []
        return await self._repos.chunks.list_for_version(
            document.active_version_id, limit=limit, offset=offset
        )

    async def download_url(self, document_id: UUID, version_id: UUID, principal: Principal) -> str:
        document = await self.get(document_id, principal)
        version = await self._repos.versions.get(version_id)
        if version is None or version.document_id != document.id:
            raise NotFoundError("Version", version_id)
        return await self._storage.presign_download(
            version.storage_uri,
            ttl=timedelta(minutes=5),
            filename=version.original_filename,
        )

    # ── mutations ───────────────────────────────────────────────────────────

    async def update(self, document_id: UUID, principal: Principal, **fields: Any) -> Document:
        """Update metadata, patching the vector payload when access control changed.

        The payload patch is what keeps an ACL edit cheap: re-embedding 300 chunks because a
        document's visibility changed would make correct ACL maintenance expensive enough
        that people would avoid it.
        """
        document = await self._owned(document_id, principal)
        if (visibility := fields.get("visibility")) is not None:
            visibility = Visibility(visibility)
            if not can_assign_visibility(principal.role, visibility):
                raise AuthorizationError(
                    f"Role {principal.role.value} cannot assign {visibility.value} visibility."
                )
        if "department_id" in fields:
            fields["department_path"] = await self._department_path(fields["department_id"])
        tags = fields.pop("tags", None)

        updated = await self._repos.documents.update(document_id, **fields)
        if tags is not None:
            await self._repos.documents.set_tags(document_id, [t.strip().lower() for t in tags])
            updated = await self._repos.documents.update(document_id)
        await self._repos.uow.commit()

        if self._acl_changed(document, updated):
            await self._queue.enqueue(
                JobName.REINDEX.value,
                str(document_id),
                acl_only=True,
                idempotency_key=f"acl:{document_id}:{updated.updated_at}",
            )
        return updated

    async def delete(self, document_id: UUID, principal: Principal) -> None:
        """Soft-delete the row and purge vectors immediately.

        The order matters. Vectors go first because they are what retrieval reads: a
        soft-deleted row whose vectors are still indexed is a document that has been deleted
        everywhere except where it counts.
        """
        document = await self._owned(document_id, principal)
        await self._queue.enqueue(
            JobName.PURGE.value, str(document_id), idempotency_key=f"purge:{document_id}"
        )
        await self._repos.documents.soft_delete(document_id, at=datetime.now(UTC))
        await self._repos.uow.commit()
        logger.info(
            "document.deleted", document_id=str(document_id), actor=str(principal.user_id or "-")
        )
        _ = document

    async def reindex(
        self, document_id: UUID, principal: Principal, *, force_reparse: bool = False
    ) -> UUID | None:
        document = await self._owned(document_id, principal)
        if document.active_version_id is None:
            raise ConflictError("This document has no indexed version to reindex.")
        job = await self._queue.enqueue(
            JobName.REINDEX.value,
            str(document_id),
            force_reparse=force_reparse,
            idempotency_key=f"reindex:{document_id}:{document.active_version_id}",
        )
        return document.active_version_id if job else None

    async def replace_acl(
        self, document_id: UUID, principal: Principal, grants: list[dict[str, Any]]
    ) -> list[DocumentAclGrant]:
        document = await self._owned(document_id, principal)
        entries = [
            DocumentAclGrant(
                id=new_id(),
                document_id=document.id,
                principal_type=grant["principal_type"],
                principal_role=grant.get("principal_role"),
                principal_id=grant.get("principal_id"),
                include_subtree=grant.get("include_subtree", True),
                expires_at=grant.get("expires_at"),
                granted_by=principal.user_id,
            )
            for grant in grants
        ]
        await self._repos.acl.replace_for_document(document.id, entries)
        await self._repos.uow.commit()
        logger.info("document.acl_replaced", document_id=str(document_id), grants=len(entries))
        return entries

    async def activate_version(
        self, document_id: UUID, version_id: UUID, principal: Principal
    ) -> None:
        """Roll back (or forward) to an already-indexed version."""
        await self._owned(document_id, principal)
        version = await self._repos.versions.get(version_id)
        if version is None or version.document_id != document_id:
            raise NotFoundError("Version", version_id)
        if not version.is_searchable:
            raise ConflictError(
                f"Version {version.version_no} is at stage '{version.status.value}' "
                f"and cannot be activated."
            )
        await self._repos.documents.set_active_version(document_id, version_id)
        await self._repos.uow.commit()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _readable(self, document: Document, principal: Principal) -> bool:
        return principal.ctx.can_read_document(
            document_id=document.id,
            visibility=document.visibility,
            department_path=document.department_path,
        )

    async def _owned(self, document_id: UUID, principal: Principal) -> Document:
        document = await self._repos.documents.get(document_id)
        if document is None or not self._readable(document, principal):
            raise NotFoundError("Document", document_id)
        if not can_administer_document(
            role=principal.role,
            actor_id=principal.user_id,
            actor_department_path=principal.ctx.department_path,
            owner_id=document.owner_id,
            document_department_path=document.department_path,
        ):
            raise AuthorizationError("You cannot modify this document.")
        return document

    async def _collection(self, collection_id: UUID) -> Collection:
        collection = await self._repos.collections.get(collection_id)
        if collection is None:
            raise NotFoundError("Collection", collection_id)
        return collection

    async def _department_path(self, department_id: UUID | None) -> str | None:
        if department_id is None:
            return None
        department = await self._repos.departments.get(department_id)
        if department is None:
            raise ValidationError(
                "Unknown department",
                errors=[{"field": "department_id", "message": "does not exist"}],
            )
        return department.path

    @staticmethod
    def _acl_changed(before: Document, after: Document) -> bool:
        return (
            before.visibility != after.visibility
            or before.department_path != after.department_path
            or before.expires_at != after.expires_at
            or before.effective_from != after.effective_from
            or before.is_archived != after.is_archived
        )

    @staticmethod
    def _encode_ticket(
        *,
        collection_id: UUID,
        key: str,
        filename: str,
        declared_mime: str,
        actor_id: UUID | None,
    ) -> bytes:
        return json.dumps(
            {
                "collection_id": str(collection_id),
                "key": key,
                "filename": filename,
                "declared_mime": declared_mime,
                "actor_id": str(actor_id) if actor_id else None,
            }
        ).encode()

    async def _consume_ticket(self, upload_id: UUID, principal: Principal) -> dict[str, Any]:
        """Redeem an upload ticket exactly once.

        Single use so a key cannot be registered twice, and bound to its issuer so one user's
        presigned upload cannot be claimed as another user's document.
        """
        raw = await self._cache.get(f"upload:{upload_id}")
        if raw is None:
            raise ConflictError(
                "This upload has expired or was already registered. Request a new upload URL."
            )
        ticket: dict[str, Any] = json.loads(raw)
        owner = ticket.get("actor_id")
        if owner and principal.user_id is not None and owner != str(principal.user_id):
            raise AuthorizationError("This upload belongs to another user.")
        await self._cache.delete(f"upload:{upload_id}")
        return ticket


def visible_collections(collections: list[Collection], mode: Mode) -> list[Collection]:
    """Collections a session may search: active, and matching the assistant mode."""
    return [c for c in collections if c.is_active and c.mode is mode]
