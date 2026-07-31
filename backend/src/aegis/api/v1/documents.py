"""Document upload, registration, and lifecycle endpoints."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Request
from pydantic import TypeAdapter

from aegis.api.deps import (
    ContainerDep,
    DocumentServiceDep,
    PrincipalDep,
    enforce_rate_limit,
)
from aegis.api.schemas import (
    AclReplaceRequest,
    DocumentOut,
    DocumentRegisterRequest,
    DocumentUpdateRequest,
    DocumentVersionOut,
    DownloadUrlResponse,
    PageOut,
    RegisteredDocumentResponse,
    ReindexRequest,
    UploadTicketRequest,
    UploadTicketResponse,
)
from aegis.core.pagination import Cursor, Page
from aegis.core.ratelimit import LimitBucket
from aegis.domain.enums import IngestStatus, Permission, Visibility
from aegis.domain.ports.repositories import DocumentQuery

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/uploads", response_model=UploadTicketResponse, status_code=201)
async def create_upload(
    body: UploadTicketRequest,
    request: Request,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
    container: ContainerDep,
) -> UploadTicketResponse:
    principal.require(Permission.DOCUMENT_WRITE, resource="documents")
    await enforce_rate_limit(
        request,
        container,
        LimitBucket.UPLOAD,
        principal_key=str(principal.user_id or principal.ctx.session_id),
        role=principal.role.value,
    )
    ticket = await documents.create_upload_ticket(
        principal,
        filename=body.filename,
        size_bytes=body.size_bytes,
        declared_mime=body.declared_mime,
        collection_id=body.collection_id,
    )
    return UploadTicketResponse(
        upload_id=ticket.upload_id,
        url=ticket.url,
        fields=ticket.fields,
        expires_at=ticket.expires_at,
        max_bytes=ticket.max_bytes,
    )


@router.post("", response_model=RegisteredDocumentResponse, status_code=202)
async def register_document(
    body: DocumentRegisterRequest,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> RegisteredDocumentResponse:
    principal.require(Permission.DOCUMENT_WRITE, resource="documents")
    result = await documents.register(
        principal,
        upload_id=body.upload_id,
        title=body.title,
        visibility=body.visibility,
        description=body.description,
        department_id=body.department_id,
        tags=body.tags,
        language=body.language,
        effective_from=body.effective_from,
        expires_at=body.expires_at,
        change_note=body.change_note,
        document_id=body.document_id,
    )
    return RegisteredDocumentResponse(
        document_id=result.document.id,
        version_id=result.version.id,
        status=result.version.status,
        job_id=result.job_id,
        poll=f"/api/v1/documents/{result.document.id}/versions",
    )


@router.get("", response_model=PageOut)
async def list_documents(
    principal: PrincipalDep,
    documents: DocumentServiceDep,
    *,
    q: str | None = None,
    collection_id: UUID | None = None,
    visibility: Visibility | None = None,
    tag: list[str] | None = Query(default=None),
    status: IngestStatus | None = None,
    department_id: UUID | None = None,
    created_after: date | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
) -> PageOut:
    principal.require(Permission.DOCUMENT_READ, resource="documents")
    decoded = Cursor.decode(cursor) if cursor else None
    query = DocumentQuery(
        collection_id=collection_id,
        visibility=visibility,
        department_id=department_id,
        tags=tuple(tag or ()),
        status=status,
        search=q,
        created_after=datetime.combine(created_after, time.min, tzinfo=UTC)
        if created_after
        else None,
    )
    rows, estimate = await documents.list_documents(
        principal,
        query,
        limit=limit + 1,
        cursor_value=decoded.last_value if decoded else None,
        cursor_id=decoded.last_id if decoded else None,
    )
    page = Page[DocumentOut].build(
        TypeAdapter(list[DocumentOut]).validate_python(rows),
        limit=limit,
        cursor_of=lambda d: Cursor(last_value=d.created_at, last_id=d.id),
        total_estimate=estimate,
    )
    return PageOut(
        items=page.items,
        next_cursor=page.next_cursor,
        has_more=page.has_more,
        total_estimate=page.total_estimate,
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: UUID,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> DocumentOut:
    principal.require(Permission.DOCUMENT_READ, resource="documents")
    document = await documents.get(document_id, principal)
    return DocumentOut.model_validate(document)


@router.patch("/{document_id}", response_model=DocumentOut)
async def patch_document(
    document_id: UUID,
    body: DocumentUpdateRequest,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> DocumentOut:
    principal.require(Permission.DOCUMENT_WRITE, resource="documents")
    fields: dict[str, Any] = body.model_dump(exclude_unset=True)
    if "visibility" in fields and fields["visibility"] is not None:
        fields["visibility"] = fields["visibility"].value
    updated = await documents.update(document_id, principal, **fields)
    return DocumentOut.model_validate(updated)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> None:
    principal.require(Permission.DOCUMENT_DELETE, resource="documents")
    await documents.delete(document_id, principal)


@router.get("/{document_id}/versions", response_model=list[DocumentVersionOut])
async def list_versions(
    document_id: UUID,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> list[DocumentVersionOut]:
    principal.require(Permission.DOCUMENT_READ, resource="documents")
    versions = await documents.versions(document_id, principal)
    return TypeAdapter(list[DocumentVersionOut]).validate_python(versions)


@router.get(
    "/{document_id}/versions/{version_id}/download",
    response_model=DownloadUrlResponse,
)
async def download_version(
    document_id: UUID,
    version_id: UUID,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> DownloadUrlResponse:
    principal.require(Permission.DOCUMENT_DOWNLOAD, resource="documents")
    url = await documents.download_url(document_id, version_id, principal)
    return DownloadUrlResponse(url=url)


@router.post("/{document_id}/reindex", status_code=202)
async def reindex(
    document_id: UUID,
    body: ReindexRequest,
    request: Request,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
    container: ContainerDep,
) -> dict[str, Any]:
    principal.require(Permission.DOCUMENT_REINDEX, resource="documents")
    await enforce_rate_limit(
        request,
        container,
        LimitBucket.REINDEX,
        principal_key=str(principal.user_id or principal.ctx.session_id),
        role=principal.role.value,
    )
    version_id = await documents.reindex(
        document_id, principal, force_reparse=body.force_reparse
    )
    return {"document_id": str(document_id), "version_id": str(version_id) if version_id else None}


@router.put("/{document_id}/acl", status_code=204)
async def replace_acl(
    document_id: UUID,
    body: AclReplaceRequest,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> None:
    principal.require(Permission.ACL_MANAGE, resource="documents")
    await documents.replace_acl(
        document_id,
        principal,
        [g.model_dump() for g in body.grants],
    )


@router.post("/{document_id}/versions/{version_id}/activate", status_code=204)
async def activate_version(
    document_id: UUID,
    version_id: UUID,
    principal: PrincipalDep,
    documents: DocumentServiceDep,
) -> None:
    principal.require(Permission.DOCUMENT_WRITE, resource="documents")
    await documents.activate_version(document_id, version_id, principal)
