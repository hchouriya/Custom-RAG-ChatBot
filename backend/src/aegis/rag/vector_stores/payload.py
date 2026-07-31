"""The payload contract: exactly what travels with a vector, and why.

Payload is the only thing a vector store can filter on, which makes this the other half of the
ACL. Every field the filter algebra references has to be here, denormalised from the document
and collection at index time, and it has to be *kept* correct when those rows change — hence
:func:`acl_payload`, which is what an ACL edit patches in place rather than re-embedding.

Two things are deliberately absent:

* **Chunk text.** It lives in PostgreSQL. Duplicating it here would double the index's memory
  for no retrieval benefit; the citation drawer reads it from the database by chunk id.
* **Anything user-supplied and unvalidated.** Payload is attacker-influenced data (a document
  title, a tag) that is later rendered in a UI, so only known keys are written.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from aegis.rag.vector_stores.filters import ancestor_paths

if TYPE_CHECKING:
    from uuid import UUID

PAYLOAD_KEYS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "version_id",
    "collection_id",
    "mode",
    "visibility_level",
    "department_path",
    "is_active",
    "expires_at",
    "effective_from",
    "chunk_type",
    "language",
    "tags",
    "ordinal",
    "page_from",
    "page_to",
    "section",
    "injection_flag",
)
"""Every key written. Anything outside this tuple is dropped rather than stored."""

ACL_KEYS: tuple[str, ...] = (
    "visibility_level",
    "department_path",
    "is_active",
    "expires_at",
    "effective_from",
    "mode",
)
"""The subset an ACL change can affect, and therefore the subset a patch has to rewrite."""


def build_payload(
    *,
    chunk_id: UUID,
    document_id: UUID,
    version_id: UUID,
    collection_id: UUID,
    mode: str,
    visibility_level: int,
    department_path: str | None,
    is_active: bool,
    expires_at: date | datetime | None = None,
    effective_from: date | datetime | None = None,
    chunk_type: str = "text",
    language: str | None = None,
    tags: tuple[str, ...] = (),
    ordinal: int = 0,
    page_from: int | None = None,
    page_to: int | None = None,
    section: str | None = None,
    injection_flag: bool = False,
) -> dict[str, Any]:
    """Assemble a point payload.

    ``department_path`` is stored as its ancestor list rather than as the path string. That is
    what lets "documents in my subtree" be an exact keyword match in an engine with no
    hierarchical type — see :func:`aegis.rag.vector_stores.filters.ancestor_paths`.

    Dates are stored as epoch seconds. Backends disagree about date parsing and about timezone
    defaults, and a comparison that is off by a day silently keeps expired documents
    retrievable — the failure mode is a customer being quoted a withdrawn price.
    """
    return {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "version_id": str(version_id),
        "collection_id": str(collection_id),
        "mode": mode,
        "visibility_level": int(visibility_level),
        "department_path": list(ancestor_paths(department_path)),
        "is_active": bool(is_active),
        "expires_at": to_epoch(expires_at),
        "effective_from": to_epoch(effective_from),
        "chunk_type": chunk_type,
        "language": language,
        "tags": list(tags),
        "ordinal": int(ordinal),
        "page_from": page_from,
        "page_to": page_to,
        "section": section,
        "injection_flag": bool(injection_flag),
    }


def acl_payload(
    *,
    visibility_level: int,
    department_path: str | None,
    is_active: bool,
    mode: str,
    expires_at: date | datetime | None = None,
    effective_from: date | datetime | None = None,
) -> dict[str, Any]:
    """The patch applied when a document's access control changes.

    Deliberately a full rewrite of :data:`ACL_KEYS` rather than a diff. A partial patch that
    forgets a key leaves the index enforcing the old rule for that field, and the whole point
    of patching instead of reindexing is that it is fast enough to close the staleness window —
    which only holds if it is also complete.
    """
    return {
        "visibility_level": int(visibility_level),
        "department_path": list(ancestor_paths(department_path)),
        "is_active": bool(is_active),
        "mode": mode,
        "expires_at": to_epoch(expires_at),
        "effective_from": to_epoch(effective_from),
    }


def to_epoch(value: date | datetime | None) -> float | None:
    """Convert to epoch seconds, treating a bare date as midnight UTC.

    A naive datetime is assumed to be UTC rather than local: the alternative makes the same
    document expire at different moments on a developer's laptop and in the cluster.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        moment = datetime(value.year, value.month, value.day, tzinfo=UTC)
    return moment.timestamp()
