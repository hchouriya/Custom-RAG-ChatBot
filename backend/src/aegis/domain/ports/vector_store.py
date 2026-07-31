"""Vector store port.

Deliberately narrow: upsert, search, hybrid search, delete by filter, plus collection
lifecycle. Anything richer would leak a specific engine's capabilities into the domain and
defeat the point of having the port at all.

Every read method takes a :class:`VectorFilter`. There is no unfiltered search in this
interface, and that is a security decision rather than an oversight — an unfiltered
overload is exactly what someone reaches for under deadline pressure, and it is how
unauthorized chunks reach a reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from aegis.domain.values import EmbeddingVector, SparseVector, VectorFilter


@dataclass(slots=True)
class VectorPoint:
    """One indexed chunk, as the vector store sees it.

    The payload carries only what is needed to filter and to render a citation. Full chunk
    text stays in PostgreSQL: duplicating it here would double the memory of the hot path
    and buy nothing for retrieval.
    """

    id: UUID
    chunk_id: UUID
    dense: EmbeddingVector
    sparse: SparseVector | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorHit:
    chunk_id: UUID
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    point_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """What a vector namespace needs in order to exist.

    ``dim`` is immutable once data is present: mixing dimensions in one namespace is
    unrecoverable corruption, so a model change means a new namespace and a backfill
    (see the index-migration procedure in docs/architecture/09 §4).
    """

    namespace: str
    dim: int
    distance: str = "cosine"
    with_sparse: bool = True
    quantization: bool = True
    on_disk_payload: bool = True
    payload_indexes: tuple[str, ...] = (
        "visibility_level",
        "department_path",
        "mode",
        "is_active",
        "collection_id",
        "document_id",
        "version_id",
        "tags",
        "chunk_type",
        "expires_at",
    )


@runtime_checkable
class VectorStore(Protocol):
    """Vector index operations."""

    name: str

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        """Create the namespace and its payload indexes if absent. Idempotent."""
        ...

    async def upsert(self, namespace: str, points: list[VectorPoint]) -> int:
        """Insert or replace points. Returns the count written.

        Point ids are deterministic (see ``core.ids.vector_point_id``), so a retried batch
        overwrites rather than duplicating.
        """
        ...

    async def search_dense(
        self,
        namespace: str,
        query: EmbeddingVector,
        *,
        limit: int,
        vfilter: VectorFilter,
        score_threshold: float | None = None,
    ) -> list[VectorHit]: ...

    async def search_sparse(
        self,
        namespace: str,
        query: SparseVector,
        *,
        limit: int,
        vfilter: VectorFilter,
    ) -> list[VectorHit]: ...

    async def search_hybrid(
        self,
        namespace: str,
        dense: EmbeddingVector,
        sparse: SparseVector | None,
        *,
        limit: int,
        vfilter: VectorFilter,
        rrf_k: int = 60,
    ) -> list[VectorHit]:
        """Dense and sparse in one round trip, fused by reciprocal rank.

        Present as a distinct method because engines that can fuse server-side should, and
        those that cannot can implement it as two searches plus local fusion — the caller
        should not have to know which.
        """
        ...

    async def delete_by_filter(self, namespace: str, vfilter: VectorFilter) -> int: ...

    async def delete_by_chunk_ids(self, namespace: str, chunk_ids: list[UUID]) -> int: ...

    async def set_payload(
        self, namespace: str, vfilter: VectorFilter, payload: dict[str, Any]
    ) -> int:
        """Patch payload in place, for ACL changes that must not require re-embedding.

        Re-embedding a 300-chunk document because its visibility changed would be absurd;
        this is the path that keeps an ACL edit cheap and therefore fast enough to close
        the staleness window quickly.
        """
        ...

    async def count(self, namespace: str, vfilter: VectorFilter | None = None) -> int: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...
