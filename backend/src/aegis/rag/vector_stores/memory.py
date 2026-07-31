"""In-process vector store.

Two jobs, and the second one is the reason it is written to the same standard as the others:

* It makes the default test suite run with no Qdrant, no Postgres, and no network, which is
  what keeps the security tests — a few hundred ACL cases — fast enough to run on every commit.
* It is the reference implementation of the port's semantics. When Qdrant and pgvector disagree
  about a filter, this is the tiebreaker, because the filter evaluation here *is* the domain
  algebra rather than a translation of it.

Not for production. Brute-force scan, no persistence, no concurrency control beyond the GIL.
Selecting it in production is rejected at startup by configuration validation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from aegis.core.errors import VectorStoreError
from aegis.core.logging import get_logger
from aegis.domain.ports.vector_store import CollectionSpec, VectorHit, VectorPoint
from aegis.rag.vector_stores.filters import matches
from aegis.rag.vector_stores.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from aegis.domain.values import EmbeddingVector, SparseVector, VectorFilter

logger = get_logger(__name__)


class InMemoryVectorStore:
    """Brute-force store over dictionaries."""

    name = "memory"
    supports_sparse = True

    def __init__(self) -> None:
        self._specs: dict[str, CollectionSpec] = {}
        self._points: dict[str, dict[UUID, VectorPoint]] = {}

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        existing = self._specs.get(spec.namespace)
        if existing and existing.dim != spec.dim and self._points.get(spec.namespace):
            raise VectorStoreError(
                f"namespace {spec.namespace} holds {existing.dim}-dimensional vectors; "
                f"cannot serve {spec.dim}"
            )
        self._specs[spec.namespace] = spec
        self._points.setdefault(spec.namespace, {})

    async def upsert(self, namespace: str, points: list[VectorPoint]) -> int:
        store = self._points.setdefault(namespace, {})
        spec = self._specs.get(namespace)
        for point in points:
            if spec and point.dense.dim != spec.dim:
                raise VectorStoreError(
                    f"vector has dimension {point.dense.dim}, namespace expects {spec.dim}"
                )
            store[point.id] = point
        return len(points)

    async def search_dense(
        self,
        namespace: str,
        query: EmbeddingVector,
        *,
        limit: int,
        vfilter: VectorFilter,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        scored: list[VectorHit] = []
        for point in self._eligible(namespace, vfilter):
            score = _cosine(query.values, point.dense.values)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(
                VectorHit(
                    chunk_id=point.chunk_id,
                    score=score,
                    payload=dict(point.payload),
                    point_id=point.id,
                )
            )
        scored.sort(key=lambda h: (-h.score, str(h.chunk_id)))
        return scored[:limit]

    async def search_sparse(
        self,
        namespace: str,
        query: SparseVector,
        *,
        limit: int,
        vfilter: VectorFilter,
    ) -> list[VectorHit]:
        weights = dict(zip(query.indices, query.values, strict=True))
        scored: list[VectorHit] = []
        for point in self._eligible(namespace, vfilter):
            if point.sparse is None:
                continue
            score = sum(
                value * weights.get(index, 0.0)
                for index, value in zip(point.sparse.indices, point.sparse.values, strict=True)
            )
            if score <= 0.0:
                continue
            scored.append(
                VectorHit(
                    chunk_id=point.chunk_id,
                    score=score,
                    payload=dict(point.payload),
                    point_id=point.id,
                )
            )
        scored.sort(key=lambda h: (-h.score, str(h.chunk_id)))
        return scored[:limit]

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
        # Each arm is over-fetched relative to `limit`: fusion can promote a result that was
        # outside the top `limit` in both arms, and truncating before fusing would discard it.
        depth = max(limit * 2, limit + 10)
        dense_hits = await self.search_dense(namespace, dense, limit=depth, vfilter=vfilter)
        if sparse is None or not sparse.nnz:
            return dense_hits[:limit]
        sparse_hits = await self.search_sparse(namespace, sparse, limit=depth, vfilter=vfilter)
        return reciprocal_rank_fusion(dense_hits, sparse_hits, k=rrf_k, limit=limit)

    async def delete_by_filter(self, namespace: str, vfilter: VectorFilter) -> int:
        store = self._points.get(namespace, {})
        doomed = [p.id for p in store.values() if matches(p.payload, vfilter)]
        for point_id in doomed:
            del store[point_id]
        return len(doomed)

    async def delete_by_chunk_ids(self, namespace: str, chunk_ids: list[UUID]) -> int:
        store = self._points.get(namespace, {})
        wanted = set(chunk_ids)
        doomed = [p.id for p in store.values() if p.chunk_id in wanted]
        for point_id in doomed:
            del store[point_id]
        return len(doomed)

    async def set_payload(
        self, namespace: str, vfilter: VectorFilter, payload: dict[str, Any]
    ) -> int:
        touched = 0
        for point in self._points.get(namespace, {}).values():
            if matches(point.payload, vfilter):
                point.payload.update(payload)
                touched += 1
        return touched

    async def count(self, namespace: str, vfilter: VectorFilter | None = None) -> int:
        store = self._points.get(namespace, {})
        if vfilter is None:
            return len(store)
        return sum(1 for p in store.values() if matches(p.payload, vfilter))

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    def _eligible(self, namespace: str, vfilter: VectorFilter) -> list[VectorPoint]:
        return [
            point
            for point in self._points.get(namespace, {}).values()
            if matches(point.payload, vfilter)
        ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise VectorStoreError(
            f"query has dimension {len(left)}, indexed vectors have {len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
