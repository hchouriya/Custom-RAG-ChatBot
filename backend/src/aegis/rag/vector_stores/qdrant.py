"""Qdrant adapter — the default backend.

Chosen over pgvector as the default for three reasons that matter at this scale: it stores
sparse vectors natively, so BM25 and dense retrieval share one index with one ACL payload; it
fuses both arms server-side in a single round trip; and it supports scalar quantization, which
cuts memory roughly four-fold for a small, measurable recall cost.

Two configuration choices are worth stating because they are easy to get wrong and expensive to
discover later:

* **Named vectors from the start.** ``dense`` and ``sparse`` are named even when sparse is
  disabled, because adding a named vector to a collection created with an unnamed one requires
  recreating the collection — that is, a full reindex during whatever incident prompted it.
* **Payload indexes are explicit.** Without them, every ACL filter is a full payload scan.
  Qdrant will happily run the query anyway, slowly, which is exactly the kind of problem that
  never shows up in development where the corpus is a hundred documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from aegis.core.errors import VectorStoreError
from aegis.core.logging import get_logger
from aegis.domain.ports.vector_store import CollectionSpec, VectorHit, VectorPoint
from aegis.rag.vector_stores.filters import to_qdrant
from aegis.rag.vector_stores.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from aegis.domain.values import EmbeddingVector, SparseVector, VectorFilter

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

KEYWORD_FIELDS = frozenset(
    {"department_path", "mode", "collection_id", "document_id", "version_id", "tags", "chunk_type"}
)
INTEGER_FIELDS = frozenset({"visibility_level", "ordinal", "page_from", "page_to"})
FLOAT_FIELDS = frozenset({"expires_at", "effective_from"})
BOOL_FIELDS = frozenset({"is_active", "injection_flag"})

UPSERT_BATCH = 256
"""Points per request.

Large enough that the per-request overhead disappears, small enough that one failed batch is a
cheap retry rather than a restarted document.
"""


class QdrantVectorStore:
    """Qdrant-backed implementation of the vector store port."""

    name = "qdrant"
    supports_sparse = True

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout_seconds: int = 30,
        client: Any = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout_seconds,
            )

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        from qdrant_client import models as qm

        if await self._client.collection_exists(spec.namespace):
            await self._verify(spec)
            await self._ensure_indexes(spec)
            return

        await self._client.create_collection(
            collection_name=spec.namespace,
            vectors_config={
                DENSE_VECTOR: qm.VectorParams(
                    size=spec.dim,
                    distance=_distance(spec.distance),
                    on_disk=spec.on_disk_payload,
                )
            },
            sparse_vectors_config=(
                {SPARSE_VECTOR: qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False))}
                if spec.with_sparse
                else None
            ),
            quantization_config=(
                # `always_ram` keeps the quantized vectors resident while the originals go to
                # disk: that combination is what makes quantization a memory win instead of an
                # extra disk read on every candidate.
                qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                )
                if spec.quantization
                else None
            ),
            on_disk_payload=spec.on_disk_payload,
        )
        await self._ensure_indexes(spec)
        logger.info("qdrant.collection_created", namespace=spec.namespace, dim=spec.dim)

    async def _verify(self, spec: CollectionSpec) -> None:
        """Refuse to serve a namespace whose dimension does not match the configuration.

        A mismatch means the embedding model changed without a reindex. Writing into it would
        fail per point; *reading* from it would silently return nonsense, because the query
        vector and the indexed vectors would come from different models.
        """
        info = await self._client.get_collection(spec.namespace)
        vectors = getattr(info.config.params, "vectors", None)
        size = None
        if isinstance(vectors, dict):
            params = vectors.get(DENSE_VECTOR)
            size = getattr(params, "size", None)
        else:
            size = getattr(vectors, "size", None)
        if size is not None and int(size) != spec.dim:
            raise VectorStoreError(
                f"namespace {spec.namespace} was created with dimension {size}, "
                f"configuration expects {spec.dim}; a model change requires a new namespace "
                "and a backfill, not an in-place edit"
            )

    async def _ensure_indexes(self, spec: CollectionSpec) -> None:
        from qdrant_client import models as qm

        for field in spec.payload_indexes:
            schema: Any
            if field in INTEGER_FIELDS:
                schema = qm.PayloadSchemaType.INTEGER
            elif field in FLOAT_FIELDS:
                schema = qm.PayloadSchemaType.FLOAT
            elif field in BOOL_FIELDS:
                schema = qm.PayloadSchemaType.BOOL
            elif field in KEYWORD_FIELDS:
                schema = qm.PayloadSchemaType.KEYWORD
            else:
                schema = qm.PayloadSchemaType.KEYWORD
            try:
                await self._client.create_payload_index(
                    collection_name=spec.namespace, field_name=field, field_schema=schema
                )
            except Exception as exc:
                # Already-exists is the common case on restart and is not an error. Anything
                # else is logged rather than raised: a missing index makes queries slow, and
                # failing startup over it would trade a slow service for no service.
                logger.debug("qdrant.index_skipped", field=field, error=str(exc)[:200])

    async def upsert(self, namespace: str, points: list[VectorPoint]) -> int:
        from qdrant_client import models as qm

        if not points:
            return 0

        written = 0
        for start in range(0, len(points), UPSERT_BATCH):
            batch = points[start : start + UPSERT_BATCH]
            structs = [
                qm.PointStruct(
                    id=str(point.id),
                    vector=self._vectors(point),
                    payload=point.payload,
                )
                for point in batch
            ]
            try:
                await self._client.upsert(collection_name=namespace, points=structs, wait=True)
            except Exception as exc:
                raise VectorStoreError(f"upsert failed: {exc}") from exc
            written += len(batch)
        return written

    def _vectors(self, point: VectorPoint) -> dict[str, Any]:
        from qdrant_client import models as qm

        vectors: dict[str, Any] = {DENSE_VECTOR: list(point.dense.values)}
        if point.sparse is not None and point.sparse.nnz:
            vectors[SPARSE_VECTOR] = qm.SparseVector(
                indices=list(point.sparse.indices), values=list(point.sparse.values)
            )
        return vectors

    async def search_dense(
        self,
        namespace: str,
        query: EmbeddingVector,
        *,
        limit: int,
        vfilter: VectorFilter,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        response = await self._query(
            namespace,
            query=list(query.values),
            using=DENSE_VECTOR,
            limit=limit,
            vfilter=vfilter,
            score_threshold=score_threshold,
        )
        return _to_hits(response)

    async def search_sparse(
        self,
        namespace: str,
        query: SparseVector,
        *,
        limit: int,
        vfilter: VectorFilter,
    ) -> list[VectorHit]:
        from qdrant_client import models as qm

        if not query.nnz:
            return []
        response = await self._query(
            namespace,
            query=qm.SparseVector(indices=list(query.indices), values=list(query.values)),
            using=SPARSE_VECTOR,
            limit=limit,
            vfilter=vfilter,
        )
        return _to_hits(response)

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
        """Both arms in one round trip, fused by the engine.

        Server-side fusion halves the latency of the retrieval stage and, more importantly,
        applies the same filter to both arms by construction — two client-side queries can
        drift apart, and a filter applied to only one arm is a leak through the other.
        """
        from qdrant_client import models as qm

        if sparse is None or not sparse.nnz:
            return await self.search_dense(namespace, dense, limit=limit, vfilter=vfilter)

        depth = max(limit * 2, limit + 10)
        query_filter = to_qdrant(vfilter)
        try:
            response = await self._client.query_points(
                collection_name=namespace,
                prefetch=[
                    qm.Prefetch(
                        query=list(dense.values),
                        using=DENSE_VECTOR,
                        limit=depth,
                        filter=query_filter,
                    ),
                    qm.Prefetch(
                        query=qm.SparseVector(
                            indices=list(sparse.indices), values=list(sparse.values)
                        ),
                        using=SPARSE_VECTOR,
                        limit=depth,
                        filter=query_filter,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            # Older servers lack the Query API. Falling back keeps retrieval working on a
            # mixed fleet during an upgrade, which is exactly when this would otherwise break.
            logger.warning("qdrant.server_fusion_unavailable", error=str(exc)[:200])
            dense_hits = await self.search_dense(namespace, dense, limit=depth, vfilter=vfilter)
            sparse_hits = await self.search_sparse(namespace, sparse, limit=depth, vfilter=vfilter)
            return reciprocal_rank_fusion(dense_hits, sparse_hits, k=rrf_k, limit=limit)
        return _to_hits(response)

    async def _query(
        self,
        namespace: str,
        *,
        query: Any,
        using: str,
        limit: int,
        vfilter: VectorFilter,
        score_threshold: float | None = None,
    ) -> Any:
        try:
            return await self._client.query_points(
                collection_name=namespace,
                query=query,
                using=using,
                limit=limit,
                query_filter=to_qdrant(vfilter),
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"search failed: {exc}") from exc

    async def delete_by_filter(self, namespace: str, vfilter: VectorFilter) -> int:
        from qdrant_client import models as qm

        before = await self.count(namespace, vfilter)
        await self._client.delete(
            collection_name=namespace,
            points_selector=qm.FilterSelector(filter=to_qdrant(vfilter)),
            wait=True,
        )
        return before

    async def delete_by_chunk_ids(self, namespace: str, chunk_ids: list[UUID]) -> int:
        from qdrant_client import models as qm

        if not chunk_ids:
            return 0
        await self._client.delete(
            collection_name=namespace,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="chunk_id", match=qm.MatchAny(any=[str(c) for c in chunk_ids])
                        )
                    ]
                )
            ),
            wait=True,
        )
        return len(chunk_ids)

    async def set_payload(
        self, namespace: str, vfilter: VectorFilter, payload: dict[str, Any]
    ) -> int:
        affected = await self.count(namespace, vfilter)
        await self._client.set_payload(
            collection_name=namespace,
            payload=payload,
            points=to_qdrant(vfilter),
            wait=True,
        )
        return affected

    async def count(self, namespace: str, vfilter: VectorFilter | None = None) -> int:
        result = await self._client.count(
            collection_name=namespace,
            count_filter=to_qdrant(vfilter) if vfilter is not None else None,
            exact=True,
        )
        return int(result.count)

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception as exc:
            logger.warning("qdrant.health_failed", error=str(exc)[:200])
            return False
        return True

    async def close(self) -> None:
        await self._client.close()


def _distance(name: str) -> Any:
    from qdrant_client import models as qm

    mapping = {
        "cosine": qm.Distance.COSINE,
        "dot": qm.Distance.DOT,
        "euclid": qm.Distance.EUCLID,
        "manhattan": qm.Distance.MANHATTAN,
    }
    try:
        return mapping[name.lower()]
    except KeyError:
        raise VectorStoreError(f"unsupported distance metric: {name}") from None


def _to_hits(response: Any) -> list[VectorHit]:
    points = getattr(response, "points", response)
    hits: list[VectorHit] = []
    for point in points:
        payload = dict(point.payload or {})
        chunk_id = payload.get("chunk_id") or point.id
        hits.append(
            VectorHit(
                chunk_id=UUID(str(chunk_id)),
                score=float(point.score),
                payload=payload,
                point_id=_as_uuid(point.id),
            )
        )
    return hits


def _as_uuid(value: Any) -> UUID | None:
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None
