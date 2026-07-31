"""pgvector adapter — one fewer service to operate.

The reason to offer it: a deployment that already runs PostgreSQL and does not want to run
Qdrant gets vector search, transactional consistency between chunks and their embeddings, and
one backup to restore. For corpora up to a few million chunks that is a good trade.

The reason it is not the default, stated plainly because it changes how retrieval is assembled:

**pgvector has no sparse vector type.** BM25 term vectors cannot be stored or searched here, so
:meth:`PgVectorStore.search_sparse` raises rather than returning an empty list. The keyword arm
of hybrid retrieval runs against the ``tsvector`` column through
``SqlDocumentRepository.keyword_search`` instead, and the retriever selects that path from
:attr:`PgVectorStore.supports_sparse`. Returning ``[]`` would have been the quiet option and
would have silently downgraded every deployment to dense-only retrieval.

Embeddings live in a per-namespace table created by :meth:`ensure_collection`. pgvector's index
types need a fixed dimension, and collections may legitimately use different models — so one
shared table cannot serve them, and a per-namespace table also makes "reindex onto a new model"
a create-and-swap rather than a destructive rewrite.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from aegis.core.errors import VectorStoreError
from aegis.core.logging import get_logger
from aegis.domain.ports.vector_store import CollectionSpec, VectorHit, VectorPoint
from aegis.rag.vector_stores.filters import to_sql

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from aegis.domain.values import EmbeddingVector, SparseVector, VectorFilter

logger = get_logger(__name__)

TABLE_PREFIX = "emb_"
_SAFE_NAMESPACE = re.compile(r"^[a-z][a-z0-9_]{0,50}$")

MAX_INDEXED_DIM = 2000
"""pgvector's limit for HNSW on ``vector``.

Above it the table still works but the index cannot be built, and an unindexed table means every
query is a sequential scan over every embedding. Rejecting at collection creation is the only
point where this is a configuration error rather than a production incident.
"""

DISTANCE_OPERATORS: dict[str, tuple[str, str]] = {
    # (operator, index opclass)
    "cosine": ("<=>", "vector_cosine_ops"),
    "dot": ("<#>", "vector_ip_ops"),
    "euclid": ("<->", "vector_l2_ops"),
}

JOIN = """
    FROM {table} e
    JOIN chunks c ON c.id = e.chunk_id
    JOIN documents d ON d.id = c.document_id
    JOIN collections col ON col.id = c.collection_id
"""
"""The join every query needs.

The ACL predicates live on three tables: visibility and department on the chunk, archival and
expiry on the document, mode and activation on the collection. Denormalising them onto
``chunks`` would make an ACL edit a rewrite of every chunk row, which is precisely the cost the
payload-patch design exists to avoid.
"""


class PgVectorStore:
    """PostgreSQL + pgvector implementation of the vector store port."""

    name = "pgvector"
    supports_sparse = False

    def __init__(self, engine: AsyncEngine, *, hnsw_ef_search: int = 100) -> None:
        self._engine = engine
        self._ef_search = hnsw_ef_search
        self._distance: dict[str, str] = {}

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        if spec.dim > MAX_INDEXED_DIM:
            raise VectorStoreError(
                f"pgvector cannot index {spec.dim} dimensions (limit {MAX_INDEXED_DIM}); "
                "use Qdrant or reduce the embedding dimension"
            )
        table = _table_name(spec.namespace)
        operator, opclass = _distance(spec.distance)
        self._distance[spec.namespace] = operator

        async with self._engine.begin() as connection:
            await connection.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id  uuid NOT NULL,
                    model     text NOT NULL,
                    embedding vector({spec.dim}) NOT NULL,
                    CONSTRAINT pk_{table} PRIMARY KEY (chunk_id),
                    CONSTRAINT fk_{table}_chunk FOREIGN KEY (chunk_id)
                        REFERENCES chunks (id) ON DELETE CASCADE
                )
                """)
            )
            await connection.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS ix_{table}_hnsw ON {table}
                USING hnsw (embedding {opclass}) WITH (m = 16, ef_construction = 200)
                """)
            )
        logger.info("pgvector.namespace_ready", namespace=spec.namespace, dim=spec.dim)

    async def upsert(self, namespace: str, points: list[VectorPoint]) -> int:
        """Write embeddings, ignoring the sparse component.

        Sparse vectors are dropped rather than rejected: the caller produces them for whichever
        backend is configured, and failing an ingest because the store cannot use one would make
        the sparse encoder a backend-specific concern.
        """
        if not points:
            return 0
        table = _table_name(namespace)
        rows = [
            {
                "chunk_id": str(point.chunk_id),
                "model": point.dense.model,
                "embedding": _literal(point.dense.values),
            }
            for point in points
        ]
        statement = text(f"""
            INSERT INTO {table} (chunk_id, model, embedding)
            VALUES (:chunk_id, :model, CAST(:embedding AS vector))
            ON CONFLICT (chunk_id) DO UPDATE
                SET embedding = EXCLUDED.embedding, model = EXCLUDED.model
        """)
        try:
            async with self._engine.begin() as connection:
                await connection.execute(statement, rows)
        except Exception as exc:
            raise VectorStoreError(f"upsert failed: {exc}") from exc
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
        table = _table_name(namespace)
        operator = self._distance.get(namespace, "<=>")
        where = to_sql(vfilter)

        # Score is reported as similarity, not distance, so that thresholds and fusion mean the
        # same thing across backends. For cosine, similarity = 1 - distance.
        statement = text(f"""
            SELECT c.id AS chunk_id,
                   1 - (e.embedding {operator} CAST(:query AS vector)) AS score,
                   c.document_id, c.version_id, c.collection_id, c.ordinal,
                   c.page_from, c.page_to, c.section, c.chunk_type::text AS chunk_type,
                   c.visibility_level, c.language
            {JOIN.format(table=table)}
            WHERE {where.sql}
            ORDER BY e.embedding {operator} CAST(:query AS vector)
            LIMIT :limit
        """)
        params = {**where.params, "query": _literal(query.values), "limit": limit}

        async with self._engine.connect() as connection:
            # Per-transaction, so a large retrieval does not change the setting for anything
            # else on the pool. Without a raised ef_search, HNSW recall at high filter
            # selectivity is poor in a way that looks like "the answer is not in the corpus".
            await connection.execute(text(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}"))
            result = await connection.execute(statement, params)
            rows = result.mappings().all()

        hits = [_to_hit(row) for row in rows]
        if score_threshold is not None:
            hits = [h for h in hits if h.score >= score_threshold]
        return hits

    async def search_sparse(
        self,
        namespace: str,
        query: SparseVector,
        *,
        limit: int,
        vfilter: VectorFilter,
    ) -> list[VectorHit]:
        raise VectorStoreError(
            "the pgvector backend does not host a sparse index; the keyword arm must use "
            "SqlDocumentRepository.keyword_search (see PgVectorStore.supports_sparse)"
        )

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
        """Dense only. Fusion with the SQL keyword arm happens in the retriever.

        The retriever owns fusion for this backend because it is the only component that can see
        both arms: one lives here, the other in the document repository.
        """
        return await self.search_dense(namespace, dense, limit=limit, vfilter=vfilter)

    async def delete_by_filter(self, namespace: str, vfilter: VectorFilter) -> int:
        table = _table_name(namespace)
        where = to_sql(vfilter)
        statement = text(f"""
            DELETE FROM {table} e
            USING chunks c, documents d, collections col
            WHERE c.id = e.chunk_id
              AND d.id = c.document_id
              AND col.id = c.collection_id
              AND {where.sql}
        """)
        async with self._engine.begin() as connection:
            result = await connection.execute(statement, where.params)
        return int(result.rowcount or 0)

    async def delete_by_chunk_ids(self, namespace: str, chunk_ids: list[UUID]) -> int:
        if not chunk_ids:
            return 0
        table = _table_name(namespace)
        statement = text(f"DELETE FROM {table} WHERE chunk_id = ANY(:ids)")
        async with self._engine.begin() as connection:
            result = await connection.execute(statement, {"ids": [str(c) for c in chunk_ids]})
        return int(result.rowcount or 0)

    async def set_payload(
        self, namespace: str, vfilter: VectorFilter, payload: dict[str, Any]
    ) -> int:
        """No-op, and correctly so.

        There is no denormalised payload in this backend: filters read the live rows on
        ``chunks``, ``documents``, and ``collections``, so an ACL change is already in effect the
        moment the transaction that made it commits. The count of affected chunks is still
        returned, because callers log it.
        """
        return await self.count(namespace, vfilter)

    async def count(self, namespace: str, vfilter: VectorFilter | None = None) -> int:
        table = _table_name(namespace)
        if vfilter is None:
            statement = text(f"SELECT count(*) FROM {table}")
            params: dict[str, Any] = {}
        else:
            where = to_sql(vfilter)
            statement = text(f"""
                SELECT count(*)
                {JOIN.format(table=table)}
                WHERE {where.sql}
            """)
            params = where.params
        async with self._engine.connect() as connection:
            result = await connection.execute(statement, params)
        return int(result.scalar() or 0)

    async def health(self) -> bool:
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:
            logger.warning("pgvector.health_failed", error=str(exc)[:200])
            return False
        return True

    async def close(self) -> None:
        # The engine is shared with the rest of the application and is disposed by whoever
        # created it. Closing it here would take the database down with the vector store.
        return None


def _table_name(namespace: str) -> str:
    """Validate and render the per-namespace table name.

    The namespace comes from a database row that an administrator can edit, and it is
    interpolated into DDL and DML that cannot be parameterised. So it is validated against a
    strict pattern rather than escaped: a rejected namespace is a configuration error, an
    escaped one is an injection waiting for the escaping to be wrong.
    """
    if not _SAFE_NAMESPACE.match(namespace):
        raise VectorStoreError(
            f"invalid vector namespace {namespace!r}: expected lowercase letters, digits, and "
            "underscores, starting with a letter"
        )
    return f"{TABLE_PREFIX}{namespace}"


def _distance(name: str) -> tuple[str, str]:
    try:
        return DISTANCE_OPERATORS[name.lower()]
    except KeyError:
        raise VectorStoreError(f"unsupported distance metric: {name}") from None


def _literal(values: tuple[float, ...] | list[float]) -> str:
    """Render as a pgvector literal.

    Sent as text and cast server-side, which avoids depending on a driver-level vector type
    registration that the async driver does not consistently provide.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _to_hit(row: Any) -> VectorHit:
    payload = {
        "chunk_id": str(row["chunk_id"]),
        "document_id": str(row["document_id"]),
        "version_id": str(row["version_id"]),
        "collection_id": str(row["collection_id"]),
        "ordinal": row["ordinal"],
        "page_from": row["page_from"],
        "page_to": row["page_to"],
        "section": row["section"],
        "chunk_type": row["chunk_type"],
        "visibility_level": row["visibility_level"],
        "language": row["language"],
    }
    return VectorHit(
        chunk_id=UUID(str(row["chunk_id"])), score=float(row["score"]), payload=payload
    )
