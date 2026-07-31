"""The retrieval pipeline.

Order of operations, and why each step is where it is:

1. **Build the ACL filter from the security context.** Before anything is embedded. The
   filter is an input to the search, not a post-processing step — a system that retrieves
   first and filters second has already loaded unauthorized content into the process, and
   one forgotten branch turns that into a leak.
2. **Embed the rewritten queries and search each namespace, dense + sparse.** Two arms
   because they fail differently: dense misses exact identifiers ("SKU-4471", "IAS 37"),
   sparse misses paraphrase ("time off" vs "leave entitlement"). Enterprise questions
   contain both.
3. **Fuse by reciprocal rank.** Scores from the two arms are not comparable; ranks are.
4. **Re-verify against PostgreSQL — enforcement layer 2.** The vector payload is a replica
   and can be stale between an ACL edit and the reindex that follows it. PostgreSQL is
   authoritative, and this is where a stale payload stops being a leak. Drops here are
   counted: a non-zero rate is a bug in payload maintenance, not a routine event.
5. **Hydrate text from PostgreSQL.** The index holds no chunk text, so this is also where
   the content that goes to the reranker and the model comes from — a single source of
   truth for what was actually cited.
6. **Rerank with a cross-encoder.** The stage that decides whether the right paragraph is
   first or seventh, and only the first few fit the context budget.

What this deliberately is not: a single vector search over an unfiltered index with a
similarity threshold. That approach cannot express "confidential, but only within my
department subtree", cannot find exact codes, and cannot tell topical similarity from an
actual answer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aegis.core.logging import get_logger
from aegis.core.telemetry import (
    acl_layer2_drops,
    retrieval_candidates,
    retrieval_top_score,
    timed_stage,
)
from aegis.domain.enums import ChunkType, Visibility
from aegis.domain.policies.acl import build_filter
from aegis.domain.values import ChunkLocator, RetrievedChunk, VectorFilter
from aegis.rag.vector_stores.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aegis.domain.entities import Collection
    from aegis.domain.ports.embeddings import EmbeddingProvider, Reranker, SparseEmbeddingProvider
    from aegis.domain.ports.repositories import ChunkRepository, DocumentRepository
    from aegis.domain.ports.vector_store import VectorHit, VectorStore
    from aegis.domain.values import SecurityContext
    from aegis.rag.retrieval.query import QueryPlan

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalConfig:
    top_k_dense: int = 40
    top_k_sparse: int = 40
    rrf_k: int = 60
    rerank_top_n: int = 8
    # Cap on what reaches the reranker after fusion. Reranking is the expensive stage, and
    # beyond ~60 candidates the marginal recall is nearly zero while latency is linear.
    max_rerank_candidates: int = 60
    hybrid: bool = True


@dataclass(slots=True)
class RetrievalResult:
    """Candidates plus the trace of how they were produced.

    The stage counts and the applied filter are not diagnostics-as-an-afterthought: they are
    persisted on ``query_traces`` and returned verbatim by ``POST /admin/retrieval/debug``,
    which is where "why did it answer that?" gets settled.
    """

    chunks: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    filter_applied: dict[str, Any] = field(default_factory=dict)
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    acl_dropped: int = 0
    reranked_count: int = 0
    namespaces: list[str] = field(default_factory=list)
    degraded_rerank: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.chunks[0].best_score if self.chunks else 0.0


class HybridRetriever:
    """Access-controlled hybrid retrieval with reranking."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        sparse_encoder: SparseEmbeddingProvider | None,
        reranker: Reranker,
        chunks: ChunkRepository,
        documents: DocumentRepository,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._reranker = reranker
        self._chunks = chunks
        self._documents = documents
        self._config = config or RetrievalConfig()

    async def retrieve(
        self,
        plan: QueryPlan,
        ctx: SecurityContext,
        collections: Sequence[Collection],
        *,
        narrowing: VectorFilter | None = None,
    ) -> RetrievalResult:
        result = RetrievalResult(queries=list(plan.queries) or [plan.original])
        if not collections:
            logger.info("retrieval.no_collections", mode=ctx.mode.value)
            return result

        vfilter = build_filter(ctx, narrowing)
        result.filter_applied = vfilter.describe()
        result.namespaces = [c.vector_namespace for c in collections]

        # `timed_stage` writes `duration_ms` into the span dict on exit, so each stage's
        # timing is read after its block rather than inside it.
        with timed_stage("search", ctx.mode.value) as span:
            hits = await self._search(result.queries, collections, vfilter, result)
            span["arms"] = len(hits)
        result.timings_ms["search"] = span["duration_ms"]

        fused = reciprocal_rank_fusion(*hits, k=self._config.rrf_k)
        result.fused_count = len(fused)
        retrieval_candidates.labels(stage="fused").observe(len(fused))
        if not fused:
            return result

        candidates = fused[: self._config.max_rerank_candidates]

        with timed_stage("acl_verify", ctx.mode.value) as span:
            candidates = await self._verify(candidates, ctx, result)
            span["kept"] = len(candidates)
        result.timings_ms["acl_verify"] = span["duration_ms"]
        if not candidates:
            return result

        with timed_stage("hydrate", ctx.mode.value) as span:
            chunks = await self._hydrate(candidates)
            span["hydrated"] = len(chunks)
        result.timings_ms["hydrate"] = span["duration_ms"]
        if not chunks:
            return result

        with timed_stage("rerank", ctx.mode.value) as span:
            reranked = await self._rerank(plan.original, chunks)
            span["kept"] = len(reranked)
        result.timings_ms["rerank"] = span["duration_ms"]

        result.chunks = reranked
        result.reranked_count = len(reranked)
        result.degraded_rerank = bool(getattr(self._reranker, "degraded", False))
        retrieval_candidates.labels(stage="reranked").observe(len(reranked))
        retrieval_top_score.labels(mode=ctx.mode.value).observe(result.top_score)
        return result

    # ── stages ──────────────────────────────────────────────────────────────

    async def _search(
        self,
        queries: list[str],
        collections: Sequence[Collection],
        vfilter: VectorFilter,
        result: RetrievalResult,
    ) -> list[list[VectorHit]]:
        """One ranked list per (query, namespace), all issued concurrently.

        Concurrency here is what makes multi-query rewriting affordable: three queries
        against two collections is six searches, which sequentially would be six round
        trips of latency and in parallel is one.
        """
        dense_vectors = await asyncio.gather(*(self._embedder.embed_query(q) for q in queries))
        sparse_vectors: list[Any] = [None] * len(queries)
        if self._config.hybrid and self._sparse is not None:
            sparse_vectors = list(
                await asyncio.gather(*(self._sparse.embed_query(q) for q in queries))
            )

        async def one(namespace: str, index: int) -> list[VectorHit]:
            sparse = sparse_vectors[index]
            if sparse is not None:
                return await self._store.search_hybrid(
                    namespace,
                    dense_vectors[index],
                    sparse,
                    limit=max(self._config.top_k_dense, self._config.top_k_sparse),
                    vfilter=vfilter,
                    rrf_k=self._config.rrf_k,
                )
            return await self._store.search_dense(
                namespace,
                dense_vectors[index],
                limit=self._config.top_k_dense,
                vfilter=vfilter,
            )

        tasks = [
            one(collection.vector_namespace, index)
            for collection in collections
            for index in range(len(queries))
        ]
        rankings = await asyncio.gather(*tasks, return_exceptions=True)

        lists: list[list[VectorHit]] = []
        for ranking in rankings:
            if isinstance(ranking, BaseException):
                # One namespace being unavailable degrades recall; failing the whole request
                # would turn a partial outage into a total one.
                logger.warning("retrieval.namespace_failed", error=str(ranking))
                continue
            lists.append(ranking)
            result.dense_count += len(ranking)
        return lists

    async def _verify(
        self, hits: list[VectorHit], ctx: SecurityContext, result: RetrievalResult
    ) -> list[VectorHit]:
        """Enforcement layer 2: ask PostgreSQL whether these chunks are really readable."""
        readable = await self._chunks.verify_readable(
            [h.chunk_id for h in hits],
            max_visibility_level=int(ctx.ceiling),
            department_path=ctx.department_path,
            granted_document_ids=list(ctx.granted_document_ids),
            mode=ctx.mode,
            now=datetime.now(UTC),
        )
        kept = [h for h in hits if h.chunk_id in readable]
        dropped = len(hits) - len(kept)
        if dropped:
            result.acl_dropped = dropped
            acl_layer2_drops.inc(dropped)
            # Loud on purpose. Steady-state this is zero; a non-zero value means the vector
            # payload disagrees with the database, which is a correctness bug with a security
            # blast radius, not a tuning issue.
            logger.warning(
                "retrieval.acl_layer2_dropped",
                dropped=dropped,
                candidates=len(hits),
                role=ctx.role.value,
                mode=ctx.mode.value,
            )
        return kept

    async def _hydrate(self, hits: list[VectorHit]) -> list[RetrievedChunk]:
        """Load chunk text and citation metadata, preserving fused order."""
        rows = await self._chunks.get_many([h.chunk_id for h in hits])
        by_id = {row.id: row for row in rows}
        titles = await self._titles({row.document_id for row in rows})

        chunks: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            row = by_id.get(hit.chunk_id)
            if row is None:
                # Indexed in the vector store but absent from PostgreSQL: a reconciliation
                # job's problem, not this request's.
                logger.warning("retrieval.orphan_vector", chunk_id=str(hit.chunk_id))
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    content=row.content,
                    locator=ChunkLocator(
                        document_id=row.document_id,
                        version_id=row.version_id,
                        document_title=titles.get(row.document_id, ""),
                        version_no=int(hit.payload.get("version_no", 1) or 1),
                        page_from=row.page_from,
                        page_to=row.page_to,
                        heading_path=tuple(row.heading_path),
                        section=row.section,
                        char_start=row.char_start,
                        char_end=row.char_end,
                    ),
                    chunk_type=row.chunk_type,
                    token_count=row.token_count,
                    visibility=Visibility.from_level(row.visibility_level),
                    department_path=row.department_path,
                    collection_id=row.collection_id,
                    content_hash=row.content_hash,
                    injection_flag=row.injection_flag,
                    score_fused=hit.score,
                    rank=rank,
                    metadata={"rrf": hit.payload.get("_rrf", {})},
                )
            )
        return chunks

    async def _titles(self, document_ids: set[UUID]) -> dict[UUID, str]:
        if not document_ids:
            return {}
        documents = await self._documents.get_many(list(document_ids))
        return {d.id: d.title for d in documents}

    async def _rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Cross-encode and keep the best ``rerank_top_n``.

        Structured chunks (tables, code, forms) are passed with their content intact rather
        than truncated to a sentence: a table row torn out of its header is not
        interpretable by the reranker any more than by a human.
        """
        passages = [self._passage(c) for c in chunks]
        scored = await self._reranker.rerank(question, passages, top_n=self._config.rerank_top_n)

        selected: list[RetrievedChunk] = []
        for rank, (index, score) in enumerate(scored, start=1):
            if index >= len(chunks):  # pragma: no cover - defensive against a bad adapter
                continue
            chunk = chunks[index]
            chunk.score_rerank = score
            chunk.rank = rank
            selected.append(chunk)
        return selected

    @staticmethod
    def _passage(chunk: RetrievedChunk) -> str:
        """What the reranker actually reads.

        The heading path is prepended for prose because a paragraph that says "20 weeks"
        without saying what of is not obviously an answer, while "Leave > Parental > 20
        weeks" is. Structured chunks keep their own layout.
        """
        if chunk.chunk_type in {ChunkType.TABLE, ChunkType.CODE, ChunkType.FORM}:
            return chunk.content[:4000]
        prefix = " > ".join(chunk.locator.heading_path[-2:])
        body = chunk.content[:2000]
        return f"{prefix}\n{body}" if prefix else body
