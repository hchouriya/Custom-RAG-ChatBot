"""Process-scoped dependency container.

Adapters are constructed once at startup and held here. Per-request services receive a
repository bundle bound to the request's session — never an adapter imported from a
service module. That split is what keeps ``services`` free of ``infrastructure`` and what
makes a worker or CLI able to reuse the same wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aegis.agents.deps import PipelineDeps
from aegis.agents.pipeline import AnswerPipeline
from aegis.core.config import Settings, get_settings
from aegis.core.logging import get_logger
from aegis.core.security import TokenCodec
from aegis.domain.policies.budget import BudgetPolicy
from aegis.domain.policies.confidence import ConfidenceGate
from aegis.infrastructure.cache import build_cache, build_rate_limiter, build_redis
from aegis.infrastructure.database.engine import create_engine, create_session_factory
from aegis.infrastructure.database.repositories import build_repositories
from aegis.infrastructure.queue import build_job_queue
from aegis.infrastructure.scanners import (
    build_injection_scanner,
    build_malware_scanner,
    build_secret_scanner,
)
from aegis.infrastructure.storage import build_object_store
from aegis.rag.chunking import build_chunk_router
from aegis.rag.embeddings import build_embedding_provider, build_sparse_encoder
from aegis.rag.guardrails import Guardrails
from aegis.rag.llm import build_llm_router
from aegis.rag.parsing import build_parser_registry
from aegis.rag.reranking import build_reranker
from aegis.rag.retrieval import HybridRetriever, QueryPlanner, RetrievalConfig
from aegis.rag.retrieval.compression import CompressionConfig
from aegis.rag.vector_stores import build_vector_store
from aegis.services.auth import AuthService
from aegis.services.chat import ChatService
from aegis.services.documents import DocumentService
from aegis.services.ingestion import IngestionService
from aegis.services.principal import PrincipalService

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from aegis.domain.ports.embeddings import EmbeddingProvider, SparseEmbeddingProvider
    from aegis.domain.ports.infrastructure import (
        Cache,
        JobQueue,
        MalwareScanner,
        ObjectStore,
        RateLimiter,
    )
    from aegis.domain.ports.repositories import Repositories
    from aegis.domain.ports.vector_store import VectorStore
    from aegis.rag.chunking.router import DefaultChunkRouter
    from aegis.rag.llm.router import LLMRouter
    from aegis.rag.parsing.registry import ParserRegistry

logger = get_logger(__name__)


@dataclass(slots=True)
class Container:
    """Long-lived clients and factories for one process."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    cache: Cache
    limiter: RateLimiter
    storage: ObjectStore
    queue: JobQueue
    vectors: VectorStore
    embedder: EmbeddingProvider
    sparse_encoder: SparseEmbeddingProvider | None
    llm: LLMRouter
    codec: TokenCodec
    parsers: ParserRegistry
    chunker: DefaultChunkRouter
    scanner: MalwareScanner
    guardrails: Guardrails
    planner: QueryPlanner
    reranker: Any
    pipeline_base: dict[str, Any] = field(default_factory=dict)
    _started: bool = False

    @classmethod
    def build(cls, settings: Settings | None = None) -> Container:
        """Construct adapters. Does not open connections — that is :meth:`startup`."""
        cfg = settings or get_settings()
        engine = create_engine(cfg)
        redis = build_redis(cfg)
        cache = build_cache(cfg, client=redis)
        limiter = build_rate_limiter(cfg, client=redis)
        embedder = build_embedding_provider(cfg, cache=cache)
        sparse = build_sparse_encoder()
        llm = build_llm_router(cfg)
        injection = build_injection_scanner(cfg)
        secrets = build_secret_scanner(cfg)
        guardrails = Guardrails(injection=injection, secrets=secrets)
        return cls(
            settings=cfg,
            engine=engine,
            session_factory=create_session_factory(engine),
            redis=redis,
            cache=cache,
            limiter=limiter,
            storage=build_object_store(cfg),
            queue=build_job_queue(cfg),
            vectors=build_vector_store(cfg, engine=engine),
            embedder=embedder,
            sparse_encoder=sparse,
            llm=llm,
            codec=TokenCodec(cfg),
            parsers=build_parser_registry(cfg),
            chunker=build_chunk_router(
                strategy=cfg.chunk_strategy,
                embedding_model=cfg.embedding_model,
                embedder=embedder,
            ),
            scanner=build_malware_scanner(cfg),
            guardrails=guardrails,
            planner=QueryPlanner(llm),
            reranker=build_reranker(cfg),
            pipeline_base={
                "gate": ConfidenceGate(
                    min_top_score=cfg.confidence_min_top_score,
                    min_supporting=cfg.confidence_min_supporting,
                    min_mean_top3=cfg.confidence_min_mean_top3,
                    min_entity_coverage=cfg.confidence_min_entity_coverage,
                ),
                "budget": BudgetPolicy(
                    prompt_cap=cfg.prompt_token_cap,
                    context_budget=cfg.context_token_budget,
                    history_budget=cfg.history_token_budget,
                    summary_budget=cfg.summary_token_budget,
                    completion_reserve=cfg.llm_max_output_tokens,
                ),
                "compression": CompressionConfig(),
                "assistant_name": cfg.app_name,
                "max_output_tokens": cfg.llm_max_output_tokens,
                "temperature": cfg.llm_temperature,
            },
        )

    async def startup(self) -> None:
        """Eagerly validate that critical dependencies are reachable."""
        if self._started:
            return
        # A ping here fails the boot rather than the first authenticated request.
        try:
            await self.redis.ping()
        except Exception as exc:
            logger.warning("container.redis_ping_failed", error=str(exc)[:200])
            if self.settings.is_production:
                raise
        self._started = True
        logger.info(
            "container.started",
            env=self.settings.app_env,
            vector_backend=self.settings.vector_backend,
            storage=self.settings.storage_backend,
            embedding=self.settings.embedding_provider,
        )

    async def shutdown(self) -> None:
        """Close pools and clients. Safe to call more than once."""
        try:
            await self.vectors.close()
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.warning("container.vectors_close_failed", error=str(exc)[:200])
        try:
            await self.redis.aclose()
        except Exception as exc:  # pragma: no cover
            logger.warning("container.redis_close_failed", error=str(exc)[:200])
        try:
            await self.engine.dispose()
        except Exception as exc:  # pragma: no cover
            logger.warning("container.engine_dispose_failed", error=str(exc)[:200])
        self._started = False
        logger.info("container.stopped")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[tuple[AsyncSession, Repositories]]:
        """Open a session and the repository bundle that shares it."""
        async with self.session_factory() as session:
            repos = build_repositories(session)
            try:
                yield session, repos
            except Exception:
                await session.rollback()
                raise

    # ── per-request service factories ───────────────────────────────────────

    def auth_service(self, repos: Repositories) -> AuthService:
        return AuthService(
            repos, settings=self.settings, codec=self.codec, cache=self.cache
        )

    def principal_service(self, repos: Repositories) -> PrincipalService:
        return PrincipalService(repos)

    def document_service(self, repos: Repositories) -> DocumentService:
        return DocumentService(
            repos,
            settings=self.settings,
            storage=self.storage,
            queue=self.queue,
            cache=self.cache,
        )

    def chat_service(self, repos: Repositories) -> ChatService:
        return ChatService(
            repos,
            pipeline=self.answer_pipeline(repos),
            settings=self.settings,
            cache=self.cache,
        )

    def ingestion_service(self, repos: Repositories) -> IngestionService:
        return IngestionService(
            repos,
            settings=self.settings,
            storage=self.storage,
            parsers=self.parsers,
            chunker=self.chunker,
            embedder=self.embedder,
            sparse_encoder=self.sparse_encoder,
            vector_store=self.vectors,
            scanner=self.scanner,
            guardrails=self.guardrails,
        )

    def answer_pipeline(self, repos: Repositories) -> AnswerPipeline:
        """Build a pipeline whose retriever is bound to this request's repositories.

        Hybrid retrieval re-verifies ACL against PostgreSQL and hydrates chunk text from
        it, so the retriever cannot be process-scoped — it must see the request's session.
        """
        retriever = HybridRetriever(
            vector_store=self.vectors,
            embedder=self.embedder,
            sparse_encoder=self.sparse_encoder,
            reranker=self.reranker,
            chunks=repos.chunks,
            documents=repos.documents,
            config=RetrievalConfig(
                top_k_dense=self.settings.retrieval_top_k_dense,
                top_k_sparse=self.settings.retrieval_top_k_sparse,
                rrf_k=self.settings.retrieval_rrf_k,
                rerank_top_n=self.settings.rerank_top_n,
            ),
        )
        deps = PipelineDeps(
            llm=self.llm,
            planner=self.planner,
            retriever=retriever,
            guardrails=self.guardrails,
            **self.pipeline_base,
        )
        return AnswerPipeline(deps)
