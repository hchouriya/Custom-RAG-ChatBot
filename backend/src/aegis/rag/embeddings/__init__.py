"""Embeddings: dense providers, the cache in front of them, and the sparse BM25 encoder.

One factory, so provider selection exists in exactly one place. Anything that constructs an
embedder directly is a second place where a missing API key can be discovered at runtime
instead of at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aegis.core.errors import ConfigurationError
from aegis.rag.embeddings.base import RemoteEmbedder, l2_normalize
from aegis.rag.embeddings.cached import CachedEmbedder
from aegis.rag.embeddings.providers import (
    CohereEmbedder,
    FakeEmbedder,
    OpenAIEmbedder,
    TeiEmbedder,
    VoyageEmbedder,
)
from aegis.rag.embeddings.sparse import (
    Bm25SparseEncoder,
    CorpusStatistics,
    term_index,
    tokenize,
)

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.domain.ports.embeddings import EmbeddingProvider
    from aegis.domain.ports.infrastructure import Cache

__all__ = [
    "Bm25SparseEncoder",
    "CachedEmbedder",
    "CohereEmbedder",
    "CorpusStatistics",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "RemoteEmbedder",
    "TeiEmbedder",
    "VoyageEmbedder",
    "build_embedding_provider",
    "build_sparse_encoder",
    "l2_normalize",
    "term_index",
    "tokenize",
]


def build_embedding_provider(
    settings: Settings, *, cache: Cache | None = None
) -> EmbeddingProvider:
    """Construct the configured provider, wrapped in the cache when one is available.

    Raises :class:`ConfigurationError` rather than returning a degraded provider. An embedder
    that cannot embed produces an index that cannot be searched, and discovering that on the
    first user query is strictly worse than failing to boot.
    """
    provider = settings.embedding_provider
    common: dict[str, Any] = {
        "model": settings.embedding_model,
        "dim": settings.embedding_dim,
        "batch_size": settings.embedding_batch_size,
        "max_concurrency": settings.embedding_max_concurrency,
    }

    embedder: EmbeddingProvider
    if provider == "fake":
        embedder = FakeEmbedder(model=settings.embedding_model, dim=settings.embedding_dim)
    elif provider == "bge_tei":
        embedder = TeiEmbedder(base_url=settings.tei_embed_url, **common)
    elif provider == "openai":
        embedder = OpenAIEmbedder(api_key=_key(settings.openai_api_key, provider), **common)
    elif provider == "voyage":
        embedder = VoyageEmbedder(api_key=_key(settings.voyage_api_key, provider), **common)
    elif provider == "cohere":
        embedder = CohereEmbedder(api_key=_key(settings.cohere_api_key, provider), **common)
    else:
        raise ConfigurationError(f"unknown embedding provider: {provider}")

    if cache is None:
        return embedder
    return CachedEmbedder(embedder, cache, ttl_seconds=settings.embedding_cache_ttl_seconds)


def build_sparse_encoder(statistics: CorpusStatistics | None = None) -> Bm25SparseEncoder:
    return Bm25SparseEncoder(statistics)


def _key(secret: object, provider: str) -> str:
    value = secret.get_secret_value() if hasattr(secret, "get_secret_value") else None
    if not value:
        raise ConfigurationError(f"EMBEDDING_PROVIDER={provider} requires its API key")
    return str(value)
