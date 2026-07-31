"""Vector stores: three backends, one port, one filter algebra.

The factory is the only place that knows which backend is configured. Everything downstream
depends on the port, which is what makes "run the retrieval test suite against the in-memory
store, then against Qdrant" a configuration change rather than a rewrite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.core.errors import ConfigurationError
from aegis.rag.vector_stores.filters import (
    PAYLOAD_COLUMNS,
    ancestor_paths,
    matches,
    to_qdrant,
    to_sql,
)
from aegis.rag.vector_stores.fusion import reciprocal_rank_fusion
from aegis.rag.vector_stores.memory import InMemoryVectorStore
from aegis.rag.vector_stores.payload import (
    ACL_KEYS,
    PAYLOAD_KEYS,
    acl_payload,
    build_payload,
    to_epoch,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from aegis.core.config import Settings
    from aegis.domain.ports.vector_store import VectorStore

__all__ = [
    "ACL_KEYS",
    "PAYLOAD_COLUMNS",
    "PAYLOAD_KEYS",
    "InMemoryVectorStore",
    "acl_payload",
    "ancestor_paths",
    "build_payload",
    "build_vector_store",
    "matches",
    "reciprocal_rank_fusion",
    "to_epoch",
    "to_qdrant",
    "to_sql",
]


def build_vector_store(settings: Settings, *, engine: AsyncEngine | None = None) -> VectorStore:
    """Construct the configured backend.

    Qdrant and pgvector are imported lazily: a deployment on one should not need the other's
    client library present, and the in-memory store must stay importable with neither.
    """
    backend = settings.vector_backend

    if backend == "memory":
        if settings.is_production:
            raise ConfigurationError(
                "VECTOR_BACKEND=memory is not persistent and cannot be used in production"
            )
        return InMemoryVectorStore()

    if backend == "qdrant":
        from aegis.rag.vector_stores.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

    if backend == "pgvector":
        if engine is None:
            raise ConfigurationError("VECTOR_BACKEND=pgvector requires a database engine")
        from aegis.rag.vector_stores.pgvector import PgVectorStore

        return PgVectorStore(engine)

    raise ConfigurationError(f"unknown vector backend: {backend}")
