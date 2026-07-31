"""pgvector-backed embedding storage.

Only used when ``VECTOR_BACKEND=pgvector``. Kept in its own table rather than as a column on
``chunks`` so the base schema carries no multi-kilobyte vector per row when Qdrant is in use —
a 3072-dimension ``float4`` vector is ~12 KB, which would make every ``SELECT`` on ``chunks``
(the citation drawer, the chunk inspector, reindex streaming) drag the whole index through
memory for nothing.

The table is created by the initial migration regardless of backend, because an empty table
costs nothing and a deployment that switches backends should not need a migration to do it.
"""

from __future__ import annotations

from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from aegis.infrastructure.database.models.base import Base, fk_uuid

PGVECTOR_MAX_DIM = 2000
"""HNSW in pgvector indexes up to 2000 dimensions.

``text-embedding-3-large`` at 3072 exceeds that, so the pgvector backend either uses a
reduced-dimension model (BGE-M3 at 1024, OpenAI's ``dimensions`` parameter) or accepts a
sequential scan. The collection's configured dimension is validated against this at startup
rather than discovered when an index build fails.
"""


class ChunkEmbeddingModel(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        Index(
            "ix_chunk_emb_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[UUID] = fk_uuid("chunks.id")
    model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(PGVECTOR_MAX_DIM), nullable=False)

    __mapper_args__ = {"primary_key": [chunk_id]}


__all__ = ["PGVECTOR_MAX_DIM", "ChunkEmbeddingModel"]
