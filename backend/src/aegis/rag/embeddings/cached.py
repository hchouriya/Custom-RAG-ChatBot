"""Caching decorator for any embedding provider.

Re-embedding is pure waste, and it happens constantly: a document is re-uploaded with one
paragraph changed, a reindex runs after a chunker tweak that left most chunks byte-identical,
the same question is asked by forty people on the same morning. Every one of those pays the
provider again for a vector that is a deterministic function of the text.

The cache key includes the model *and* the dimension. Cheap to add, and it is what prevents the
worst possible cache hit: a vector produced by the previous embedding model being served after
a model change, poisoning an index that then looks fine.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING

from aegis.core.logging import get_logger
from aegis.domain.values import EmbeddingVector

if TYPE_CHECKING:
    from aegis.domain.ports.embeddings import EmbeddingProvider
    from aegis.domain.ports.infrastructure import Cache

logger = get_logger(__name__)

KEY_PREFIX = "emb"


class CachedEmbedder:
    """Wraps a provider with a read-through cache.

    Cache failures are swallowed by design. Redis being unavailable must slow ingestion down,
    not stop it — the cache is an optimisation, and treating it as a dependency would make an
    optional component load-bearing.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        cache: Cache,
        *,
        ttl_seconds: int = 604_800,
        cache_queries: bool = True,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds
        self._cache_queries = cache_queries
        self.name = f"cached:{inner.name}"
        self.model = inner.model
        self.dim = inner.dim

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []

        keys = [self._key(t) for t in texts]
        cached = await self._get_many(keys)

        # Deduplicate within the batch as well as against the cache. A document with a
        # repeated boilerplate paragraph would otherwise pay for it once per occurrence.
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached and key not in missing:
                missing[key] = text

        if missing:
            fresh = await self._inner.embed_documents(list(missing.values()))
            produced = dict(zip(missing.keys(), fresh, strict=True))
            cached.update(produced)
            await self._set_many({k: _encode(v) for k, v in produced.items()})

        return [cached[key] for key in keys]

    async def embed_query(self, text: str) -> EmbeddingVector:
        if not self._cache_queries:
            return await self._inner.embed_query(text)

        # Queries and documents are cached under different keys: asymmetric models embed them
        # differently, so serving one for the other would return a subtly wrong vector.
        key = self._key(text, kind="q")
        hit = await self._get_many([key])
        if key in hit:
            return hit[key]
        vector = await self._inner.embed_query(text)
        await self._set_many({key: _encode(vector)})
        return vector

    async def health(self) -> bool:
        return await self._inner.health()

    async def close(self) -> None:
        await self._inner.close()

    def _key(self, text: str, *, kind: str = "d") -> str:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]
        return f"{KEY_PREFIX}:{self.model}:{self.dim}:{kind}:{digest}"

    async def _get_many(self, keys: list[str]) -> dict[str, EmbeddingVector]:
        try:
            raw = await self._cache.get_many(keys)
        except Exception as exc:
            logger.warning("embedding_cache_read_failed", error=str(exc))
            return {}

        found: dict[str, EmbeddingVector] = {}
        for key, blob in raw.items():
            vector = _decode(blob, self.model, self.dim)
            if vector is not None:
                found[key] = vector
        return found

    async def _set_many(self, items: dict[str, bytes]) -> None:
        if not items:
            return
        try:
            await self._cache.set_many(items, ttl_seconds=self._ttl)
        except Exception as exc:
            logger.warning("embedding_cache_write_failed", error=str(exc))


def _encode(vector: EmbeddingVector) -> bytes:
    """Pack as little-endian float32.

    JSON would be roughly six times larger for the same vector and would spend real CPU on
    parsing during a reindex. float32 also matches what the vector stores keep, so nothing is
    lost by narrowing here.
    """
    return struct.pack(f"<{len(vector.values)}f", *vector.values)


def _decode(blob: bytes, model: str, dim: int) -> EmbeddingVector | None:
    """Unpack, rejecting anything the wrong size.

    A wrong-sized entry means the model or dimension changed under a key that did not, or the
    value was truncated in transit. Either way it is discarded rather than repaired: a
    plausible-looking wrong vector is worse than a cache miss.
    """
    if len(blob) != dim * 4:
        return None
    try:
        values = struct.unpack(f"<{dim}f", blob)
    except struct.error:
        return None
    return EmbeddingVector(values=values, model=model)
