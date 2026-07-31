"""Embedding and reranking ports.

Documents and queries are embedded through separate methods even though several providers
treat them identically. Asymmetric models (E5, BGE with instruction prefixes, Voyage's
``input_type``) require different handling for each, and encoding a query with the document
prefix silently degrades recall — a failure that produces no error and no log line, only
worse answers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aegis.domain.values import EmbeddingVector, SparseVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces dense vectors."""

    name: str
    model: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed chunk texts, batched and concurrency-limited by the adapter.

        Order of the result must match the input, since callers zip it back onto chunks.
        """
        ...

    async def embed_query(self, text: str) -> EmbeddingVector: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """Produces term-weighted sparse vectors for keyword retrieval.

    Separate from the dense port because the two are independently swappable: a deployment
    may keep BM25 while changing its dense model, and vice versa.
    """

    name: str

    async def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    async def embed_query(self, text: str) -> SparseVector: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores ``(query, passage)`` pairs with a cross-encoder.

    Returns indices into the input list paired with scores, ordered best first, rather than
    reordered passages. The caller keeps its own richer objects and only applies the new
    order, so no metadata is lost in translation.
    """

    name: str
    model: str

    async def rerank(
        self, query: str, passages: list[str], *, top_n: int
    ) -> list[tuple[int, float]]: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...
