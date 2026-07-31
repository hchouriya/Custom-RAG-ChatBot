"""The concrete embedding providers.

Four adapters over :class:`RemoteEmbedder`, each a payload shape and a response path. They are
in one module because that is all they are; splitting forty lines each across four files would
hide how little they differ, and how much the difference matters.

What does differ, and is the reason the port has separate document and query methods:

* **OpenAI** is symmetric. Documents and queries are embedded identically.
* **Voyage** requires ``input_type``. Omitting it costs several points of recall.
* **Cohere** requires ``input_type`` with different values again, and rejects the request
  without it.
* **BGE via TEI** is symmetric for ``bge-m3`` but needs a query instruction prefix for the
  ``v1.5`` family. Getting this wrong is invisible: no error, just worse retrieval.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from aegis.core.errors import ProviderError
from aegis.domain.values import EmbeddingVector
from aegis.rag.embeddings.base import RemoteEmbedder, l2_normalize

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
"""Required by ``bge-*-v1.5``, harmful to ``bge-m3``.

The v1.5 models were trained with this prefix on the query side only. Applying it to documents,
or omitting it for queries, degrades recall by a margin that shows up in evaluation but never
in an error message.
"""


class OpenAIEmbedder(RemoteEmbedder):
    """``text-embedding-3-*``.

    ``dimensions`` is sent explicitly because the 3-series supports Matryoshka truncation. It
    lets a deployment trade a little accuracy for a third of the storage, and it means the
    configured dimension is what the index gets rather than whatever the default is this month.
    """

    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", "https://api.openai.com/v1")
        super().__init__(**kwargs)

    @property
    def _endpoint(self) -> str:
        return "/embeddings"

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self.dim
        return payload

    def _parse(self, body: Any) -> list[list[float]]:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise ProviderError(self.name, "malformed response: no data array", retryable=False)
        # Sorted by index because the API documents order but does not guarantee it, and a
        # misordered batch silently attaches every vector to the wrong chunk.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]


class VoyageEmbedder(RemoteEmbedder):
    """``voyage-3`` family. Strong on long-context retrieval and code."""

    name = "voyage"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", "https://api.voyageai.com/v1")
        super().__init__(**kwargs)

    @property
    def _endpoint(self) -> str:
        return "/embeddings"

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": texts,
            "input_type": "query" if is_query else "document",
            "truncation": True,
            "output_dimension": self.dim,
        }

    def _parse(self, body: Any) -> list[list[float]]:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise ProviderError(self.name, "malformed response: no data array", retryable=False)
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]


class CohereEmbedder(RemoteEmbedder):
    """``embed-*`` v2. ``input_type`` is mandatory, not advisory."""

    name = "cohere"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", "https://api.cohere.com/v2")
        super().__init__(**kwargs)

    @property
    def _endpoint(self) -> str:
        return "/embed"

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "texts": texts,
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
            "truncate": "END",
        }

    def _parse(self, body: Any) -> list[list[float]]:
        embeddings = body.get("embeddings") if isinstance(body, dict) else None
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float")
        if not isinstance(embeddings, list):
            raise ProviderError(self.name, "malformed response: no embeddings", retryable=False)
        return embeddings


class TeiEmbedder(RemoteEmbedder):
    """Self-hosted BGE (or any model) behind a HuggingFace text-embeddings-inference sidecar.

    The default provider, and the reason the test suite needs no API keys. It also keeps
    document text inside the deployment's own network, which for a corpus of internal policies
    is usually a compliance requirement rather than a preference.
    """

    name = "bge_tei"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", "http://localhost:8080")
        # A local sidecar needs no key, and sending an empty bearer token makes TEI reject
        # the request outright.
        kwargs.setdefault("api_key", None)
        super().__init__(**kwargs)

    @property
    def _endpoint(self) -> str:
        return "/embed"

    @property
    def _needs_query_instruction(self) -> bool:
        lowered = self.model.lower()
        return "bge" in lowered and "m3" not in lowered

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        if is_query and self._needs_query_instruction:
            texts = [f"{BGE_QUERY_INSTRUCTION}{t}" for t in texts]
        return {"inputs": texts, "normalize": True, "truncate": True}

    def _parse(self, body: Any) -> list[list[float]]:
        if not isinstance(body, list):
            raise ProviderError(self.name, "malformed response: expected a list", retryable=False)
        return body


class FakeEmbedder:
    """Deterministic pseudo-embeddings for tests and offline development.

    Not random: the vector is derived from the text's own token hashes, so equal text embeds
    equally and texts sharing vocabulary land closer together than texts that do not. That is
    the minimum needed for retrieval tests to assert on ranking rather than on mock call
    counts, which is the difference between a test that catches a regression and one that
    passes after the pipeline breaks.
    """

    name = "fake"

    def __init__(self, *, model: str = "fake", dim: int = 1024) -> None:
        self.model = model
        self.dim = dim

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(values=self._vector(t), model=self.model) for t in texts]

    async def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=self._vector(text), model=self.model)

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    def _vector(self, text: str) -> tuple[float, ...]:
        buckets = [0.0] * self.dim
        for term in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            buckets[index] += sign
        if not any(buckets):
            buckets[0] = 1.0
        return l2_normalize(buckets)
