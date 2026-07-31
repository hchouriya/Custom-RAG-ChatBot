"""Cross-encoder reranking.

Reranking is the single highest-leverage stage in this pipeline. A bi-encoder embeds the
query and the passage independently, so it can only measure "are these about the same
topic". A cross-encoder reads both together and can tell that a passage about *paternity*
leave is not an answer to a question about *parental* leave for managers. In published
benchmarks and in this system's own evaluation set it moves nDCG@5 by 10-20 points, which is
the difference between the right paragraph being first and being seventh — and only the first
few survive the context budget.

The cost is a forward pass per candidate, which is why the funnel is wide then narrow:
40 + 40 candidates retrieved, ~60 unique fused, 8 kept. Reranking 60 short passages on a
GPU-backed TEI sidecar is 40-80 ms; reranking 500 would not be.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from aegis.core.errors import ProviderError
from aegis.core.logging import get_logger

if TYPE_CHECKING:
    from aegis.core.config import Settings

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=None)
# A single request with hundreds of pairs can exceed the sidecar's own batch limit and is
# slower than two concurrent halves.
_MAX_PAIRS_PER_REQUEST = 64


def _top(scores: list[tuple[int, float]], top_n: int) -> list[tuple[int, float]]:
    # Ties break on the original index so the output is deterministic, which matters for
    # evaluation runs and for snapshot tests.
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return scores[:top_n]


class TeiReranker:
    """Text Embeddings Inference ``/rerank``.

    The default because it runs the BGE reranker locally with no per-call cost and no data
    egress — a document corpus is the last thing an enterprise wants to ship to a third
    party one passage at a time.
    """

    name = "bge_tei"

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.model = model
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout or _TIMEOUT)

    async def _batch(self, query: str, passages: list[str], offset: int) -> list[tuple[int, float]]:
        try:
            response = await self._client.post(
                "/rerank",
                json={"query": query, "texts": passages, "raw_scores": False},
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(
                self.name, f"rerank transport failure: {exc}", retryable=True
            ) from exc
        if response.status_code >= 400:
            raise ProviderError(
                self.name,
                f"rerank HTTP {response.status_code}: {response.text[:200]}",
                retryable=response.status_code >= 500,
                status=response.status_code,
            )
        payload: list[dict[str, Any]] = response.json()
        return [(offset + int(item["index"]), float(item["score"])) for item in payload]

    async def rerank(
        self, query: str, passages: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        if not passages:
            return []
        windows = [
            (start, passages[start : start + _MAX_PAIRS_PER_REQUEST])
            for start in range(0, len(passages), _MAX_PAIRS_PER_REQUEST)
        ]
        results = await asyncio.gather(
            *(self._batch(query, window, start) for start, window in windows)
        )
        return _top([pair for batch in results for pair in batch], top_n)

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=3.0)
        except (httpx.TimeoutException, httpx.TransportError):
            return False
        return response.status_code < 500

    async def close(self) -> None:
        await self._client.aclose()


class CohereReranker:
    """Cohere ``/v2/rerank``.

    Strong quality with no infrastructure, at the price of sending passage text to a third
    party — acceptable for a customer-facing knowledge base of public content, generally not
    for an internal one. Which is why the default is local.
    """

    name = "cohere"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "rerank-v3.5",
        base_url: str = "https://api.cohere.com",
    ) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=_TIMEOUT,
        )

    async def rerank(
        self, query: str, passages: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        if not passages:
            return []
        try:
            response = await self._client.post(
                "/v2/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": passages,
                    "top_n": min(top_n, len(passages)),
                },
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(
                self.name, f"rerank transport failure: {exc}", retryable=True
            ) from exc
        if response.status_code >= 400:
            raise ProviderError(
                self.name,
                f"rerank HTTP {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in {429, 500, 502, 503, 504},
                status=response.status_code,
            )
        payload = response.json()
        return _top(
            [
                (int(item["index"]), float(item["relevance_score"]))
                for item in payload.get("results", [])
            ],
            top_n,
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        await self._client.aclose()


class CrossEncoderReranker:
    """In-process sentence-transformers cross-encoder.

    For deployments that want local reranking without a second container. The model runs in
    a thread so the event loop is not blocked, but it shares the API process's CPU with
    request handling — under load that contention shows up as latency on *every* endpoint,
    not just chat. The TEI sidecar is the better shape; this exists for single-box installs.
    """

    name = "cross_encoder"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", *, max_length: int = 512) -> None:
        self.model = model
        self._max_length = max_length
        self._encoder: Any | None = None
        self._lock = asyncio.Lock()

    async def _load(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        async with self._lock:
            if self._encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:  # pragma: no cover - optional extra
                    raise ProviderError(
                        self.name,
                        "sentence-transformers is not installed; "
                        "install the 'local-models' extra or use RERANKER_PROVIDER=bge_tei",
                    ) from exc
                self._encoder = await asyncio.to_thread(
                    CrossEncoder, self.model, max_length=self._max_length
                )
        return self._encoder

    async def rerank(
        self, query: str, passages: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        if not passages:
            return []
        encoder = await self._load()
        scores = await asyncio.to_thread(encoder.predict, [(query, p) for p in passages])
        return _top([(i, float(s)) for i, s in enumerate(scores)], top_n)

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self._encoder = None


class NoopReranker:
    """Keeps fusion order.

    Used when reranking is disabled and as the degraded path when the reranker is down.
    Retrieval without reranking is measurably worse but still useful; refusing to answer
    because a *ranking refinement* is unavailable would be the wrong tradeoff.
    """

    name = "noop"
    model = "none"

    async def rerank(
        self, query: str, passages: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        # Descending pseudo-scores preserve the incoming order after any later sort.
        return [(i, 1.0 - i / max(len(passages), 1)) for i in range(min(top_n, len(passages)))]

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class ResilientReranker:
    """Wraps a reranker so its failure degrades ranking instead of the answer."""

    def __init__(self, inner: Any, *, fallback: Any | None = None) -> None:
        self._inner = inner
        self._fallback = fallback or NoopReranker()
        self.name = inner.name
        self.model = inner.model
        self.degraded = False

    async def rerank(
        self, query: str, passages: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        try:
            result = await self._inner.rerank(query, passages, top_n=top_n)
        except ProviderError as exc:
            self.degraded = True
            logger.warning("rerank.degraded", provider=self.name, error=str(exc))
            return await self._fallback.rerank(query, passages, top_n=top_n)
        self.degraded = False
        return result

    async def health(self) -> bool:
        healthy: bool = await self._inner.health()
        return healthy

    async def close(self) -> None:
        await self._inner.close()


def build_reranker(settings: Settings) -> Any:
    """Construct the configured reranker, wrapped for resilience."""
    match settings.reranker_provider:
        case "bge_tei":
            inner: Any = TeiReranker(settings.tei_rerank_url, model=settings.reranker_model)
        case "cohere":
            key = settings.cohere_api_key
            inner = CohereReranker(
                key.get_secret_value() if key else "", model=settings.reranker_model
            )
        case "cross_encoder":
            inner = CrossEncoderReranker(settings.reranker_model)
        case _:
            return NoopReranker()
    return ResilientReranker(inner)
