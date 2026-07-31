"""What every remote embedding provider needs, implemented once.

The providers differ in a URL, a payload shape, and a response path. They do *not* differ in
what makes an embedding client production-worthy: batching, bounded concurrency, retry with
jitter on the failures worth retrying, a token-aware truncation guard, dimension validation,
and never logging the text being embedded. Putting that in each adapter would mean four
implementations of retry, three of which are subtly wrong.

Two decisions here are load-bearing:

* **Order is preserved and validated.** Callers zip vectors back onto chunks by position. A
  provider that returns results out of order, or drops one, would attach the wrong embedding
  to the wrong chunk — a corruption that produces no error and shows up months later as
  "search is bad".
* **Dimension is checked on every response.** A provider silently changing default dimensions
  is a real event, and a mixed-dimension index cannot be repaired by anything short of a full
  reindex.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
from typing import TYPE_CHECKING, Any

import httpx

from aegis.core.errors import ProviderError
from aegis.core.logging import get_logger
from aegis.domain.values import EmbeddingVector

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0
DEFAULT_TIMEOUT_SECONDS = 60.0

MAX_CHARS_PER_INPUT = 32_000
"""Hard truncation before the request is sent.

Every provider rejects or silently truncates over-long inputs, and both outcomes are worse
than truncating here: the rejection fails a whole batch because of one chunk, and the silent
version indexes a chunk whose vector describes only its first half. The chunker's max_tokens
keeps normal content far below this; this only catches the pathological case.
"""


class RemoteEmbedder:
    """Base class for HTTP embedding providers.

    Subclasses implement :meth:`_payload` and :meth:`_parse`, and nothing else.
    """

    name: str = "remote"

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        base_url: str,
        api_key: str | None = None,
        batch_size: int = 64,
        max_concurrency: int = 4,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=max_concurrency * 2),
        )

    # ── Subclass contract ───────────────────────────────────────────────────

    @property
    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, texts: list[str], *, is_query: bool) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, body: Any) -> list[list[float]]:
        raise NotImplementedError

    # ── Port implementation ─────────────────────────────────────────────────

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        results = await asyncio.gather(
            *(self._embed_batch(batch, is_query=False) for batch in batches)
        )
        return [vector for batch in results for vector in batch]

    async def embed_query(self, text: str) -> EmbeddingVector:
        vectors = await self._embed_batch([text], is_query=True)
        return vectors[0]

    async def health(self) -> bool:
        try:
            await self.embed_query("health")
        except Exception:
            return False
        return True

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── Internals ───────────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[EmbeddingVector]:
        prepared = [_truncate(t) for t in texts]
        async with self._semaphore:
            body = await self._post(self._payload(prepared, is_query=is_query))

        raw = self._parse(body)
        if len(raw) != len(prepared):
            raise ProviderError(
                self.name,
                f"expected {len(prepared)} embeddings, received {len(raw)}",
                retryable=False,
            )
        return [EmbeddingVector(values=tuple(self._validate(v)), model=self.model) for v in raw]

    def _validate(self, values: Sequence[float]) -> Sequence[float]:
        if len(values) != self.dim:
            raise ProviderError(
                self.name,
                f"model {self.model} returned dimension {len(values)}, expected {self.dim}; "
                "the index cannot mix dimensions, so this is a configuration error",
                retryable=False,
            )
        if not all(math.isfinite(v) for v in values):
            raise ProviderError(self.name, "embedding contained NaN or infinity", retryable=False)
        return values

    async def _post(self, payload: dict[str, Any]) -> Any:
        """POST with retry on the failures that retrying can fix.

        Jittered backoff, not fixed: a worker embedding thousands of chunks retries in lockstep
        otherwise, and synchronised retries are how a provider's rate limit turns into an
        outage that never recovers.
        """
        url = f"{self._base_url}{self._endpoint}"
        last: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.post(url, json=payload, headers=self._headers())
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ProviderError(self.name, f"transport error: {exc}", retryable=True)
            else:
                if response.status_code < 400:
                    return response.json()
                error = ProviderError(
                    self.name,
                    _error_detail(response),
                    retryable=response.status_code in RETRYABLE_STATUS,
                    status=response.status_code,
                )
                # A 4xx is a bad key, a bad model name, or a payload the provider rejects.
                # Retrying burns quota and delays the real error by half a minute.
                if not error.retryable:
                    raise error
                last = error

            if attempt == MAX_ATTEMPTS:
                break
            delay = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * 2 ** (attempt - 1))
            delay += random.random() * delay  # noqa: S311 - jitter, not a security decision
            logger.warning(
                "embedding_retry",
                provider=self.name,
                attempt=attempt,
                delay_seconds=round(delay, 2),
                error=str(last),
            )
            await asyncio.sleep(delay)

        raise last or ProviderError(self.name, "embedding request failed", retryable=True)


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_CHARS_PER_INPUT else text[:MAX_CHARS_PER_INPUT]


def _error_detail(response: httpx.Response) -> str:
    """Extract a provider's error message without echoing the request.

    The request body is the document text. It must never reach a log line, which is why the
    detail is built from the response alone.
    """
    detail = f"HTTP {response.status_code}"
    with contextlib.suppress(Exception):
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error") or body.get("message") or body.get("detail")
            if isinstance(error, dict):
                error = error.get("message")
            if error:
                detail = f"{detail}: {str(error)[:200]}"
    return detail


def l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    """Scale to unit length so cosine similarity equals a dot product.

    Applied where a provider does not already normalise. It matters because the vector stores
    are configured for cosine distance, and an unnormalised vector makes similarity depend on
    text length — long chunks would rank above short ones regardless of relevance.
    """
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return tuple(values)
    return tuple(v / norm for v in values)
