"""Shared machinery for LLM providers.

Three providers with three different wire formats sit behind one port. The parts that are
genuinely common live here: the HTTP client with sane timeouts, retry classification, token
accounting, and cost estimation.

Timeouts are split rather than global. A single 90 s timeout would apply to the whole
response, which for a streaming call is wrong in both directions: it lets a stalled stream
hang for 90 s after the last token, and it kills a legitimately long answer. So the connect
and read timeouts are short and the overall budget is enforced by the caller.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

import httpx

from aegis.core.errors import ProviderError
from aegis.core.logging import get_logger
from aegis.core.telemetry import cost_usd_total, llm_ttft, provider_errors, tokens_total

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# USD per million tokens, (prompt, completion). Estimates: exact billing is the provider's
# invoice, and this exists to make an expensive prompt visible on a dashboard before the
# invoice arrives, not to reconcile against it.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.1": (2.50, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
DEFAULT_PRICE = (1.00, 4.00)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = PRICING.get(model, DEFAULT_PRICE)
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000


def record_usage(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Update the token and cost counters, returning the estimated cost."""
    tokens_total.labels(provider=provider, model=model, kind="prompt").inc(prompt_tokens)
    tokens_total.labels(provider=provider, model=model, kind="completion").inc(completion_tokens)
    cost = estimate_cost(model, prompt_tokens, completion_tokens)
    cost_usd_total.labels(provider=provider, model=model).inc(cost)
    return cost


def observe_ttft(provider: str, model: str, seconds: float) -> None:
    llm_ttft.labels(provider=provider, model=model).observe(seconds)


def build_client(
    base_url: str, headers: dict[str, str], *, timeout_seconds: int
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(
            connect=5.0,
            read=float(timeout_seconds),
            write=30.0,
            # No pool timeout: waiting for a connection is a queueing problem, and failing
            # fast here would surface as a provider error when the provider is fine.
            pool=None,
        ),
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        follow_redirects=False,
    )


def classify(provider: str, status: int, body: str) -> ProviderError:
    """Map an HTTP failure onto a retry decision.

    4xx other than 408/409/425/429 is a bug or a bad key: retrying burns quota and delays
    the real error by the length of the backoff. 5xx and 429 are worth retrying.
    """
    retryable = status in RETRYABLE_STATUS
    provider_errors.labels(provider=provider, code=str(status)).inc()
    detail = body[:300].replace("\n", " ")
    return ProviderError(provider, f"HTTP {status}: {detail}", retryable=retryable, status=status)


async def with_retries(
    call: Callable[[], Awaitable[Any]],
    *,
    provider: str,
    attempts: int = 3,
    base_delay: float = 0.5,
) -> Any:
    """Retry with full jitter.

    Full jitter rather than fixed backoff because retries from many workers after a shared
    429 arrive in lockstep otherwise, which re-creates the burst that caused the throttle.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt == attempts:
                raise
            delay = random.uniform(0, base_delay * 2 ** (attempt - 1))  # noqa: S311 - jitter, not crypto
            logger.warning(
                "llm.retrying",
                provider=provider,
                attempt=attempt,
                delay_ms=round(delay * 1000),
                error=str(exc),
            )
            await asyncio.sleep(delay)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt == attempts:
                provider_errors.labels(provider=provider, code="transport").inc()
                raise ProviderError(provider, f"transport failure: {exc}", retryable=True) from exc
            await asyncio.sleep(random.uniform(0, base_delay * 2 ** (attempt - 1)))  # noqa: S311
    raise ProviderError(provider, f"exhausted retries: {last}", retryable=True)


class HttpProvider:
    """Behaviour shared by the three HTTP-backed providers.

    Token counting uses the tiktoken encodings for every provider, including Anthropic and
    Gemini whose tokenizers are not public. That is an approximation, but a much better one
    than characters-divided-by-four, and it is only used for prompt *budgeting* — the
    authoritative counts come back in the response's usage block.
    """

    name: str
    model: str
    _client: httpx.AsyncClient
    _health_path: str = "/models"

    def count_tokens(self, text: str, *, model: str | None = None) -> int:
        from aegis.rag.chunking.tokenizer import get_token_counter

        return get_token_counter(model or self.model).count(text)

    async def health(self) -> bool:
        try:
            response = await self._client.get(self._health_path, timeout=5.0)
        except (httpx.TimeoutException, httpx.TransportError):
            return False
        # 401/403 means the endpoint is reachable and our key is wrong, which is a
        # configuration failure rather than a dependency outage; readiness reports it
        # separately so an operator is not sent to look at the wrong system.
        return response.status_code < 500

    async def close(self) -> None:
        await self._client.aclose()


def approx_tokens(text: str) -> int:
    """Cheap token estimate for providers that do not report usage on a stream.

    Deliberately an over-estimate: budgeting with a low number overflows the context window,
    which fails the request, while a high number only wastes a little headroom.
    """
    return max(1, len(text) // 3)


def sse_lines(chunk: str) -> list[str]:
    """Split an SSE chunk into complete ``data:`` payloads."""
    return [
        line[5:].strip()
        for line in chunk.splitlines()
        if line.startswith("data:") and line[5:].strip()
    ]
