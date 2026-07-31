"""OpenAI, Anthropic, and Gemini behind one port.

Written against the HTTP APIs rather than the three vendor SDKs. That is a deliberate
tradeoff. The SDKs give convenience helpers and stay current with new parameters; they also
each bring a dependency tree, their own retry and timeout behaviour, their own logging, and
their own opinions about event loops. Three of them in one process means three sources of
truth for "what happens on a 429". The wire formats here are stable, small, and the parts we
use are the parts least likely to change.

Streaming is the primary path. ``complete`` exists for the short internal calls — intent
classification, query rewriting, title generation — where the caller wants a value rather
than an iterator.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import httpx

from aegis.core.errors import ProviderError
from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import ChatMessage, Completion, CompletionUsage
from aegis.rag.llm.base import (
    HttpProvider,
    approx_tokens,
    build_client,
    classify,
    observe_ttft,
    record_usage,
    with_retries,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


def _split_system(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Anthropic and Gemini take the system prompt out of band, OpenAI takes it inline."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    rest = [m for m in messages if m.role != "system"]
    return system, rest


class OpenAIProvider(HttpProvider):
    """OpenAI chat completions.

    Also the adapter for every OpenAI-compatible endpoint — vLLM, Ollama, TGI, Together,
    Groq, Azure with a rewritten base URL. That compatibility is the practical reason this
    is the default provider: a self-hosted deployment with no external egress uses the same
    code path as a hosted one.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-5.1",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 90,
        organization: str | None = None,
    ) -> None:
        self.model = model
        self._timeout = timeout_seconds
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if organization:
            headers["OpenAI-Organization"] = organization
        self._client = build_client(base_url, headers, timeout_seconds=timeout_seconds)

    def _body(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        model: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if stream:
            # Without this, a streamed response carries no usage block and cost has to be
            # estimated from character counts.
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Completion:
        async def call() -> Completion:
            started = time.perf_counter()
            response = await self._client.post(
                "/chat/completions",
                json=self._body(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    stream=False,
                ),
            )
            if response.status_code >= 400:
                raise classify(self.name, response.status_code, response.text)
            payload = response.json()
            choice = payload["choices"][0]
            usage = payload.get("usage", {})
            used_model = str(payload.get("model", model or self.model))
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            cost = record_usage(self.name, used_model, prompt_tokens, completion_tokens)
            return Completion(
                text=choice["message"]["content"] or "",
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    model=used_model,
                    provider=self.name,
                    ttft_ms=int((time.perf_counter() - started) * 1000),
                    finish_reason=choice.get("finish_reason"),
                ),
            )

        result: Completion = await with_retries(call, provider=self.name)
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        usage_sink: CompletionUsage | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas.

        ``usage_sink`` is mutated with token counts and TTFT as they become known. An
        async generator cannot return a value, and the alternative — yielding a union of
        text and usage objects — pushes type narrowing onto every consumer.
        """
        used_model = model or self.model
        started = time.perf_counter()
        first_token_at: float | None = None
        text_length = 0

        body = self._body(
            messages, temperature=temperature, max_tokens=max_tokens, model=model, stream=True
        )
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    raise classify(
                        self.name, response.status_code, (await response.aread()).decode()
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    if usage := event.get("usage"):
                        prompt_tokens = int(usage.get("prompt_tokens", 0))
                        completion_tokens = int(usage.get("completion_tokens", 0))
                        if usage_sink is not None:
                            usage_sink.prompt_tokens = prompt_tokens
                            usage_sink.completion_tokens = completion_tokens
                    for choice in event.get("choices", []):
                        reason = choice.get("finish_reason")
                        if reason and usage_sink is not None:
                            usage_sink.finish_reason = reason
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                                observe_ttft(self.name, used_model, first_token_at - started)
                            text_length += len(delta)
                            yield delta
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(self.name, f"stream failed: {exc}", retryable=True) from exc
        finally:
            if usage_sink is not None:
                _finalise(usage_sink, self.name, used_model, started, first_token_at, text_length)


class AnthropicProvider(HttpProvider):
    """Claude messages API."""

    name = "anthropic"
    # Anthropic has no unauthenticated listing endpoint; /models needs the key, which is
    # exactly what we want to verify.
    _health_path = "/models"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-sonnet-4",
        base_url: str = "https://api.anthropic.com/v1",
        timeout_seconds: int = 90,
        api_version: str = "2023-06-01",
    ) -> None:
        self.model = model
        self._client = build_client(
            base_url,
            {
                "x-api-key": api_key,
                "anthropic-version": api_version,
                "Content-Type": "application/json",
            },
            timeout_seconds=timeout_seconds,
        )

    def _body(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        model: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        system, rest = _split_system(messages)
        body: dict[str, Any] = {
            "model": model or self.model,
            # Required by the API, unlike OpenAI where it is optional.
            "max_tokens": max_tokens or 2048,
            "messages": [{"role": m.role, "content": m.content} for m in rest],
            "stream": stream,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        return body

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Completion:
        async def call() -> Completion:
            started = time.perf_counter()
            response = await self._client.post(
                "/messages",
                json=self._body(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    stream=False,
                ),
            )
            if response.status_code >= 400:
                raise classify(self.name, response.status_code, response.text)
            payload = response.json()
            text = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            usage = payload.get("usage", {})
            used_model = str(payload.get("model", model or self.model))
            prompt_tokens = int(usage.get("input_tokens", 0))
            completion_tokens = int(usage.get("output_tokens", 0))
            cost = record_usage(self.name, used_model, prompt_tokens, completion_tokens)
            return Completion(
                text=text,
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    model=used_model,
                    provider=self.name,
                    ttft_ms=int((time.perf_counter() - started) * 1000),
                    finish_reason=payload.get("stop_reason"),
                ),
            )

        result: Completion = await with_retries(call, provider=self.name)
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        usage_sink: CompletionUsage | None = None,
    ) -> AsyncIterator[str]:
        used_model = model or self.model
        started = time.perf_counter()
        first_token_at: float | None = None
        text_length = 0
        body = self._body(
            messages, temperature=temperature, max_tokens=max_tokens, model=model, stream=True
        )
        try:
            async with self._client.stream("POST", "/messages", json=body) as response:
                if response.status_code >= 400:
                    raise classify(
                        self.name, response.status_code, (await response.aread()).decode()
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    event = json.loads(data)
                    kind = event.get("type")
                    if kind == "content_block_delta":
                        delta = (event.get("delta") or {}).get("text", "")
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                                observe_ttft(self.name, used_model, first_token_at - started)
                            text_length += len(delta)
                            yield delta
                    elif kind == "message_start" and usage_sink is not None:
                        usage = (event.get("message") or {}).get("usage", {})
                        usage_sink.prompt_tokens = int(usage.get("input_tokens", 0))
                    elif kind == "message_delta" and usage_sink is not None:
                        usage_sink.completion_tokens = int(
                            (event.get("usage") or {}).get("output_tokens", 0)
                        )
                        usage_sink.finish_reason = (event.get("delta") or {}).get("stop_reason")
                    elif kind == "error":
                        raise ProviderError(
                            self.name, str(event.get("error", {}).get("message", "stream error"))
                        )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(self.name, f"stream failed: {exc}", retryable=True) from exc
        finally:
            if usage_sink is not None:
                _finalise(usage_sink, self.name, used_model, started, first_token_at, text_length)


class GeminiProvider(HttpProvider):
    """Gemini generateContent / streamGenerateContent."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-pro",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: int = 90,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = build_client(
            base_url, {"Content-Type": "application/json"}, timeout_seconds=timeout_seconds
        )

    def _body(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        system, rest = _split_system(messages)
        body: dict[str, Any] = {
            "contents": [
                # Gemini calls the assistant "model"; everything else maps directly.
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [{"text": m.content}],
                }
                for m in rest
            ],
            "generationConfig": {},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if temperature is not None:
            body["generationConfig"]["temperature"] = temperature
        if max_tokens is not None:
            body["generationConfig"]["maxOutputTokens"] = max_tokens
        return body

    @staticmethod
    def _text(payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for candidate in payload.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if text := part.get("text"):
                    parts.append(text)
        return "".join(parts)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Completion:
        used_model = model or self.model

        async def call() -> Completion:
            started = time.perf_counter()
            response = await self._client.post(
                f"/models/{used_model}:generateContent",
                params={"key": self._api_key},
                json=self._body(messages, temperature=temperature, max_tokens=max_tokens),
            )
            if response.status_code >= 400:
                raise classify(self.name, response.status_code, response.text)
            payload = response.json()
            usage = payload.get("usageMetadata", {})
            prompt_tokens = int(usage.get("promptTokenCount", 0))
            completion_tokens = int(usage.get("candidatesTokenCount", 0))
            cost = record_usage(self.name, used_model, prompt_tokens, completion_tokens)
            candidates = payload.get("candidates") or [{}]
            return Completion(
                text=self._text(payload),
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    model=used_model,
                    provider=self.name,
                    ttft_ms=int((time.perf_counter() - started) * 1000),
                    finish_reason=candidates[0].get("finishReason"),
                ),
            )

        result: Completion = await with_retries(call, provider=self.name)
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        usage_sink: CompletionUsage | None = None,
    ) -> AsyncIterator[str]:
        used_model = model or self.model
        started = time.perf_counter()
        first_token_at: float | None = None
        text_length = 0
        try:
            async with self._client.stream(
                "POST",
                f"/models/{used_model}:streamGenerateContent",
                params={"key": self._api_key, "alt": "sse"},
                json=self._body(messages, temperature=temperature, max_tokens=max_tokens),
            ) as response:
                if response.status_code >= 400:
                    raise classify(
                        self.name, response.status_code, (await response.aread()).decode()
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    event = json.loads(data)
                    usage = event.get("usageMetadata")
                    if usage and usage_sink is not None:
                        usage_sink.prompt_tokens = int(usage.get("promptTokenCount", 0))
                        usage_sink.completion_tokens = int(usage.get("candidatesTokenCount", 0))
                    delta = self._text(event)
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            observe_ttft(self.name, used_model, first_token_at - started)
                        text_length += len(delta)
                        yield delta
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(self.name, f"stream failed: {exc}", retryable=True) from exc
        finally:
            if usage_sink is not None:
                _finalise(usage_sink, self.name, used_model, started, first_token_at, text_length)


class FakeLLM:
    """Deterministic provider for tests and offline development.

    Echoes a grounded-looking answer with citation markers so the citation validator, the
    SSE protocol, and the frontend can all be exercised without a key or a network.
    """

    name = "fake"

    def __init__(self, model: str = "fake-1", *, reply: str | None = None) -> None:
        self.model = model
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    def _answer(self, messages: list[ChatMessage]) -> str:
        if self._reply is not None:
            return self._reply
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context = "\n".join(m.content for m in messages if m.role == "system")
        markers = "[1]" if "[1]" in context else ""
        return f"Based on the provided sources{markers}, here is the answer to: {question[:160]}"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        text = self._answer(messages)
        return Completion(
            text=text,
            usage=CompletionUsage(
                prompt_tokens=sum(approx_tokens(m.content) for m in messages),
                completion_tokens=approx_tokens(text),
                model=model or self.model,
                provider=self.name,
                ttft_ms=1,
                finish_reason="stop",
            ),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        usage_sink: CompletionUsage | None = None,
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        text = self._answer(messages)
        for word in text.split(" "):
            yield word + " "
        if usage_sink is not None:
            usage_sink.prompt_tokens = sum(approx_tokens(m.content) for m in messages)
            usage_sink.completion_tokens = approx_tokens(text)
            usage_sink.model = model or self.model
            usage_sink.provider = self.name
            usage_sink.finish_reason = "stop"

    def count_tokens(self, text: str, *, model: str | None = None) -> int:
        return approx_tokens(text)

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _finalise(
    usage: CompletionUsage,
    provider: str,
    model: str,
    started: float,
    first_token_at: float | None,
    text_length: int,
) -> None:
    """Fill in whatever the provider did not report, and record the counters once."""
    usage.model = model
    usage.provider = provider
    if first_token_at is not None:
        usage.ttft_ms = int((first_token_at - started) * 1000)
    if usage.completion_tokens == 0 and text_length:
        usage.completion_tokens = max(1, text_length // 4)
    usage.cost_usd = record_usage(provider, model, usage.prompt_tokens, usage.completion_tokens)


__all__ = [
    "AnthropicProvider",
    "FakeLLM",
    "GeminiProvider",
    "OpenAIProvider",
]
