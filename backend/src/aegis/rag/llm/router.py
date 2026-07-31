"""Model selection, circuit breaking, and provider fallback.

Three concerns that are usually tangled into the call site:

**Which model may this caller use.** A customer session must not be able to select the
most expensive model, and an admin debugging a bad answer wants to. The allowlist is
role-scoped and checked here, not in the router's caller, because there is more than one
caller (chat, regeneration, evaluation runs) and a check per caller is a check that will
eventually be forgotten.

**What happens when a provider is down.** A hard failure of the primary provider should
degrade to the secondary, not to an error page. But retrying a provider that is failing
every request adds latency to every subsequent request, so a breaker opens after repeated
failures and the fallback becomes the primary until it closes.

**Cheap calls should use a cheap model.** Intent classification and query rewriting are
short, structured, and latency-critical; running them on the flagship model triples their
cost and their time-to-first-token for no measurable gain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.core.errors import ConfigurationError, ProviderError, ValidationError
from aegis.core.logging import get_logger
from aegis.core.telemetry import circuit_breaker_state
from aegis.domain.enums import Role

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aegis.core.config import Settings
    from aegis.domain.ports.infrastructure import (
        ChatMessage,
        Completion,
        CompletionUsage,
        LLMProvider,
    )

logger = get_logger(__name__)

# Which models each role may ask for by name. Absent an explicit request, the mode default
# applies, so this only constrains the `options.model` escape hatch.
MODEL_ALLOWLIST: dict[Role, frozenset[str]] = {
    Role.GUEST: frozenset(),
    Role.CUSTOMER: frozenset(),
    Role.INTERNAL_EMPLOYEE: frozenset({"gpt-5-mini", "gemini-2.5-flash", "claude-haiku-4"}),
    Role.MANAGER: frozenset(
        {
            "gpt-5-mini",
            "gpt-5.1",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "claude-haiku-4",
            "claude-sonnet-4",
        }
    ),
    Role.ADMIN: frozenset(),  # empty means "no restriction" for admin, see `resolve_model`
}


@dataclass(slots=True)
class Breaker:
    """Per-provider circuit breaker.

    Three failures inside the window opens it for ``cooldown``; the next call after the
    cooldown is a single probe (half-open), and one success closes it. The numbers are
    deliberately small: an LLM call is expensive enough that five failed attempts is
    already a visible latency regression for the user who triggered them.
    """

    threshold: int = 3
    cooldown: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        # Once the cooldown elapses the breaker is half-open: one request is let through, and
        # its outcome either closes the breaker or starts the cooldown again.
        return time.monotonic() - self.opened_at < self.cooldown

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()

    def state(self) -> int:
        if self.opened_at is None:
            return 0
        return 2 if self.is_open else 1


@dataclass(slots=True)
class LLMRouter:
    """Ordered provider chain with per-provider breakers."""

    providers: dict[str, LLMProvider]
    order: tuple[str, ...]
    default_model: str
    fast_model: str
    max_output_tokens: int = 2048
    breakers: dict[str, Breaker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.providers:
            raise ConfigurationError("no LLM provider is configured")
        for name in self.order:
            self.breakers.setdefault(name, Breaker())

    # ── selection ───────────────────────────────────────────────────────────

    def resolve_model(self, requested: str | None, *, role: Role, fast: bool = False) -> str:
        """Validate a caller's model request against their allowlist.

        Rejects rather than silently substituting. A customer who asks for a model they may
        not have should get a clear 422; quietly downgrading them makes an evaluation run
        that pins a model produce results for a different one.
        """
        if requested is None:
            return self.fast_model if fast else self.default_model
        allowed = MODEL_ALLOWLIST.get(role, frozenset())
        if role is Role.ADMIN:
            return requested
        if requested not in allowed:
            raise ValidationError(
                f"model {requested!r} is not available for role {role.value}",
                errors=[{"field": "options.model", "message": "not permitted for this role"}],
            )
        return requested

    def clamp_temperature(self, requested: float | None) -> float:
        """Grounded answering is not a creative writing task.

        Sampling entropy is hallucination surface: at temperature 0.7 a model will
        cheerfully rephrase "20 weeks" as "about five months", which is a different claim
        than the source made.
        """
        if requested is None:
            return 0.1
        return max(0.0, min(0.3, requested))

    def _chain(self, provider: str | None) -> list[str]:
        if provider is not None:
            if provider not in self.providers:
                raise ConfigurationError(f"provider {provider!r} is not configured")
            return [provider]
        healthy = [name for name in self.order if not self.breakers[name].is_open]
        # Everything is open: try the primary anyway rather than refusing outright, since a
        # breaker is a heuristic and the alternative is a guaranteed failure.
        return healthy or [self.order[0]]

    def _observe(self) -> None:
        for name, breaker in self.breakers.items():
            circuit_breaker_state.labels(provider=name).set(breaker.state())

    # ── calls ───────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
    ) -> Completion:
        last: ProviderError | None = None
        for name in self._chain(provider):
            breaker = self.breakers[name]
            try:
                result = await self.providers[name].complete(
                    messages,
                    model=model if name == self.order[0] else None,
                    temperature=temperature,
                    max_tokens=max_tokens or self.max_output_tokens,
                )
            except ProviderError as exc:
                breaker.record_failure()
                self._observe()
                last = exc
                logger.warning("llm.provider_failed", provider=name, error=str(exc))
                continue
            breaker.record_success()
            self._observe()
            return result
        self._observe()
        raise last or ProviderError("llm", "no provider produced a completion")

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
        usage_sink: CompletionUsage | None = None,
    ) -> AsyncIterator[str]:
        """Stream from the first provider that yields a token.

        Fallback only applies *before* the first token. Once bytes have reached the client,
        switching providers would splice two different answers together — the user would
        read a sentence that no model actually wrote. After first token, a failure is
        surfaced as a stream error and the partial answer is persisted as such.
        """
        chain = self._chain(provider)
        last: ProviderError | None = None
        for index, name in enumerate(chain):
            breaker = self.breakers[name]
            started = False
            try:
                iterator = self.providers[name].stream(
                    messages,
                    model=model if index == 0 else None,
                    temperature=temperature,
                    max_tokens=max_tokens or self.max_output_tokens,
                    usage_sink=usage_sink,  # type: ignore[call-arg]
                )
                async for delta in iterator:
                    started = True
                    yield delta
            except ProviderError as exc:
                breaker.record_failure()
                self._observe()
                last = exc
                if started:
                    raise
                logger.warning(
                    "llm.stream_failed_before_first_token", provider=name, error=str(exc)
                )
                continue
            breaker.record_success()
            self._observe()
            return
        self._observe()
        raise last or ProviderError("llm", "no provider produced a stream")

    def count_tokens(self, text: str, *, model: str | None = None) -> int:
        return self.providers[self.order[0]].count_tokens(text, model=model)

    async def health(self) -> dict[str, bool]:
        return {name: await p.health() for name, p in self.providers.items()}

    async def close(self) -> None:
        for provider in self.providers.values():
            await provider.close()


def _make_provider(name: str, model: str, settings: Settings) -> LLMProvider | None:
    """Instantiate one provider, or ``None`` when its credential is absent.

    Absence is not an error here: the chain is built from whatever is configured, and a
    fallback entry for a provider with no key is a deployment that simply has one fewer
    fallback.
    """
    from aegis.rag.llm.providers import AnthropicProvider, GeminiProvider, OpenAIProvider

    timeout = settings.llm_timeout_seconds
    match name:
        case "openai" if settings.openai_api_key:
            return OpenAIProvider(
                settings.openai_api_key.get_secret_value(),
                model=model,
                base_url=settings.openai_base_url,
                timeout_seconds=timeout,
            )
        case "anthropic" if settings.anthropic_api_key:
            return AnthropicProvider(
                settings.anthropic_api_key.get_secret_value(), model=model, timeout_seconds=timeout
            )
        case "gemini" if settings.google_api_key:
            return GeminiProvider(
                settings.google_api_key.get_secret_value(), model=model, timeout_seconds=timeout
            )
    return None


def build_llm_router(settings: Settings) -> LLMRouter:
    """Construct the chain from configuration.

    ``LLM_PROVIDER``/``LLM_MODEL`` is the primary; ``LLM_FALLBACK_CHAIN`` (a
    ``provider:model`` list) supplies the rest in order. With no credentials at all the
    fake provider is used outside production, which is what lets the whole stack boot,
    ingest, retrieve, and answer offline — the difference between a demo you can run and
    a demo you can read about.
    """
    from aegis.rag.llm.providers import FakeLLM

    providers: dict[str, LLMProvider] = {}
    order: list[str] = []

    for name, model in [(settings.llm_provider, settings.llm_model), *settings.fallback_models()]:
        if name in providers:
            continue
        if (provider := _make_provider(name, model, settings)) is not None:
            providers[name] = provider
            order.append(name)

    if not providers:
        if settings.is_production:
            raise ConfigurationError(
                "no LLM provider configured: set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or GOOGLE_API_KEY"
            )
        logger.warning(
            "llm.using_fake_provider",
            reason="no api keys configured",
            consequence="answers are stubbed; retrieval and citations are real",
        )
        providers["fake"] = FakeLLM()
        order.append("fake")

    primary = providers[order[0]]
    fast = settings.intent_model if order[0] != "fake" else primary.model
    return LLMRouter(
        providers=providers,
        order=tuple(order),
        default_model=primary.model,
        fast_model=fast,
        max_output_tokens=settings.llm_max_output_tokens,
    )
