"""The answer pipeline.

An async generator of domain events. The API layer frames them as SSE; a worker running an
evaluation set consumes the same events and ignores the token deltas. Nothing about the
transport is decided here.

Event order is a contract — ``meta`` → ``citations`` → ``token``* → ``usage`` → ``done`` — and
it exists because the client needs the source list before the prose starts referring to
``[2]``. A stream always ends with exactly one terminal event (``done`` or ``error``), so the
client's state machine has no ambiguous end state.

Cancellation is cooperative: the caller passes ``is_cancelled``, which is polled between
deltas. A user who closes the tab, or presses stop, must stop the spend; an abandoned tab that
keeps a completion running is a bill for output nobody will read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.agents.nodes import (
    build_prompt,
    finalise,
    gate,
    guard_input,
    nearest_documents,
    plan_query,
    retrieve,
)
from aegis.agents.state import (
    AnswerState,
    CitationsEvent,
    ClarifyEvent,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    RefusalEvent,
    TokenEvent,
    UsageEvent,
)
from aegis.core.errors import GuardrailViolationError, ProviderError
from aegis.core.logging import get_logger
from aegis.core.telemetry import answers_total, current_trace_id, timed_stage
from aegis.domain.enums import AnswerStatus
from aegis.domain.policies.confidence import REFUSAL_MESSAGE
from aegis.domain.ports.infrastructure import ChatMessage, CompletionUsage
from aegis.domain.values import Citation
from aegis.rag.retrieval.citations import best_quote

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from aegis.agents.deps import PipelineDeps
    from aegis.agents.state import AnswerRequest, StreamEvent

logger = get_logger(__name__)

SUGGESTION_FALLBACK = (
    "Try naming the specific policy, product, or document you have in mind.",
    "Ask about a narrower time period or region.",
)


class AnswerPipeline:
    """Runs the nodes, then generates."""

    def __init__(self, deps: PipelineDeps) -> None:
        self._deps = deps

    async def prepare(self, request: AnswerRequest) -> AnswerState:
        """Everything up to but excluding generation.

        Exposed separately because it is exactly what ``POST /admin/retrieval/debug`` needs:
        the full pipeline, read-only, with no completion to pay for.
        """
        state = AnswerState(request=request, trace_id=current_trace_id())
        await guard_input(state, self._deps)
        await plan_query(state, self._deps)
        with timed_stage("retrieve", state.mode.value) as span:
            await retrieve(state, self._deps)
            span["candidates"] = state.retrieval.fused_count if state.retrieval else 0
        state.timings_ms["retrieve"] = span["duration_ms"]
        await gate(state, self._deps)
        await build_prompt(state, self._deps)
        return state

    async def run(
        self,
        request: AnswerRequest,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AsyncIterator[tuple[StreamEvent, AnswerState]]:
        """Produce the event stream for one question.

        Each event is yielded with the state so the caller can persist the final message
        without a second pass over the pipeline's internals.
        """
        deps = self._deps
        try:
            state = await self.prepare(request)
        except GuardrailViolationError as exc:
            # A blocked question is a normal outcome, not a server fault: the caller turns
            # this into a refusal event on a 200 response.
            answers_total.labels(status="blocked", mode=request.ctx.mode.value).inc()
            blocked = AnswerState(request=request)
            blocked.status = AnswerStatus.REFUSED
            blocked.refusal_reason = exc.code
            yield (
                RefusalEvent(
                    reason=exc.code,
                    message=str(exc.detail or exc.title),
                    detail="",
                    escalation_available=False,
                ),
                blocked,
            )
            yield DoneEvent(status="refused", message_id=request.message_id), blocked
            return

        model = deps.llm.resolve_model(request.model, role=request.ctx.role)
        sources = state.context.used if state.context else []

        yield (
            MetaEvent(
                message_id=request.message_id,
                trace_id=state.trace_id,
                model=model,
                mode=state.mode.value,
                intent=state.intent.value,
                rewritten=tuple(state.plan.queries if state.plan else ()),
                retrieved=state.retrieval.fused_count if state.retrieval else 0,
                reranked=state.retrieval.reranked_count if state.retrieval else 0,
            ),
            state,
        )

        # ── refusal ─────────────────────────────────────────────────────────
        if state.status is AnswerStatus.NO_ANSWER:
            state.answer = REFUSAL_MESSAGE
            answers_total.labels(status="no_answer", mode=state.mode.value).inc()
            yield (
                RefusalEvent(
                    reason=state.refusal_reason or "insufficient_context",
                    message=REFUSAL_MESSAGE,
                    detail=_refusal_detail(state),
                    nearest_documents=tuple(nearest_documents(state)),
                    suggestions=SUGGESTION_FALLBACK,
                    escalation_available=True,
                ),
                state,
            )
            yield DoneEvent(status="no_answer", message_id=request.message_id), state
            return

        # ── sources up front ────────────────────────────────────────────────
        if sources:
            offered = tuple(
                Citation(
                    marker=index,
                    chunk_id=chunk.chunk_id,
                    locator=chunk.locator,
                    quote=best_quote(
                        (state.context.contents[index - 1] if state.context else chunk.content),
                        request.question,
                    )[0],
                    score_rerank=chunk.score_rerank,
                    rank=chunk.rank,
                )
                for index, chunk in enumerate(sources, start=1)
            )
            yield CitationsEvent(citations=offered), state

        if not request.generate:
            yield DoneEvent(status="prepared", message_id=request.message_id), state
            return

        # ── generate ────────────────────────────────────────────────────────
        messages = [
            ChatMessage(role="system", content=state.system_prompt),
            *(ChatMessage(role=role, content=content) for role, content in request.history),
            ChatMessage(role="user", content=state.user_prompt),
        ]
        usage = CompletionUsage()
        state.usage = usage
        chunks: list[str] = []

        try:
            iterator = deps.llm.stream(
                messages,
                model=model,
                temperature=deps.llm.clamp_temperature(request.temperature or deps.temperature),
                max_tokens=deps.max_output_tokens,
                usage_sink=usage,
            )
            async for delta in iterator:
                if is_cancelled is not None and is_cancelled():
                    state.status = AnswerStatus.ERROR
                    state.refusal_reason = "cancelled"
                    state.answer = "".join(chunks)
                    usage.finish_reason = "cancelled"
                    answers_total.labels(status="cancelled", mode=state.mode.value).inc()
                    yield DoneEvent(status="cancelled", message_id=request.message_id), state
                    return
                chunks.append(delta)
                yield TokenEvent(delta=delta), state
        except ProviderError as exc:
            state.status = AnswerStatus.ERROR
            state.refusal_reason = "provider_error"
            state.answer = "".join(chunks)
            answers_total.labels(status="error", mode=state.mode.value).inc()
            logger.error("answer.provider_failed", error=str(exc), provider=exc.provider)
            yield (
                ErrorEvent(
                    code="PROVIDER_UNAVAILABLE",
                    message="The model provider is unavailable. Please try again.",
                    retryable=True,
                ),
                state,
            )
            return

        state.answer = "".join(chunks)

        # ── validate ────────────────────────────────────────────────────────
        try:
            await finalise(state, deps)
        except GuardrailViolationError as exc:
            state.status = AnswerStatus.ERROR
            state.refusal_reason = exc.code
            answers_total.labels(status="blocked_output", mode=state.mode.value).inc()
            yield ErrorEvent(code=exc.code, message=str(exc.detail or exc.title)), state
            return

        if state.status is AnswerStatus.CLARIFY:
            yield ClarifyEvent(question=state.answer), state

        confidence = state.confidence.score if state.confidence else 0.0
        used = tuple(c.marker for c in state.citations if c.was_used)
        answers_total.labels(status=state.status.value, mode=state.mode.value).inc()
        yield (
            UsageEvent(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=round(usage.cost_usd, 6),
                ttft_ms=usage.ttft_ms,
                total_ms=state.elapsed_ms,
                confidence=round(confidence, 4),
                grounded=bool(used),
            ),
            state,
        )
        yield (
            DoneEvent(
                status=state.status.value, message_id=request.message_id, citations_used=used
            ),
            state,
        )

    async def answer(self, request: AnswerRequest) -> AnswerState:
        """Non-streaming convenience wrapper.

        Used by evaluation runs and by any client that would rather have one JSON object.
        Shares the streaming path exactly, so the two cannot drift.
        """
        request.stream = False
        final: AnswerState | None = None
        buffer: list[str] = []
        async for event, state in self.run(request):
            final = state
            if isinstance(event, TokenEvent):
                buffer.append(event.delta)
        if final is None:  # pragma: no cover - run always yields
            raise RuntimeError("pipeline produced no events")
        if buffer and not final.answer:
            final.answer = "".join(buffer)
        return final


def _refusal_detail(state: AnswerState) -> str:
    """One sentence explaining *why* nothing was answerable.

    Users accept "I could not find this" when it comes with what was searched and what was
    closest; without that it reads as the system being broken.
    """
    if not state.retrieval or not state.retrieval.chunks:
        if state.retrieval and state.retrieval.acl_dropped:
            return "Nothing you have access to covers this."
        return "No document in the searched collections covers this."
    titles = ", ".join(
        dict.fromkeys(c.locator.document_title or "Untitled" for c in state.retrieval.chunks[:3])
    )
    reason = (state.confidence.reason if state.confidence else "") or "insufficient_context"
    if "entity_not_covered" in reason:
        return (
            f"The closest matches ({titles}) discuss the topic "
            f"but not the specific case you asked about."
        )
    return f"The closest matches ({titles}) were not specific enough to answer."


def clarify_options(state: AnswerState) -> tuple[str, ...]:
    """Concrete options for a clarifying question, taken from what was actually retrieved."""
    if not state.retrieval:
        return ()
    seen: dict[str, None] = {}
    for chunk in state.retrieval.chunks[:5]:
        label = chunk.locator.section or (
            chunk.locator.heading_path[-1]
            if chunk.locator.heading_path
            else chunk.locator.document_title
        )
        if label:
            seen.setdefault(label, None)
    return tuple(seen)


ANSWER_PIPELINE_EVENTS = (
    "meta",
    "citations",
    "token",
    "usage",
    "refusal",
    "clarify",
    "done",
    "error",
)
