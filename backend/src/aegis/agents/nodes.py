"""Pipeline nodes.

Each node is an async function over :class:`AnswerState`: it reads what earlier nodes wrote
and writes its own result. No node performs I/O it does not own, no node returns a value, and
no node knows what comes after it.

That shape is what makes the graph swappable. The default runner in ``pipeline.py`` calls
these in sequence with two conditional edges; ``langgraph_runner.py`` hands the identical
functions to a ``StateGraph`` when LangGraph is installed. Because the state is one plain
object and the nodes are pure over it, neither runner is privileged and the tests exercise
the nodes directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aegis.core.logging import get_logger
from aegis.core.telemetry import confidence_gate, timed_stage
from aegis.domain.enums import AnswerStatus, Intent
from aegis.domain.policies.confidence import REFUSAL_MESSAGE
from aegis.rag.prompts.templates import (
    CLARIFY_PROMPT,
    GREETING_PROMPT,
    scope_description,
    system_prompt,
    user_prompt,
)
from aegis.rag.retrieval.context import allocate_budget, assemble

if TYPE_CHECKING:
    from aegis.agents.deps import PipelineDeps
    from aegis.agents.state import AnswerState

logger = get_logger(__name__)


async def guard_input(state: AnswerState, deps: PipelineDeps) -> None:
    """Reject hostile input before spending anything on it."""
    with timed_stage("guardrail_input", state.mode.value):
        verdict = deps.guardrails.check_input(state.request.question, mode=state.mode)
    if verdict.suspicious:
        state.guardrail_notes.append(f"input_suspicious:{','.join(verdict.categories)}")
    state.request.question = verdict.question


async def plan_query(state: AnswerState, deps: PipelineDeps) -> None:
    state.plan = await deps.planner.plan(
        state.request.question, history=state.request.history, mode=state.mode
    )


async def retrieve(state: AnswerState, deps: PipelineDeps) -> None:
    """Search, unless the intent says there is nothing to search for."""
    if state.plan is None or not state.plan.intent.needs_retrieval:
        return
    state.retrieval = await deps.retriever.retrieve(
        state.plan,
        state.request.ctx,
        state.request.collections,
        narrowing=state.request.narrowing,
    )


async def gate(state: AnswerState, deps: PipelineDeps) -> None:
    """Decide answer / clarify / refuse **before** generating.

    Refusing after generation would mean the model has already been shown context that did
    not support an answer, and we would be paying to discard its output.
    """
    if state.plan is not None and not state.plan.intent.needs_retrieval:
        return
    chunks = state.retrieval.chunks if state.retrieval else []
    with timed_stage("confidence", state.mode.value) as span:
        state.confidence = deps.gate.evaluate(
            state.request.question,
            chunks,
            degraded_rerank=bool(state.retrieval and state.retrieval.degraded_rerank),
        )
        span["decision"] = state.confidence.decision
    state.timings_ms["confidence"] = span["duration_ms"]
    confidence_gate.labels(decision=state.confidence.decision).inc()

    if state.confidence.decision == "refuse":
        state.status = AnswerStatus.NO_ANSWER
        state.refusal_reason = state.confidence.reason or "insufficient_context"
    elif state.confidence.decision == "clarify":
        state.status = AnswerStatus.CLARIFY
        state.refusal_reason = state.confidence.reason


async def build_prompt(state: AnswerState, deps: PipelineDeps) -> None:
    """Assemble the context block and the two prompt messages.

    Runs after the gate so a refusal costs no assembly, and before generation so the debug
    endpoint can stop right here.
    """
    intent = state.intent
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    if state.status is AnswerStatus.CLARIFY:
        coverage = "\n".join(
            f"- {c.locator.display()}"
            for c in (state.retrieval.chunks if state.retrieval else [])[:5]
        )
        state.system_prompt = system_prompt(
            state.request.ctx,
            intent=intent,
            assistant_name=deps.assistant_name,
            today=today,
            refusal=REFUSAL_MESSAGE,
            extra_rules=CLARIFY_PROMPT.format(
                reason=state.refusal_reason or "ambiguous", coverage=coverage or "- (nothing close)"
            ),
        )
        state.user_prompt = state.request.question
        return

    if intent in {Intent.GREETING, Intent.FEEDBACK}:
        state.system_prompt = GREETING_PROMPT.format(
            assistant_name=deps.assistant_name, scope=scope_description(state.mode)
        )
        state.user_prompt = state.request.question
        return

    budget, history = allocate_budget(
        question=state.request.question,
        history=state.request.history,
        summary=state.request.summary,
        count_tokens=deps.count_tokens,
        policy=deps.budget,
    )
    state.budget = budget
    state.request.history = history

    chunks = state.retrieval.chunks if state.retrieval else []
    state.context = assemble(
        chunks,
        state.request.question,
        budget=budget,
        count_tokens=deps.count_tokens,
        compression=deps.compression,
    )
    state.system_prompt = system_prompt(
        state.request.ctx,
        intent=intent,
        assistant_name=deps.assistant_name,
        today=today,
        refusal=REFUSAL_MESSAGE,
    )
    state.user_prompt = user_prompt(
        state.request.question, state.context.text, summary=state.request.summary
    )


async def finalise(state: AnswerState, deps: PipelineDeps) -> None:
    """Validate citations and run the output guardrail.

    Called after the answer text is complete, whether it arrived in one piece or as a stream.
    An answer that cites nothing while claiming facts is downgraded here rather than in the
    API layer, because "was this grounded?" is a property of the pipeline's own output.
    """
    from aegis.rag.retrieval.citations import extract, mark_unused

    sources = state.context.used if state.context else []
    if sources and state.answer:
        result = extract(
            state.answer,
            sources,
            state.context.contents if state.context else None,
            max_citations=state.request.max_citations,
            renumber=not state.request.stream,
        )
        state.answer = result.answer
        state.citations = mark_unused(result.citations, sources)
        state.invalid_markers = result.invalid_markers
        if result.invalid_markers:
            state.guardrail_notes.append(
                f"invalid_markers:{','.join(str(m) for m in result.invalid_markers)}"
            )
        # A grounded-answer product that produced an uncited answer has failed its own
        # contract, even when the prose looks right. Record it as such.
        if not result.citations and state.status is AnswerStatus.OK:
            state.status = AnswerStatus.NO_ANSWER
            state.refusal_reason = "no_valid_citations"
            state.answer = REFUSAL_MESSAGE
            logger.warning(
                "answer.ungrounded",
                mode=state.mode.value,
                sources=len(sources),
                intent=state.intent.value,
            )

    verdict = deps.guardrails.check_output(state.answer, mode=state.mode)
    state.answer = verdict.answer
    if verdict.redacted:
        state.guardrail_notes.append(f"secrets_redacted:{','.join(verdict.secret_categories)}")
    state.guardrail_notes.extend(verdict.notes)


def nearest_documents(state: AnswerState, limit: int = 3) -> list[dict[str, object]]:
    """The closest misses, for a refusal that is useful rather than merely correct.

    Telling the user what *is* covered turns a dead end into a next step, and the same list
    feeds the no-answer dashboard as a ranked content-gap backlog.
    """
    seen: dict[str, dict[str, object]] = {}
    for chunk in state.retrieval.chunks if state.retrieval else []:
        title = chunk.locator.document_title or "Untitled"
        if title not in seen:
            seen[title] = {"title": title, "score": round(chunk.best_score, 3)}
        if len(seen) >= limit:
            break
    return list(seen.values())
