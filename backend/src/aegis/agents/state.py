"""Pipeline state and the events it emits.

The pipeline is a state machine, and this is its state. Every stage reads and writes one
object, which is what makes ``/admin/retrieval/debug`` possible: run the same nodes with
``generate=False`` and return the state, and you have the rewritten queries, the exact
filter, every candidate with every score, the compressed context, and the assembled prompt —
without paying for a completion.

Events are domain objects rather than pre-formatted SSE strings. The transport (SSE today, a
WebSocket or a batch response tomorrow) belongs to the API layer; the pipeline should not know
how its output is framed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from aegis.domain.enums import AnswerStatus, Intent, Mode

if TYPE_CHECKING:
    from aegis.domain.entities import Collection
    from aegis.domain.ports.infrastructure import CompletionUsage
    from aegis.domain.values import (
        Citation,
        Confidence,
        SecurityContext,
        TokenBudget,
        VectorFilter,
    )
    from aegis.rag.retrieval.context import AssembledContext
    from aegis.rag.retrieval.hybrid import RetrievalResult
    from aegis.rag.retrieval.query import QueryPlan


@dataclass(slots=True)
class AnswerRequest:
    """What the caller asked for, already validated and narrowed."""

    question: str
    ctx: SecurityContext
    collections: list[Collection]
    conversation_id: UUID | None = None
    message_id: UUID | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    summary: str | None = None
    narrowing: VectorFilter | None = None
    model: str | None = None
    temperature: float | None = None
    max_citations: int = 8
    stream: bool = True
    generate: bool = True
    """False for the retrieval-debug path: run every stage up to the prompt, then stop."""


@dataclass(slots=True)
class AnswerState:
    """Mutable state threaded through the pipeline."""

    request: AnswerRequest
    trace_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    plan: QueryPlan | None = None
    retrieval: RetrievalResult | None = None
    context: AssembledContext | None = None
    budget: TokenBudget | None = None
    confidence: Confidence | None = None

    system_prompt: str = ""
    user_prompt: str = ""

    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    status: AnswerStatus = AnswerStatus.OK
    refusal_reason: str | None = None
    clarification: str | None = None
    usage: CompletionUsage | None = None
    guardrail_notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def mode(self) -> Mode:
        return self.request.ctx.mode

    @property
    def intent(self) -> Intent:
        return self.plan.intent if self.plan else Intent.FACTUAL_LOOKUP

    @property
    def elapsed_ms(self) -> int:
        return int((datetime.now(UTC) - self.started_at).total_seconds() * 1000)

    def debug_payload(self) -> dict[str, Any]:
        """Everything an operator needs to explain one answer.

        Deliberately verbose and deliberately not the shape of a user-facing response: this
        is read by a human investigating a complaint, and omitting a stage to keep the JSON
        tidy is how investigations turn into guesswork.
        """
        retrieval = self.retrieval
        context = self.context
        return {
            "trace_id": self.trace_id,
            "mode": self.mode.value,
            "intent": self.intent.value,
            "question": self.request.question,
            "queries": list(self.plan.queries) if self.plan else [],
            "keywords": list(self.plan.keywords) if self.plan else [],
            "used_llm_for_planning": bool(self.plan and self.plan.used_llm),
            "filter_applied": retrieval.filter_applied if retrieval else {},
            "namespaces": retrieval.namespaces if retrieval else [],
            "counts": {
                "dense": retrieval.dense_count if retrieval else 0,
                "fused": retrieval.fused_count if retrieval else 0,
                "acl_dropped": retrieval.acl_dropped if retrieval else 0,
                "reranked": retrieval.reranked_count if retrieval else 0,
                "in_context": context.marker_count if context else 0,
                "dropped_for_budget": len(context.dropped) if context else 0,
            },
            "degraded_rerank": bool(retrieval and retrieval.degraded_rerank),
            "candidates": [
                {
                    "marker": index,
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.locator.document_id),
                    "document_title": chunk.locator.document_title,
                    "page": chunk.locator.page_from,
                    "section": chunk.locator.section,
                    "chunk_type": chunk.chunk_type.value,
                    "visibility": chunk.visibility.value,
                    "department_path": chunk.department_path,
                    "tokens": chunk.token_count,
                    "score_dense": chunk.score_dense,
                    "score_sparse": chunk.score_sparse,
                    "score_fused": chunk.score_fused,
                    "score_rerank": chunk.score_rerank,
                    "rrf": chunk.metadata.get("rrf", {}),
                    "excerpt": chunk.content[:400],
                }
                for index, chunk in enumerate(retrieval.chunks if retrieval else [], start=1)
            ],
            "confidence": None
            if self.confidence is None
            else {
                "decision": self.confidence.decision,
                "score": round(self.confidence.score, 4),
                "top_score": round(self.confidence.top_score, 4),
                "mean_top3": round(self.confidence.mean_top3, 4),
                "supporting_chunks": self.confidence.supporting_chunks,
                "entity_coverage": round(self.confidence.entity_coverage, 4),
                "reason": self.confidence.reason,
            },
            "budget": None
            if self.budget is None
            else {
                "context": self.budget.context,
                "history": self.budget.history,
                "summary": self.budget.summary,
                "question": self.budget.question,
                "completion_reserve": self.budget.completion_reserve,
            },
            "compression": None
            if context is None
            else {
                "chunks_compressed": context.compression.chunks_compressed,
                "tokens_before": context.compression.tokens_before,
                "tokens_after": context.compression.tokens_after,
                "saved_pct": round(context.compression.saved_pct, 1),
            },
            "prompt": {
                "system": self.system_prompt,
                "user": self.user_prompt,
                "context_tokens": context.tokens if context else 0,
            },
            "timings_ms": self.timings_ms | (retrieval.timings_ms if retrieval else {}),
            "status": self.status.value,
            "refusal_reason": self.refusal_reason,
        }


# ── Events ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MetaEvent:
    message_id: UUID | None
    trace_id: str | None
    model: str
    mode: str
    intent: str
    rewritten: tuple[str, ...]
    retrieved: int
    reranked: int


@dataclass(frozen=True, slots=True)
class CitationsEvent:
    citations: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    delta: str


@dataclass(frozen=True, slots=True)
class UsageEvent:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    ttft_ms: int | None
    total_ms: int
    confidence: float
    grounded: bool


@dataclass(frozen=True, slots=True)
class RefusalEvent:
    reason: str
    message: str
    detail: str
    nearest_documents: tuple[dict[str, Any], ...] = ()
    suggestions: tuple[str, ...] = ()
    escalation_available: bool = True


@dataclass(frozen=True, slots=True)
class ClarifyEvent:
    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DoneEvent:
    status: str
    message_id: UUID | None
    citations_used: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    code: str
    message: str
    retryable: bool = False


StreamEvent = (
    MetaEvent
    | CitationsEvent
    | TokenEvent
    | UsageEvent
    | RefusalEvent
    | ClarifyEvent
    | DoneEvent
    | ErrorEvent
)

EventName = Literal[
    "meta", "citations", "token", "usage", "refusal", "clarify", "done", "error", "heartbeat"
]
