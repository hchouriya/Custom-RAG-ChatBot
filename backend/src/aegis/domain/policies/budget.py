"""Token budgeting.

Prompt size drives both cost and latency, and past a well-reranked handful of chunks it
stops buying accuracy — so the budget is a hard constraint decided here rather than an
emergent property of however much text retrieval happened to return.

The allocation is fixed-share with borrowing: each segment has a nominal allowance, unused
allowance flows to context (the segment that most benefits from more room), and overflow is
resolved by a defined order of sacrifice. The completion reserve is never borrowed from —
truncating the model's output to fit more input produces a cut-off answer, which is a worse
failure than a slightly thinner context.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.domain.values import TokenBudget

# Nominal allowances, from docs/architecture/05 §3.
SYSTEM_PROMPT_TOKENS = 900
MODE_RULES_TOKENS = 200
QUESTION_TOKENS = 300


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Budget parameters, sourced from ``Settings`` (or a runtime override)."""

    prompt_cap: int = 16_000
    context_budget: int = 10_000
    history_budget: int = 2_000
    summary_budget: int = 600
    completion_reserve: int = 2_000

    def allocate(
        self,
        *,
        question_tokens: int,
        summary_tokens: int = 0,
        history_tokens: int = 0,
        available_context_tokens: int | None = None,
    ) -> TokenBudget:
        """Compute the budget for one request.

        ``available_context_tokens`` is what retrieval actually produced. When it is less
        than the context allowance, the surplus is *not* redistributed to history: older
        conversation turns dilute attention and invite the model to treat its own prior
        answers as sources. Surplus is simply left unused, which is cheaper and safer.
        """
        question = min(max(question_tokens, 1), QUESTION_TOKENS * 4)
        summary = min(summary_tokens, self.summary_budget)
        history = min(history_tokens, self.history_budget)

        fixed = SYSTEM_PROMPT_TOKENS + MODE_RULES_TOKENS + question + summary + history
        room = self.prompt_cap - self.completion_reserve - fixed
        context_allowance = min(self.context_budget, max(0, room))
        if available_context_tokens is not None:
            context_allowance = min(context_allowance, available_context_tokens)

        return TokenBudget(
            system=SYSTEM_PROMPT_TOKENS,
            mode_rules=MODE_RULES_TOKENS,
            summary=summary,
            history=history,
            context=context_allowance,
            question=question,
            completion_reserve=self.completion_reserve,
        )

    def fit_context(
        self,
        chunk_tokens: list[int],
        budget: TokenBudget,
        *,
        min_chunks: int = 1,
    ) -> int:
        """How many chunks fit, given they arrive in descending relevance order.

        Returns the count to keep. Chunks are dropped from the *end* — the lowest reranked
        — because that sacrifices the least relevant evidence. A chunk is never partially
        included: half a source cannot support the citation that points at it.

        ``min_chunks`` guarantees at least one source even if it exceeds the allowance; the
        alternative is generating with no context at all, which the confidence gate would
        then have to refuse. Better to run one oversized prompt than to manufacture a
        refusal out of an arithmetic edge case.
        """
        kept = 0
        used = 0
        for tokens in chunk_tokens:
            if used + tokens > budget.context and kept >= min_chunks:
                break
            used += tokens
            kept += 1
        return kept

    def history_turns_that_fit(
        self, turn_tokens: list[int], budget: TokenBudget, *, max_turns: int = 6
    ) -> int:
        """How many of the most recent messages fit the history allowance.

        ``turn_tokens`` is ordered newest first, so dropping from the end drops the oldest.
        """
        kept = 0
        used = 0
        for tokens in turn_tokens[:max_turns]:
            if used + tokens > budget.history:
                break
            used += tokens
            kept += 1
        return kept

    def needs_summarization(self, message_count: int, *, threshold: int = 8) -> bool:
        """Whether the rolling summary should be refreshed.

        Summarizing incrementally past a threshold keeps per-turn cost O(1) instead of
        growing with conversation length.
        """
        return message_count > threshold
