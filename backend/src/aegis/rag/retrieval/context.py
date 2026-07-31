"""Context assembly: turn ranked chunks into the block the model reads.

Three decisions are encoded here.

**Sources are numbered, and the number is the citation contract.** The model is told to cite
``[n]``, where ``n`` is the number printed above the source. That makes a citation
machine-checkable: marker → source → chunk id → document, version, page, and character span.
An answer that cites ``[9]`` when eight sources were supplied is caught by the validator
rather than shown to a user.

**Provenance travels with the text.** Each source header carries the document title, version,
page, and section. Without it the model cannot say "according to the 2026 Leave Policy" and
users cannot tell whether an answer came from a current document or a superseded one.

**Untrusted content is fenced.** Document text is attacker-influenced — anyone who can upload
a file can write "ignore your instructions" into it. The fence plus an explicit statement that
everything inside is data, not instructions, is the cheap and effective mitigation. It is not
a guarantee, which is why the injection scanner flags at ingest and the system prompt states
the rule again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.domain.policies.budget import BudgetPolicy
from aegis.rag.retrieval.compression import CompressionConfig, CompressionStats, compress

if TYPE_CHECKING:
    from collections.abc import Callable

    from aegis.domain.values import RetrievedChunk, TokenBudget

SOURCE_OPEN = "<<<SOURCE {marker}>>>"
SOURCE_CLOSE = "<<<END SOURCE {marker}>>>"


@dataclass(slots=True)
class AssembledContext:
    text: str
    """The rendered context block, ready to be embedded in the prompt."""

    used: list[RetrievedChunk] = field(default_factory=list)
    """Chunks that survived the budget, in marker order — ``used[0]`` is marker 1."""

    dropped: list[RetrievedChunk] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)
    """Post-compression text per marker, so a citation quote is validated against exactly
    what the model was shown."""

    tokens: int = 0
    compression: CompressionStats = field(default_factory=CompressionStats)

    @property
    def marker_count(self) -> int:
        return len(self.used)


def _header(chunk: RetrievedChunk, marker: int) -> str:
    locator = chunk.locator
    parts = [f"[{marker}] {locator.document_title or 'Untitled document'}"]
    if locator.version_no > 1:
        parts.append(f"version {locator.version_no}")
    if locator.page_from:
        parts.append(
            f"page {locator.page_from}"
            if not locator.page_to or locator.page_to == locator.page_from
            else f"pages {locator.page_from}-{locator.page_to}"
        )
    if locator.section:
        parts.append(f"section {locator.section}")
    elif locator.heading_path:
        parts.append(" > ".join(locator.heading_path))
    return " · ".join(parts)


def assemble(
    chunks: list[RetrievedChunk],
    question: str,
    *,
    budget: TokenBudget,
    count_tokens: Callable[[str], int],
    compression: CompressionConfig | None = None,
) -> AssembledContext:
    """Render the context block within ``budget.context`` tokens.

    Chunks arrive in reranked order and are dropped from the end, so the least relevant
    evidence is what is sacrificed. Dropped chunks are kept in the result rather than
    discarded: they are what ``/admin/retrieval/debug`` shows as "retrieved but did not fit",
    which is the answer to a surprising number of "why didn't it use document X" questions.
    """
    if not chunks:
        return AssembledContext(text="")

    texts, stats = compress(chunks, question, config=compression)

    rendered: list[str] = []
    used: list[RetrievedChunk] = []
    used_texts: list[str] = []
    total = 0

    for chunk, text in zip(chunks, texts, strict=True):
        marker = len(used) + 1
        block = "\n".join(
            [
                SOURCE_OPEN.format(marker=marker),
                _header(chunk, marker),
                "",
                text.strip(),
                SOURCE_CLOSE.format(marker=marker),
            ]
        )
        cost = count_tokens(block)
        # `min_chunks=1`: one oversized source beats an empty context, which the confidence
        # gate would then have to refuse for a reason that has nothing to do with the corpus.
        if total + cost > budget.context and used:
            break
        rendered.append(block)
        used.append(chunk)
        used_texts.append(text.strip())
        total += cost

    # Markers are assigned as sources are accepted, so the numbering the model sees is
    # always 1..n with no gaps and never needs renumbering after a drop.
    dropped = chunks[len(used) :]

    return AssembledContext(
        text="\n\n".join(rendered),
        used=used,
        dropped=dropped,
        contents=used_texts,
        tokens=total,
        compression=stats,
    )


def allocate_budget(
    *,
    question: str,
    history: list[tuple[str, str]],
    summary: str | None,
    count_tokens: Callable[[str], int],
    policy: BudgetPolicy,
) -> tuple[TokenBudget, list[tuple[str, str]]]:
    """Compute the budget and the history turns that fit inside it.

    History is trimmed newest-first. Older turns are the ones worth losing: they are the
    least likely to contain the referent of a pronoun in the current question, and the most
    likely to contain a previous *answer*, which the model should not be treating as a source.
    """
    question_tokens = count_tokens(question)
    summary_tokens = count_tokens(summary) if summary else 0

    newest_first = list(reversed(history))
    turn_tokens = [count_tokens(content) for _role, content in newest_first]
    budget = policy.allocate(
        question_tokens=question_tokens,
        summary_tokens=summary_tokens,
        history_tokens=sum(turn_tokens),
    )
    keep = policy.history_turns_that_fit(turn_tokens, budget)
    return budget, list(reversed(newest_first[:keep]))
