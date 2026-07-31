"""Citation extraction and validation.

The product claim is "every answer is grounded and cited". This module is what makes that
claim testable, and it treats the model's output as untrusted:

* A marker outside the supplied range is **removed from the answer**, not merely ignored. A
  visible ``[9]`` that links nowhere is worse than no citation, because it looks like
  provenance while providing none.
* A quote is located in the source text rather than taken from the model. The model is asked
  for markers, not for quotes; the quote shown in the UI is the sentence from the document
  that best matches the claim, found here by lexical overlap. That way a citation's quote can
  never be a paraphrase the source did not contain.
* An answer that asserts facts with **no** valid citations is treated as ungrounded by the
  caller, which is the only reliable way to catch a model that ignored its instructions.

Markers are renumbered so the user sees ``[1][2][3]`` with no gaps even when the model cited
sources 2, 5, and 7 — gaps invite the reader to wonder what they are not being shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.core.telemetry import citation_validation_failures, citations_per_answer
from aegis.domain.values import Citation
from aegis.rag.chunking.base import split_sentences, term_counts

if TYPE_CHECKING:
    from aegis.domain.values import RetrievedChunk

# `[1]`, `[1, 3]`, `[1][3]`, `[1-3]` — models produce all of these regardless of the
# instruction, so all of them are parsed.
_MARKER = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
MAX_QUOTE_CHARS = 400
MAX_MARKER = 50


@dataclass(slots=True)
class CitationResult:
    answer: str
    """The answer with invalid markers stripped and valid ones renumbered."""

    citations: list[Citation] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    uncited_sources: list[int] = field(default_factory=list)
    """Sources offered to the model that it did not cite. Retained because "what did the
    model ignore" is often the explanation for a wrong answer."""

    @property
    def is_grounded(self) -> bool:
        return bool(self.citations)


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in group.split(","):
        part = part.strip()
        if match := _RANGE.match(part):
            start, end = int(match.group(1)), int(match.group(2))
            if 0 < end - start < MAX_MARKER:
                numbers.extend(range(start, end + 1))
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def find_markers(text: str) -> list[int]:
    """Every marker referenced by the answer, in order of first appearance."""
    seen: dict[int, None] = {}
    for match in _MARKER.finditer(text):
        for number in _expand(match.group(1)):
            seen.setdefault(number, None)
    return list(seen)


def best_quote(
    source_text: str, claim: str, *, max_chars: int = MAX_QUOTE_CHARS
) -> tuple[str, int, int]:
    """Pick the sentence in the source that best supports the claim.

    Returns the sentence with its character offsets inside ``source_text`` so the UI can
    highlight the exact span rather than string-matching it again — a search that breaks on
    any text that repeats, which in a policy document is most of it.
    """
    sentences = split_sentences(source_text)
    if not sentences:
        excerpt = source_text[:max_chars]
        return excerpt, 0, len(excerpt)

    claim_terms = term_counts(claim)
    best_index = 0
    best_score = -1.0
    for index, (sentence, _start, _end) in enumerate(sentences):
        terms = term_counts(sentence)
        if not terms:
            continue
        overlap = sum(count for term, count in terms.items() if term in claim_terms)
        score = overlap / (sum(terms.values()) ** 0.5)
        if score > best_score:
            best_score, best_index = score, index

    text, start, end = sentences[best_index]
    # Include the following sentence when the chosen one is short: a bare "It is 20 weeks."
    # is not a usable quote without the sentence that says what "it" is.
    if len(text) < 80 and best_index + 1 < len(sentences):
        _next_text, _next_start, next_end = sentences[best_index + 1]
        if next_end - start <= max_chars:
            end = next_end
            text = source_text[start:end]
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
        end = start + len(text)
    return text.strip(), start, end


def extract(
    answer: str,
    sources: list[RetrievedChunk],
    contents: list[str] | None = None,
    *,
    max_citations: int = 8,
    renumber: bool = False,
) -> CitationResult:
    """Validate the markers in ``answer`` against the sources that were supplied.

    ``contents`` is the post-compression text the model actually saw, used for quote
    selection so a quote can never come from a passage the model was not shown.

    ``renumber`` closes gaps in the markers. It must stay **off** on the streaming path: the
    client has already received the numbered source list in the ``citations`` event, and
    renumbering afterwards would repoint every marker in the text the user is reading. It is
    on for non-streaming callers, where the answer and its citations are delivered together.
    """
    if not answer.strip():
        return CitationResult(answer=answer)

    shown = contents if contents and len(contents) == len(sources) else [s.content for s in sources]
    referenced = find_markers(answer)
    valid = [m for m in referenced if 1 <= m <= len(sources)][:max_citations]
    invalid = [m for m in referenced if m not in valid]

    for marker in invalid:
        citation_validation_failures.labels(
            reason="out_of_range" if marker > len(sources) else "invalid"
        ).inc()

    mapping = (
        {old: new for new, old in enumerate(valid, start=1)} if renumber else {m: m for m in valid}
    )
    citations: list[Citation] = []
    for old, new in mapping.items():
        chunk = sources[old - 1]
        quote, start, end = best_quote(shown[old - 1], answer)
        citations.append(
            Citation(
                marker=new,
                chunk_id=chunk.chunk_id,
                locator=chunk.locator,
                quote=quote,
                quote_start=start,
                quote_end=end,
                score_rerank=chunk.score_rerank,
                rank=chunk.rank,
                was_used=True,
            )
        )

    citations_per_answer.observe(len(citations))
    citations.sort(key=lambda c: c.marker)
    return CitationResult(
        answer=_rewrite_markers(answer, mapping),
        citations=citations,
        invalid_markers=invalid,
        uncited_sources=[i for i in range(1, len(sources) + 1) if i not in mapping],
    )


def _rewrite_markers(answer: str, mapping: dict[int, int]) -> str:
    """Renumber valid markers and delete invalid ones.

    Deletion tidies up the punctuation it leaves behind — "as stated ." reads as a bug to a
    user, and it is one.
    """

    def replace(match: re.Match[str]) -> str:
        numbers = [mapping[n] for n in _expand(match.group(1)) if n in mapping]
        if not numbers:
            return ""
        return "".join(f"[{n}]" for n in sorted(set(numbers)))

    cleaned = _MARKER.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def mark_unused(citations: list[Citation], sources: list[RetrievedChunk]) -> list[Citation]:
    """Append the uncited sources as ``was_used=False`` records.

    Persisted alongside the used citations so a stored answer carries the whole evidence set
    it was offered, not just the part it took. Without this, an investigation into a bad
    answer cannot distinguish "retrieval failed" from "the model ignored the right source".
    """
    cited = {c.chunk_id for c in citations}
    extras = [
        Citation(
            marker=0,
            chunk_id=chunk.chunk_id,
            locator=chunk.locator,
            quote="",
            score_rerank=chunk.score_rerank,
            rank=chunk.rank,
            was_used=False,
        )
        for chunk in sources
        if chunk.chunk_id not in cited
    ]
    return [*citations, *extras]
