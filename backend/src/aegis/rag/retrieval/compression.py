"""Context compression: keep the sentences that answer the question, drop the rest.

A reranked chunk is relevant as a whole, but often only two of its eight sentences bear on
the question. Sending all eight costs tokens and, more importantly, dilutes attention — the
"lost in the middle" effect is real, and a prompt padded with near-miss prose measurably
degrades the answer that a tighter prompt would have produced.

Compression is extractive, never generative. An LLM-based compressor would be a second place
where facts can be invented, and it would sit *between* the source and the citation quote —
so a hallucination there would be indistinguishable from a correctly cited fact. Selecting
whole source sentences keeps every character in the context traceable to the document.

Structured chunks are exempt. A table with rows removed is worse than no table: the reader,
human or model, has no way to know that rows are missing, so a partial table reads as a
complete one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aegis.rag.chunking.base import split_sentences, term_counts

if TYPE_CHECKING:
    from aegis.domain.values import RetrievedChunk

_NUMERIC = re.compile(r"\d")
# Sentences that only make sense with their neighbour: a bare "This applies to:" followed by
# a list, or "See below." Keeping the neighbour is cheaper than losing the meaning.
_DANGLING = re.compile(r"(:|;)\s*$")


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    enabled: bool = True
    min_chunk_tokens: int = 120
    """Below this a chunk is left alone: the savings are trivial and the risk of cutting the
    one sentence that mattered is not."""

    keep_ratio: float = 0.6
    """Upper bound on how much of a chunk may be dropped. Compression that routinely removes
    most of a chunk is a sign the chunker is producing chunks that are too broad, and hiding
    that behind aggressive filtering makes the real problem invisible."""

    min_sentences: int = 2
    neighbour_window: int = 1
    """Sentences of context kept either side of a selected sentence, so a quote does not
    arrive stripped of the clause that qualifies it."""


@dataclass(slots=True)
class CompressionStats:
    chunks_compressed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def saved_pct(self) -> float:
        if not self.tokens_before:
            return 0.0
        return 100.0 * (1 - self.tokens_after / self.tokens_before)


def _score_sentences(sentences: list[str], query_terms: dict[str, int]) -> list[float]:
    """Lexical overlap with the question, with a bonus for numbers.

    Numbers earn a bonus because in policy and pricing corpora the answer usually *is* a
    number, and a sentence containing one is far more likely to be the load-bearing sentence
    than a paraphrase of the heading.
    """
    scores: list[float] = []
    for sentence in sentences:
        terms = term_counts(sentence)
        overlap = sum(count for term, count in terms.items() if term in query_terms)
        total = sum(terms.values()) or 1
        score = overlap / (total**0.5)
        if _NUMERIC.search(sentence):
            score += 0.15
        scores.append(score)
    return scores


def compress_chunk(
    chunk: RetrievedChunk,
    query: str,
    *,
    config: CompressionConfig | None = None,
) -> str:
    """Return the compressed content for one chunk.

    Selected sentences are returned **in document order**, not in score order. Reordering
    would produce text that reads as if the source said things in a sequence it never did,
    which is a subtle form of misquotation.
    """
    cfg = config or CompressionConfig()
    if not cfg.enabled or chunk.chunk_type.is_structured:
        return chunk.content
    if chunk.token_count and chunk.token_count < cfg.min_chunk_tokens:
        return chunk.content

    sentences = [text for text, _start, _end in split_sentences(chunk.content)]
    if len(sentences) <= cfg.min_sentences:
        return chunk.content

    query_terms = term_counts(query)
    if not query_terms:
        return chunk.content

    scores = _score_sentences(sentences, query_terms)
    keep_count = max(cfg.min_sentences, round(len(sentences) * cfg.keep_ratio))
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:keep_count]

    keep: set[int] = set()
    for index in ranked:
        for offset in range(-cfg.neighbour_window, cfg.neighbour_window + 1):
            neighbour = index + offset
            if 0 <= neighbour < len(sentences):
                keep.add(neighbour)
    # A trailing colon promises the next sentence; keep it.
    for index in sorted(keep):
        if _DANGLING.search(sentences[index]) and index + 1 < len(sentences):
            keep.add(index + 1)

    if len(keep) >= len(sentences):
        return chunk.content
    return " ".join(sentences[i] for i in sorted(keep))


def compress(
    chunks: list[RetrievedChunk],
    query: str,
    *,
    config: CompressionConfig | None = None,
) -> tuple[list[str], CompressionStats]:
    """Compress every chunk, returning the texts alongside what was saved.

    The chunk objects are not mutated: the full text is still needed to locate and validate
    a citation quote against the original, and to show the user the surrounding passage in
    the citation drawer.
    """
    cfg = config or CompressionConfig()
    stats = CompressionStats()
    texts: list[str] = []
    for chunk in chunks:
        compressed = compress_chunk(chunk, query, config=cfg)
        texts.append(compressed)
        stats.tokens_before += chunk.token_count or max(1, len(chunk.content) // 4)
        stats.tokens_after += max(1, len(compressed) // 4)
        if compressed != chunk.content:
            stats.chunks_compressed += 1
    return texts, stats
