"""Semantic chunking: boundaries where the topic changes, not where the budget runs out.

Two implementations of the same idea, because the good one costs money. Both measure the
distance between consecutive sentence windows and cut at the largest gaps; they differ in how
distance is computed:

* :meth:`SemanticChunker.asplit` embeds each window and uses cosine distance. This is the real
  thing, and it costs one embedding call per sentence in the document — for a 200-page PDF,
  more than embedding the chunks themselves.
* :meth:`SemanticChunker.split` uses IDF-weighted lexical overlap. No network, no cost,
  deterministic, and it catches the boundaries that matter most in enterprise documents, which
  announce topic changes by changing vocabulary ("Termination" → "Confidentiality").

The threshold is a percentile of the observed distances, never an absolute value. Absolute
thresholds do not transfer between embedding models, or even between a well-written policy and
a transcript, and the failure is silent: every document becomes one chunk, or every sentence
becomes one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from aegis.core.logging import get_logger
from aegis.rag.chunking.base import (
    Segment,
    link_heading_layout,
    pack,
    segments_from_block,
    term_counts,
)

if TYPE_CHECKING:
    from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk
    from aegis.domain.ports.embeddings import EmbeddingProvider
    from aegis.domain.ports.parser import DocumentBlock
    from aegis.rag.chunking.tokenizer import TokenCounter

logger = get_logger(__name__)

MAX_EMBEDDED_SENTENCES = 600
"""Above this, :meth:`asplit` degrades to the lexical measure.

A cap rather than a warning: the embedding call is per sentence, so an unbounded document
turns one upload into a five-figure API bill. Documents this long are also the ones with real
headings, where the Markdown strategy is the better router choice anyway.
"""


class SemanticChunker:
    """Breaks where consecutive sentence windows are least similar."""

    name = "semantic"

    def __init__(
        self,
        counter: TokenCounter,
        embedder: EmbeddingProvider | None = None,
        *,
        breakpoint_percentile: float = 90.0,
        buffer_size: int = 1,
    ) -> None:
        self._counter = counter
        self._embedder = embedder
        self._percentile = breakpoint_percentile
        self._buffer = max(0, buffer_size)

    def supports(self, block: DocumentBlock) -> bool:
        return not block.block_type.is_structured

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]:
        segments = self._segments(blocks)
        if len(segments) > 2:
            self._mark_breaks(segments, self._lexical_distances(segments))
        return pack(segments, config, self._counter)

    async def asplit(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]:
        segments = self._segments(blocks)
        if len(segments) <= 2:
            return pack(segments, config, self._counter)

        distances = None
        if self._embedder is not None and len(segments) <= MAX_EMBEDDED_SENTENCES:
            distances = await self._embedded_distances(segments)
        if distances is None:
            distances = self._lexical_distances(segments)

        self._mark_breaks(segments, distances)
        return pack(segments, config, self._counter)

    def _segments(self, blocks: list[DocumentBlock]) -> list[Segment]:
        segments: list[Segment] = []
        for index, block in enumerate(blocks):
            segments.extend(
                segments_from_block(
                    block, self._counter, break_before=block.is_heading and index > 0
                )
            )
        link_heading_layout(segments)
        return segments

    def _gap(self, count: int, index: int) -> tuple[range, range]:
        """The sentences on either side of the gap after ``index``.

        Deliberately asymmetric. Comparing "window centred on i" with "window centred on i+1"
        puts the neighbouring sentence on both sides of the comparison, which smears the
        boundary by one sentence — the topic change gets detected one sentence late, and the
        chunk before it ends with a sentence about the next topic. Comparing what is *before*
        the gap with what is *after* it measures the gap itself.
        """
        left = range(max(0, index - self._buffer), index + 1)
        right = range(index + 1, min(count, index + 2 + self._buffer))
        return left, right

    async def _embedded_distances(self, segments: list[Segment]) -> list[float] | None:
        """Cosine distance across each gap, or ``None`` if embedding failed.

        One embedding per sentence, then mean-pooled per side of the gap. Embedding the windows
        directly would be marginally more accurate and would cost twice as many calls for every
        document, which is not a trade worth making on a per-upload path.

        A failure here is not an ingest failure: the lexical measure produces slightly worse
        boundaries, while propagating the error produces no document at all.
        """
        try:
            vectors = await self._embedder.embed_documents(  # type: ignore[union-attr]
                [s.text for s in segments]
            )
        except Exception as exc:
            logger.warning("semantic_chunking_embedding_failed", error=str(exc))
            return None
        if len(vectors) != len(segments):
            logger.warning("semantic_chunking_length_mismatch", got=len(vectors))
            return None

        dense = [v.values for v in vectors]
        distances: list[float] = []
        for index in range(len(segments) - 1):
            left, right = self._gap(len(segments), index)
            distances.append(
                1.0 - _cosine(_mean([dense[i] for i in left]), _mean([dense[i] for i in right]))
            )
        return distances

    def _lexical_distances(self, segments: list[Segment]) -> list[float]:
        """IDF-weighted cosine distance across each gap."""
        counts = [term_counts(s.text) for s in segments]
        document_frequency: dict[str, int] = {}
        for terms in counts:
            for term in terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        total = len(counts)
        weighted = [
            {
                term: count * math.log(1 + total / document_frequency[term])
                for term, count in terms.items()
            }
            for terms in counts
        ]

        distances: list[float] = []
        for index in range(total - 1):
            left, right = self._gap(total, index)
            distances.append(
                1.0
                - _sparse_cosine(
                    _merge([weighted[i] for i in left]), _merge([weighted[i] for i in right])
                )
            )
        return distances

    def _mark_breaks(self, segments: list[Segment], distances: list[float]) -> None:
        """Set ``break_before`` on the segments following the largest distances."""
        if not distances:
            return
        threshold = _percentile(distances, self._percentile)
        for index, distance in enumerate(distances):
            if distance >= threshold and distance > 0:
                segments[index + 1].break_before = True


def _mean(vectors: list[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    if len(vectors) == 1:
        return list(vectors[0])
    return [sum(column) / len(vectors) for column in zip(*vectors, strict=True)]


def _merge(term_maps: list[dict[str, float]]) -> dict[str, float]:
    if len(term_maps) == 1:
        return term_maps[0]
    merged: dict[str, float] = {}
    for terms in term_maps:
        for term, weight in terms.items():
            merged[term] = merged.get(term, 0.0) + weight
    return merged


def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())
    if dot == 0.0:
        return 0.0
    norm = math.sqrt(sum(v * v for v in left.values())) * math.sqrt(
        sum(v * v for v in right.values())
    )
    return dot / norm if norm else 0.0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile.

    Written out rather than pulled from numpy because this package must import in the worker
    image without a numerics stack, and a five-line function is cheaper than that dependency.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
