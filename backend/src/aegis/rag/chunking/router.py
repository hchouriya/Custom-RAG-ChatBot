"""Chunk routing: which strategy handles which part of a document.

Selection is per *region*, not per document, and that is the whole point of this module. A
benefits policy is prose with a rate table in the middle and a code sample in an appendix.
Choosing one strategy for the file means either the table gets split across chunks or the prose
gets chunked by row. Regions of consecutive same-kind blocks each go to the strategy that
understands them, and the results are stitched back into document order.

The router also owns the decisions that need the whole document in view: adaptive chunk sizing,
deduplication of repeated boilerplate, keyword extraction, and ordinal assignment.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aegis.core.logging import get_logger
from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk
from aegis.rag.chunking.base import term_counts
from aegis.rag.chunking.code import CodeChunker
from aegis.rag.chunking.markdown import MarkdownChunker
from aegis.rag.chunking.recursive import RecursiveChunker
from aegis.rag.chunking.semantic import SemanticChunker
from aegis.rag.chunking.table import TableChunker
from aegis.rag.chunking.tokenizer import TokenCounter, get_token_counter

if TYPE_CHECKING:
    from aegis.domain.ports.embeddings import EmbeddingProvider
    from aegis.domain.ports.parser import DocumentBlock, ParsedDocument

logger = get_logger(__name__)

MAX_CHUNKS_PER_DOCUMENT = 20_000
"""Hard ceiling, independent of size limits.

A 200 MB CSV is a legitimate upload that produces millions of chunks and would take the
embedding budget and the vector store with it. Truncating with a warning on the document is
recoverable; an out-of-memory worker mid-batch is not.
"""

MIN_HEADINGS_FOR_MARKDOWN = 3
DENSE_BLOCK_TOKENS = 400
SPARSE_BLOCK_TOKENS = 40
KEYWORDS_PER_CHUNK = 6
KEYWORD_MIN_LENGTH = 3


@runtime_checkable
class AsyncCapable(Protocol):
    """A chunker whose boundary decision needs to await something (an embedding call)."""

    async def asplit(
        self, blocks: list[DocumentBlock], config: ChunkingConfig
    ) -> list[ProtoChunk]: ...


@runtime_checkable
class SyncChunker(Protocol):
    name: str

    def supports(self, block: DocumentBlock) -> bool: ...

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]: ...


@dataclass(slots=True)
class _Region:
    chunker: SyncChunker
    blocks: list[DocumentBlock]


class DefaultChunkRouter:
    """Routes regions to strategies and assembles the document's chunk list."""

    def __init__(
        self,
        counter: TokenCounter | None = None,
        *,
        strategy: str = "adaptive",
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._counter = counter or get_token_counter()
        self._strategy = strategy
        self._table = TableChunker(self._counter)
        self._code = CodeChunker(self._counter)
        self._markdown = MarkdownChunker(self._counter)
        self._recursive = RecursiveChunker(self._counter)
        self._semantic = SemanticChunker(self._counter, embedder)

    async def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ProtoChunk]:
        blocks = [b for b in document.blocks if b.text.strip()]
        if not blocks:
            return []

        effective = self._resolve_config(document, config)
        prose = self._prose_chunker(document)

        chunks: list[ProtoChunk] = []
        for region in _regions(blocks, prose, self._table, self._code):
            chunks.extend(await self._run(region, effective))

        chunks = _deduplicate(chunks)
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            document.warnings.append(
                f"Document produced {len(chunks)} chunks; only the first "
                f"{MAX_CHUNKS_PER_DOCUMENT} were indexed."
            )
            chunks = chunks[:MAX_CHUNKS_PER_DOCUMENT]

        _assign_keywords(chunks)
        for ordinal, chunk in enumerate(chunks):
            chunk.ordinal = ordinal
            if chunk.language is None:
                chunk.language = document.language
            chunk.metadata.setdefault("strategy", self._strategy)
        return chunks

    async def _run(self, region: _Region, config: ChunkingConfig) -> list[ProtoChunk]:
        """Run one region's strategy off the event loop.

        Chunking a large document is hundreds of milliseconds of tokenizing. Left inline it
        would block every other request on the worker, which is the class of bug that only
        shows up under load.
        """
        if isinstance(region.chunker, AsyncCapable):
            return await region.chunker.asplit(region.blocks, config)
        return await asyncio.to_thread(region.chunker.split, region.blocks, config)

    def _prose_chunker(self, document: ParsedDocument) -> SyncChunker:
        """Pick the strategy for unstructured regions."""
        if self._strategy == "recursive":
            return self._recursive
        if self._strategy == "semantic":
            return self._semantic
        if self._strategy == "markdown":
            return self._markdown

        headings = sum(1 for b in document.blocks if b.is_heading)
        if headings >= MIN_HEADINGS_FOR_MARKDOWN:
            # Real headings beat inferred boundaries every time: the author already told us
            # where the topics change.
            return self._markdown
        if document.used_ocr:
            # OCR output has no reliable heading or paragraph structure, so a measured
            # boundary is better than a guessed one.
            return self._semantic
        return self._recursive

    def _resolve_config(self, document: ParsedDocument, config: ChunkingConfig) -> ChunkingConfig:
        """Apply the document title and adaptive sizing."""
        base = (
            config
            if config.document_title
            else ChunkingConfig(
                target_tokens=config.target_tokens,
                overlap_pct=config.overlap_pct,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
                contextual_headers=config.contextual_headers,
                document_title=document.title or "",
            )
        )
        if self._strategy != "adaptive":
            return base
        return base.scaled(self._size_factor(document))

    def _size_factor(self, document: ParsedDocument) -> float:
        """Scale the target chunk size to the document's shape.

        The three cases are the ones that measurably move retrieval quality:

        * OCR text is noisy, and a smaller chunk limits how much good content a bad page can
          drag down with it.
        * Slide decks and bullet lists have tiny blocks; packing more of them per chunk avoids
          a corpus of one-line fragments that match everything and answer nothing.
        * Undifferentiated walls of text need smaller chunks for precision, because there is
          no structure to make a large chunk coherent.
        """
        factor = 1.0
        if document.used_ocr:
            factor *= 0.85

        texts = [b.text for b in document.blocks if b.text.strip()]
        if texts:
            density = sum(self._counter.count(t) for t in texts) / len(texts)
            if density <= SPARSE_BLOCK_TOKENS:
                factor *= 1.15
            elif density >= DENSE_BLOCK_TOKENS:
                factor *= 0.9
        return factor


def _regions(
    blocks: list[DocumentBlock],
    prose: SyncChunker,
    table: SyncChunker,
    code: SyncChunker,
) -> list[_Region]:
    """Split the document into maximal runs of blocks handled by one strategy."""
    regions: list[_Region] = []
    for block in blocks:
        chunker = table if table.supports(block) else code if code.supports(block) else prose
        if regions and regions[-1].chunker is chunker:
            regions[-1].blocks.append(block)
            continue
        regions.append(_Region(chunker=chunker, blocks=[block]))
    return regions


def _deduplicate(chunks: list[ProtoChunk]) -> list[ProtoChunk]:
    """Drop exact repeats, keeping the first occurrence.

    Repeated content is not hypothetical: a confidentiality footer extracted as a block on
    every page of a 90-page contract becomes 90 near-identical chunks that crowd out the
    answer for any query mentioning confidentiality.
    """
    seen: set[str] = set()
    unique: list[ProtoChunk] = []
    for chunk in chunks:
        digest = hashlib.sha256(" ".join(chunk.content.split()).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(chunk)
    return unique


def _assign_keywords(chunks: list[ProtoChunk]) -> None:
    """Attach the terms that distinguish each chunk from the rest of the document.

    Document-scoped TF-IDF, so the terms are the ones that separate this chunk from its
    siblings rather than the ones that are merely frequent. They are stored on the chunk for
    filter-free keyword boosting and to give the admin UI something inspectable when a
    document retrieves badly.
    """
    if not chunks:
        return
    frequencies = [term_counts(c.content, min_length=KEYWORD_MIN_LENGTH) for c in chunks]
    document_frequency: dict[str, int] = {}
    for counts in frequencies:
        for term in counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    total = len(chunks)
    for chunk, counts in zip(chunks, frequencies, strict=True):
        if not counts:
            continue
        scored = sorted(
            counts.items(),
            key=lambda item: (
                -item[1] * math.log(1 + total / document_frequency[item[0]]),
                item[0],
            ),
        )
        chunk.keywords = tuple(term for term, _ in scored[:KEYWORDS_PER_CHUNK])


def build_chunk_router(
    *,
    strategy: str = "adaptive",
    embedding_model: str | None = None,
    embedder: EmbeddingProvider | None = None,
) -> DefaultChunkRouter:
    """Construct the router with the tokenizer that matches the embedding model."""
    counter = get_token_counter(embedding_model) if embedding_model else get_token_counter()
    return DefaultChunkRouter(counter, strategy=strategy, embedder=embedder)
