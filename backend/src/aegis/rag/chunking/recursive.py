"""Recursive prose chunking: the default, and the fallback for every other strategy.

"Recursive" is the LangChain term for descending a separator hierarchy — paragraphs, then
sentences, then words — and stopping at the coarsest level that fits the budget. This
implementation descends the same hierarchy but does it by *building up* from sentences instead
of splitting down from the whole text, which gives every boundary an offset into the source
document for free. Splitting down and recovering offsets afterwards means a substring search
per chunk, and that search is ambiguous in exactly the documents worth indexing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.domain.enums import ChunkType
from aegis.rag.chunking.base import Segment, link_heading_layout, pack, segments_from_block

if TYPE_CHECKING:
    from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk
    from aegis.domain.ports.parser import DocumentBlock
    from aegis.rag.chunking.tokenizer import TokenCounter


class RecursiveChunker:
    """Packs sentences into token-budgeted chunks, breaking at paragraph boundaries."""

    name = "recursive"

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def supports(self, block: DocumentBlock) -> bool:
        """Handles anything unstructured, which is what makes it the fallback."""
        return not block.block_type.is_structured

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]:
        segments: list[Segment] = []
        for index, block in enumerate(blocks):
            # A heading is not a chunk of its own. It is the label for what follows, so it
            # opens the next chunk and becomes its context header — a chunk containing only
            # "4.2 Termination" retrieves nothing and answers nothing.
            segments.extend(
                segments_from_block(
                    block, self._counter, break_before=block.is_heading and index > 0
                )
            )
        link_heading_layout(segments)

        chunk_type = ChunkType.OCR if _is_ocr(blocks) else None
        return pack(segments, config, self._counter, chunk_type=chunk_type)


def _is_ocr(blocks: list[DocumentBlock]) -> bool:
    """Whether this region came from OCR, which downstream confidence scoring needs to know."""
    return bool(blocks) and all(b.confidence is not None for b in blocks)
