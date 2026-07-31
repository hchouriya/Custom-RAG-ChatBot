"""Heading-aware chunking for documents that carry a real hierarchy.

Used for Markdown, HTML, and DOCX — anything whose parser produced heading levels rather than
guessed them. The rule is that a section boundary outranks the token budget: a chunk should
end where a section ends, even at 300 tokens, because the alternative is a chunk whose second
half answers a different question than its first.

Sections that overflow the budget are split by the recursive strategy, with the heading path
repeated on every piece so the third chunk of §4.2 still knows it is in §4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aegis.rag.chunking.base import Segment, link_heading_layout, pack, segments_from_block

if TYPE_CHECKING:
    from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk
    from aegis.domain.ports.parser import DocumentBlock
    from aegis.rag.chunking.tokenizer import TokenCounter

MAX_BREAK_LEVEL = 3
"""Deepest heading level that forces a chunk boundary.

Below this, headings are frequently one-line labels inside a list ("### Windows", "### macOS"),
and breaking on them produces chunks too small to answer anything. They still contribute to
the heading path, which is where their terms are needed.
"""


@dataclass(slots=True)
class _Section:
    heading: DocumentBlock | None
    body: list[DocumentBlock]
    level: int


class MarkdownChunker:
    """Splits on heading boundaries, then packs each section to the token budget."""

    name = "markdown"

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def supports(self, block: DocumentBlock) -> bool:
        return block.is_heading or not block.block_type.is_structured

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]:
        segments: list[Segment] = []
        for index, section in enumerate(_sections(blocks)):
            hard = index > 0 and section.level <= MAX_BREAK_LEVEL
            if section.heading is not None:
                segments.extend(
                    segments_from_block(section.heading, self._counter, break_before=hard)
                )
                hard = False
            for position, block in enumerate(section.body):
                segments.extend(
                    segments_from_block(block, self._counter, break_before=hard and position == 0)
                )
        link_heading_layout(segments)
        return pack(segments, config, self._counter)


def _sections(blocks: list[DocumentBlock]) -> list[_Section]:
    """Group blocks under the heading that introduces them."""
    sections: list[_Section] = []
    current = _Section(heading=None, body=[], level=0)
    for block in blocks:
        if block.is_heading:
            if current.heading is not None or current.body:
                sections.append(current)
            current = _Section(heading=block, body=[], level=block.heading_level or 1)
            continue
        current.body.append(block)
    if current.heading is not None or current.body:
        sections.append(current)
    return sections
