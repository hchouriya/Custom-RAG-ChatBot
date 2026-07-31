"""Table-aware chunking.

Two invariants, both learned from what breaks without them:

1. A row is never split. Half a row is not partial information, it is wrong information — the
   value ends up under the wrong column.
2. Every chunk of a table repeats the header row. A row group without its header is a grid of
   numbers whose meaning is in a chunk that did not get retrieved, and the model will invent
   the column names it needs.

Tables arrive here as Markdown from the parsers, which is deliberate: it is the format the
models read most reliably, and it survives a round trip through the vector store's payload
without an escaping scheme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.domain.enums import ChunkType
from aegis.domain.ports.chunker import ProtoChunk
from aegis.rag.chunking.base import Segment, build_context_header, pack

if TYPE_CHECKING:
    from aegis.domain.ports.chunker import ChunkingConfig
    from aegis.domain.ports.parser import DocumentBlock
    from aegis.rag.chunking.tokenizer import TokenCounter

_DIVIDER_CHARS = set("-: |")


class TableChunker:
    """Splits tables at row boundaries, repeating the header on every chunk."""

    name = "table"

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def supports(self, block: DocumentBlock) -> bool:
        return block.block_type in {ChunkType.TABLE, ChunkType.FORM}

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]:
        chunks: list[ProtoChunk] = []
        for block in blocks:
            chunks.extend(self._split_one(block, config))
        for ordinal, chunk in enumerate(chunks):
            chunk.ordinal = ordinal
        return chunks

    def _split_one(self, block: DocumentBlock, config: ChunkingConfig) -> list[ProtoChunk]:
        header, rows = _split_header(block.text)
        header_tokens = self._counter.count(header)

        # A table that fits is one chunk. Splitting it would gain nothing and cost the
        # relationship between its rows.
        if self._counter.count(block.text) <= config.max_tokens or not rows:
            return pack(
                [self._segment(block, block.text, block.char_start)],
                config,
                self._counter,
                chunk_type=block.block_type,
                overlap=False,
            )

        # Budget for rows is what remains after the header, which is duplicated into every
        # chunk. Ignoring that is how a "1400 token max" table chunk reaches 1600.
        row_budget = max(config.min_tokens, config.target_tokens - header_tokens)
        chunks: list[ProtoChunk] = []
        group: list[str] = []
        used = 0
        group_start = block.char_start
        group_end = block.char_start

        def flush() -> None:
            nonlocal group, used
            if not group:
                return
            body = "\n".join(group)
            content = f"{header}\n{body}"
            chunks.append(
                self._chunk(block, content, group_start, group_end, config, rows=len(group))
            )
            group, used = [], 0

        for row, row_start, row_end in rows:
            cost = self._counter.count(row)
            if group and used + cost > row_budget:
                flush()
            if not group:
                group_start = block.char_start + row_start
            group.append(row)
            used += cost
            group_end = block.char_start + row_end
        flush()

        for index, chunk in enumerate(chunks):
            chunk.metadata["part"] = index + 1
            chunk.metadata["parts"] = len(chunks)
        return chunks

    def _segment(self, block: DocumentBlock, text: str, char_start: int) -> Segment:
        return Segment(
            text=text,
            tokens=self._counter.count(text),
            char_start=char_start,
            char_end=char_start + len(text),
            page=block.page,
            block_type=block.block_type,
            heading_path=block.heading_path,
            separator="\n",
            atomic=True,
            bbox=block.bbox,
            language=block.language,
        )

    def _chunk(
        self,
        block: DocumentBlock,
        content: str,
        char_start: int,
        char_end: int,
        config: ChunkingConfig,
        *,
        rows: int,
    ) -> ProtoChunk:
        return ProtoChunk(
            content=content,
            chunk_type=block.block_type,
            token_count=self._counter.count(content),
            page_from=block.page,
            page_to=block.page,
            heading_path=block.heading_path,
            section=block.heading_path[-1] if block.heading_path else None,
            char_start=char_start,
            char_end=char_end,
            bbox=block.bbox,
            context_header=(
                build_context_header(config.document_title, block.heading_path, self._counter)
                if config.contextual_headers
                else None
            ),
            language=block.language,
            metadata={"rows": rows, "has_header": True},
        )


def _split_header(text: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Separate a Markdown table's header from its body rows, with offsets.

    Falls back to treating the first line as the header when there is no divider row, which is
    what Excel-derived and OCR-derived tables look like.
    """
    lines = text.split("\n")
    if not lines:
        return "", []

    header_end = 1
    if len(lines) > 1 and _is_divider(lines[1]):
        header_end = 2

    header = "\n".join(lines[:header_end])
    rows: list[tuple[str, int, int]] = []
    position = sum(len(line) + 1 for line in lines[:header_end])
    for line in lines[header_end:]:
        if line.strip():
            rows.append((line, position, position + len(line)))
        position += len(line) + 1
    return header, rows


def _is_divider(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= _DIVIDER_CHARS and "-" in stripped
