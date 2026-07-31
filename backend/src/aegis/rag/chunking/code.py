"""Code-aware chunking.

Code is prose with hard boundaries. A function split down the middle produces two chunks that
both look like code and neither of which compiles or explains anything, and the half without
the signature has lost the only name a user would search for.

Boundaries are found by indentation and declaration keywords rather than by parsing. A real
parser per language would be more precise and is the wrong trade here: the corpus is
documentation with embedded snippets, often incomplete or pseudo-code, and a parser that raises
on a fragment gives up on exactly the content that needs indexing. Indentation is the one
structural signal every language in a docs corpus actually respects.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

from aegis.domain.enums import ChunkType
from aegis.rag.chunking.base import Segment, pack

if TYPE_CHECKING:
    from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk
    from aegis.domain.ports.parser import DocumentBlock
    from aegis.rag.chunking.tokenizer import TokenCounter

_DECLARATION = re.compile(
    r"""^(?:
        (?:export\s+|public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+
          |abstract\s+|async\s+|pub\s+|default\s+)*
        (?:def|class|function|func|fn|interface|struct|enum|impl|trait|module|namespace
          |type|const\s+\w+\s*=\s*(?:async\s*)?\(|var|let)
        \b
      | (?:@|\#\[)          # decorators and attributes belong to the declaration below them
      | /\*\*                # doc comment opening a definition
      | [A-Z][A-Za-z0-9_]*\s*\([^)]*\)\s*\{   # C-family method signature
    )""",
    re.VERBOSE,
)

_COMMENT = re.compile(r"^\s*(?:#|//|/\*|\*|--)")


class CodeChunker:
    """Splits code blocks at top-level declarations, never mid-line."""

    name = "code"

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    def supports(self, block: DocumentBlock) -> bool:
        return block.block_type is ChunkType.CODE

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]:
        segments: list[Segment] = []
        for block in blocks:
            segments.extend(self._segments(block, config))
        # No overlap: a repeated function body is a duplicate result, and code is retrieved by
        # the names in its declaration line, which every unit already starts with.
        return pack(segments, config, self._counter, chunk_type=ChunkType.CODE, overlap=False)

    def _segments(self, block: DocumentBlock, config: ChunkingConfig) -> list[Segment]:
        """One segment per logical unit: a declaration and the lines indented under it."""
        segments: list[Segment] = []
        for index, (text, start, end) in enumerate(_declaration_units(block.text)):
            unit = Segment(
                text=text,
                tokens=self._counter.count(text),
                char_start=block.char_start + start,
                char_end=block.char_start + end,
                page=block.page,
                block_type=ChunkType.CODE,
                heading_path=block.heading_path,
                separator="\n",
                break_before=index > 0,
                atomic=True,
                bbox=block.bbox if index == 0 else None,
                language=block.language or block.metadata.get("language"),
            )
            if unit.tokens > config.max_tokens:
                # A generated file or a minified bundle has no declaration structure to
                # respect. It still has to be indexed, and a line boundary is the last split
                # point that leaves readable code.
                segments.extend(self._split_lines(unit, config))
            else:
                segments.append(unit)
        return segments

    def _split_lines(self, segment: Segment, config: ChunkingConfig) -> list[Segment]:
        pieces: list[Segment] = []
        buffer: list[str] = []
        used = 0
        cursor = segment.char_start
        for line in segment.text.split("\n"):
            cost = self._counter.count(line) or 1
            if buffer and used + cost > config.target_tokens:
                text = "\n".join(buffer)
                pieces.append(_line_group(segment, text, cursor, used, first=not pieces))
                cursor += len(text) + 1
                buffer, used = [], 0
            buffer.append(line)
            used += cost
        if buffer:
            text = "\n".join(buffer)
            pieces.append(_line_group(segment, text, cursor, used, first=not pieces))
        return pieces or [segment]


def _line_group(
    segment: Segment, text: str, char_start: int, tokens: int, *, first: bool
) -> Segment:
    return Segment(
        text=text,
        tokens=tokens,
        char_start=char_start,
        char_end=char_start + len(text),
        page=segment.page,
        block_type=ChunkType.CODE,
        heading_path=segment.heading_path,
        separator="\n",
        break_before=first and segment.break_before,
        atomic=False,
        language=segment.language,
    )


def _declaration_units(text: str) -> list[tuple[str, int, int]]:
    """Group lines into declaration-rooted units, with offsets into ``text``.

    A unit starts at a line that declares something at the lowest indentation seen in the
    block, and runs until the next such line. Leading comments and decorators stay attached to
    the declaration they describe, because a docstring separated from its function is noise in
    one chunk and an unexplained signature in another.
    """
    lines = text.split("\n")
    if not lines:
        return []

    base_indent = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )

    starts: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip() or len(line) - len(line.lstrip()) > base_indent:
            continue
        if _DECLARATION.match(line.strip()):
            starts.append(_with_leading_comments(lines, index, base_indent))

    boundaries = [0, *sorted({s for s in starts if s > 0}), len(lines)]

    units: list[tuple[str, int, int]] = []
    offsets = _line_offsets(lines)
    for low, high in itertools.pairwise(boundaries):
        if low >= high:
            continue
        body = "\n".join(lines[low:high])
        if not body.strip():
            continue
        start = offsets[low]
        units.append((body, start, start + len(body)))
    return units or [(text, 0, len(text))]


def _with_leading_comments(lines: list[str], index: int, base_indent: int) -> int:
    """Walk back over the comment and decorator lines that introduce ``index``."""
    start = index
    while start > 0:
        candidate = lines[start - 1]
        if not candidate.strip():
            break
        if len(candidate) - len(candidate.lstrip()) > base_indent:
            break
        if not _COMMENT.match(candidate) and not candidate.lstrip().startswith(("@", "#[")):
            break
        start -= 1
    return start


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1
    return offsets
