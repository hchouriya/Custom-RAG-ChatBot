"""Heading detection and hierarchy assignment.

PDF and plain text carry no structural markup, yet section context is what makes a retrieved
chunk interpretable: "the limit is 30 days" means nothing until you know it sits under
"Refunds → International orders". Every chunk therefore needs a heading path, and for
unstructured formats it has to be inferred.

The inference here is intentionally conservative. A false heading pollutes the path of every
following chunk until the next real one, so each rule requires several agreeing signals, and
the fallback is "this is body text" rather than "this might be a heading".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from aegis.domain.enums import ChunkType
from aegis.domain.ports.parser import DocumentBlock

# "4.2.1 Scope" — numbered headings are the strongest signal in technical documents.
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,4})[.)]?\s+(\S.{0,120})$")
# "Article IV — Termination"
_ROMAN = re.compile(r"^((?:IX|IV|V?I{1,3}|X{1,3})(?:\.\d+)*)[.)]?\s+(\S.{0,120})$")
# "Appendix B: Rate card", "Chapter 3", "Section 12"
_LABELLED = re.compile(
    r"^(appendix|annex|chapter|section|part|schedule|exhibit)\s+([A-Z0-9]{1,4})\b[:.\-—\s]*(.{0,120})$",
    re.IGNORECASE,
)
_MARKDOWN_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_SENTENCE_END = re.compile(r"[.!?;:]$")
_ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 \-&/(),.']{2,79}$")

MAX_HEADING_WORDS = 14
MAX_HEADING_CHARS = 120
MAX_DEPTH = 6


@dataclass(slots=True)
class HeadingCandidate:
    text: str
    level: int
    numbering: str | None = None
    confidence: float = 0.5


@dataclass(slots=True)
class _HierarchyState:
    """Running heading stack while walking blocks in order."""

    stack: list[tuple[int, str]] = field(default_factory=list)

    def push(self, level: int, text: str) -> tuple[str, ...]:
        # Drop anything at or below this level: a level-2 heading ends the previous level-2
        # section and everything nested inside it.
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((min(level, MAX_DEPTH), text))
        return self.path

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self.stack)


def detect_heading(
    text: str,
    *,
    font_size: float | None = None,
    body_font_size: float | None = None,
    is_bold: bool = False,
    is_markdown: bool = False,
) -> HeadingCandidate | None:
    """Classify one line as a heading, or return ``None``.

    Font metrics are used when the parser can supply them (PDF), and are the single most
    reliable signal: a line 1.2 times the body size is a heading far more often than any textual
    pattern implies. Without them, the decision rests on numbering, length, capitalisation,
    and the absence of terminal punctuation.
    """
    line = text.strip()
    if not line or len(line) > MAX_HEADING_CHARS:
        return None

    if is_markdown:
        atx = _MARKDOWN_ATX.match(line)
        if atx:
            return HeadingCandidate(atx.group(2).strip(), len(atx.group(1)), confidence=1.0)

    labelled = _LABELLED.match(line)
    if labelled:
        title = " ".join(part for part in labelled.groups() if part).strip()
        return HeadingCandidate(title, 1, numbering=labelled.group(2), confidence=0.9)

    numbered = _NUMBERED.match(line)
    if numbered and not _SENTENCE_END.search(numbered.group(2)):
        numbering = numbered.group(1)
        level = numbering.count(".") + 1
        return HeadingCandidate(
            f"{numbering} {numbered.group(2)}".strip(),
            min(level, MAX_DEPTH),
            numbering=numbering,
            confidence=0.85,
        )

    roman = _ROMAN.match(line)
    if roman and not _SENTENCE_END.search(roman.group(2)):
        return HeadingCandidate(line, 1, numbering=roman.group(1), confidence=0.7)

    words = line.split()
    if len(words) > MAX_HEADING_WORDS or _SENTENCE_END.search(line):
        return None

    # Font-based: a size jump over body text, optionally reinforced by bold.
    if font_size and body_font_size and body_font_size > 0:
        ratio = font_size / body_font_size
        if ratio >= 1.35:
            return HeadingCandidate(line, 1, confidence=0.9)
        if ratio >= 1.15:
            return HeadingCandidate(line, 2, confidence=0.75)
        if ratio >= 1.05 and is_bold:
            return HeadingCandidate(line, 3, confidence=0.6)
        # Explicitly *not* a heading: same size as body. Returning here prevents the weaker
        # textual heuristics below from overriding a reliable negative.
        return None

    if _ALL_CAPS.match(line) and len(words) <= 10:
        return HeadingCandidate(line.title() if line.isupper() else line, 2, confidence=0.65)

    if is_bold and len(words) <= 10:
        return HeadingCandidate(line, 3, confidence=0.55)

    return None


def assign_hierarchy(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Fill ``heading_path`` on every block, in document order.

    Mutates in place and returns the same list; blocks are large and there is no reason to
    copy them. Headings themselves receive the path *including* their own text, so a chunk
    made from a heading plus its body reads correctly on its own.
    """
    state = _HierarchyState()
    for block in blocks:
        if block.is_heading:
            level = block.heading_level or 1
            block.heading_path = state.push(level, block.text.strip())
        else:
            block.heading_path = state.path
    return blocks


def infer_blocks_from_text(
    text: str, *, page: int | None = None, is_markdown: bool = False
) -> list[DocumentBlock]:
    """Split flat text into typed blocks, detecting headings and fenced code.

    Used for plain text and as the PDF path when font metrics are unavailable. Paragraph
    boundaries come from blank lines, which is the one layout signal every text format shares.
    """
    blocks: list[DocumentBlock] = []
    offset = 0
    in_fence = False
    fence_lines: list[str] = []
    fence_start = 0
    paragraph: list[str] = []
    paragraph_start = 0

    def flush_paragraph() -> None:
        nonlocal paragraph, paragraph_start
        if not paragraph:
            return
        body = "\n".join(paragraph).strip()
        if body:
            blocks.append(
                DocumentBlock(
                    text=body,
                    block_type=ChunkType.LIST if _is_list(body) else ChunkType.TEXT,
                    page=page,
                    char_start=paragraph_start,
                    char_end=paragraph_start + len(body),
                )
            )
        paragraph = []

    for raw_line in text.split("\n"):
        line_length = len(raw_line) + 1
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            if in_fence:
                body = "\n".join(fence_lines)
                blocks.append(
                    DocumentBlock(
                        text=body,
                        block_type=ChunkType.CODE,
                        page=page,
                        char_start=fence_start,
                        char_end=fence_start + len(body),
                    )
                )
                fence_lines = []
                in_fence = False
            else:
                flush_paragraph()
                in_fence = True
                fence_start = offset
            offset += line_length
            continue

        if in_fence:
            fence_lines.append(raw_line)
            offset += line_length
            continue

        if not line.strip():
            flush_paragraph()
            offset += line_length
            continue

        candidate = detect_heading(line, is_markdown=is_markdown)
        if candidate is not None and candidate.confidence >= 0.65:
            flush_paragraph()
            blocks.append(
                DocumentBlock(
                    text=candidate.text,
                    block_type=ChunkType.HEADING,
                    page=page,
                    heading_level=candidate.level,
                    char_start=offset,
                    char_end=offset + len(line),
                    metadata={"numbering": candidate.numbering} if candidate.numbering else {},
                )
            )
            offset += line_length
            continue

        if not paragraph:
            paragraph_start = offset
        paragraph.append(line)
        offset += line_length

    if in_fence and fence_lines:
        # Unterminated fence: keep the content as code rather than losing it.
        body = "\n".join(fence_lines)
        blocks.append(
            DocumentBlock(
                text=body,
                block_type=ChunkType.CODE,
                page=page,
                char_start=fence_start,
                char_end=fence_start + len(body),
            )
        )
    flush_paragraph()
    return blocks


def _is_list(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    markers = sum(1 for line in lines if re.match(r"^(?:[-*+•]|\d{1,3}[.)])\s+", line))
    return markers >= max(2, len(lines) // 2)


def document_title(blocks: list[DocumentBlock], *, fallback: str | None = None) -> str | None:
    """Best guess at the document's title.

    The first heading on the first page, or the first substantial line. Used only as a
    default for the upload form, which the uploader can overwrite — so a wrong guess costs
    an edit, not a mis-titled document.
    """
    for block in blocks[:12]:
        if block.is_heading and 3 <= len(block.text) <= MAX_HEADING_CHARS:
            return block.text.strip()
    for block in blocks[:4]:
        first_line = block.text.strip().split("\n")[0]
        if 8 <= len(first_line) <= MAX_HEADING_CHARS:
            return first_line
    return fallback
