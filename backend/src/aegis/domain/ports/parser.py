"""Document parsing port and the structures every parser must produce.

A parser's job is to turn bytes into a :class:`ParsedDocument`: an ordered list of typed
blocks that each know their page and their position in the heading hierarchy. Chunking
never sees the original file, and the parser never decides chunk boundaries — that
separation is what lets nine formats share five chunking strategies.

The block-with-locators shape is the reason citations can say "page 12, §4.2" rather than
"somewhere in this document". Locator data has to be captured at parse time; it cannot be
reconstructed later from plain text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aegis.domain.enums import ChunkType


@dataclass(slots=True)
class DocumentBlock:
    """One contiguous, typed region of a document."""

    text: str
    block_type: ChunkType = ChunkType.TEXT
    page: int | None = None
    heading_path: tuple[str, ...] = ()
    heading_level: int | None = None
    char_start: int = 0
    char_end: int = 0
    bbox: dict[str, float] | None = None
    language: str | None = None
    confidence: float | None = None
    """OCR confidence, 0-1. ``None`` for digitally extracted text.

    Recorded per block so a low-confidence scan can be flagged for review instead of
    being indexed as authoritative text.
    """
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_heading(self) -> bool:
        return self.block_type is ChunkType.HEADING

    @property
    def section(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None


@dataclass(slots=True)
class PageInfo:
    """Per-page bookkeeping, needed for OCR quality reporting and page mapping."""

    number: int
    char_count: int = 0
    used_ocr: bool = False
    ocr_confidence: float | None = None
    width: float | None = None
    height: float | None = None


@dataclass(slots=True)
class ParsedDocument:
    """Everything a parser extracted from one file."""

    blocks: list[DocumentBlock]
    pages: list[PageInfo] = field(default_factory=list)
    title: str | None = None
    language: str | None = None
    parser: str = ""
    used_ocr: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    """Non-fatal problems: unreadable pages, dropped macros, low-confidence OCR.

    Surfaced in the admin UI so a content owner can see *why* their document retrieves
    poorly, instead of filing a ticket about it.
    """

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    @property
    def page_count(self) -> int:
        return len(self.pages) or (max((b.page or 0) for b in self.blocks) if self.blocks else 0)

    @property
    def low_confidence_pages(self) -> list[int]:
        return [
            p.number for p in self.pages if p.ocr_confidence is not None and p.ocr_confidence < 0.6
        ]


@runtime_checkable
class DocumentParser(Protocol):
    """Turns file bytes into a :class:`ParsedDocument`."""

    name: str

    def supports(self, mime_type: str, filename: str) -> bool:
        """Whether this parser handles the format.

        Both arguments are provided because MIME detection is imperfect for text-ish
        formats — Markdown and CSV both sniff as ``text/plain``, and only the extension
        distinguishes them.
        """
        ...

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        """Parse, or raise ``ParserError``.

        Async because a parser may need OCR in a thread pool or a network call; CPU-bound
        work must be offloaded rather than run inline on the event loop.
        """
        ...
