"""Chunking port.

A chunker converts parsed blocks into :class:`ProtoChunk` objects — chunk content plus the
locators and metadata needed for retrieval and citation, but without database identity.
Persistence is the ingestion service's job.

``ProtoChunk`` keeps ``char_start``/``char_end`` pointing into the *cleaned* document text.
Those offsets are what let the citation drawer highlight the exact supporting sentence
inside its surrounding paragraph, so they must survive cleaning and chunking rather than
being recomputed from a substring search later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aegis.domain.enums import ChunkType
from aegis.domain.ports.parser import DocumentBlock, ParsedDocument


@dataclass(slots=True)
class ProtoChunk:
    """A chunk before it has a database row."""

    content: str
    ordinal: int = 0
    chunk_type: ChunkType = ChunkType.TEXT
    token_count: int = 0
    page_from: int | None = None
    page_to: int | None = None
    heading_path: tuple[str, ...] = ()
    section: str | None = None
    char_start: int = 0
    char_end: int = 0
    bbox: dict[str, Any] | None = None
    context_header: str | None = None
    summary: str | None = None
    keywords: tuple[str, ...] = ()
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return (
            f"{self.context_header}\n---\n{self.content}" if self.context_header else self.content
        )


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Per-collection chunking parameters.

    ``target_tokens`` is the single highest-impact retrieval knob in the system, which is
    exactly why it is configuration measured against a golden set rather than a constant
    compiled into a splitter.
    """

    target_tokens: int = 800
    overlap_pct: int = 15
    min_tokens: int = 250
    max_tokens: int = 1400
    contextual_headers: bool = True
    document_title: str = ""

    @property
    def overlap_tokens(self) -> int:
        return max(0, round(self.target_tokens * self.overlap_pct / 100))

    def scaled(self, factor: float) -> ChunkingConfig:
        """Adaptive sizing: clamp ``target * factor`` into the configured bounds."""
        target = round(self.target_tokens * factor)
        return ChunkingConfig(
            target_tokens=max(self.min_tokens, min(self.max_tokens, target)),
            overlap_pct=self.overlap_pct,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            contextual_headers=self.contextual_headers,
            document_title=self.document_title,
        )


@runtime_checkable
class Chunker(Protocol):
    """Splits blocks into chunks."""

    name: str

    def supports(self, block: DocumentBlock) -> bool:
        """Whether this strategy should handle the block.

        Selection is per *block*, not per document: a policy PDF with an embedded rate
        table needs prose splitting and table handling in the same file.
        """
        ...

    def split(self, blocks: list[DocumentBlock], config: ChunkingConfig) -> list[ProtoChunk]: ...


@runtime_checkable
class ChunkRouter(Protocol):
    """Dispatches each region of a document to the right chunker and orders the result."""

    async def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ProtoChunk]: ...
