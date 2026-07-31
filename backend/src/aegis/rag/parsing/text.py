"""Parsers for text-shaped formats: plain text, Markdown, CSV/TSV, JSON, and HTML.

These share one property that PDF and Office formats lack: the bytes already are the content.
The work is not extraction but *structure recovery* — finding the headings, tables, and code
blocks that a chunker needs in order to split along meaning rather than at arbitrary offsets.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
from typing import Any, ClassVar

from aegis.core.errors import ParserError
from aegis.core.logging import get_logger
from aegis.domain.enums import ChunkType
from aegis.domain.ports.parser import DocumentBlock, PageInfo, ParsedDocument
from aegis.rag.parsing.cleaning import clean_text, is_meaningful, is_noise
from aegis.rag.parsing.sniff import CSV, HTML, JSON, MARKDOWN, TSV, TXT
from aegis.rag.parsing.structure import (
    assign_hierarchy,
    document_title,
    infer_blocks_from_text,
)
from aegis.rag.parsing.tables import looks_like_header, to_markdown, to_row_records

logger = get_logger(__name__)

MAX_CSV_ROWS = 50_000
MAX_JSON_DEPTH = 8
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def decode(data: bytes) -> str:
    """Decode bytes to text, trying the encodings that actually occur.

    ``latin-1`` is last and cannot fail, which makes this total: a document with one corrupt
    byte is ingested with one wrong character rather than rejected.
    """
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ParserError("The file could not be decoded as text.")


class PlainTextParser:
    """Plain text. Structure is inferred from blank lines and heading heuristics."""

    name = "plaintext"

    def supports(self, mime_type: str, filename: str) -> bool:
        return mime_type == TXT

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        text = clean_text(decode(data))
        blocks = [b for b in infer_blocks_from_text(text) if b.is_heading or is_meaningful(b.text)]
        assign_hierarchy(blocks)
        return ParsedDocument(
            blocks=blocks,
            pages=[PageInfo(number=1, char_count=len(text))],
            title=document_title(blocks, fallback=filename),
            parser=self.name,
        )


class MarkdownParser:
    """Markdown. Headings, fenced code, and pipe tables are all explicit here.

    Markdown is treated as a first-class format rather than converted to plain text because
    its structure is exactly what the Markdown-aware chunker needs, and throwing it away
    would mean re-inferring it worse.
    """

    name = "markdown"

    def supports(self, mime_type: str, filename: str) -> bool:
        return mime_type == MARKDOWN

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        text = decode(data)
        front_matter, body = _split_front_matter(text)
        blocks: list[DocumentBlock] = []

        for segment, is_table in _split_markdown_tables(body):
            if is_table:
                blocks.append(
                    DocumentBlock(
                        text=clean_text(segment, preserve_layout=True),
                        block_type=ChunkType.TABLE,
                        metadata={"format": "markdown"},
                    )
                )
                continue
            for block in infer_blocks_from_text(clean_text(segment), is_markdown=True):
                # Headings and code survive the meaningfulness filter: a one-word heading
                # carries the section path, and a three-line snippet is legitimate content.
                if (
                    block.is_heading
                    or block.block_type is ChunkType.CODE
                    or is_meaningful(block.text)
                ):
                    blocks.append(block)

        assign_hierarchy(blocks)
        title = front_matter.get("title") if isinstance(front_matter.get("title"), str) else None
        return ParsedDocument(
            blocks=blocks,
            pages=[PageInfo(number=1, char_count=len(text))],
            title=title or document_title(blocks, fallback=filename),
            parser=self.name,
            metadata={"front_matter": front_matter} if front_matter else {},
        )


class DelimitedTextParser:
    """CSV and TSV.

    Emits one table block per bounded group of rows plus a schema block. The schema block
    exists because questions about a dataset are as often about its shape ("which columns are
    in the export?") as about its values, and no row chunk can answer that.
    """

    name = "delimited"

    def supports(self, mime_type: str, filename: str) -> bool:
        return mime_type in {CSV, TSV}

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        text = decode(data)
        delimiter = "\t" if mime_type == TSV else ","
        return await asyncio.to_thread(self._parse_sync, text, delimiter, filename)

    def _parse_sync(self, text: str, delimiter: str, filename: str) -> ParsedDocument:
        try:
            rows = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter) if any(row)]
        except csv.Error as exc:
            raise ParserError(f"The delimited file is malformed: {exc}") from exc

        if not rows:
            raise ParserError("The file contains no rows.")

        warnings: list[str] = []
        if len(rows) > MAX_CSV_ROWS:
            warnings.append(f"Only the first {MAX_CSV_ROWS:,} of {len(rows):,} rows were indexed.")
            rows = rows[:MAX_CSV_ROWS]

        header = rows[0] if looks_like_header(rows[0]) else []
        body = rows[1:] if header else rows
        columns = header or [f"column_{i + 1}" for i in range(max(len(r) for r in rows))]

        blocks: list[DocumentBlock] = [
            DocumentBlock(
                text=(
                    f"Dataset: {filename}\n"
                    f"Columns ({len(columns)}): {', '.join(str(c) for c in columns)}\n"
                    f"Rows: {len(body):,}"
                ),
                block_type=ChunkType.CAPTION,
                metadata={"role": "schema"},
            )
        ]

        # Row records rather than raw rows: a chunk of bare values shares almost no
        # vocabulary with the question it should answer. See rag.parsing.tables.
        records = to_row_records(body, header=columns)
        group_size = 25
        for start in range(0, len(records), group_size):
            group = records[start : start + group_size]
            blocks.append(
                DocumentBlock(
                    text="\n".join(group),
                    block_type=ChunkType.TABLE,
                    metadata={
                        "role": "rows",
                        "row_from": start + 1,
                        "row_to": start + len(group),
                        "columns": [str(c) for c in columns],
                    },
                )
            )

        return ParsedDocument(
            blocks=blocks,
            pages=[PageInfo(number=1, char_count=len(text))],
            title=filename,
            parser=self.name,
            metadata={"row_count": len(body), "columns": [str(c) for c in columns]},
            warnings=warnings,
        )


class JsonParser:
    """JSON. Flattened to readable ``path: value`` lines.

    Raw JSON embeds poorly — braces and quotes carry no meaning to an embedding model, and
    nesting separates a key from its value by hundreds of tokens. Flattened paths keep each
    value adjacent to the names that identify it.
    """

    name = "json"

    def supports(self, mime_type: str, filename: str) -> bool:
        return mime_type == JSON

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        try:
            payload = json.loads(decode(data))
        except json.JSONDecodeError as exc:
            raise ParserError(f"Invalid JSON at line {exc.lineno}: {exc.msg}") from exc

        lines = list(_flatten_json(payload))
        blocks = [
            DocumentBlock(text="\n".join(lines[i : i + 40]), block_type=ChunkType.TEXT)
            for i in range(0, len(lines), 40)
        ]
        return ParsedDocument(
            blocks=blocks or [DocumentBlock(text="(empty document)")],
            pages=[PageInfo(number=1, char_count=sum(len(line) for line in lines))],
            title=filename,
            parser=self.name,
        )


class HtmlParser:
    """HTML. Boilerplate removed, headings and tables preserved.

    ``lxml`` via BeautifulSoup rather than a regex or ``html.parser``: real-world HTML is
    malformed in ways that only a recovering parser survives, and the alternative is losing
    the second half of a page to one unclosed ``<div>``.
    """

    name = "html"

    # Elements that never contain document content. Dropped wholesale, because a nav menu
    # repeated on every page is the single largest source of BM25 noise in a web corpus.
    _DROP: ClassVar[tuple[str, ...]] = (
        "script",
        "style",
        "nav",
        "footer",
        "aside",
        "form",
        "noscript",
        "svg",
        "iframe",
    )
    _HEADINGS: ClassVar[dict[str, int]] = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    _CONTENT_TAGS: ClassVar[list[str]] = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "pre",
        "code",
        "table",
        "blockquote",
    ]

    def supports(self, mime_type: str, filename: str) -> bool:
        return mime_type == HTML

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        return await asyncio.to_thread(self._parse_sync, data, filename)

    def _parse_sync(self, data: bytes, filename: str) -> ParsedDocument:
        from bs4 import BeautifulSoup, Tag

        try:
            soup = BeautifulSoup(data, "lxml")
        except Exception as exc:
            raise ParserError(f"The HTML could not be parsed: {exc}") from exc

        for element in soup(list(self._DROP)):
            element.decompose()

        # Prefer the semantic content root; fall back to <body> when the page has none.
        root = soup.find("main") or soup.find("article") or soup.body or soup
        blocks: list[DocumentBlock] = []

        if isinstance(root, Tag):
            for element in root.find_all(self._CONTENT_TAGS):
                if not isinstance(element, Tag):
                    continue
                block = self._element_to_block(element)
                if block is not None:
                    blocks.append(block)

        assign_hierarchy(blocks)
        title_tag = soup.find("title")
        title = clean_text(title_tag.get_text()) if isinstance(title_tag, Tag) else None
        text_length = sum(len(b.text) for b in blocks)
        return ParsedDocument(
            blocks=blocks,
            pages=[PageInfo(number=1, char_count=text_length)],
            title=title or document_title(blocks, fallback=filename),
            parser=self.name,
        )

    def _element_to_block(self, element: Any) -> DocumentBlock | None:
        name = element.name

        if name == "table":
            rows = [
                [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                for row in element.find_all("tr")
            ]
            rows = [r for r in rows if any(r)]
            if not rows:
                return None
            return DocumentBlock(
                text=to_markdown(rows),
                block_type=ChunkType.TABLE,
                metadata={"format": "markdown", "rows": len(rows)},
            )

        if name in {"pre", "code"}:
            body = element.get_text("\n", strip=False)
            if not body.strip():
                return None
            return DocumentBlock(
                text=clean_text(body, preserve_layout=True), block_type=ChunkType.CODE
            )

        text = clean_text(element.get_text(" ", strip=True))
        if not text:
            return None

        if name in self._HEADINGS:
            return DocumentBlock(
                text=text, block_type=ChunkType.HEADING, heading_level=self._HEADINGS[name]
            )
        if name == "li":
            # Nested lists would otherwise be emitted once per level of nesting.
            if element.find_parent("li") is not None:
                return None
            return DocumentBlock(text=f"- {text}", block_type=ChunkType.LIST)
        if is_noise(text):
            return None
        return DocumentBlock(text=text, block_type=ChunkType.TEXT)


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML front matter without a YAML dependency.

    Only flat ``key: value`` pairs are read, which covers what front matter is actually used
    for (title, tags, dates). Anything more structured is left in the body rather than
    parsed wrongly.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    meta: dict[str, Any] = {}
    for line in text[3:end].strip().split("\n"):
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        cleaned = value.strip().strip("\"'")
        if cleaned.startswith("[") and cleaned.endswith("]"):
            meta[key.strip()] = [v.strip().strip("\"'") for v in cleaned[1:-1].split(",") if v]
        elif cleaned:
            meta[key.strip()] = cleaned

    body_start = text.find("\n", end + 1)
    return meta, text[body_start + 1 :] if body_start != -1 else ""


def _split_markdown_tables(text: str) -> list[tuple[str, bool]]:
    """Separate pipe tables from prose, preserving order.

    Tables are isolated so they reach the table-aware chunker whole; splitting a table across
    chunks leaves rows without their header, which is the same as deleting them.
    """
    segments: list[tuple[str, bool]] = []
    buffer: list[str] = []
    table: list[str] = []

    def flush(target: list[str], is_table: bool) -> None:
        if target:
            body = "\n".join(target).strip()
            if body:
                segments.append((body, is_table))
            target.clear()

    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        is_table_start = (
            _MD_TABLE_ROW.match(line)
            and index + 1 < len(lines)
            and _MD_TABLE_DIVIDER.match(lines[index + 1])
        )
        if is_table_start:
            flush(buffer, False)
            while index < len(lines) and _MD_TABLE_ROW.match(lines[index]):
                table.append(lines[index])
                index += 1
            flush(table, True)
            continue
        buffer.append(line)
        index += 1

    flush(buffer, False)
    return segments


def _flatten_json(value: Any, prefix: str = "", depth: int = 0) -> list[str]:
    """Flatten to ``a.b[0].c: value`` lines, bounded in depth."""
    if depth > MAX_JSON_DEPTH:
        return [f"{prefix}: (nested content omitted)"]
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(item, child, depth + 1))
        return lines
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value[:200]):
            lines.extend(_flatten_json(item, f"{prefix}[{index}]", depth + 1))
        if len(value) > 200:
            lines.append(f"{prefix}: ({len(value) - 200} more items omitted)")
        return lines
    return [f"{prefix or 'value'}: {value}"]
