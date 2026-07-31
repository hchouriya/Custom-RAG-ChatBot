"""PDF parsing with layout awareness, table extraction, and OCR fallback.

PDF is the hardest and most common format, and the one where citation quality is decided. A
PDF has no paragraphs, no headings, and no reading order — only glyphs at coordinates. Every
structural fact a citation needs ("page 12, §4.2, these three lines") has to be reconstructed
from geometry and font metrics at parse time; none of it can be recovered later from extracted
text.

Library choice: ``pdfplumber`` (MIT, over ``pdfminer.six``) for text with per-word font
metrics and for table detection, and ``pypdf`` (BSD) for document metadata and encryption
checks. **PyMuPDF is deliberately not used**: it is faster and better, but it is AGPL, which
for a proprietary enterprise product means either publishing the source or buying a
commercial licence. That is a legal decision, not a technical one, so it is recorded here
next to the code it constrains.

Extraction strategy per page:

1. Detect tables and record their bounding boxes.
2. Extract words *outside* those boxes, with size and font name.
3. Group words into lines by baseline, lines into paragraphs by vertical gap.
4. Classify each line against the document's modal body font size.
5. If the page yielded almost no text, treat it as scanned and OCR it.

Running headers and footers are identified across pages and removed, because a footer
repeated on 200 pages otherwise dominates BM25 for its own words.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any

from aegis.core.errors import ParserError
from aegis.core.logging import get_logger
from aegis.domain.enums import ChunkType
from aegis.domain.ports.parser import DocumentBlock, PageInfo, ParsedDocument
from aegis.rag.parsing.cleaning import (
    clean_text,
    detect_repeated_lines,
    is_meaningful,
    normalize_whitespace,
    strip_boilerplate,
)
from aegis.rag.parsing.ocr import ocr_pdf_page
from aegis.rag.parsing.sniff import PDF
from aegis.rag.parsing.structure import assign_hierarchy, detect_heading, document_title
from aegis.rag.parsing.tables import looks_like_header, to_markdown, to_row_records

logger = get_logger(__name__)

MAX_PAGES = 2000
# A page with fewer characters than this is treated as scanned. 100 is high enough to catch a
# scan bearing only a stamped page number, and low enough not to OCR a genuine title page.
DEFAULT_OCR_MIN_CHARS = 100
# Two lines are on the same baseline if their tops differ by less than this many points.
_LINE_TOLERANCE = 2.5
# A vertical gap larger than this multiple of the line height starts a new paragraph.
_PARAGRAPH_GAP_RATIO = 1.6
_LARGE_TABLE_ROWS = 12


@dataclass(slots=True)
class _Line:
    text: str
    top: float
    bottom: float
    size: float
    is_bold: bool
    x0: float

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1.0)


@dataclass(slots=True)
class _PageExtract:
    number: int
    lines: list[_Line] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    width: float | None = None
    height: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line.text) for line in self.lines)


class PdfParser:
    """Parses PDFs, falling back to OCR per page rather than per document.

    Per-page fallback matters in practice: real documents are mixed, with a scanned signature
    page appended to a digital contract. Deciding OCR for the whole document would either
    waste minutes on 200 digital pages or silently drop the one page that carries the
    signature.
    """

    name = "pdfplumber"

    def __init__(
        self,
        *,
        ocr_enabled: bool = True,
        ocr_min_chars_per_page: int = DEFAULT_OCR_MIN_CHARS,
        ocr_dpi: int = 300,
        ocr_languages: str = "eng",
        max_pages: int = MAX_PAGES,
    ) -> None:
        self._ocr_enabled = ocr_enabled
        self._ocr_min_chars = ocr_min_chars_per_page
        self._ocr_dpi = ocr_dpi
        self._ocr_languages = ocr_languages
        self._max_pages = max_pages

    def supports(self, mime_type: str, filename: str) -> bool:
        return mime_type == PDF

    async def parse(self, data: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        # Layout extraction is CPU-bound for whole seconds on a large document; keeping it
        # off the event loop is what lets one worker serve other requests meanwhile.
        extracts, metadata, warnings = await asyncio.to_thread(self._extract_sync, data)

        pages: list[PageInfo] = []
        ocr_used = False
        for extract in extracts:
            if (
                self._ocr_enabled
                and extract.char_count < self._ocr_min_chars
                and not extract.tables
            ):
                result = await ocr_pdf_page(
                    data,
                    extract.number - 1,
                    dpi=self._ocr_dpi,
                    languages=self._ocr_languages,
                )
                if result.warning:
                    warnings.append(f"Page {extract.number}: {result.warning}")
                if result.is_usable:
                    ocr_used = True
                    extract.lines = [
                        _Line(
                            text=clean_text(result.text),
                            top=0.0,
                            bottom=10.0,
                            size=0.0,
                            is_bold=False,
                            x0=0.0,
                        )
                    ]
                    pages.append(
                        PageInfo(
                            number=extract.number,
                            char_count=len(result.text),
                            used_ocr=True,
                            ocr_confidence=result.confidence,
                            width=extract.width,
                            height=extract.height,
                        )
                    )
                    continue
                warnings.append(
                    f"Page {extract.number} appears to be scanned and could not be read."
                )
            pages.append(
                PageInfo(
                    number=extract.number,
                    char_count=extract.char_count,
                    width=extract.width,
                    height=extract.height,
                )
            )

        boilerplate = detect_repeated_lines([e.text for e in extracts])
        body_size = _modal_body_size(extracts)
        blocks = self._build_blocks(extracts, boilerplate=boilerplate, body_size=body_size)

        form_fields = metadata.get("form_fields")
        if isinstance(form_fields, dict) and form_fields:
            blocks.insert(
                0,
                DocumentBlock(
                    text="\n".join(f"{name}: {value}" for name, value in form_fields.items()),
                    block_type=ChunkType.FORM,
                    page=1,
                    metadata={"role": "form_fields"},
                ),
            )

        assign_hierarchy(blocks)

        title = metadata.get("title") or document_title(blocks, fallback=filename)
        return ParsedDocument(
            blocks=blocks,
            pages=pages,
            title=normalize_whitespace(str(title)) if title else None,
            parser=self.name,
            used_ocr=ocr_used,
            metadata=metadata,
            warnings=warnings,
        )

    # ── Extraction ──────────────────────────────────────────────────────────

    def _extract_sync(self, data: bytes) -> tuple[list[_PageExtract], dict[str, Any], list[str]]:
        import io

        import pdfplumber

        warnings: list[str] = []
        metadata = self._document_metadata(data, warnings)

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                page_count = len(pdf.pages)
                if page_count > self._max_pages:
                    warnings.append(
                        f"Only the first {self._max_pages} of {page_count} pages were indexed."
                    )
                extracts = [
                    self._extract_page(page, index + 1, warnings)
                    for index, page in enumerate(pdf.pages[: self._max_pages])
                ]
        except ParserError:
            raise
        except Exception as exc:
            raise ParserError(f"The PDF could not be read: {exc}") from exc

        metadata["page_count"] = len(extracts)
        return extracts, metadata, warnings

    def _document_metadata(self, data: bytes, warnings: list[str]) -> dict[str, Any]:
        """Read document info with pypdf, and reject encrypted files clearly.

        An encrypted PDF extracts as empty text rather than raising, so without this check the
        document would be ingested successfully with no content — the worst possible outcome,
        because nothing looks wrong until a search returns nothing.
        """
        import io

        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data))
        except PdfReadError as exc:
            raise ParserError(f"The PDF is malformed: {exc}") from exc

        if reader.is_encrypted:
            try:
                # Owner-password-only PDFs decrypt with an empty user password and are
                # legitimately readable.
                if reader.decrypt("") == 0:
                    raise ParserError(
                        "The PDF is password-protected. Remove the password and upload again."
                    )
            except NotImplementedError as exc:
                raise ParserError("The PDF uses an unsupported encryption scheme.") from exc

        info: dict[str, Any] = dict(reader.metadata or {})
        metadata: dict[str, Any] = {}
        for key, target in (
            ("/Title", "title"),
            ("/Author", "author"),
            ("/Subject", "subject"),
            ("/Producer", "producer"),
            ("/CreationDate", "created"),
        ):
            value = info.get(key)
            if value:
                metadata[target] = str(value)[:500]

        # Filled form fields hold content that glyph extraction never sees — an AcroForm value
        # lives in the field dictionary, not the page content stream. Without this, a
        # completed application form ingests as a blank template.
        try:
            fields = reader.get_fields() or {}
        except Exception:
            fields = {}
        filled = {
            str(name): str(field.get("/V"))
            for name, field in fields.items()
            if isinstance(field, dict) and field.get("/V") not in (None, "")
        }
        if filled:
            metadata["form_fields"] = filled
            warnings.append(f"{len(filled)} filled form field(s) were indexed as text.")
        return metadata

    def _extract_page(self, page: Any, number: int, warnings: list[str]) -> _PageExtract:
        extract = _PageExtract(number=number, width=page.width, height=page.height)
        try:
            tables = page.find_tables()
        except Exception:
            tables = []

        boxes: list[tuple[float, float, float, float]] = []
        for table in tables:
            try:
                rows = [[(cell or "").strip() for cell in row] for row in table.extract()]
            except Exception as exc:
                logger.debug("pdf.table_extract_failed", page=number, error=str(exc)[:200])
                continue
            rows = [row for row in rows if any(row)]
            if len(rows) >= 2:
                extract.tables.append(rows)
                x0, top, x1, bottom = table.bbox
                boxes.append((x0, top, x1, bottom))

        try:
            words = page.extract_words(
                extra_attrs=["size", "fontname"], use_text_flow=False, keep_blank_chars=False
            )
        except Exception as exc:
            warnings.append(f"Page {number}: text extraction failed ({exc}).")
            return extract

        outside = [w for w in words if not _in_any_box(w, boxes)]
        extract.lines = _group_words_into_lines(outside)
        return extract

    # ── Block assembly ──────────────────────────────────────────────────────

    def _build_blocks(
        self,
        extracts: list[_PageExtract],
        *,
        boilerplate: set[str],
        body_size: float | None,
    ) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        for extract in extracts:
            blocks.extend(
                self._page_text_blocks(extract, boilerplate=boilerplate, body_size=body_size)
            )
            blocks.extend(self._page_table_blocks(extract))
        return blocks

    def _page_text_blocks(
        self, extract: _PageExtract, *, boilerplate: set[str], body_size: float | None
    ) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        lines = [line for line in extract.lines if line.text.strip() not in boilerplate]
        if not lines:
            return blocks

        heights = [line.height for line in lines]
        median_height = statistics.median(heights) if heights else 12.0
        paragraph: list[_Line] = []

        def flush() -> None:
            nonlocal paragraph
            if not paragraph:
                return
            body = strip_boilerplate(
                clean_text("\n".join(line.text for line in paragraph)), boilerplate
            )
            if body and is_meaningful(body):
                blocks.append(
                    DocumentBlock(
                        text=body,
                        block_type=ChunkType.TEXT,
                        page=extract.number,
                        bbox={
                            "x0": min(line.x0 for line in paragraph),
                            "top": paragraph[0].top,
                            "bottom": paragraph[-1].bottom,
                        },
                    )
                )
            paragraph = []

        previous: _Line | None = None
        for line in lines:
            heading = detect_heading(
                line.text,
                font_size=line.size or None,
                body_font_size=body_size,
                is_bold=line.is_bold,
            )
            if heading is not None and heading.confidence >= 0.6:
                flush()
                blocks.append(
                    DocumentBlock(
                        text=heading.text,
                        block_type=ChunkType.HEADING,
                        page=extract.number,
                        heading_level=heading.level,
                        bbox={"x0": line.x0, "top": line.top, "bottom": line.bottom},
                    )
                )
                previous = line
                continue

            if previous is not None and (line.top - previous.bottom) > median_height * (
                _PARAGRAPH_GAP_RATIO - 1
            ):
                flush()
            paragraph.append(line)
            previous = line

        flush()
        return blocks

    def _page_table_blocks(self, extract: _PageExtract) -> list[DocumentBlock]:
        """Emit each table, choosing an encoding based on its size.

        Small tables stay as Markdown, which keeps them readable and compact. Large ones become
        row records so that no chunk boundary can separate a value from its column name — see
        ``rag.parsing.tables`` for why that changes whether the table is retrievable at all.
        """
        blocks: list[DocumentBlock] = []
        for index, rows in enumerate(extract.tables, start=1):
            header = rows[0] if looks_like_header(rows[0]) else []
            body = rows[1:] if header else rows
            columns = header or [f"column_{i + 1}" for i in range(len(rows[0]))]

            if len(rows) <= _LARGE_TABLE_ROWS:
                blocks.append(
                    DocumentBlock(
                        text=to_markdown(body, header=columns),
                        block_type=ChunkType.TABLE,
                        page=extract.number,
                        metadata={
                            "format": "markdown",
                            "table_index": index,
                            "rows": len(body),
                            "columns": [str(c) for c in columns],
                        },
                    )
                )
                continue

            records = to_row_records(body, header=columns)
            group = 20
            for start in range(0, len(records), group):
                window = records[start : start + group]
                blocks.append(
                    DocumentBlock(
                        text="\n".join(window),
                        block_type=ChunkType.TABLE,
                        page=extract.number,
                        metadata={
                            "format": "records",
                            "table_index": index,
                            "row_from": start + 1,
                            "row_to": start + len(window),
                            "columns": [str(c) for c in columns],
                        },
                    )
                )
        return blocks


def _in_any_box(word: dict[str, Any], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Whether a word falls inside a detected table.

    Table text is emitted separately with its structure intact; leaving it in the prose stream
    as well would duplicate it and produce paragraphs of orphaned cell values.
    """
    x_center = (float(word["x0"]) + float(word["x1"])) / 2
    y_center = (float(word["top"]) + float(word["bottom"])) / 2
    return any(x0 <= x_center <= x1 and top <= y_center <= bottom for x0, top, x1, bottom in boxes)


def _group_words_into_lines(words: list[dict[str, Any]]) -> list[_Line]:
    """Group words into visual lines by baseline, then sort left to right.

    Necessary because ``extract_words`` returns words in content-stream order, which for
    multi-column layouts and generated PDFs is not reading order.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    lines: list[_Line] = []
    bucket: list[dict[str, Any]] = []
    current_top = float(ordered[0]["top"])

    def flush() -> None:
        if not bucket:
            return
        row = sorted(bucket, key=lambda w: float(w["x0"]))
        text = " ".join(str(w["text"]) for w in row).strip()
        if not text:
            return
        sizes = [float(w.get("size") or 0) for w in row]
        fonts = " ".join(str(w.get("fontname") or "") for w in row).lower()
        lines.append(
            _Line(
                text=text,
                top=min(float(w["top"]) for w in row),
                bottom=max(float(w["bottom"]) for w in row),
                size=max(sizes) if sizes else 0.0,
                is_bold="bold" in fonts or "black" in fonts,
                x0=min(float(w["x0"]) for w in row),
            )
        )

    for word in ordered:
        top = float(word["top"])
        if abs(top - current_top) > _LINE_TOLERANCE:
            flush()
            bucket = []
            current_top = top
        bucket.append(word)
    flush()
    return lines


def _modal_body_size(extracts: list[_PageExtract]) -> float | None:
    """The document's body font size, as the most common rounded line size.

    Weighted by characters rather than by line count, so that a document with many short
    heading lines does not mistake heading size for body size.
    """
    weights: dict[float, int] = {}
    for extract in extracts:
        for line in extract.lines:
            if line.size <= 0:
                continue
            key = round(line.size, 1)
            weights[key] = weights.get(key, 0) + len(line.text)
    if not weights:
        return None
    return max(weights.items(), key=lambda item: item[1])[0]
