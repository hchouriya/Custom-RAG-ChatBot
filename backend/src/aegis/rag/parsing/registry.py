"""Parser selection.

One place decides which parser handles a file, and it decides from sniffed bytes rather than
from the filename. Adding a format means adding a parser and one registry entry; no caller
changes.

The registry also owns the post-parse invariants that every format must satisfy, because
"did we actually get any text out of this?" is a question with the same answer for all nine
formats and is the difference between a silently empty document and a clear rejection.
"""

from __future__ import annotations

from aegis.core.config import Settings
from aegis.core.errors import ParserError, UnsupportedMediaTypeError
from aegis.core.logging import get_logger
from aegis.core.telemetry import timed_stage
from aegis.domain.ports.parser import DocumentParser, ParsedDocument
from aegis.rag.parsing.office import DocxParser, PptxParser, XlsxParser
from aegis.rag.parsing.pdf import PdfParser
from aegis.rag.parsing.sniff import PDF, SniffResult, detect_content_type
from aegis.rag.parsing.text import (
    DelimitedTextParser,
    HtmlParser,
    JsonParser,
    MarkdownParser,
    PlainTextParser,
)

logger = get_logger(__name__)

MIN_EXTRACTED_CHARS = 20
"""Below this, a document is treated as empty.

A "successful" ingest that produced no text is the worst failure mode available: nothing looks
wrong, the document appears in the admin list as indexed, and it silently never retrieves. It
is better to fail the upload and say why.
"""


class ParserRegistry:
    """Routes a file to its parser.

    Parsers are constructed once and reused. They are stateless — configuration is injected at
    construction, and per-call state lives in locals — so sharing them across concurrent
    ingests is safe and avoids re-reading settings per document.
    """

    def __init__(self, settings: Settings) -> None:
        self._parsers: tuple[DocumentParser, ...] = (
            PdfParser(
                ocr_enabled=settings.ocr_enabled,
                ocr_min_chars_per_page=settings.ocr_min_chars_per_page,
                ocr_dpi=settings.ocr_dpi,
                ocr_languages=settings.ocr_languages,
            ),
            DocxParser(),
            PptxParser(),
            XlsxParser(),
            HtmlParser(),
            MarkdownParser(),
            DelimitedTextParser(),
            JsonParser(),
            # Plain text is last: it would otherwise claim files that a more specific parser
            # handles better.
            PlainTextParser(),
        )

    @property
    def parsers(self) -> tuple[DocumentParser, ...]:
        return self._parsers

    def identify(
        self, data: bytes, *, filename: str, declared_mime: str | None = None
    ) -> SniffResult:
        """Determine the true content type, rejecting unsupported formats with a reason."""
        result = detect_content_type(data, filename=filename, declared_mime=declared_mime)
        if not result.is_supported:
            raise UnsupportedMediaTypeError(
                result.detail or f"{result.mime_type} files are not supported."
            )
        if result.mismatch:
            # Not fatal — browsers send wrong types constantly — but a deliberate mismatch is a
            # probe, so it is recorded on the audit entry for the upload.
            logger.info(
                "upload.mime_mismatch",
                declared=result.declared_mime,
                detected=result.mime_type,
                filename=filename,
            )
        return result

    def for_type(self, mime_type: str, filename: str) -> DocumentParser:
        for parser in self._parsers:
            if parser.supports(mime_type, filename):
                return parser
        raise UnsupportedMediaTypeError(f"No parser is registered for {mime_type}.")

    async def parse(
        self, data: bytes, *, filename: str, declared_mime: str | None = None
    ) -> tuple[ParsedDocument, SniffResult]:
        """Identify, parse, and validate. Returns the document and what it was detected as."""
        detected = self.identify(data, filename=filename, declared_mime=declared_mime)
        parser = self.for_type(detected.mime_type, filename)

        with timed_stage("parse") as span:
            document = await parser.parse(data, filename=filename, mime_type=detected.mime_type)
            span["parser"] = parser.name

        self._validate(document, filename=filename, mime_type=detected.mime_type)
        logger.info(
            "parse.completed",
            parser=parser.name,
            mime_type=detected.mime_type,
            blocks=len(document.blocks),
            pages=document.page_count,
            chars=document.char_count,
            used_ocr=document.used_ocr,
            warnings=len(document.warnings),
        )
        return document, detected

    def _validate(self, document: ParsedDocument, *, filename: str, mime_type: str) -> None:
        if document.char_count >= MIN_EXTRACTED_CHARS:
            return
        if mime_type == PDF:
            raise ParserError(
                "No text could be extracted. The PDF appears to be a scan — enable OCR, "
                "or upload a text-based version."
            )
        raise ParserError(f"No text could be extracted from {filename}.")


def build_parser_registry(settings: Settings) -> ParserRegistry:
    return ParserRegistry(settings)
