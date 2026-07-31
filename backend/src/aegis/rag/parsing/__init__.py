"""Document parsing: bytes in, typed blocks with locators out."""

from aegis.rag.parsing.office import DocxParser, PptxParser, XlsxParser
from aegis.rag.parsing.pdf import PdfParser
from aegis.rag.parsing.registry import ParserRegistry, build_parser_registry
from aegis.rag.parsing.sniff import SniffResult, detect_content_type
from aegis.rag.parsing.text import (
    DelimitedTextParser,
    HtmlParser,
    JsonParser,
    MarkdownParser,
    PlainTextParser,
)

__all__ = [
    "DelimitedTextParser",
    "DocxParser",
    "HtmlParser",
    "JsonParser",
    "MarkdownParser",
    "ParserRegistry",
    "PdfParser",
    "PlainTextParser",
    "PptxParser",
    "SniffResult",
    "XlsxParser",
    "build_parser_registry",
    "detect_content_type",
]
