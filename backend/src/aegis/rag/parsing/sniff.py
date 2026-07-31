"""Content-type detection from magic bytes.

The client-declared ``Content-Type`` is never trusted for anything that decides how a file
is opened. It is attacker-controlled, and "this .pdf is really a zip full of symlinks" is a
classic upload exploit. Extensions are used only to disambiguate formats whose bytes are
genuinely identical (Markdown, CSV, and plain text are all just text).

``filetype`` is used rather than ``python-magic`` because it is pure Python: no ``libmagic``
system package, which means the same code path in the container, on a developer's Windows
machine, and in CI.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

import filetype

from aegis.core.logging import get_logger

logger = get_logger(__name__)

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOC = "application/msword"
XLS = "application/vnd.ms-excel"
PPT = "application/vnd.ms-powerpoint"
HTML = "text/html"
MARKDOWN = "text/markdown"
CSV = "text/csv"
TSV = "text/tab-separated-values"
JSON = "application/json"
TXT = "text/plain"
RTF = "application/rtf"

SUPPORTED: frozenset[str] = frozenset({PDF, DOCX, PPTX, XLSX, HTML, MARKDOWN, CSV, TSV, JSON, TXT})
"""Formats the platform will ingest.

Legacy binary Office formats (``.doc``, ``.xls``, ``.ppt``) are detected so the rejection
can say "convert this to .docx" instead of "unsupported file", which is the difference
between a user solving their problem and filing a ticket.
"""

LEGACY_OFFICE: frozenset[str] = frozenset({DOC, XLS, PPT})

EXTENSION_MIME: dict[str, str] = {
    ".pdf": PDF,
    ".docx": DOCX,
    ".pptx": PPTX,
    ".xlsx": XLSX,
    ".doc": DOC,
    ".xls": XLS,
    ".ppt": PPT,
    ".htm": HTML,
    ".html": HTML,
    ".md": MARKDOWN,
    ".markdown": MARKDOWN,
    ".csv": CSV,
    ".tsv": TSV,
    ".json": JSON,
    ".txt": TXT,
    ".text": TXT,
    ".log": TXT,
    ".rtf": RTF,
}

# OOXML containers are all zip archives; the entry layout is what distinguishes them.
_OOXML_MARKERS: tuple[tuple[str, str], ...] = (
    ("word/", DOCX),
    ("ppt/", PPTX),
    ("xl/", XLSX),
)

_MAX_SNIFF_BYTES = 8192
_MAX_HEADER_CELL_CHARS = 48
_MAX_HEADER_CELL_WORDS = 5


@dataclass(frozen=True, slots=True)
class SniffResult:
    mime_type: str
    extension: str
    declared_mime: str | None = None
    is_supported: bool = True
    detail: str | None = None
    """Why an unsupported result was reached, phrased for the uploader."""

    @property
    def mismatch(self) -> bool:
        """Whether the declared type disagreed with the bytes.

        Not fatal on its own — browsers send ``application/octet-stream`` constantly — but
        recorded on the audit entry, because a deliberate mismatch is a probe.
        """
        if not self.declared_mime or self.declared_mime == "application/octet-stream":
            return False
        return self.declared_mime.split(";")[0].strip().lower() != self.mime_type


def detect_content_type(
    data: bytes, *, filename: str, declared_mime: str | None = None
) -> SniffResult:
    """Determine the real content type of ``data``.

    Order matters: signature-based detection first, then container inspection for OOXML,
    then a text-shape heuristic, and only then the extension. The extension is a hint of
    last resort because it is the one thing the uploader fully controls.
    """
    extension = PurePosixPath(filename.lower()).suffix

    if data.startswith(b"%PDF-"):
        return SniffResult(PDF, extension, declared_mime)

    if data[:4] == b"PK\x03\x04":
        ooxml = _sniff_ooxml(data)
        if ooxml:
            return SniffResult(ooxml, extension, declared_mime)
        return SniffResult(
            "application/zip",
            extension,
            declared_mime,
            is_supported=False,
            detail="Archives are not ingested. Upload the documents individually.",
        )

    # Legacy OLE2 compound files: .doc/.xls/.ppt all share this signature, so the extension
    # is what names them — acceptable here because the result is a rejection either way.
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        legacy = EXTENSION_MIME.get(extension, DOC)
        return SniffResult(
            legacy,
            extension,
            declared_mime,
            is_supported=False,
            detail=(
                "Legacy Office formats are not supported. "
                "Save as .docx, .xlsx, or .pptx and upload again."
            ),
        )

    if data.startswith(b"{\\rtf"):
        return SniffResult(
            RTF,
            extension,
            declared_mime,
            is_supported=False,
            detail="RTF is not supported. Save as .docx and upload again.",
        )

    guess = filetype.guess(data[:_MAX_SNIFF_BYTES])
    if guess is not None:
        mime = guess.mime
        if mime not in SUPPORTED:
            return SniffResult(
                mime,
                extension,
                declared_mime,
                is_supported=False,
                detail=f"{mime} files are not ingested.",
            )
        return SniffResult(mime, extension, declared_mime)

    text = _decode_text(data[:_MAX_SNIFF_BYTES])
    if text is None:
        return SniffResult(
            "application/octet-stream",
            extension,
            declared_mime,
            is_supported=False,
            detail="The file is not readable as a document.",
        )

    return SniffResult(_classify_text(text, extension), extension, declared_mime)


def _sniff_ooxml(data: bytes) -> str | None:
    """Identify an OOXML flavour by its archive layout.

    Reads the central directory only. A malformed or bomb-like archive returns ``None``
    rather than raising, so the caller's rejection path handles it uniformly.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return None

    for prefix, mime in _OOXML_MARKERS:
        if any(name.startswith(prefix) for name in names):
            return mime
    return None


def _decode_text(data: bytes) -> str | None:
    """Decode as text, or return ``None`` if it is binary.

    A byte-order mark settles the encoding outright and has to be checked first: UTF-16 text
    is half NUL bytes, so the binary test below would reject every UTF-16 document. Windows
    tools still emit those, notably "Save As → Unicode text".

    Without a BOM, a NUL byte in the first block is the practical binary test: no single-byte
    text document contains one, and every binary format hits one almost immediately.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        # The sample is a byte prefix and may end mid-code-unit, hence the even-length trim
        # and the lenient error handling.
        return data[: len(data) - len(data) % 2].decode("utf-16", errors="ignore")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="ignore")
    if b"\x00" in data:
        return None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _classify_text(text: str, extension: str) -> str:
    """Distinguish HTML, Markdown, CSV, JSON, and plain text.

    Content wins over extension here, unlike elsewhere: a ``.txt`` that is unmistakably a
    Markdown document should be chunked by heading, and a ``.csv`` that is actually prose
    should not be parsed as a table.
    """
    stripped = text.lstrip()
    lowered = stripped[:2048].lower()

    if lowered.startswith(("<!doctype html", "<html", "<?xml")) or ("<body" in lowered):
        return HTML
    if stripped.startswith(("{", "[")) and extension == ".json":
        return JSON

    declared = EXTENSION_MIME.get(extension)
    if declared in {CSV, TSV} and _looks_tabular(text, "," if declared == CSV else "\t"):
        return declared
    if declared == MARKDOWN:
        return MARKDOWN

    if _looks_markdown(text):
        return MARKDOWN
    if declared in {CSV, TSV}:
        # Extension says tabular but the shape disagrees; treat as prose so the table
        # parser does not emit one-cell rows.
        return TXT
    if _looks_tabular(text, ","):
        return CSV
    return TXT


def _looks_markdown(text: str) -> bool:
    """Heuristic: ATX headings, fenced code, or list markers on several lines."""
    lines = text.splitlines()[:200]
    if not lines:
        return False
    signals = sum(
        1
        for line in lines
        if line.startswith(("# ", "## ", "### ", "- ", "* ", "> ", "```", "| "))
        or (line.startswith("[") and "](" in line)
    )
    return signals >= max(2, len(lines) // 20)


def _looks_tabular(text: str, delimiter: str) -> bool:
    """Whether the first rows parse as a consistent, multi-column table.

    ``csv.Sniffer`` is avoided: it raises on ambiguous input and is generous with prose
    containing commas. Column-count stability is stricter and quieter, but not sufficient on
    its own — two sentences with one comma each also have a stable width of two. So the first
    row must additionally look like a header: short, complete, unpunctuated field names.
    """
    sample = "\n".join(text.splitlines()[:20])
    if not sample.strip():
        return False
    try:
        rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
    except csv.Error:
        return False
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return False
    widths = {len(r) for r in rows}
    if len(widths) != 1 or next(iter(widths)) < 2:
        return False
    return _looks_like_field_names(rows[0])


def _looks_like_field_names(row: list[str]) -> bool:
    """Whether a row reads as column names rather than the first sentence of a document."""
    for cell in row:
        value = cell.strip()
        if not value or len(value) > _MAX_HEADER_CELL_CHARS:
            return False
        if value.endswith((".", "!", "?", ":", ";")):
            return False
        if len(value.split()) > _MAX_HEADER_CELL_WORDS:
            return False
    return True
