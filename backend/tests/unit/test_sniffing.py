"""Content-type detection.

The security-relevant cases are the ones where the declared type and the bytes disagree, and
the ones where an unsupported format must be rejected with an actionable message rather than
parsed by whichever parser claims it first.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from aegis.rag.parsing import sniff
from aegis.rag.parsing.sniff import SniffResult


def _ooxml(prefix: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(f"{prefix}/document.xml", "<document/>")
    return buffer.getvalue()


class TestMagicBytes:
    def test_pdf_is_detected_from_signature(self) -> None:
        assert sniff.detect_content_type(b"%PDF-1.7\n...", filename="x.pdf").mime_type == sniff.PDF

    def test_docx_is_detected_from_archive_layout(self) -> None:
        result = sniff.detect_content_type(_ooxml("word"), filename="report.docx")
        assert result.mime_type == sniff.DOCX

    def test_xlsx_and_pptx_are_distinguished_from_docx(self) -> None:
        assert sniff.detect_content_type(_ooxml("xl"), filename="a.xlsx").mime_type == sniff.XLSX
        assert sniff.detect_content_type(_ooxml("ppt"), filename="a.pptx").mime_type == sniff.PPTX

    def test_extension_does_not_override_bytes(self) -> None:
        """A PDF renamed to .docx is still parsed as a PDF.

        This is the whole point of sniffing: the extension is uploader-controlled.
        """
        result = sniff.detect_content_type(b"%PDF-1.4\n", filename="invoice.docx")
        assert result.mime_type == sniff.PDF


class TestRejections:
    def test_plain_zip_is_rejected_with_guidance(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.txt", "hello")
        result = sniff.detect_content_type(buffer.getvalue(), filename="bundle.zip")
        assert not result.is_supported
        assert "individually" in (result.detail or "")

    def test_legacy_office_says_how_to_fix_it(self) -> None:
        ole2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32
        result = sniff.detect_content_type(ole2, filename="old.doc")
        assert not result.is_supported
        assert ".docx" in (result.detail or "")

    def test_binary_is_rejected(self) -> None:
        result = sniff.detect_content_type(bytes(range(256)), filename="blob.bin")
        assert not result.is_supported


class TestTextClassification:
    def test_markdown_is_recognised_by_content(self) -> None:
        body = b"# Title\n\nSome text.\n\n- one\n- two\n\n## Section\n\nMore.\n"
        assert sniff.detect_content_type(body, filename="notes.txt").mime_type == sniff.MARKDOWN

    def test_csv_requires_consistent_columns(self) -> None:
        table = b"name,role,dept\nAda,eng,core\nGrace,eng,core\n"
        assert sniff.detect_content_type(table, filename="people.csv").mime_type == sniff.CSV

    def test_prose_named_csv_is_treated_as_text(self) -> None:
        """A misnamed file must not reach the table parser.

        Otherwise every comma in a sentence becomes a column and the document indexes as
        one-cell rows.
        """
        prose = b"Hello, this is a sentence.\nAnd another one, with commas.\n"
        assert sniff.detect_content_type(prose, filename="notes.csv").mime_type == sniff.TXT

    def test_html_is_recognised(self) -> None:
        page = b"<!DOCTYPE html><html><body><h1>Hi</h1></body></html>"
        assert sniff.detect_content_type(page, filename="page.html").mime_type == sniff.HTML

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "cp1252"])
    def test_various_encodings_decode(self, encoding: str) -> None:
        data = "Naïve café résumé — text.".encode(encoding)
        assert sniff.detect_content_type(data, filename="a.txt").is_supported


class TestMismatchReporting:
    def test_mismatch_is_flagged(self) -> None:
        result = sniff.detect_content_type(
            b"%PDF-1.4\n", filename="a.pdf", declared_mime="image/png"
        )
        assert result.mismatch

    def test_octet_stream_is_not_a_mismatch(self) -> None:
        """Browsers send this constantly; treating it as suspicious would be pure noise."""
        result = SniffResult(sniff.PDF, ".pdf", "application/octet-stream")
        assert not result.mismatch
