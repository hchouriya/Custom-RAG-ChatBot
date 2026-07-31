"""Optical character recognition for scanned pages.

OCR is opt-in and degrades gracefully. ``pytesseract`` needs the ``tesseract`` binary, which
the container image installs but a developer's laptop generally does not; when it is missing,
a scanned document is ingested with a warning attached rather than failing the upload. A
document that says "3 pages could not be read" is useful; a failed ingest with no explanation
is not.

Two decisions worth noting:

* Pages are rasterised with ``pypdfium2`` rather than ``pdf2image``, so there is no poppler
  system dependency.
* Confidence is captured per page and stored. Text recognised at 45% confidence is not
  equivalent to extracted text, and marking it lets the admin UI show *why* a document
  retrieves badly instead of leaving the content owner guessing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import cache

from aegis.core.logging import get_logger

logger = get_logger(__name__)

# 300 DPI is the accuracy floor for tesseract on body text; 150 loses small print and 600
# quadruples the time for no measurable gain.
DEFAULT_DPI = 300
MIN_ACCEPTABLE_CONFIDENCE = 0.45


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float | None
    engine: str
    warning: str | None = None

    @property
    def is_usable(self) -> bool:
        if not self.text.strip():
            return False
        return self.confidence is None or self.confidence >= MIN_ACCEPTABLE_CONFIDENCE


@cache
def ocr_available() -> bool:
    """Whether OCR can actually run in this process.

    Cached because it shells out to the tesseract binary, and the answer cannot change
    without a restart.
    """
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception as exc:
        logger.info("ocr.unavailable", reason=str(exc)[:200])
        return False
    return True


def render_page(pdf_bytes: bytes, page_index: int, *, dpi: int = DEFAULT_DPI) -> bytes:
    """Rasterise one page to PNG bytes.

    Rendering one page at a time keeps peak memory proportional to a single page rather than
    to the document: a 400-page scan at 300 DPI is about 12 GB if rendered eagerly.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        page = document[page_index]
        bitmap = page.render(scale=dpi / 72)
        image = bitmap.to_pil()
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        document.close()


def ocr_image(image_bytes: bytes, *, languages: str = "eng", psm: int = 3) -> OcrResult:
    """Recognise text in one rasterised page.

    ``image_to_data`` is used instead of ``image_to_string`` because it returns per-word
    confidence, which is the only way to distinguish "this page is blank" from "this page was
    unreadable" — the two produce identical empty strings otherwise.
    """
    if not ocr_available():
        return OcrResult(
            text="",
            confidence=None,
            engine="none",
            warning="OCR is unavailable: the tesseract binary is not installed.",
        )

    import io

    import pytesseract
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            data = pytesseract.image_to_data(
                image,
                lang=languages,
                config=f"--psm {psm}",
                output_type=pytesseract.Output.DICT,
            )
    except Exception as exc:
        logger.warning("ocr.failed", error=str(exc)[:200])
        return OcrResult(text="", confidence=None, engine="tesseract", warning=str(exc)[:200])

    words: list[str] = []
    confidences: list[float] = []
    for text, confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        words.append(cleaned)
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value / 100.0)

    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    result = OcrResult(
        text=" ".join(words),
        confidence=mean_confidence,
        engine="tesseract",
    )
    if mean_confidence is not None and mean_confidence < MIN_ACCEPTABLE_CONFIDENCE:
        result.warning = f"Low OCR confidence ({mean_confidence:.0%}); text may be unreliable."
    return result


async def ocr_pdf_page(
    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_DPI,
    languages: str = "eng",
) -> OcrResult:
    """Render and recognise one PDF page off the event loop.

    Both stages are CPU-bound and take seconds. Running them inline would block every other
    request served by the same worker, which for a streaming chat API is fatal.
    """

    def _run() -> OcrResult:
        try:
            image = render_page(pdf_bytes, page_index, dpi=dpi)
        except Exception as exc:
            logger.warning("ocr.render_failed", page=page_index + 1, error=str(exc)[:200])
            return OcrResult(
                text="",
                confidence=None,
                engine="none",
                warning=f"Page {page_index + 1} could not be rendered for OCR.",
            )
        return ocr_image(image, languages=languages)

    return await asyncio.to_thread(_run)
