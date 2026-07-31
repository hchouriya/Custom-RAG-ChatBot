"""Text normalisation applied to every parsed block.

Cleaning happens once, at ingest, and its output is what gets embedded, indexed, cited, and
shown in the citation drawer. That has one hard consequence: cleaning must never delete
content, only normalise its representation. A step that drops a line to make text tidier
also makes a citation point at the wrong place, and there is no way to notice from the
outside.

The specific problems each function fixes are all things that measurably degrade retrieval:
soft hyphens split a word into two tokens that match neither query form, repeated page
headers dominate BM25 for their own words, and ligatures make "workflow" unsearchable.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

# Zero-width and formatting characters. Invisible in the UI, but they break tokenisation and
# are a favourite way to smuggle instructions past a naive injection scanner.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_NEWLINES = re.compile(r"\n{3,}")
# "author-\nity" → "authority". Requires a lowercase letter on both sides so that
# "COVID-\n19" and "state-\nof-the-art" survive.
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")
_BULLET = re.compile(r"^[\u2022\u2023\u25aa\u25cf\u00b7\u2043\u2219]\s*", re.MULTILINE)
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*/\s*\d{1,4})?\s*$", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2032": "'",
    "\u2033": '"',
    "\u00ab": '"',
    "\u00bb": '"',
}

DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"}

_TRANSLATION = str.maketrans({**LIGATURES, **QUOTES, **DASHES})


def clean_text(text: str, *, preserve_layout: bool = False) -> str:
    """Normalise one block of extracted text.

    ``preserve_layout`` keeps internal runs of spaces and line breaks intact, and is used for
    tables, code, and forms where alignment carries meaning. Collapsing whitespace inside a
    code block changes what the code means; inside a table it destroys the column structure
    that makes the rows readable.
    """
    if not text:
        return ""

    # NFKC folds full-width and other compatibility forms onto their ASCII equivalents, so
    # full-width text tokenises the same as normal text. Applied before the explicit maps
    # below so they see canonical input.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    if not preserve_layout:
        text = _BULLET.sub("- ", text)
        text = _SPACES.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _NEWLINES.sub("\n\n", text)

    return text.strip()


def detect_repeated_lines(
    pages: Sequence[str], *, min_pages: int = 3, threshold: float = 0.6
) -> set[str]:
    """Find running headers and footers by looking across pages.

    Cross-page frequency is the only reliable signal: a header is not distinguishable from a
    heading by looking at one page. A line that appears at the top or bottom of most pages is
    boilerplate; the same line appearing once is content.

    Left deliberately conservative — it only inspects the first and last two lines of each
    page and requires the line to repeat on a majority of them, because deleting a real
    heading is a worse outcome than keeping a footer.
    """
    if len(pages) < min_pages:
        return set()

    counter: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        if not lines:
            continue
        for line in {*lines[:2], *lines[-2:]}:
            if 3 <= len(line) <= 120:
                counter[line] += 1

    cutoff = max(min_pages - 1, int(len(pages) * threshold))
    return {line for line, count in counter.items() if count >= cutoff}


def strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    """Remove known running headers, footers, and bare page numbers.

    Applied per page after :func:`detect_repeated_lines` has seen the whole document.
    """
    if not text:
        return text
    kept = [
        line
        for line in text.split("\n")
        if line.strip() not in boilerplate and not _PAGE_NUMBER_LINE.match(line)
    ]
    return "\n".join(kept).strip()


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces. For titles and headings only."""
    return _SPACES.sub(" ", text.replace("\n", " ")).strip()


def is_meaningful(text: str, *, min_chars: int = 24, min_alpha_ratio: float = 0.35) -> bool:
    """Whether a block carries enough real text to be worth embedding.

    Filters the artefacts every extractor produces: a stray "3", a line of dots from a table
    of contents, a page of separator glyphs. Embedding these costs money and adds noise that
    competes with real content at retrieval time.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alpha = sum(1 for ch in stripped if ch.isalnum())
    if alpha / len(stripped) < min_alpha_ratio:
        return False
    # A "word" here is a run of 2+ alphanumerics; three of them is the floor for a sentence
    # fragment that could plausibly answer something.
    return len(re.findall(r"[^\W_]{2,}", stripped, flags=re.UNICODE)) >= 3


def is_noise(text: str, *, min_alpha_ratio: float = 0.35) -> bool:
    """Whether a block is an extraction artefact rather than content.

    Weaker than :func:`is_meaningful`, for the formats that mark their own block boundaries
    (an HTML ``<p>``, a DOCX paragraph). There a two-word paragraph is still content the
    user wrote, so length is not disqualifying; only the length-independent signals are —
    no letters at all, or mostly separator glyphs.
    """
    stripped = text.strip()
    if not stripped or not any(ch.isalpha() for ch in stripped):
        return True
    alpha = sum(1 for ch in stripped if ch.isalnum())
    return alpha / len(stripped) < min_alpha_ratio


def truncate_urls(text: str, *, max_length: int = 80) -> str:
    """Shorten long URLs, which otherwise consume a chunk's token budget with noise.

    Tracking parameters routinely make a link longer than the sentence containing it. The
    host and path are kept because those are what a reader recognises.
    """

    def _shorten(match: re.Match[str]) -> str:
        url = match.group(0)
        return url if len(url) <= max_length else url[: max_length - 1] + "…"

    return _URL.sub(_shorten, text)
