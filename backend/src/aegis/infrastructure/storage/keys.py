"""Object key construction.

Keys are derived, never taken from the client. A filename arrives from a browser and may
contain ``../``, a drive letter, a NUL byte, a right-to-left override that makes
``exe.txt`` render as ``txt.exe``, or four kilobytes of Unicode — none of which belongs in
a storage path. The original filename is kept in PostgreSQL for display; the key is built
here from identifiers we minted ourselves.

The layout is versioned rather than mutable::

    documents/<document_id>/v<version_no>/<checksum-prefix>/<safe-filename>
    uploads/<yyyy>/<mm>/<upload_id>/<safe-filename>

Including the version number and a checksum prefix means a re-upload never overwrites the
bytes an existing citation points at, which is what makes "show me the source as it was
when the answer was given" answerable.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

MAX_FILENAME_LEN = 96
MAX_KEY_LEN = 900
"""S3 allows 1024 bytes. The margin absorbs the prefixes above without a length check
at every call site."""

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_DOTS = re.compile(r"\.{2,}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]")

DEFAULT_NAME = "file"


def sanitize_filename(name: str) -> str:
    """Reduce a client-supplied filename to a safe, recognisable basename.

    Recognisable matters: the key appears in operator tooling and in storage browsers, and
    ``documents/<uuid>/v3/9f2c/handbook-2026.pdf`` is diagnosable where a bare hash is not.
    """
    name = _CONTROL.sub("", unicodedata.normalize("NFKC", name)).strip()
    # Windows and POSIX separators both, since the browser sends whatever the OS gave it.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = _DOTS.sub(".", name).strip(". ")

    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    stem = _UNSAFE.sub("-", stem).strip("-") or DEFAULT_NAME
    suffix = _UNSAFE.sub("", suffix)[:16].lower()

    keep = MAX_FILENAME_LEN - (len(suffix) + 1 if suffix else 0)
    stem = stem[:keep]
    return f"{stem}.{suffix}" if suffix else stem


def document_key(
    *, document_id: UUID, version_no: int, checksum: str, filename: str, prefix: str = "documents"
) -> str:
    """Immutable key for the stored bytes of one document version."""
    if version_no < 1:
        raise ValueError("version_no starts at 1")
    key = (
        f"{prefix}/{document_id}/v{version_no}/{_checksum_prefix(checksum)}/"
        f"{sanitize_filename(filename)}"
    )
    return key[:MAX_KEY_LEN]


def upload_key(*, upload_id: UUID, filename: str, now: datetime | None = None) -> str:
    """Key for bytes that have been offered but not yet accepted.

    Staged under a date-partitioned prefix so that a lifecycle rule can expire abandoned
    uploads by prefix, and so a day's worth of them can be listed without scanning the
    whole bucket.
    """
    moment = now or datetime.now(UTC)
    return f"uploads/{moment:%Y/%m}/{upload_id}/{sanitize_filename(filename)}"[:MAX_KEY_LEN]


def derived_key(key: str, *, kind: str, suffix: str) -> str:
    """Key for something generated from an object: an OCR sidecar, a thumbnail, a preview.

    Derived artefacts live beside the source under a ``.derived`` sibling so that deleting
    a document version's prefix removes them too.
    """
    base, _, _ = key.rpartition("/")
    return f"{base}/.derived/{kind}{suffix}"


def is_safe_key(key: str) -> bool:
    """Whether a key is one we could have produced.

    Used on any path that turns a caller-supplied key back into storage access — a
    presigned download, a local-disk read. Traversal is the obvious risk; absolute paths
    and empty segments are the ones that get forgotten.
    """
    if not key or len(key) > MAX_KEY_LEN or key.startswith("/"):
        return False
    if _CONTROL.search(key) or "//" in key:
        return False
    segments = key.split("/")
    return all(s not in {"", ".", ".."} for s in segments)


def version_prefix(document_id: UUID, version_no: int, *, prefix: str = "documents") -> str:
    """Prefix covering every object belonging to one version, for bulk deletion."""
    return f"{prefix}/{document_id}/v{version_no}/"


def document_prefix(document_id: UUID, *, prefix: str = "documents") -> str:
    return f"{prefix}/{document_id}/"


def _checksum_prefix(checksum: str) -> str:
    digest = _UNSAFE.sub("", checksum).lower()
    if len(digest) < 8:
        raise ValueError("checksum is too short to identify content")
    return digest[:16]
