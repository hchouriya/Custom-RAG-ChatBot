"""Filesystem object storage for development and tests.

Its reason to exist is ``docker compose up`` without MinIO, and unit tests that exercise the
real upload path without a container. It is refused in production by ``Settings``: local disk
cannot be shared between two API replicas, and a presigned URL issued by one of them would
404 on the other.

Presigning is emulated with an HMAC-signed URL pointing back at our own API. The signature
covers the key, the operation, the deadline, and the size limit, so a development URL cannot
be edited into a grant for a different object — which keeps the local path honest enough that
the API contract is identical to the S3 one.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis.core.errors import StorageError
from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import ObjectMetadata, PresignedUpload
from aegis.infrastructure.storage.keys import is_safe_key

if TYPE_CHECKING:
    from aegis.core.config import Settings

logger = get_logger(__name__)

UPLOAD_PATH = "/api/v1/documents/local-upload"
DOWNLOAD_PATH = "/api/v1/documents/local-download"


class LocalObjectStore:
    """Stores objects as files under ``root``, with metadata in a sidecar.

    Blocking file I/O is pushed to a thread. Writing a 200 MB file inside the event loop
    would stall every other request on the process for the duration, and this adapter exists
    to make development resemble production rather than to be quietly worse than it.
    """

    def __init__(
        self,
        root: Path,
        *,
        secret: str,
        base_url: str = "http://localhost:8000",
        presign_ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        self._root = root.resolve()
        self._secret = secret.encode("utf-8")
        self._base_url = base_url.rstrip("/")
        self._presign_ttl = presign_ttl
        self._root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> LocalObjectStore:
        return cls(
            settings.local_storage_path,
            secret=settings.secret_key.get_secret_value(),
            base_url=settings.public_base_url,
            presign_ttl=timedelta(seconds=settings.s3_presign_ttl_seconds),
        )

    # ── Presigning ──────────────────────────────────────────────────────────

    async def presign_upload(
        self, key: str, *, content_type: str, max_bytes: int, ttl: timedelta | None = None
    ) -> PresignedUpload:
        expiry = datetime.now(UTC) + (ttl or self._presign_ttl)
        token = self.sign(
            key, operation="put", expires_at=expiry, content_type=content_type, max_bytes=max_bytes
        )
        return PresignedUpload(
            url=f"{self._base_url}{UPLOAD_PATH}",
            fields={"token": token, "Content-Type": content_type},
            key=key,
            expires_at=expiry,
            max_bytes=max_bytes,
        )

    async def presign_download(
        self, key: str, *, ttl: timedelta | None = None, filename: str | None = None
    ) -> str:
        expiry = datetime.now(UTC) + (ttl or self._presign_ttl)
        token = self.sign(key, operation="get", expires_at=expiry, filename=filename)
        return f"{self._base_url}{DOWNLOAD_PATH}?token={token}"

    # ── Signed grants ───────────────────────────────────────────────────────

    def sign(
        self,
        key: str,
        *,
        operation: str,
        expires_at: datetime,
        content_type: str | None = None,
        max_bytes: int | None = None,
        filename: str | None = None,
    ) -> str:
        self._require_safe(key)
        claims: dict[str, Any] = {
            "k": key,
            "op": operation,
            "exp": int(expires_at.timestamp()),
        }
        if content_type is not None:
            claims["ct"] = content_type
        if max_bytes is not None:
            claims["max"] = max_bytes
        if filename is not None:
            claims["fn"] = filename
        payload = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
        return f"{payload}.{_b64(self._mac(payload))}"

    def verify(self, token: str, *, operation: str) -> dict[str, Any]:
        """Validate a grant and return its claims.

        Raises rather than returning ``None``: every caller of this is about to touch the
        filesystem with the key it contains, and a falsy return is too easy to skip.
        """
        payload, _, signature = token.partition(".")
        if not payload or not signature:
            raise StorageError("Malformed storage grant.")
        if not hmac.compare_digest(_b64(self._mac(payload)), signature):
            raise StorageError("Storage grant signature is invalid.")
        try:
            claims = json.loads(_unb64(payload))
        except (ValueError, UnicodeDecodeError) as exc:
            raise StorageError("Malformed storage grant.") from exc

        if claims.get("op") != operation:
            raise StorageError("Storage grant is for a different operation.")
        if int(claims.get("exp", 0)) < int(datetime.now(UTC).timestamp()):
            raise StorageError("Storage grant has expired.")
        if not is_safe_key(str(claims.get("k", ""))):
            raise StorageError("Storage grant names an unsafe key.")
        return dict(claims)

    def _mac(self, payload: str) -> bytes:
        return hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest()

    # ── Objects ─────────────────────────────────────────────────────────────

    async def head(self, key: str) -> ObjectMetadata | None:
        path = self._path(key)
        return await asyncio.to_thread(self._head_sync, key, path)

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise StorageError(f"Object not found: {key[:64]!r}") from exc
        except OSError as exc:
            raise self._wrap(exc, "get", key) from exc

    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectMetadata:
        path = self._path(key)
        try:
            await asyncio.to_thread(self._put_sync, path, data, content_type)
        except OSError as exc:
            raise self._wrap(exc, "put", key) from exc
        return ObjectMetadata(
            key=key,
            size_bytes=len(data),
            etag=hashlib.md5(data, usedforsecurity=False).hexdigest(),
            content_type=content_type,
            last_modified=datetime.now(UTC),
        )

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(self._delete_sync, path)

    async def delete_prefix(self, prefix: str) -> int:
        if not is_safe_key(prefix.rstrip("/")):
            raise StorageError(f"Rejected an unsafe prefix: {prefix[:64]!r}")
        target = self._root / prefix.rstrip("/")
        return await asyncio.to_thread(self._delete_tree_sync, target)

    async def health(self) -> bool:
        return await asyncio.to_thread(lambda: self._root.is_dir())

    async def close(self) -> None:
        return None

    # ── Internals ───────────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        self._require_safe(key)
        candidate = (self._root / key).resolve()
        # Resolution happens before the containment check so that a symlink planted inside
        # the root cannot point outwards.
        if not candidate.is_relative_to(self._root):
            raise StorageError(f"Object key escapes the storage root: {key[:64]!r}")
        return candidate

    def _require_safe(self, key: str) -> None:
        if not is_safe_key(key):
            raise StorageError(f"Rejected an unsafe object key: {key[:64]!r}")

    def _head_sync(self, key: str, path: Path) -> ObjectMetadata | None:
        if not path.is_file():
            return None
        stat = path.stat()
        meta = _sidecar(path)
        content_type: str | None = None
        if meta.is_file():
            try:
                content_type = json.loads(meta.read_text("utf-8")).get("content_type")
            except (OSError, ValueError):
                content_type = None
        return ObjectMetadata(
            key=key,
            size_bytes=stat.st_size,
            etag=hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
            content_type=content_type,
            last_modified=datetime.fromtimestamp(stat.st_mtime, UTC),
        )

    def _put_sync(self, path: Path, data: bytes, content_type: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a neighbour and rename, so a reader never observes a half-written object
        # and a crash leaves no truncated file that would pass a size check.
        temporary = path.with_name(f".{path.name}.partial")
        temporary.write_bytes(data)
        temporary.replace(path)
        _sidecar(path).write_text(json.dumps({"content_type": content_type}), encoding="utf-8")

    def _delete_sync(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        _sidecar(path).unlink(missing_ok=True)

    def _delete_tree_sync(self, target: Path) -> int:
        if not target.is_dir():
            return 0
        count = sum(1 for p in target.rglob("*") if p.is_file() and not p.name.endswith(".meta"))
        shutil.rmtree(target, ignore_errors=True)
        return count

    def _wrap(self, exc: OSError, operation: str, key: str) -> StorageError:
        logger.error("storage.operation_failed", operation=operation, key=key, error=str(exc))
        return StorageError(f"Storage operation '{operation}' failed.")


def _sidecar(path: Path) -> Path:
    return path.with_name(f".{path.name}.meta")


def _b64(raw: bytes | str) -> str:
    data = raw.encode() if isinstance(raw, str) else raw
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
