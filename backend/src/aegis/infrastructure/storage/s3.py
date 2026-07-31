"""S3-compatible object storage.

One adapter serves MinIO in development and S3 (or R2, or GCS in interoperability mode) in
production. That is a deliberate constraint rather than a convenience: presigned signature
versions, path-versus-virtual-host addressing, and CORS on ``PUT`` are exactly the things
that behave differently between a local stub and a real bucket, and each of them fails for
the first time in production if development never exercised them.

Uploads are presigned POSTs, not proxied through the API. Two reasons, both structural:

* A 200 MB file streamed through the API occupies a worker for the whole transfer and
  bounds concurrent uploads by worker count.
* The size limit is enforced by the storage service through a POST policy condition. A
  limit the client can decline to honour is not a limit — and a limit enforced only after
  the bytes have been received has already cost the bandwidth it was meant to save.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import aioboto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from aegis.core.errors import StorageError
from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import ObjectMetadata, PresignedUpload
from aegis.infrastructure.storage.keys import is_safe_key

if TYPE_CHECKING:
    from aegis.core.config import Settings

logger = get_logger(__name__)

_MISSING = frozenset({"404", "NoSuchKey", "NotFound"})


class S3ObjectStore:
    """Async S3 client with one shared connection pool.

    The client is created on first use and reused. Creating one per call costs a TLS
    handshake and a credential resolution per object, which is measurable on a document
    with fifty derived artefacts and pointless on every other request.
    """

    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: str | None = None,
        force_path_style: bool = True,
        presign_ttl: timedelta = timedelta(minutes=15),
        max_pool_connections: int = 32,
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint_url
        self._presign_ttl = presign_ttl
        self._session = aioboto3.Session(
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
            region_name=region,
        )
        self._config = Config(
            # s3v4 is required for presigned POST policies and by every non-AWS
            # implementation worth supporting.
            signature_version="s3v4",
            s3={"addressing_style": "path" if force_path_style else "auto"},
            # Adaptive retries back off on throttling instead of amplifying it, which
            # matters when a reindex is pushing thousands of objects.
            retries={"max_attempts": 4, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
            connect_timeout=5,
            read_timeout=60,
        )
        self._stack = AsyncExitStack()
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> S3ObjectStore:
        return cls(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            access_key_id=settings.s3_access_key_id.get_secret_value(),
            secret_access_key=settings.s3_secret_access_key.get_secret_value(),
            endpoint_url=settings.s3_endpoint_url,
            force_path_style=settings.s3_force_path_style,
            presign_ttl=timedelta(seconds=settings.s3_presign_ttl_seconds),
        )

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                self._client = await self._stack.enter_async_context(
                    self._session.client("s3", endpoint_url=self._endpoint, config=self._config)
                )
        return self._client

    async def close(self) -> None:
        await self._stack.aclose()
        self._client = None

    # ── Presigning ──────────────────────────────────────────────────────────

    async def presign_upload(
        self, key: str, *, content_type: str, max_bytes: int, ttl: timedelta | None = None
    ) -> PresignedUpload:
        """Grant one-time permission to place exactly one object.

        The policy pins the key, the content type, and the size range. Pinning the content
        type is what stops a grant for ``report.pdf`` being redeemed with an HTML file that
        the browser would then render from our own origin on download.
        """
        self._require_safe(key)
        expiry = ttl or self._presign_ttl
        seconds = int(expiry.total_seconds())
        client = await self._get_client()
        try:
            post = await client.generate_presigned_post(
                Bucket=self._bucket,
                Key=key,
                Fields={"Content-Type": content_type},
                Conditions=[
                    {"Content-Type": content_type},
                    ["content-length-range", 1, max_bytes],
                ],
                ExpiresIn=seconds,
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap(exc, "presign_upload", key) from exc

        return PresignedUpload(
            url=str(post["url"]),
            fields={str(k): str(v) for k, v in post["fields"].items()},
            key=key,
            expires_at=datetime.now(UTC) + expiry,
            max_bytes=max_bytes,
        )

    async def presign_download(
        self, key: str, *, ttl: timedelta | None = None, filename: str | None = None
    ) -> str:
        """Short-lived download URL.

        ``attachment`` is forced in the content disposition. Serving a stored HTML or SVG
        document inline would execute its script under the storage origin, which for a
        single-origin deployment is the application's own origin.
        """
        self._require_safe(key)
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if filename:
            params["ResponseContentDisposition"] = _content_disposition(filename)
        client = await self._get_client()
        try:
            url = await client.generate_presigned_url(
                "get_object",
                Params=params,
                ExpiresIn=int((ttl or self._presign_ttl).total_seconds()),
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap(exc, "presign_download", key) from exc
        return str(url)

    # ── Objects ─────────────────────────────────────────────────────────────

    async def head(self, key: str) -> ObjectMetadata | None:
        """Metadata, or ``None`` when absent. Absence is not an error here.

        Callers use this to verify that a presigned upload actually happened, where "the
        client never completed it" is an expected outcome rather than a failure.
        """
        self._require_safe(key)
        client = await self._get_client()
        try:
            response = await client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _is_missing(exc):
                return None
            raise self._wrap(exc, "head", key) from exc
        except BotoCoreError as exc:
            raise self._wrap(exc, "head", key) from exc
        return _metadata(key, response)

    async def get(self, key: str) -> bytes:
        self._require_safe(key)
        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self._bucket, Key=key)
            async with response["Body"] as stream:
                return bytes(await stream.read())
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap(exc, "get", key) from exc

    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectMetadata:
        self._require_safe(key)
        client = await self._get_client()
        try:
            response = await client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                # Encryption at rest is set per object as well as per bucket: a bucket
                # policy that is silently relaxed later must not retroactively expose
                # objects written while it was in force.
                ServerSideEncryption="AES256",
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap(exc, "put", key) from exc
        return ObjectMetadata(
            key=key,
            size_bytes=len(data),
            etag=str(response.get("ETag", "")).strip('"'),
            content_type=content_type,
            last_modified=datetime.now(UTC),
        )

    async def delete(self, key: str) -> None:
        """Idempotent: deleting an absent object succeeds, as S3 itself does."""
        self._require_safe(key)
        client = await self._get_client()
        try:
            await client.delete_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _is_missing(exc):
                return
            raise self._wrap(exc, "delete", key) from exc
        except BotoCoreError as exc:
            raise self._wrap(exc, "delete", key) from exc

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every object under ``prefix``, in batches. Returns the count removed.

        Used by hard deletion and by purging superseded versions, where the set of derived
        artefacts under a version is not enumerated anywhere else.
        """
        client = await self._get_client()
        removed = 0
        try:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if not keys:
                    continue
                # 1000 is the API's per-request maximum.
                for batch in _batched(keys, 1000):
                    await client.delete_objects(
                        Bucket=self._bucket, Delete={"Objects": batch, "Quiet": True}
                    )
                    removed += len(batch)
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap(exc, "delete_prefix", prefix) from exc
        return removed

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            await client.head_bucket(Bucket=self._bucket)
        except (ClientError, BotoCoreError, OSError) as exc:
            logger.warning("storage.unhealthy", error=str(exc), bucket=self._bucket)
            return False
        return True

    # ── Internals ───────────────────────────────────────────────────────────

    def _require_safe(self, key: str) -> None:
        if not is_safe_key(key):
            raise StorageError(f"Rejected an unsafe object key: {key[:64]!r}")

    def _wrap(self, exc: Exception, operation: str, key: str) -> StorageError:
        code = ""
        if isinstance(exc, ClientError):
            code = str(exc.response.get("Error", {}).get("Code", ""))
        logger.error(
            "storage.operation_failed",
            operation=operation,
            key=key,
            bucket=self._bucket,
            aws_code=code,
            error=str(exc),
        )
        return StorageError(f"Storage operation '{operation}' failed.")


def _is_missing(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    return str(error.get("Code", "")) in _MISSING


def _metadata(key: str, response: dict[str, Any]) -> ObjectMetadata:
    return ObjectMetadata(
        key=key,
        size_bytes=int(response.get("ContentLength", 0)),
        etag=str(response.get("ETag", "")).strip('"'),
        content_type=response.get("ContentType"),
        last_modified=response.get("LastModified"),
    )


def _content_disposition(filename: str) -> str:
    """RFC 6266 disposition with an ASCII fallback.

    Non-ASCII filenames need the ``filename*`` form; older clients need the plain one, and
    quoting the plain one is what keeps a filename containing a space or a semicolon from
    truncating the header.
    """
    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def _batched(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
