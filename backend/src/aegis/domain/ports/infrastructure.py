"""Ports for the remaining infrastructure capabilities.

Object storage, cache, rate limiting, job queue, scanners, and the LLM. Grouped in one
module because each is a handful of methods and splitting them into six files would add
imports without adding clarity.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

# ── Object storage ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PresignedUpload:
    url: str
    fields: dict[str, str]
    key: str
    expires_at: datetime
    max_bytes: int


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    key: str
    size_bytes: int
    etag: str
    content_type: str | None = None
    last_modified: datetime | None = None


@runtime_checkable
class ObjectStore(Protocol):
    """S3-compatible object storage.

    One implementation serves MinIO locally and S3 in cloud. Divergence between dev and
    production storage produces exactly the bugs that only appear in production —
    multipart behaviour, presigned signature versions, CORS on ``PUT`` — so both speak the
    same API.
    """

    async def presign_upload(
        self, key: str, *, content_type: str, max_bytes: int, ttl: timedelta
    ) -> PresignedUpload:
        """Grant one-time permission to place an object.

        ``max_bytes`` is enforced by the storage service through the policy, not by us:
        a size limit the client can ignore is not a limit.
        """
        ...

    async def presign_download(
        self, key: str, *, ttl: timedelta, filename: str | None = None
    ) -> str: ...

    async def head(self, key: str) -> ObjectMetadata | None: ...

    async def get(self, key: str) -> bytes: ...

    async def put(self, key: str, data: bytes, *, content_type: str) -> ObjectMetadata: ...

    async def delete(self, key: str) -> None: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


# ── Cache and rate limiting ─────────────────────────────────────────────────


@runtime_checkable
class Cache(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None: ...

    async def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Batch read. One round trip for a whole embedding batch instead of N."""
        ...

    async def set_many(
        self, items: dict[str, bytes], *, ttl_seconds: int | None = None
    ) -> None: ...

    async def delete(self, *keys: str) -> int: ...

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int: ...

    async def add(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        """Set only if absent. Returns whether it was set.

        The primitive behind idempotency keys, TOTP replay protection, and distributed
        locks — all of which need "first writer wins" semantics rather than last-write-wins.
        """
        ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    reset_after_seconds: int


@runtime_checkable
class RateLimiter(Protocol):
    async def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        """Consume one unit against a sliding window."""
        ...

    async def peek(self, key: str, *, limit: int, window_seconds: int) -> RateLimitResult:
        """Read the current state without consuming.

        Needed so a rejected request does not itself count towards the limit, which would
        let a client extend their own lockout indefinitely by retrying.
        """
        ...

    async def reset(self, key: str) -> None: ...


# ── Job queue ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EnqueuedJob:
    job_id: str
    queued_at: datetime


@runtime_checkable
class JobQueue(Protocol):
    async def enqueue(
        self,
        name: str,
        *args: Any,
        idempotency_key: str | None = None,
        delay: timedelta | None = None,
        **kwargs: Any,
    ) -> EnqueuedJob | None:
        """Enqueue a job. Returns ``None`` when ``idempotency_key`` was already queued.

        Deduplication belongs here rather than in each caller: ingestion is keyed on
        content checksum, so re-uploading identical bytes is a no-op by construction.
        """
        ...

    async def queue_depth(self, queue: str = "default") -> int: ...

    async def oldest_job_age_seconds(self, queue: str = "default") -> float | None: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


# ── Scanners ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScanResult:
    clean: bool
    threat: str | None = None
    scanner: str = ""
    skipped: bool = False


@runtime_checkable
class MalwareScanner(Protocol):
    async def scan(self, data: bytes, *, filename: str) -> ScanResult: ...

    async def health(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    category: str
    pattern: str
    excerpt: str
    confidence: float


@dataclass(frozen=True, slots=True)
class InjectionScanResult:
    findings: tuple[InjectionFinding, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def max_confidence(self) -> float:
        return max((f.confidence for f in self.findings), default=0.0)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.category for f in self.findings))


@runtime_checkable
class InjectionScanner(Protocol):
    """Detects instruction-like text in user input or document content.

    Used at two boundaries with different consequences. On user input a high-confidence
    match blocks the request. On document content it only *flags* — a security policy that
    legitimately discusses "ignore previous instructions" must stay findable, and blocking
    on ingest would hand anyone with upload rights a censorship tool.
    """

    def scan(self, text: str) -> InjectionScanResult: ...


@runtime_checkable
class SecretScanner(Protocol):
    """Finds credentials in text. Used on document content and on model output."""

    def scan(self, text: str) -> tuple[str, ...]:
        """Returns the categories of secret found, never the secret itself."""
        ...

    def redact(self, text: str) -> str: ...


# ── LLM ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class CompletionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""
    ttft_ms: int | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class Completion:
    text: str
    usage: CompletionUsage = field(default_factory=CompletionUsage)


@runtime_checkable
class LLMProvider(Protocol):
    """Text generation.

    Streaming is a separate method rather than a flag so that the return types stay honest:
    one returns a value, the other an async iterator, and a boolean parameter that changes
    the return type is a type error waiting to happen.
    """

    name: str
    model: str

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Completion: ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]: ...

    def count_tokens(self, text: str, *, model: str | None = None) -> int:
        """Count with the provider's own tokenizer.

        A four-characters-per-token heuristic is off by around 30 % on code and tables —
        precisely the content that causes context overflow — so the real tokenizer is used
        for budgeting.
        """
        ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...
