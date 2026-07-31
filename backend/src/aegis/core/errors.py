"""Domain exception hierarchy, mapped to RFC 9457 problem documents at the edge.

Services raise these; only ``api.middleware.errors`` knows about HTTP. That split is
what lets the same services run inside a worker or a CLI command, where an
``HTTPException`` would be meaningless.

Two rules that are easy to get wrong and expensive to get wrong:

* ``detail`` is shown to the caller, so it must never contain SQL, provider payloads,
  stack traces, or content the caller is not authorized to see.
* ``NotFoundError`` is raised instead of ``AuthorizationError`` wherever the existence
  of the resource is itself sensitive (see ``docs/architecture/06`` §1).
"""

from __future__ import annotations

from typing import Any


class AegisError(Exception):
    """Base class for every deliberate failure in the platform."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    title: str = "Internal server error"

    def __init__(
        self,
        detail: str | None = None,
        *,
        code: str | None = None,
        errors: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.detail = detail or self.title
        if code:
            self.code = code
        self.errors = errors or []
        self.headers = headers or {}
        super().__init__(self.detail)

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


# ── 400 family ──────────────────────────────────────────────────────────────


class ValidationError(AegisError):
    status_code = 422
    code = "VALIDATION_ERROR"
    title = "Request validation failed"


class BadRequestError(AegisError):
    status_code = 400
    code = "BAD_REQUEST"
    title = "Malformed request"


class AuthenticationError(AegisError):
    status_code = 401
    code = "AUTHENTICATION_FAILED"
    title = "Authentication failed"

    def __init__(self, detail: str | None = None, **kw: Any) -> None:
        super().__init__(detail, **kw)
        self.headers.setdefault("WWW-Authenticate", "Bearer")


class MFARequiredError(AegisError):
    """Credentials were correct but a second factor is outstanding.

    Distinct from ``AuthenticationError`` because the client must render a TOTP step
    rather than an error, and it carries a short-lived challenge token instead of a
    session.
    """

    status_code = 401
    code = "MFA_REQUIRED"
    title = "Second factor required"

    def __init__(self, challenge_token: str, *, enrolment_required: bool = False) -> None:
        super().__init__("A time-based one-time password is required to continue.")
        self.challenge_token = challenge_token
        self.enrolment_required = enrolment_required


class AuthorizationError(AegisError):
    status_code = 403
    code = "FORBIDDEN"
    title = "Not permitted"


class NotFoundError(AegisError):
    status_code = 404
    code = "NOT_FOUND"
    title = "Resource not found"

    def __init__(self, resource: str = "Resource", identifier: Any = None, **kw: Any) -> None:
        detail = (
            f"{resource} not found" if identifier is None else f"{resource} {identifier} not found"
        )
        super().__init__(detail, **kw)


class ConflictError(AegisError):
    status_code = 409
    code = "CONFLICT"
    title = "Conflicting state"


class PreconditionFailedError(AegisError):
    """``If-Match`` did not match: someone else changed the resource first."""

    status_code = 412
    code = "PRECONDITION_FAILED"
    title = "Resource was modified by someone else"


class PayloadTooLargeError(AegisError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"
    title = "Payload too large"


class UnsupportedMediaTypeError(AegisError):
    status_code = 415
    code = "UNSUPPORTED_MEDIA_TYPE"
    title = "Unsupported file type"


class RateLimitError(AegisError):
    status_code = 429
    code = "RATE_LIMITED"
    title = "Too many requests"

    def __init__(self, retry_after_seconds: int, detail: str | None = None) -> None:
        super().__init__(
            detail or f"Rate limit exceeded. Retry in {retry_after_seconds}s.",
            headers={"Retry-After": str(retry_after_seconds)},
        )
        self.retry_after_seconds = retry_after_seconds


# ── Guardrails ──────────────────────────────────────────────────────────────


class GuardrailViolationError(AegisError):
    """Input or output was blocked by a guardrail.

    ``category`` feeds the ``aegis_guardrail_blocks_total`` metric, so it must stay a
    small, stable set of values rather than free text.
    """

    status_code = 400
    code = "GUARDRAIL_BLOCKED"
    title = "Request blocked by a safety policy"

    def __init__(self, category: str, detail: str | None = None, *, layer: str = "input") -> None:
        super().__init__(detail or "This request was blocked by a safety policy.")
        self.category = category
        self.layer = layer


# ── Document lifecycle ──────────────────────────────────────────────────────


class DocumentNotIndexedError(ConflictError):
    code = "DOCUMENT_NOT_INDEXED"
    title = "Document is not searchable yet"


class IngestionError(AegisError):
    """A pipeline stage failed. ``stage`` and ``retryable`` drive worker retry policy."""

    status_code = 500
    code = "INGESTION_FAILED"
    title = "Document processing failed"

    def __init__(self, stage: str, detail: str, *, retryable: bool = True) -> None:
        super().__init__(detail)
        self.stage = stage
        self.retryable = retryable


class MalwareDetectedError(AegisError):
    status_code = 422
    code = "MALWARE_DETECTED"
    title = "File rejected by malware scan"


class ParserError(IngestionError):
    code = "PARSE_FAILED"
    title = "Document could not be parsed"

    def __init__(self, detail: str, *, retryable: bool = False) -> None:
        super().__init__("parse", detail, retryable=retryable)


# ── Upstream dependencies ───────────────────────────────────────────────────


class ProviderError(AegisError):
    """An external model provider failed.

    ``retryable`` distinguishes 429/5xx (retry with jitter) from 4xx (a bug or a bad
    key — retrying only burns quota and delays the real error).
    """

    status_code = 502
    code = "PROVIDER_ERROR"
    title = "Upstream model provider failed"

    def __init__(
        self, provider: str, detail: str, *, retryable: bool = True, status: int | None = None
    ) -> None:
        super().__init__(f"{provider}: {detail}")
        self.provider = provider
        self.retryable = retryable
        self.upstream_status = status


class VectorStoreError(AegisError):
    status_code = 503
    code = "VECTOR_STORE_UNAVAILABLE"
    title = "Search index is unavailable"


class StorageError(AegisError):
    """Object storage refused or failed an operation.

    Deliberately opaque to the caller: bucket names, endpoints, and provider request ids
    are operational detail that belongs in the log line, not in the response body.
    """

    status_code = 503
    code = "STORAGE_UNAVAILABLE"
    title = "Document storage is unavailable"


class ServiceUnavailableError(AegisError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"
    title = "Service temporarily unavailable"


class ConfigurationError(AegisError):
    """A configuration mistake discovered too late to fail at boot.

    Always a bug: a missing dimension, a provider without a key, an unregistered
    adapter. Never surfaced to end users with detail.
    """

    status_code = 500
    code = "CONFIGURATION_ERROR"
    title = "Server misconfiguration"
