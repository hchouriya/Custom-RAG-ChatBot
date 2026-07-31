"""Structured logging with request correlation and secret redaction.

Every log line carries the request id, and in an authenticated request also the user,
role, and mode. The binding lives in a ``contextvar``, so it survives ``await`` and is
inherited by tasks — including worker jobs, which receive the ids in their payload and
re-bind them.

Question text is deliberately *not* logged. It can contain personal data, and it already
has a home in ``query_traces``, which is access-controlled and retention-managed. Logs
get a length and a hash so two occurrences of the same question can be correlated
without storing it twice.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from contextvars import ContextVar
from typing import Any

import structlog

from aegis.core.config import Settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
role_var: ContextVar[str | None] = ContextVar("role", default=None)
mode_var: ContextVar[str | None] = ContextVar("mode", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_REDACT_KEYS = frozenset(
    {
        "password",
        "new_password",
        "old_password",
        "secret",
        "secret_key",
        "token",
        "access_token",
        "refresh_token",
        "challenge_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "set-cookie",
        "totp",
        "totp_code",
        "mfa_secret",
        "session_secret",
        "private_key",
        "credentials",
    }
)

_REDACT_KEY_PARTS = ("password", "secret", "token", "api_key", "apikey", "credential")

# Value-shaped detection, for secrets that arrive inside a free-text field where the
# key name gives no warning (a pasted key in an error message, for instance).
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI-style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"postgres(?:ql)?://[^:]+:[^@\s]+@"),  # connection string with password
)

REDACTED = "[redacted]"


def hash_text(text: str) -> str:
    """Short stable digest, for correlating repeated values without logging them."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        for pattern in _SECRET_VALUE_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value
    if isinstance(value, dict):
        return _redact_mapping(value)
    if isinstance(value, list | tuple):
        return type(value)(_redact_value(v) for v in value)
    return value


def _redact_mapping(mapping: dict[Any, Any]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, value in mapping.items():
        lowered = str(key).lower()
        if lowered in _REDACT_KEYS or any(part in lowered for part in _REDACT_KEY_PARTS):
            out[key] = REDACTED
        else:
            out[key] = _redact_value(value)
    return out


def redact_processor(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Strip secrets by key name and by value shape.

    Runs on every event, including ones emitted by third-party libraries through the
    stdlib bridge, because the library that logs your connection string is never the
    one you expected.
    """
    return _redact_mapping(dict(event_dict))


def context_processor(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Attach the ambient request context to every event."""
    for key, var in (
        ("request_id", request_id_var),
        ("user_id", user_id_var),
        ("role", role_var),
        ("mode", mode_var),
        ("trace_id", trace_id_var),
    ):
        value = var.get()
        if value is not None:
            event_dict.setdefault(key, value)
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Install structlog and route stdlib logging through it.

    Called once at startup. Third-party loggers are routed through the same pipeline so
    that an SQLAlchemy warning is redacted and correlated exactly like our own events.
    """
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        context_processor,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redact_processor,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *shared,
            # Hand off to the stdlib handler for rendering — avoids double-printing when
            # LoggerFactory is paired with a ProcessorFormatter on the root logger.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        # LoggerFactory (not PrintLogger): add_logger_name needs a stdlib logger with `.name`.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    # Noise control. Uvicorn's access log duplicates our timing middleware, and
    # SQLAlchemy's INFO stream is the full SQL text of every statement.
    for name, level in (
        ("uvicorn.access", logging.WARNING),
        ("uvicorn.error", logging.INFO),
        ("sqlalchemy.engine", logging.WARNING),
        ("httpx", logging.WARNING),
        ("httpcore", logging.WARNING),
        ("botocore", logging.WARNING),
        ("aiobotocore", logging.WARNING),
        ("asyncio", logging.WARNING),
    ):
        logging.getLogger(name).setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Module-level logger. Convention: ``logger = get_logger(__name__)``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_request_context(
    *,
    request_id: str,
    user_id: str | None = None,
    role: str | None = None,
    mode: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Bind correlation ids for the current task and everything it awaits."""
    request_id_var.set(request_id)
    if user_id is not None:
        user_id_var.set(user_id)
    if role is not None:
        role_var.set(role)
    if mode is not None:
        mode_var.set(mode)
    if trace_id is not None:
        trace_id_var.set(trace_id)


def clear_request_context() -> None:
    for var in (request_id_var, user_id_var, role_var, mode_var, trace_id_var):
        var.set(None)
