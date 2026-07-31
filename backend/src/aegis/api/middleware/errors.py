"""RFC 9457 problem+json error middleware.

Services raise :class:`~aegis.core.errors.AegisError`; this is the only place that knows
about HTTP status codes and response shape. Unhandled exceptions become a generic 500 —
``detail`` must never carry a stack trace or provider payload to the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from aegis.core.errors import AegisError, MFARequiredError
from aegis.core.logging import get_logger, request_id_var

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

logger = get_logger(__name__)

PROBLEM_TYPE_BASE = "https://aegis.local/errors"


class ErrorMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except AegisError as exc:
            return _problem_response(request, exc)
        except Exception:
            logger.exception("unhandled_exception", path=request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "type": f"{PROBLEM_TYPE_BASE}/internal-error",
                    "title": "Internal server error",
                    "status": 500,
                    "detail": "An unexpected error occurred.",
                    "instance": request.url.path,
                    "request_id": request_id_var.get(),
                    "code": "INTERNAL_ERROR",
                    "errors": [],
                },
                media_type="application/problem+json",
            )


def _problem_response(request: Request, exc: AegisError) -> JSONResponse:
    slug = exc.code.lower().replace("_", "-")
    body: dict[str, Any] = {
        "type": f"{PROBLEM_TYPE_BASE}/{slug}",
        "title": exc.title,
        "status": exc.status_code,
        "detail": exc.detail,
        "instance": request.url.path,
        "request_id": request_id_var.get(),
        "code": exc.code,
        "errors": exc.errors,
    }
    if isinstance(exc, MFARequiredError):
        body["challenge_token"] = exc.challenge_token
        body["enrolment_required"] = exc.enrolment_required
    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=dict(exc.headers),
        media_type="application/problem+json",
    )
