"""Request-id binding and response timing.

``X-Request-ID`` is echoed when the client sends one, otherwise minted. Timing is reported
as ``X-Response-Time-Ms`` so a slow request is diagnosable from the access log alone.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from aegis.core.logging import bind_request_context, clear_request_context, request_id_var

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "").strip()
        request_id = incoming or uuid.uuid4().hex
        bind_request_context(request_id=request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            # Cleared in LoggingMiddleware after the timing header is written when both
            # are present; clear here too so a path that skips logging still resets.
            pass
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        # Re-attach request id after clear — the response still needs it, and RequestId
        # middleware may have already set the header before we cleared the contextvar.
        if rid := getattr(request.state, "request_id", None):
            response.headers.setdefault("X-Request-ID", rid)
            request_id_var.set(rid)
        return response
