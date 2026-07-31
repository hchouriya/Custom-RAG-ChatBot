"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from aegis.api.middleware import ErrorMiddleware, LoggingMiddleware, RequestIdMiddleware
from aegis.api.v1 import health
from aegis.api.v1.router import api_router
from aegis.core.config import get_settings
from aegis.core.container import Container
from aegis.core.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: Container = app.state.container
    await container.startup()
    logger.info("app.started", env=container.settings.app_env)
    try:
        yield
    finally:
        await container.shutdown()
        logger.info("app.stopped")


def create_app() -> FastAPI:
    """Build the ASGI application.

    Middleware order (outermost first): request-id → logging → errors → routes.
    Auth is a dependency, not middleware, so public routes stay public without a
    special-case path list.
    """
    settings = get_settings()
    configure_logging(settings)
    container = Container.build(settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
    )
    app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Response-Time-Ms",
            "X-Trace-Id",
            "Retry-After",
        ],
    )
    # Starlette wraps in reverse order of addition: last added is outermost.
    app.add_middleware(ErrorMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIdMiddleware)

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(health.router)

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/version", include_in_schema=False)
    async def version() -> dict[str, str]:
        return {"name": settings.app_name, "version": "0.2.0"}

    @app.get("/", include_in_schema=False)
    async def root() -> PlainTextResponse:
        return PlainTextResponse(f"{settings.app_name} API")

    return app


app = create_app()
