"""Liveness and readiness probes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response

from aegis.api.deps import ContainerDep
from aegis.api.schemas import HealthLiveResponse, HealthReadyResponse
from aegis.infrastructure.database.engine import check_database

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthLiveResponse)
async def live() -> HealthLiveResponse:
    """Process is up. Touches no dependencies — safe for Kubernetes liveness."""
    return HealthLiveResponse(status="ok")


@router.get("/health/ready", response_model=HealthReadyResponse)
async def ready(container: ContainerDep, response: Response) -> HealthReadyResponse:
    """PostgreSQL, Redis, and the vector store within a 500 ms budget each."""

    async def _db() -> bool:
        return await check_database(container.engine)

    async def _redis() -> bool:
        try:
            return bool(await asyncio.wait_for(container.redis.ping(), timeout=0.5))
        except Exception:
            return False

    async def _vectors() -> bool:
        try:
            return bool(await asyncio.wait_for(container.vectors.health(), timeout=0.5))
        except Exception:
            return False

    db_ok, redis_ok, vectors_ok = await asyncio.gather(_db(), _redis(), _vectors())
    checks = {"postgres": db_ok, "redis": redis_ok, "vectors": vectors_ok}
    ok = all(checks.values())
    if not ok:
        response.status_code = 503
    return HealthReadyResponse(status="ok" if ok else "degraded", checks=checks)
