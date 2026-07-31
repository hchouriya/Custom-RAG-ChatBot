"""Shared fixtures.

Nothing here touches the network or a database. Integration fixtures live in
``tests/integration/conftest.py`` behind the ``integration`` marker, so the default
``pytest`` run is fast enough to be worth running on every save.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from aegis.core.config import Settings


@pytest.fixture(scope="session", autouse=True)
def _test_environment() -> Iterator[None]:
    """Force test settings before any module reads configuration.

    Set through the process environment rather than by constructing ``Settings`` directly
    because ``get_settings`` is cached and the first caller wins — including a module-level
    caller in an imported adapter.
    """
    original = dict(os.environ)
    os.environ.update(
        {
            "APP_ENV": "test",
            "SECRET_KEY": "test-secret-key-that-is-long-enough-to-pass-validation",
            "DATABASE_URL": "postgresql+asyncpg://aegis:aegis@localhost:5432/aegis_test",
            "REDIS_URL": "redis://localhost:6379/15",
            "VECTOR_BACKEND": "memory",
            "EMBEDDING_PROVIDER": "fake",
            "RERANKER_PROVIDER": "noop",
            "RATE_LIMIT_ENABLED": "false",
            "MALWARE_SCAN_ENABLED": "false",
            "OCR_ENABLED": "false",
            "METRICS_ENABLED": "true",
            "OTEL_ENABLED": "false",
        }
    )
    from aegis.core.config import get_settings

    get_settings.cache_clear()
    yield
    os.environ.clear()
    os.environ.update(original)
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    from aegis.core.config import get_settings

    return get_settings()
