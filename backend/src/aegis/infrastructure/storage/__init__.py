"""Object storage adapters and their factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.core.errors import ConfigurationError
from aegis.infrastructure.storage.keys import (
    derived_key,
    document_key,
    document_prefix,
    is_safe_key,
    sanitize_filename,
    upload_key,
    version_prefix,
)
from aegis.infrastructure.storage.local import LocalObjectStore

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.domain.ports.infrastructure import ObjectStore

__all__ = [
    "LocalObjectStore",
    "build_object_store",
    "derived_key",
    "document_key",
    "document_prefix",
    "is_safe_key",
    "sanitize_filename",
    "upload_key",
    "version_prefix",
]


def build_object_store(settings: Settings) -> ObjectStore:
    """Construct the configured store.

    ``aioboto3`` is imported lazily so that a test run, or a deployment on local disk, does
    not pay for loading botocore's service model — which is a measurable fraction of import
    time for the whole application.
    """
    if settings.storage_backend == "local":
        if settings.is_production:  # pragma: no cover - Settings rejects this at boot
            raise ConfigurationError("STORAGE_BACKEND=local cannot be used in production")
        return LocalObjectStore.from_settings(settings)

    if settings.storage_backend == "s3":
        from aegis.infrastructure.storage.s3 import S3ObjectStore

        return S3ObjectStore.from_settings(settings)

    raise ConfigurationError(f"unknown storage backend: {settings.storage_backend}")
