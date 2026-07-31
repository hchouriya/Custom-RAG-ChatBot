"""Secret resolution behind one port.

Locally secrets come from the environment; in cloud they come from a managed store. The
port exists so that no calling code cares which, and so that rotating a credential is an
operational action rather than a redeploy.

The AWS and Vault adapters are import-guarded: their SDKs are not runtime dependencies
of a deployment that does not use them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aegis.core.config import Settings
from aegis.core.errors import ConfigurationError
from aegis.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class SecretsProvider(Protocol):
    """Resolves a named secret to its value."""

    async def get(self, name: str) -> str | None: ...

    async def require(self, name: str) -> str: ...

    async def close(self) -> None: ...


class _Base:
    async def require(self, name: str) -> str:
        value = await self.get(name)  # type: ignore[attr-defined]
        if not value:
            raise ConfigurationError(f"Required secret {name!r} is not available")
        return str(value)

    async def close(self) -> None:
        return None


class EnvSecretsProvider(_Base):
    """Reads from the validated ``Settings`` object.

    Note that it reads *Settings*, not the environment: that keeps the ban on
    ``os.getenv`` intact and means every secret has already been through validation.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def get(self, name: str) -> str | None:
        value = getattr(self._settings, name.lower(), None)
        if value is None:
            return None
        getter = getattr(value, "get_secret_value", None)
        return str(getter()) if callable(getter) else str(value)


class AwsSecretsManagerProvider(_Base):
    """AWS Secrets Manager, with an in-process cache.

    Caching matters: Secrets Manager bills per API call and is rate-limited, so an
    uncached read inside a request path becomes both a cost and a latency problem.
    Rotation is handled by restarting workers or by ``invalidate()``.
    """

    def __init__(self, settings: Settings, *, prefix: str = "aegis/") -> None:
        self._settings = settings
        self._prefix = prefix
        self._cache: dict[str, str] = {}
        self._session: object | None = None

    async def get(self, name: str) -> str | None:
        if name in self._cache:
            return self._cache[name]
        try:
            import aioboto3
        except ImportError as exc:  # pragma: no cover - deployment-specific
            raise ConfigurationError(
                "SECRETS_PROVIDER=aws requires aioboto3 to be installed"
            ) from exc

        session = aioboto3.Session()
        async with session.client("secretsmanager", region_name=self._settings.s3_region) as client:
            try:
                response = await client.get_secret_value(SecretId=f"{self._prefix}{name}")
            except Exception as exc:
                logger.warning("secrets.aws.miss", secret=name, error=type(exc).__name__)
                return None
        value = response.get("SecretString")
        if not value:
            return None
        self._cache[name] = str(value)
        return str(value)

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)


class VaultSecretsProvider(_Base):
    """HashiCorp Vault KV v2 over its HTTP API.

    Uses httpx directly rather than the ``hvac`` client because hvac is synchronous, and
    a blocking call inside the event loop stalls every concurrent request on the worker.
    """

    def __init__(
        self, settings: Settings, *, address: str, token: str, mount: str = "secret"
    ) -> None:
        self._settings = settings
        self._address = address.rstrip("/")
        self._token = token
        self._mount = mount
        self._cache: dict[str, str] = {}
        self._client: object | None = None

    async def get(self, name: str) -> str | None:
        if name in self._cache:
            return self._cache[name]
        import httpx

        url = f"{self._address}/v1/{self._mount}/data/aegis/{name}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, headers={"X-Vault-Token": self._token})
        if response.status_code != 200:
            logger.warning("secrets.vault.miss", secret=name, status=response.status_code)
            return None
        value = response.json().get("data", {}).get("data", {}).get("value")
        if value:
            self._cache[name] = str(value)
        return str(value) if value else None


def build_secrets_provider(settings: Settings) -> SecretsProvider:
    """Select the adapter named by ``SECRETS_PROVIDER``."""
    match settings.secrets_provider:
        case "env":
            return EnvSecretsProvider(settings)
        case "aws":
            return AwsSecretsManagerProvider(settings)
        case "vault":
            # Vault address and token are themselves environment-provided; a Vault
            # deployment supplies them through the pod's service account.
            env = EnvSecretsProvider(settings)
            address = getattr(settings, "vault_addr", "http://vault:8200")
            token = getattr(settings, "vault_token", "")
            if not token:
                raise ConfigurationError("SECRETS_PROVIDER=vault requires VAULT_TOKEN")
            _ = env
            return VaultSecretsProvider(settings, address=str(address), token=str(token))
