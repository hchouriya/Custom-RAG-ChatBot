"""FastAPI dependencies: container, session, principal, rate limits."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aegis.core.config import Settings, get_settings
from aegis.core.container import Container
from aegis.core.errors import AuthenticationError, RateLimitError
from aegis.core.logging import bind_request_context
from aegis.core.ratelimit import LimitBucket, RateLimitPolicies
from aegis.core.security import AccessTokenClaims, TokenCodec
from aegis.domain.enums import Mode
from aegis.domain.ports.repositories import Repositories
from aegis.services.auth import AuthService
from aegis.services.chat import ChatService
from aegis.services.documents import DocumentService
from aegis.services.principal import Principal, PrincipalService

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Application container is not initialised")
    return container


def get_settings_dep() -> Settings:
    return get_settings()


async def get_repos(
    container: Annotated[Container, Depends(get_container)],
) -> AsyncIterator[Repositories]:
    """One session and repository bundle per request.

    Commits are explicit inside services. On unclean exit the session rolls back so a
    half-written turn cannot leak into the next request that reuses a pooled connection.
    """
    async with container.session_factory() as session:
        from aegis.infrastructure.database.repositories import build_repositories

        repos = build_repositories(session)
        try:
            yield repos
        except Exception:
            await session.rollback()
            raise


def get_auth_service(
    repos: Annotated[Repositories, Depends(get_repos)],
    container: Annotated[Container, Depends(get_container)],
) -> AuthService:
    return container.auth_service(repos)


def get_document_service(
    repos: Annotated[Repositories, Depends(get_repos)],
    container: Annotated[Container, Depends(get_container)],
) -> DocumentService:
    return container.document_service(repos)


def get_chat_service(
    repos: Annotated[Repositories, Depends(get_repos)],
    container: Annotated[Container, Depends(get_container)],
) -> ChatService:
    return container.chat_service(repos)


def get_principal_service(
    repos: Annotated[Repositories, Depends(get_repos)],
    container: Annotated[Container, Depends(get_container)],
) -> PrincipalService:
    return container.principal_service(repos)


def _decode_access_or_guest(codec: TokenCodec, token: str) -> AccessTokenClaims:
    """Accept either a normal access token or a guest session token."""
    try:
        return codec.decode(token, expect_type="access")
    except AuthenticationError as exc:
        if getattr(exc, "code", None) != "TOKEN_WRONG_TYPE":
            raise
        return codec.decode(token, expect_type="guest")


async def get_principal(
    request: Request,
    container: Annotated[Container, Depends(get_container)],
    repos: Annotated[Repositories, Depends(get_repos)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    x_assistant_mode: Annotated[str | None, Header(alias="X-Assistant-Mode")] = None,
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Bearer token required")
    claims = _decode_access_or_guest(container.codec, credentials.credentials)
    requested = Mode(x_assistant_mode) if x_assistant_mode else None

    auth = container.auth_service(repos)
    current_epoch: int | None = None
    if claims.token_type != "guest":  # noqa: S105 - token kind, not a secret
        current_epoch = await auth.current_epoch(claims.subject)

    principal = await container.principal_service(repos).resolve(
        claims, requested_mode=requested, current_epoch=current_epoch
    )
    bind_request_context(
        user_id=str(principal.user_id) if principal.user_id else None,
        role=principal.role.value,
        mode=principal.ctx.mode.value,
    )
    request.state.principal = principal
    return principal


async def get_optional_principal(
    request: Request,
    container: Annotated[Container, Depends(get_container)],
    repos: Annotated[Repositories, Depends(get_repos)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    x_assistant_mode: Annotated[str | None, Header(alias="X-Assistant-Mode")] = None,
) -> Principal | None:
    if credentials is None:
        return None
    return await get_principal(
        request, container, repos, credentials, x_assistant_mode=x_assistant_mode
    )


async def enforce_rate_limit(
    request: Request,
    container: Annotated[Container, Depends(get_container)],
    bucket: LimitBucket,
    *,
    principal_key: str | None = None,
    role: str | None = None,
) -> None:
    """Raise :class:`RateLimitError` when any scoped window is exhausted."""
    settings = container.settings
    policies = RateLimitPolicies(settings)
    if not policies.enabled:
        return
    policy = policies.resolve(bucket, role=role)
    ip = request.client.host if request.client else None
    for key in policies.scopes(policy, principal=principal_key, ip=ip):
        result = await container.limiter.check(
            key, limit=policy.limit, window_seconds=policy.window_seconds
        )
        if not result.allowed:
            raise RateLimitError(policy.retry_after)


ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ReposDep = Annotated[Repositories, Depends(get_repos)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]
OptionalPrincipalDep = Annotated[Principal | None, Depends(get_optional_principal)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
