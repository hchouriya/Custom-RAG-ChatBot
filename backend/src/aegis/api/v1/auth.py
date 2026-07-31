"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from aegis.api.deps import AuthServiceDep, ContainerDep, PrincipalDep, ReposDep, enforce_rate_limit
from aegis.api.schemas import (
    LoginRequest,
    LogoutRequest,
    MeResponse,
    PasswordChangeRequest,
    RefreshRequest,
    TokenResponse,
)
from aegis.core.errors import AuthenticationError, ValidationError
from aegis.core.ratelimit import LimitBucket
from aegis.domain.enums import Mode
from aegis.services.auth import LoginContext, SessionTokens

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_context(request: Request) -> LoginContext:
    return LoginContext(
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )


def _token_response(tokens: SessionTokens) -> TokenResponse:
    return TokenResponse(
        access_token=tokens.access_token,
        access_expires_at=tokens.access_expires_at,
        refresh_token=tokens.refresh_token,
        refresh_expires_at=tokens.refresh_expires_at,
        mode=tokens.mode,
        is_guest=tokens.is_guest,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    auth: AuthServiceDep,
    repos: ReposDep,
    container: ContainerDep,
) -> TokenResponse:
    await enforce_rate_limit(
        request,
        container,
        LimitBucket.LOGIN,
        principal_key=body.email.lower(),
    )
    tokens = await auth.login(
        body.email,
        body.password,
        mode=body.mode,
        totp_code=body.totp_code,
        context=_client_context(request),
    )
    await repos.uow.commit()
    return _token_response(tokens)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    request: Request,
    auth: AuthServiceDep,
    repos: ReposDep,
) -> TokenResponse:
    tokens = await auth.refresh(
        body.refresh_token, mode=body.mode, context=_client_context(request)
    )
    await repos.uow.commit()
    return _token_response(tokens)


@router.post("/logout", status_code=204)
async def logout(body: LogoutRequest, auth: AuthServiceDep, repos: ReposDep) -> None:
    await auth.logout(body.refresh_token)
    await repos.uow.commit()


@router.post("/logout-all", status_code=204)
async def logout_all(
    principal: PrincipalDep, auth: AuthServiceDep, repos: ReposDep
) -> None:
    if principal.user_id is None:
        raise AuthenticationError("Guest sessions have no persistent sessions to revoke.")
    await auth.logout_all(principal.user_id)
    await repos.uow.commit()


@router.post("/guest", response_model=TokenResponse)
async def guest(
    request: Request,
    auth: AuthServiceDep,
    container: ContainerDep,
) -> TokenResponse:
    await enforce_rate_limit(request, container, LimitBucket.GUEST_SESSION)
    tokens = await auth.guest_session()
    return _token_response(tokens)


@router.get("/me", response_model=MeResponse)
async def me(principal: PrincipalDep) -> MeResponse:
    user = principal.user
    return MeResponse(
        id=principal.user_id,
        email=principal.email,
        full_name=principal.full_name,
        role=principal.role,
        mode=principal.ctx.mode,
        permissions=sorted(p.value for p in principal.permissions),
        allowed_modes=list(user.allowed_modes) if user else [Mode.CUSTOMER],
        must_change_password=principal.must_change_password,
        is_guest=principal.is_guest,
        department_id=user.department_id if user else None,
    )


@router.post("/password/change", status_code=204)
async def change_password(
    body: PasswordChangeRequest,
    principal: PrincipalDep,
    auth: AuthServiceDep,
    repos: ReposDep,
) -> None:
    if principal.user_id is None:
        raise ValidationError("Guest sessions cannot change a password.")
    await auth.change_password(
        principal.user_id,
        current_password=body.current_password,
        new_password=body.new_password,
    )
    await repos.uow.commit()
