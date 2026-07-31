"""Turning a verified token into a principal.

One function, ``resolve``, is the only sanctioned way to obtain a :class:`SecurityContext`.
Everything security-critical downstream — the ACL filter, permission checks, rate-limit keys —
reads from the object it returns, so if this is right, a mistake elsewhere cannot widen access;
if it is wrong, nothing downstream can save it.

Three things happen here that are easy to get wrong:

* **The epoch is checked.** A token whose ``ver`` is behind the user's current epoch is
  rejected, which is what makes a role change take effect in about a second.
* **A requested mode may only narrow.** ``X-Assistant-Mode: internal`` from a customer token
  is refused; ``customer`` from an internal token is honoured. That asymmetry is the whole
  guarantee, and it lives in exactly one place.
* **Explicit grants are loaded, bounded.** A principal with more grants than the cap is an
  error rather than a truncation, because a silently truncated ACL produces answers that look
  complete while omitting documents the user is entitled to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from aegis.core.errors import AuthenticationError, AuthorizationError
from aegis.core.logging import get_logger
from aegis.core.telemetry import authz_denials
from aegis.domain.enums import Mode, Permission, Role
from aegis.domain.policies.acl import ModeNotPermittedError, build_security_context
from aegis.domain.policies.permissions import PermissionResolver
from aegis.domain.values import MAX_EXPLICIT_GRANTS, SecurityContext

if TYPE_CHECKING:
    from aegis.core.security import AccessTokenClaims
    from aegis.domain.entities import User
    from aegis.domain.ports.repositories import Repositories

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Principal:
    """The caller, fully resolved.

    Carries both the security context (what may be *read*) and the permission set (what may
    be *done*). They are separate concepts: reading a confidential document and being allowed
    to reclassify it are different authorities, and merging them into one list produces either
    an unbounded retrieval filter or an admin panel that cannot model a real organisation.
    """

    ctx: SecurityContext
    permissions: frozenset[Permission]
    user: User | None = None
    email: str | None = None
    full_name: str | None = None
    must_change_password: bool = False

    @property
    def role(self) -> Role:
        return self.ctx.role

    @property
    def user_id(self) -> UUID | None:
        return self.ctx.user_id

    @property
    def is_guest(self) -> bool:
        return self.ctx.is_guest

    def has(self, permission: Permission) -> bool:
        return permission in self.permissions

    def require(self, permission: Permission, *, resource: str = "-") -> None:
        """Assert a permission, or raise 403.

        Endpoints where the *existence* of the resource is sensitive must catch this and
        raise ``NotFoundError`` instead — a 403 on a confidential document confirms that the
        document exists, which is itself the leak.
        """
        if permission not in self.permissions:
            authz_denials.labels(role=self.role.value, resource=resource).inc()
            raise AuthorizationError(f"This action requires the {permission.value} permission.")


class PrincipalService:
    def __init__(self, repos: Repositories, *, resolver: PermissionResolver | None = None) -> None:
        self._repos = repos
        self._resolver = resolver or PermissionResolver()

    async def resolve(
        self,
        claims: AccessTokenClaims,
        *,
        requested_mode: Mode | None = None,
        collection_ids: tuple[UUID, ...] = (),
        current_epoch: int | None = None,
    ) -> Principal:
        if claims.token_type == "guest":
            return self._guest(claims, requested_mode)

        user = await self._repos.users.get(claims.subject)
        if user is None or not user.is_active:
            raise AuthenticationError("Account is not active", code="ACCOUNT_INACTIVE")

        epoch = current_epoch if current_epoch is not None else user.permission_epoch
        if claims.permission_epoch < epoch:
            # Deliberately its own code: the client's correct response is to refresh, not to
            # show a login screen.
            raise AuthenticationError(
                "Your permissions changed. Please retry.", code="TOKEN_STALE_EPOCH"
            )

        mode = self._narrow(user, claims, requested_mode)
        grants = await self._grants(user)

        try:
            ctx = build_security_context(
                user_id=user.id,
                role=user.role,
                requested_mode=mode,
                department_id=user.department_id,
                department_path=user.department_path,
                granted_document_ids=grants,
                collection_ids=collection_ids,
                session_id=claims.session_id,
                permission_epoch=epoch,
            )
        except ModeNotPermittedError as exc:
            raise AuthorizationError(
                f"Role {user.role.value} cannot use the {mode.value} assistant."
            ) from exc

        overrides = await self._repos.permissions.overrides_for(user.id)
        return Principal(
            ctx=ctx,
            permissions=self._resolver.resolve(user.role, overrides),
            user=user,
            email=user.email,
            full_name=user.full_name,
            must_change_password=user.must_change_password,
        )

    def _guest(self, claims: AccessTokenClaims, requested_mode: Mode | None) -> Principal:
        if requested_mode is not None and requested_mode is not Mode.CUSTOMER:
            raise AuthorizationError("Guest sessions are limited to the customer assistant.")
        ctx = build_security_context(
            user_id=None,
            role=Role.GUEST,
            requested_mode=Mode.CUSTOMER,
            session_id=claims.session_id,
            is_guest=True,
        )
        return Principal(ctx=ctx, permissions=self._resolver.for_role(Role.GUEST))

    @staticmethod
    def _narrow(user: User, claims: AccessTokenClaims, requested: Mode | None) -> Mode:
        """Resolve the effective mode from the token and the request header."""
        token_mode = Mode(claims.mode)
        if requested is None or requested is token_mode:
            return token_mode
        if requested not in user.allowed_modes:
            raise AuthorizationError(
                f"Role {user.role.value} cannot use the {requested.value} assistant."
            )
        # Internal → customer narrows and is allowed; customer → internal widens and is not.
        if token_mode is Mode.CUSTOMER and requested is Mode.INTERNAL:
            raise AuthorizationError(
                "This session is limited to the customer assistant. Sign in again to switch."
            )
        return requested

    async def _grants(self, user: User) -> tuple[UUID, ...]:
        paths = [user.department_path] if user.department_path else []
        granted = await self._repos.acl.granted_document_ids(
            user_id=user.id,
            role=user.role,
            department_paths=paths,
            # Fetch one past the cap so the overflow is detectable rather than silent.
            limit=MAX_EXPLICIT_GRANTS + 1,
        )
        if len(granted) > MAX_EXPLICIT_GRANTS:
            logger.error(
                "principal.too_many_grants",
                user_id=str(user.id),
                grants=len(granted),
                cap=MAX_EXPLICIT_GRANTS,
            )
            raise AuthorizationError(
                "This account has too many individual document grants to evaluate. "
                "Ask an administrator to convert them to a role or department grant."
            )
        return tuple(granted)
