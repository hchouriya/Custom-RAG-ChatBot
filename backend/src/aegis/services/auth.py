"""Authentication and session lifecycle.

The design points that matter, each of which is a decision rather than a default:

**Refresh tokens rotate and reuse is detected.** Every refresh issues a new token and
revokes the old one. Presenting an already-spent token means it leaked, so the response is to
revoke the entire family — every descendant session — rather than to reject one request. The
alternative leaves an attacker holding a working credential alongside the victim.

**Access tokens carry a permission epoch.** Demoting a user bumps the epoch, and every token
minted before it stops verifying within one Redis lookup. Without this, a revocation waits out
the token TTL; "you are an admin for another fourteen minutes" is not access control.

**Failed logins are counted atomically and the enumeration answer is uniform.** A wrong
password and an unknown email produce the same error and comparable timing, because the
difference is a free list of valid accounts.

**Guest sessions are real sessions.** They get a signed token with role ``guest``, a session
id, and their own rate limit, rather than an unauthenticated path with parallel logic. One
authorization path is far easier to keep correct than two.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aegis.core.errors import (
    AuthenticationError,
    AuthorizationError,
    MFARequiredError,
    NotFoundError,
    ValidationError,
)
from aegis.core.ids import new_id
from aegis.core.logging import get_logger
from aegis.core.security import (
    generate_mfa_secret,
    generate_opaque_token,
    generate_recovery_codes,
    hash_password,
    hash_token,
    mfa_provisioning_uri,
    password_needs_rehash,
    validate_password_strength,
    verify_password,
    verify_totp,
)
from aegis.core.telemetry import login_failures
from aegis.domain.enums import Mode, Role
from aegis.domain.policies.acl import ModeNotPermittedError, default_mode, resolve_ceiling

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.core.security import TokenCodec
    from aegis.domain.entities import User
    from aegis.domain.ports.infrastructure import Cache
    from aegis.domain.ports.repositories import Repositories

logger = get_logger(__name__)

EPOCH_CACHE_TTL = 60
GUEST_SESSION_TTL_HOURS = 12


@dataclass(frozen=True, slots=True)
class SessionTokens:
    access_token: str
    access_expires_at: datetime
    refresh_token: str | None
    refresh_expires_at: datetime | None
    mode: Mode
    user: User | None = None
    is_guest: bool = False


@dataclass(frozen=True, slots=True)
class LoginContext:
    """Request facts recorded with a session, for the "your sessions" list and for forensics."""

    ip: str | None = None
    user_agent: str | None = None


class AuthService:
    def __init__(
        self,
        repos: Repositories,
        *,
        settings: Settings,
        codec: TokenCodec,
        cache: Cache,
    ) -> None:
        self._repos = repos
        self._settings = settings
        self._codec = codec
        self._cache = cache

    # ── login ───────────────────────────────────────────────────────────────

    async def login(
        self,
        email: str,
        password: str,
        *,
        mode: Mode | None = None,
        totp_code: str | None = None,
        context: LoginContext | None = None,
    ) -> SessionTokens:
        """Verify credentials and mint a session.

        The order is: find user, check lockout, verify password, then MFA. Verifying the
        password before checking MFA is what lets a wrong password and a wrong TOTP code be
        distinguished in the audit log without being distinguishable to the caller.
        """
        settings = self._settings
        pepper = settings.secret_key.get_secret_value()
        user = await self._repos.users.get_by_email(email.strip().lower())

        if user is None:
            # Hash anyway so a missing account and a wrong password take comparable time.
            hash_password(password, pepper=pepper)
            login_failures.labels(reason="unknown_email").inc()
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            login_failures.labels(reason="inactive").inc()
            raise AuthenticationError("Invalid email or password")
        if user.is_locked:
            login_failures.labels(reason="locked").inc()
            raise AuthenticationError(
                "This account is temporarily locked after repeated failed attempts.",
                code="ACCOUNT_LOCKED",
            )

        credentials = await self._repos.users.get_credentials(user.id)
        if credentials is None or not credentials.password_hash:
            login_failures.labels(reason="no_password").inc()
            raise AuthenticationError("Invalid email or password")

        if not verify_password(password, credentials.password_hash, pepper=pepper):
            updated = await self._repos.users.record_login_failure(
                user.id,
                max_failures=settings.max_failed_logins,
                lockout_minutes=settings.lockout_minutes,
            )
            login_failures.labels(reason="bad_password").inc()
            logger.info(
                "auth.login_failed",
                user_id=str(user.id),
                failed_logins=updated.failed_logins,
                locked=updated.is_locked,
            )
            raise AuthenticationError("Invalid email or password")

        if password_needs_rehash(credentials.password_hash):
            await self._repos.users.set_password(user.id, hash_password(password, pepper=pepper))

        requires_mfa = settings.mfa_enabled and user.role.value in settings.mfa_required_role_list
        if user.has_mfa or requires_mfa:
            await self._check_second_factor(
                user, credentials.mfa_secret, totp_code, enrolment_required=not user.has_mfa
            )

        resolved = self._resolve_mode(user, mode)
        await self._repos.users.record_login_success(user.id, at=datetime.now(UTC))
        tokens = await self._issue(user, resolved, context=context)
        await self._cache_epoch(user.id, user.permission_epoch)
        logger.info("auth.login", user_id=str(user.id), role=user.role.value, mode=resolved.value)
        return tokens

    async def _check_second_factor(
        self,
        user: User,
        secret: str | None,
        code: str | None,
        *,
        enrolment_required: bool,
    ) -> None:
        if not user.has_mfa or not secret:
            challenge, _ = self._codec.encode(
                subject=user.id,
                role=user.role.value,
                mode=default_mode(user.role).value,
                session_id=new_id(),
                permission_epoch=user.permission_epoch,
                token_type="mfa_challenge",
                ttl=timedelta(minutes=10),
            )
            raise MFARequiredError(challenge, enrolment_required=enrolment_required)

        if not code:
            challenge, _ = self._codec.encode(
                subject=user.id,
                role=user.role.value,
                mode=default_mode(user.role).value,
                session_id=new_id(),
                permission_epoch=user.permission_epoch,
                token_type="mfa_challenge",
                ttl=timedelta(minutes=10),
            )
            raise MFARequiredError(challenge)

        normalised = code.strip().replace(" ", "").replace("-", "")
        if len(normalised) > 6 and await self._repos.users.consume_recovery_code(
            user.id, hash_token(normalised).hex()
        ):
            logger.warning("auth.recovery_code_used", user_id=str(user.id))
            return

        if not verify_totp(secret, normalised):
            login_failures.labels(reason="bad_totp").inc()
            raise AuthenticationError("That code is not valid.", code="MFA_INVALID")

        # TOTP alone cannot detect replay inside its own 30-second window; a first-writer-wins
        # marker closes it.
        from aegis.core.security import totp_counter

        marker = f"mfa:used:{user.id}:{totp_counter()}"
        if not await self._cache.add(marker, b"1", ttl_seconds=90):
            login_failures.labels(reason="totp_replay").inc()
            raise AuthenticationError("That code has already been used.", code="MFA_REPLAY")

    # ── guest ───────────────────────────────────────────────────────────────

    async def guest_session(self) -> SessionTokens:
        """Anonymous customer-mode session.

        No refresh token: a guest session that could be silently extended forever is a
        rate-limit bypass, and 12 hours is longer than any real support conversation.
        """
        if not self._settings.guest_access_enabled:
            raise AuthorizationError("Guest access is disabled.")
        session_id = new_id()
        token, expires = self._codec.encode(
            subject=session_id,
            role=Role.GUEST.value,
            mode=Mode.CUSTOMER.value,
            session_id=session_id,
            permission_epoch=0,
            token_type="guest",
            ttl=timedelta(hours=GUEST_SESSION_TTL_HOURS),
        )
        return SessionTokens(
            access_token=token,
            access_expires_at=expires,
            refresh_token=None,
            refresh_expires_at=None,
            mode=Mode.CUSTOMER,
            is_guest=True,
        )

    # ── refresh ─────────────────────────────────────────────────────────────

    async def refresh(
        self,
        refresh_token: str,
        *,
        mode: Mode | None = None,
        context: LoginContext | None = None,
    ) -> SessionTokens:
        digest = hash_token(refresh_token)
        record = await self._repos.refresh_tokens.find_active(digest)

        if record is None:
            # Not active. If it was ever used, this is a replay of a rotated token: the
            # token leaked, and the whole family goes.
            if family := await self._repos.refresh_tokens.was_used(digest):
                revoked = await self._repos.refresh_tokens.revoke_family(family)
                await self._repos.uow.commit()
                logger.warning(
                    "auth.refresh_reuse_detected", family_id=str(family), revoked=revoked
                )
                raise AuthenticationError(
                    "This session has been ended for security reasons. Please sign in again.",
                    code="REFRESH_REUSED",
                )
            raise AuthenticationError("Invalid refresh token", code="REFRESH_INVALID")

        user = await self._repos.users.get(UUID(str(record["user_id"])))
        if user is None or not user.can_authenticate:
            await self._repos.refresh_tokens.revoke(UUID(str(record["jti"])))
            await self._repos.uow.commit()
            raise AuthenticationError("Account is not active", code="ACCOUNT_INACTIVE")

        await self._repos.refresh_tokens.revoke(UUID(str(record["jti"])))
        resolved = self._resolve_mode(user, mode)
        tokens = await self._issue(
            user,
            resolved,
            context=context,
            family_id=UUID(str(record["family_id"])),
        )
        await self._cache_epoch(user.id, user.permission_epoch)
        return tokens

    async def logout(self, refresh_token: str | None) -> None:
        """Revoke the presented session, if any.

        Silent on an unknown token: logout is not an oracle for whether a token was valid,
        and a client retrying a logout should not receive an error.
        """
        if not refresh_token:
            return
        record = await self._repos.refresh_tokens.find_active(hash_token(refresh_token))
        if record is not None:
            await self._repos.refresh_tokens.revoke(UUID(str(record["jti"])))

    async def logout_all(self, user_id: UUID) -> int:
        revoked = await self._repos.refresh_tokens.revoke_all_for_user(user_id)
        await self._repos.users.bump_permission_epoch(user_id)
        await self._cache.delete(self._epoch_key(user_id))
        return revoked

    # ── passwords ───────────────────────────────────────────────────────────

    async def change_password(
        self, user_id: UUID, *, current_password: str, new_password: str
    ) -> None:
        """Change a password and end every other session.

        Ending other sessions is the point of a password change for most users — someone
        else knows the old one. Keeping them alive would make the change cosmetic.
        """
        pepper = self._settings.secret_key.get_secret_value()
        user = await self._repos.users.get(user_id)
        credentials = await self._repos.users.get_credentials(user_id)
        if user is None or credentials is None or not credentials.password_hash:
            raise NotFoundError("User", user_id)
        if not verify_password(current_password, credentials.password_hash, pepper=pepper):
            raise AuthenticationError("Current password is incorrect", code="PASSWORD_MISMATCH")
        if current_password == new_password:
            raise ValidationError("The new password must differ from the current one.")

        validate_password_strength(new_password, self._settings, email=user.email)
        await self._repos.users.set_password(user_id, hash_password(new_password, pepper=pepper))
        await self._repos.users.update(user_id, must_change_password=False)
        await self._repos.refresh_tokens.revoke_all_for_user(user_id)
        await self._repos.users.bump_permission_epoch(user_id)
        await self._cache.delete(self._epoch_key(user_id))
        logger.info("auth.password_changed", user_id=str(user_id))

    # ── MFA enrolment ───────────────────────────────────────────────────────

    async def begin_mfa_enrolment(self, user_id: UUID) -> dict[str, Any]:
        user = await self._repos.users.get(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        secret = generate_mfa_secret()
        # Held in the cache, not the user row: an unconfirmed secret must not be able to
        # lock someone out of their own account if enrolment is abandoned.
        await self._cache.set(f"mfa:pending:{user_id}", secret.encode(), ttl_seconds=900)
        return {
            "secret": secret,
            "otpauth_uri": mfa_provisioning_uri(
                secret, account=user.email, issuer=self._settings.mfa_issuer
            ),
        }

    async def confirm_mfa_enrolment(self, user_id: UUID, code: str) -> list[str]:
        pending = await self._cache.get(f"mfa:pending:{user_id}")
        if pending is None:
            raise ValidationError("Enrolment expired. Start again.")
        secret = pending.decode()
        if not verify_totp(secret, code):
            raise AuthenticationError("That code is not valid.", code="MFA_INVALID")
        codes = generate_recovery_codes()
        await self._repos.users.set_mfa(
            user_id,
            secret=secret,
            recovery_code_hashes=[hash_token(c.replace("-", "")).hex() for c in codes],
        )
        await self._repos.users.update(user_id, has_mfa=True)
        await self._cache.delete(f"mfa:pending:{user_id}")
        logger.info("auth.mfa_enrolled", user_id=str(user_id))
        # Shown exactly once. Storing only digests means we could not display them later
        # even if asked to.
        return codes

    # ── epoch cache ─────────────────────────────────────────────────────────

    @staticmethod
    def _epoch_key(user_id: UUID) -> str:
        return f"user:{user_id}:ver"

    async def _cache_epoch(self, user_id: UUID, epoch: int) -> None:
        await self._cache.set(
            self._epoch_key(user_id), str(epoch).encode(), ttl_seconds=EPOCH_CACHE_TTL
        )

    async def current_epoch(self, user_id: UUID) -> int:
        """The user's permission epoch, cached for a minute.

        One Redis read per request instead of one database read, and a 60-second worst case
        between a demotion and its enforcement. Bumping the epoch deletes the key, so the
        usual case is immediate.
        """
        cached = await self._cache.get(self._epoch_key(user_id))
        if cached is not None:
            try:
                return int(cached)
            except ValueError:  # pragma: no cover - corrupt cache entry
                pass
        user = await self._repos.users.get(user_id)
        if user is None:
            raise AuthenticationError("Account no longer exists", code="ACCOUNT_MISSING")
        await self._cache_epoch(user_id, user.permission_epoch)
        return user.permission_epoch

    # ── helpers ─────────────────────────────────────────────────────────────

    def _resolve_mode(self, user: User, requested: Mode | None) -> Mode:
        """Honour a mode request only when the role permits it."""
        if requested is None:
            return user.default_mode
        try:
            resolve_ceiling(user.role, requested)
        except ModeNotPermittedError as exc:
            raise AuthorizationError(
                f"Role {user.role.value} cannot use the {requested.value} assistant."
            ) from exc
        return requested

    async def _issue(
        self,
        user: User,
        mode: Mode,
        *,
        context: LoginContext | None,
        family_id: UUID | None = None,
    ) -> SessionTokens:
        session_id = new_id()
        access, access_expires = self._codec.encode(
            subject=user.id,
            role=user.role.value,
            mode=mode.value,
            session_id=session_id,
            permission_epoch=user.permission_epoch,
            department_id=user.department_id,
        )
        refresh = generate_opaque_token(32)
        refresh_expires = datetime.now(UTC) + timedelta(days=self._settings.refresh_token_ttl_days)
        await self._repos.refresh_tokens.create(
            user_id=user.id,
            jti=session_id,
            token_hash=hash_token(refresh),
            family_id=family_id or session_id,
            expires_at=refresh_expires,
            user_agent=(context.user_agent if context else None),
            ip=(context.ip if context else None),
        )
        return SessionTokens(
            access_token=access,
            access_expires_at=access_expires,
            refresh_token=refresh,
            refresh_expires_at=refresh_expires,
            mode=mode,
            user=user,
        )
