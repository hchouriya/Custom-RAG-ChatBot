"""Cryptographic primitives: password hashing, JWTs, opaque tokens, and TOTP.

Everything here is deliberately small, dependency-thin, and free of I/O so it can be
tested exhaustively without a database. Nothing in this module reads or writes state;
session lifecycle lives in ``services.auth_service``.

Design notes worth keeping in view:

* Access tokens carry a permission epoch (``ver``). A token whose epoch is stale is
  rejected even if unexpired, which is how a demotion takes effect in about a second
  instead of after the full 15-minute TTL.
* Refresh tokens are opaque random strings, never JWTs, and only their SHA-256 digest is
  stored. A leaked database therefore yields no usable session, and revocation is a real
  server-side operation rather than a hope that the client discards a token.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID

import jwt
import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from aegis.core.config import Settings
from aegis.core.errors import AuthenticationError, ValidationError

TokenType = Literal["access", "guest", "mfa_challenge", "password_reset"]

ISSUER: Final = "aegis"
AUDIENCE: Final = "aegis-api"

# Argon2id at ~64 MiB / 3 iterations: the OWASP-recommended balance. Raising memory is
# the most effective lever against GPU cracking, which is why memory (not time) is high.
_hasher = PasswordHasher(time_cost=3, memory_cost=65_536, parallelism=4, hash_len=32, salt_len=16)


# ── Passwords ───────────────────────────────────────────────────────────────


def hash_password(password: str, *, pepper: str = "") -> str:
    """Hash a password with Argon2id.

    The pepper is a server-side secret concatenated before hashing. Unlike the salt it
    is not stored with the hash, so a database-only compromise cannot be attacked
    offline without also obtaining the application secret.
    """
    return _hasher.hash(password + pepper)


def verify_password(password: str, stored_hash: str, *, pepper: str = "") -> bool:
    """Constant-time verification. Returns ``False`` rather than raising on any failure."""
    try:
        return _hasher.verify(stored_hash, password + pepper)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(stored_hash: str) -> bool:
    """True when the hash predates the current cost parameters.

    Called on successful login so that raising the cost parameters silently upgrades
    existing users instead of requiring a password reset for everyone.
    """
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True


_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "passw0rd",
        "welcome",
        "welcome1",
        "letmein",
        "changeme",
        "iloveyou",
        "admin",
        "administrator",
        "qwerty",
        "qwertyuiop",
        "12345678",
        "123456789",
        "1234567890",
        "abc123",
        "monkey",
        "dragon",
        "football",
        "baseball",
        "sunshine",
        "princess",
        "companyname",
        "aegis",
        "aegis123",
    }
)


def validate_password_strength(password: str, settings: Settings, *, email: str = "") -> None:
    """Reject weak passwords, raising ``ValidationError`` with every failure at once.

    Length dominates strength, so the floor is 12 rather than 8, and the composition
    rules are kept mild: rules that are too fussy push people towards ``Summer2026!``,
    which satisfies every checkbox and is trivially guessable.
    """
    problems: list[str] = []
    if len(password) < settings.password_min_length:
        problems.append(f"must be at least {settings.password_min_length} characters")
    if len(password) > 256:
        problems.append("must be at most 256 characters")
    if not any(c.isalpha() for c in password):
        problems.append("must contain a letter")
    if not any(c.isdigit() for c in password):
        problems.append("must contain a digit")

    lowered = password.lower()
    if lowered in _COMMON_PASSWORDS:
        problems.append("is among the most commonly used passwords")
    if email:
        local = email.split("@", 1)[0].lower()
        if len(local) >= 3 and local in lowered:
            problems.append("must not contain your email address")

    if len(set(password)) <= 3:
        problems.append("must not repeat only a few characters")

    if problems:
        raise ValidationError(
            "Password is too weak: " + "; ".join(problems),
            errors=[{"field": "password", "message": p} for p in problems],
        )


# ── Opaque tokens (refresh tokens, API keys, upload ids) ────────────────────


def generate_opaque_token(n_bytes: int = 32) -> str:
    """URL-safe random token. 32 bytes = 256 bits of entropy."""
    return secrets.token_urlsafe(n_bytes)


def hash_token(token: str) -> bytes:
    """SHA-256 of a high-entropy token.

    Plain SHA-256 is correct here, unlike for passwords: the input already has 256 bits
    of entropy, so there is nothing for a slow KDF to protect against, and a fast digest
    keeps refresh verification cheap.
    """
    return hashlib.sha256(token.encode("utf-8")).digest()


def constant_time_equals(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


API_KEY_PREFIX_LEN: Final = 8


def generate_api_key() -> tuple[str, str, bytes]:
    """Return ``(full_key, prefix, digest)``.

    The prefix is stored in clear so the admin UI can identify a key without holding it;
    the full key is shown exactly once at creation.
    """
    raw = generate_opaque_token(32)
    full = f"ak_{raw}"
    return full, full[:API_KEY_PREFIX_LEN], hash_token(full)


# ── JWT ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AccessTokenClaims:
    """Decoded access token. Minimal by design: a JWT is a cache, not a database."""

    subject: UUID
    role: str
    mode: str
    session_id: UUID
    permission_epoch: int
    department_id: UUID | None
    token_type: TokenType
    expires_at: datetime
    issued_at: datetime
    jti: str


class TokenCodec:
    """Encodes and decodes signed tokens for one configuration.

    Instantiated once and injected, rather than exposing module functions, so that key
    rotation and algorithm choice live in a single object that tests can substitute.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._algorithm = settings.jwt_algorithm
        if self._algorithm == "RS256":
            # Validated to exist by Settings, so these reads cannot fail at runtime.
            self._signing_key: str = Path(str(settings.jwt_private_key_path)).read_text("utf-8")
            self._verify_keys: list[str] = [
                Path(str(settings.jwt_public_key_path)).read_text("utf-8")
            ]
        else:
            secret = settings.secret_key.get_secret_value()
            self._signing_key = secret
            self._verify_keys = [secret]

    def add_verification_key(self, key: str) -> None:
        """Accept an additional key during a rotation window.

        Rotation without this is a forced logout for every active session: the new
        signing key cannot verify tokens minted by the old one.
        """
        if key not in self._verify_keys:
            self._verify_keys.append(key)

    def encode(
        self,
        *,
        subject: UUID,
        role: str,
        mode: str,
        session_id: UUID,
        permission_epoch: int,
        department_id: UUID | None = None,
        token_type: TokenType = "access",  # noqa: S107 - a claim value, not a secret
        ttl: timedelta | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, datetime]:
        """Sign a token. Returns ``(token, expires_at)``."""
        now = datetime.now(UTC)
        expires = now + (ttl or timedelta(minutes=self._settings.access_token_ttl_minutes))
        payload: dict[str, Any] = {
            "sub": str(subject),
            "role": role,
            "mode": mode,
            "sid": str(session_id),
            "ver": permission_epoch,
            "typ": token_type,
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "exp": int(expires.timestamp()),
            "iss": ISSUER,
            "aud": AUDIENCE,
            "jti": generate_opaque_token(16),
        }
        if department_id is not None:
            payload["dept"] = str(department_id)
        if extra:
            payload.update(extra)
        return jwt.encode(payload, self._signing_key, algorithm=self._algorithm), expires

    def decode(self, token: str, *, expect_type: TokenType = "access") -> AccessTokenClaims:
        """Verify and decode, raising ``AuthenticationError`` on any problem.

        ``algorithms`` is pinned to the single configured algorithm: accepting a list
        the attacker can influence is the classic JWT confusion vulnerability.
        """
        last_error: Exception | None = None
        for key in self._verify_keys:
            try:
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=[self._algorithm],
                    audience=AUDIENCE,
                    issuer=ISSUER,
                    options={"require": ["exp", "iat", "sub", "typ", "jti"]},
                )
                break
            except jwt.ExpiredSignatureError as exc:
                raise AuthenticationError("Token has expired", code="TOKEN_EXPIRED") from exc
            except jwt.InvalidTokenError as exc:
                last_error = exc
                continue
        else:
            raise AuthenticationError("Invalid token", code="TOKEN_INVALID") from last_error

        if payload.get("typ") != expect_type:
            raise AuthenticationError(f"Expected a {expect_type} token", code="TOKEN_WRONG_TYPE")

        try:
            department = payload.get("dept")
            return AccessTokenClaims(
                subject=UUID(payload["sub"]),
                role=str(payload["role"]),
                mode=str(payload["mode"]),
                session_id=UUID(payload["sid"]),
                permission_epoch=int(payload["ver"]),
                department_id=UUID(department) if department else None,
                token_type=expect_type,
                expires_at=datetime.fromtimestamp(payload["exp"], UTC),
                issued_at=datetime.fromtimestamp(payload["iat"], UTC),
                jti=str(payload["jti"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise AuthenticationError("Malformed token claims", code="TOKEN_INVALID") from exc


# ── TOTP (RFC 6238) ─────────────────────────────────────────────────────────

TOTP_DIGITS: Final = 6
TOTP_INTERVAL: Final = 30
TOTP_VALID_WINDOW: Final = 1  # ±30 s, tolerating modest clock drift


def generate_mfa_secret() -> str:
    """Base32 secret for an authenticator app."""
    return pyotp.random_base32()


def mfa_provisioning_uri(secret: str, *, account: str, issuer: str) -> str:
    """``otpauth://`` URI that the frontend renders as a QR code.

    Returned as a URI rather than a rendered image so no image library is needed
    server-side and the secret never touches disk or a log as binary.
    """
    return pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL).provisioning_uri(
        name=account, issuer_name=issuer
    )


def verify_totp(secret: str, code: str) -> bool:
    """Verify a TOTP code with a ±1 step window.

    Replay within the same window must be prevented by the caller (``auth_service``
    marks the accepted counter in Redis); TOTP alone cannot detect it.
    """
    code = code.strip().replace(" ", "")
    if not code.isdigit() or len(code) != TOTP_DIGITS:
        return False
    return pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL).verify(
        code, valid_window=TOTP_VALID_WINDOW
    )


def totp_counter(at: datetime | None = None) -> int:
    """Current TOTP time step, used as the replay-protection key."""
    moment = at or datetime.now(UTC)
    return int(moment.timestamp()) // TOTP_INTERVAL


def generate_recovery_codes(count: int = 10) -> list[str]:
    """Single-use recovery codes, formatted for legibility when written down."""
    codes: list[str] = []
    for _ in range(count):
        raw = base64.b32encode(secrets.token_bytes(10)).decode("ascii").rstrip("=").lower()
        codes.append(f"{raw[:5]}-{raw[5:10]}-{raw[10:15]}")
    return codes
