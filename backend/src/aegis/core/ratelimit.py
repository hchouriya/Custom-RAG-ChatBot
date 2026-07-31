"""Rate limit policy definitions.

Pure data and resolution logic — no I/O. The sliding-window counter itself is a
``RateLimiter`` adapter in ``infrastructure.cache``, so policies can be unit-tested and
reasoned about without Redis.

Every policy is keyed by *both* principal and client address. Per-principal alone lets
one attacker register ten accounts; per-IP alone punishes everyone behind a corporate
NAT. Requiring both to pass is the only combination that behaves sensibly in an
enterprise network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aegis.core.config import Settings


class LimitBucket(StrEnum):
    LOGIN = "login"
    CHAT = "chat"
    UPLOAD = "upload"
    REINDEX = "reindex"
    TICKET = "ticket"
    GUEST_SESSION = "guest_session"
    PASSWORD_RESET = "password_reset"  # noqa: S105 - a bucket name, not a credential
    EXPORT = "export"


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """``limit`` requests per ``window_seconds``, counted per scope."""

    bucket: LimitBucket
    limit: int
    window_seconds: int
    per_ip: bool = True
    per_principal: bool = True

    @property
    def retry_after(self) -> int:
        """Conservative hint: a sliding window frees capacity gradually.

        Advertising the full window would over-delay a caller who is only slightly over,
        while advertising a single second invites a hot retry loop.
        """
        return max(1, self.window_seconds // 4)


_MINUTE = 60
_QUARTER_HOUR = 900
_HOUR = 3600


class RateLimitPolicies:
    """Resolves the applicable policy for a bucket and role."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat_by_role: dict[str, int] = {
            "guest": settings.rate_limit_chat_per_min_guest,
            "customer": settings.rate_limit_chat_per_min_customer,
            "internal_employee": settings.rate_limit_chat_per_min_internal,
            "manager": settings.rate_limit_chat_per_min_internal,
            "admin": settings.rate_limit_chat_per_min_admin,
        }

    @property
    def enabled(self) -> bool:
        return self._settings.rate_limit_enabled

    def resolve(self, bucket: LimitBucket, role: str | None = None) -> RateLimitPolicy:
        s = self._settings
        match bucket:
            case LimitBucket.LOGIN:
                # Per-IP and per-email both, so credential stuffing across many accounts
                # from one host is throttled as effectively as brute force on one account.
                return RateLimitPolicy(bucket, s.rate_limit_login_per_15min, _QUARTER_HOUR)
            case LimitBucket.CHAT:
                per_min = self._chat_by_role.get(role or "guest", s.rate_limit_chat_per_min_guest)
                return RateLimitPolicy(bucket, per_min, _MINUTE)
            case LimitBucket.UPLOAD:
                limit = (
                    s.rate_limit_upload_per_hour
                    if role != "admin"
                    else s.rate_limit_upload_per_hour * 5
                )
                return RateLimitPolicy(bucket, limit, _HOUR)
            case LimitBucket.REINDEX:
                return RateLimitPolicy(bucket, 20 if role == "admin" else 5, _HOUR)
            case LimitBucket.TICKET:
                return RateLimitPolicy(
                    bucket, s.rate_limit_ticket_per_hour, _HOUR, per_principal=False
                )
            case LimitBucket.GUEST_SESSION:
                return RateLimitPolicy(bucket, 20, _HOUR, per_principal=False)
            case LimitBucket.PASSWORD_RESET:
                return RateLimitPolicy(bucket, 3, _HOUR)
            case LimitBucket.EXPORT:
                return RateLimitPolicy(bucket, 10, _HOUR)

    def scopes(
        self, policy: RateLimitPolicy, *, principal: str | None, ip: str | None
    ) -> list[str]:
        """Redis keys to check for one request.

        A request passes only if every returned scope is under its limit.
        """
        keys: list[str] = []
        if policy.per_principal and principal:
            keys.append(f"rl:{policy.bucket}:p:{principal}")
        if policy.per_ip and ip:
            keys.append(f"rl:{policy.bucket}:i:{ip}")
        return keys
