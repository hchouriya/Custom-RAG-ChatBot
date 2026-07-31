"""Administrative permissions.

The default matrix here is the seed for the ``role_permissions`` table and the fallback
when that table is empty. Keeping it in the domain — rather than only in a migration —
means the capability matrix is diffable in version control and testable without a
database, which is what an auditor asks to see.

Resolution order for a principal: role grants, then per-user overrides, with ``deny``
beating ``allow``. Deny-wins is the only defensible precedence: a revocation that a
later-evaluated grant can undo is not a revocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from aegis.domain.enums import Permission as P
from aegis.domain.enums import Role

_CUSTOMER_BASE: frozenset[P] = frozenset({P.CHAT_CUSTOMER, P.TICKET_CREATE})

_INTERNAL_BASE: frozenset[P] = _CUSTOMER_BASE | {
    P.CHAT_INTERNAL,
    P.DOCUMENT_READ,
    P.DOCUMENT_WRITE,
    P.DOCUMENT_DELETE,  # scoped to own uploads by `acl.can_administer_document`
    P.DOCUMENT_DOWNLOAD,
    P.DOCUMENT_REINDEX,
    P.COLLECTION_READ,
    P.ANALYTICS_READ,  # own usage only
}

_MANAGER_BASE: frozenset[P] = _INTERNAL_BASE | {
    P.USER_READ,
    P.TICKET_MANAGE,
    P.AUDIT_READ,  # department scope
    P.TRACE_READ,
    P.RETRIEVAL_DEBUG,
}

DEFAULT_ROLE_PERMISSIONS: dict[Role, frozenset[P]] = {
    Role.GUEST: frozenset({P.CHAT_CUSTOMER, P.TICKET_CREATE}),
    Role.CUSTOMER: _CUSTOMER_BASE,
    Role.INTERNAL_EMPLOYEE: _INTERNAL_BASE,
    Role.MANAGER: _MANAGER_BASE,
    Role.ADMIN: frozenset(P),
}
"""Coarse capability by role.

Several of these are *scoped* rather than absolute — an internal employee holds
``document:delete`` but only for their own uploads, and ``analytics:read`` but only for
their own usage. The scope is applied by the service that owns the resource, because it
depends on the resource's owner and department, which a permission code cannot express.
"""

PERMISSION_DESCRIPTIONS: dict[P, tuple[str, str]] = {
    P.CHAT_CUSTOMER: ("chat", "Use the customer support assistant"),
    P.CHAT_INTERNAL: ("chat", "Use the internal assistant"),
    P.TICKET_CREATE: ("support", "Create a support ticket"),
    P.TICKET_MANAGE: ("support", "Triage and resolve support tickets"),
    P.DOCUMENT_READ: ("documents", "View document metadata in the admin panel"),
    P.DOCUMENT_WRITE: ("documents", "Upload documents and edit metadata"),
    P.DOCUMENT_DELETE: ("documents", "Delete and replace documents"),
    P.DOCUMENT_DOWNLOAD: ("documents", "Download original files"),
    P.DOCUMENT_REINDEX: ("documents", "Trigger re-chunking or re-embedding"),
    P.ACL_MANAGE: ("documents", "Edit per-document access grants"),
    P.COLLECTION_READ: ("collections", "View collections"),
    P.COLLECTION_MANAGE: ("collections", "Create and configure collections"),
    P.USER_READ: ("users", "View users"),
    P.USER_MANAGE: ("users", "Create, edit, and deactivate users"),
    P.ROLE_MANAGE: ("users", "Edit the role permission matrix"),
    P.APIKEY_MANAGE: ("users", "Issue and revoke service API keys"),
    P.ANALYTICS_READ: ("analytics", "View analytics within scope"),
    P.ANALYTICS_READ_ALL: ("analytics", "View analytics across the organisation"),
    P.AUDIT_READ: ("audit", "Read audit logs"),
    P.TRACE_READ: ("audit", "Read query traces"),
    P.RETRIEVAL_DEBUG: ("audit", "Run the retrieval debugger"),
    P.PROMPT_MANAGE: ("platform", "Edit and activate prompt templates"),
    P.SETTINGS_MANAGE: ("platform", "Change runtime settings"),
    P.EVAL_RUN: ("platform", "Trigger evaluation runs"),
    P.INDEX_MANAGE: ("platform", "Rebuild and reconcile the search index"),
}


@dataclass(frozen=True, slots=True)
class PermissionOverride:
    permission: P
    allow: bool
    expires_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.expires_at is None or self.expires_at > datetime.now(UTC)


class PermissionResolver:
    """Resolves the effective permission set for a principal.

    Constructed with the matrix loaded from the database at startup; falls back to
    :data:`DEFAULT_ROLE_PERMISSIONS` when that load returns nothing, so a freshly
    migrated but unseeded database is usable rather than inert.
    """

    def __init__(self, matrix: dict[Role, frozenset[P]] | None = None) -> None:
        self._matrix = matrix or DEFAULT_ROLE_PERMISSIONS

    @property
    def matrix(self) -> dict[Role, frozenset[P]]:
        return self._matrix

    def for_role(self, role: Role) -> frozenset[P]:
        return self._matrix.get(role, frozenset())

    def resolve(
        self, role: Role, overrides: list[PermissionOverride] | None = None
    ) -> frozenset[P]:
        """Effective permissions: role grants, plus allows, minus denies.

        Denies are applied last and unconditionally.
        """
        effective = set(self.for_role(role))
        if overrides:
            active = [o for o in overrides if o.is_active]
            effective |= {o.permission for o in active if o.allow}
            effective -= {o.permission for o in active if not o.allow}
        return frozenset(effective)

    def has(
        self,
        role: Role,
        permission: P,
        overrides: list[PermissionOverride] | None = None,
    ) -> bool:
        return permission in self.resolve(role, overrides)
