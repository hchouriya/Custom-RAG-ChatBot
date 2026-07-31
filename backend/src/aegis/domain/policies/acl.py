"""Access control policy — the implementation of invariant I1.

This is the most security-critical module in the codebase. Everything it needs arrives in
a :class:`SecurityContext` that was derived server-side from a verified token; no field of
an HTTP request reaches it except through ``narrowing``, which can only intersect.

Two properties are worth stating because they are what make the design defensible:

1. **The ceiling table is a total function with deny-by-default on absence.** A role added
   without an entry gets no access rather than inheriting someone else's.
2. **Mode is a ceiling, not a preference.** ``(admin, customer)`` caps at
   :attr:`VisibilityLevel.CUSTOMER`, so an administrator exercising the customer
   assistant cannot pull internal content into a customer-facing transcript.

Pure functions, no I/O, no imports outside the domain — which is why the 200-case
permission matrix in ``tests/security`` runs in milliseconds without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from aegis.domain.enums import Mode, Role, Visibility, VisibilityLevel
from aegis.domain.values import (
    MAX_EXPLICIT_GRANTS,
    And,
    In,
    IsNull,
    Match,
    Or,
    PrefixMatch,
    Range,
    SecurityContext,
    VectorFilter,
    is_in_subtree,
)

CEILING: dict[tuple[Role, Mode], VisibilityLevel] = {
    (Role.GUEST, Mode.CUSTOMER): VisibilityLevel.PUBLIC,
    (Role.CUSTOMER, Mode.CUSTOMER): VisibilityLevel.CUSTOMER,
    (Role.INTERNAL_EMPLOYEE, Mode.CUSTOMER): VisibilityLevel.CUSTOMER,
    (Role.MANAGER, Mode.CUSTOMER): VisibilityLevel.CUSTOMER,
    (Role.ADMIN, Mode.CUSTOMER): VisibilityLevel.CUSTOMER,
    (Role.INTERNAL_EMPLOYEE, Mode.INTERNAL): VisibilityLevel.INTERNAL,
    (Role.MANAGER, Mode.INTERNAL): VisibilityLevel.CONFIDENTIAL,
    (Role.ADMIN, Mode.INTERNAL): VisibilityLevel.RESTRICTED,
}
"""``(role, mode) → highest readable visibility level``.

Absent pairs — ``(customer, internal)`` and ``(guest, internal)`` — are denied, which is
enforced by :func:`resolve_ceiling` raising rather than defaulting.
"""


class ModeNotPermittedError(Exception):
    """The principal's role cannot use the requested assistant at all."""

    def __init__(self, role: Role, mode: Mode) -> None:
        super().__init__(f"role {role} may not use {mode} mode")
        self.role = role
        self.mode = mode


def resolve_ceiling(role: Role, mode: Mode) -> VisibilityLevel:
    """Highest visibility level readable by ``role`` in ``mode``.

    Raises :class:`ModeNotPermittedError` for a combination that is not allowed, rather
    than returning a safe-looking default. A silent fallback here would be indefensible:
    it would mean a customer requesting internal mode receives *something* instead of a
    refusal, and the difference would be invisible in the code.
    """
    try:
        return CEILING[(role, mode)]
    except KeyError:
        raise ModeNotPermittedError(role, mode) from None


def default_mode(role: Role) -> Mode:
    return Mode.INTERNAL if role.is_internal else Mode.CUSTOMER


def build_security_context(
    *,
    user_id: UUID | None,
    role: Role,
    requested_mode: Mode | None,
    department_id: UUID | None = None,
    department_path: str | None = None,
    granted_document_ids: tuple[UUID, ...] = (),
    collection_ids: tuple[UUID, ...] = (),
    session_id: UUID | None = None,
    permission_epoch: int = 0,
    is_guest: bool = False,
) -> SecurityContext:
    """Assemble the context. The only sanctioned way to construct one."""
    mode = requested_mode or default_mode(role)
    ceiling = resolve_ceiling(role, mode)

    if len(granted_document_ids) > MAX_EXPLICIT_GRANTS:
        raise ValueError(
            f"principal has {len(granted_document_ids)} explicit document grants, "
            f"above the {MAX_EXPLICIT_GRANTS} limit; express this as a role or "
            "department grant instead"
        )

    return SecurityContext(
        user_id=user_id,
        role=role,
        mode=mode,
        ceiling=ceiling,
        department_id=department_id,
        department_path=department_path,
        granted_document_ids=granted_document_ids,
        collection_ids=collection_ids,
        session_id=session_id,
        permission_epoch=permission_epoch,
        is_guest=is_guest,
    )


def build_filter(ctx: SecurityContext, narrowing: VectorFilter | None = None) -> VectorFilter:
    """Translate a security context into a vector-store filter.

    Structure of the result:

    ``must``
        Invariants that hold for every readable chunk: the assistant mode, the document
        being active and unexpired, the collection restriction, and the ceiling as a
        range predicate on ``visibility_level``.

    ``should`` with ``min_should=1``
        The alternative ways a chunk may qualify — by level alone, by level *and*
        department containment (confidential), or by explicit grant (restricted). At least
        one must hold; expressed as ``should`` rather than a nested ``Or`` in ``must``
        because backends optimize a top-level minimum-should-match well.

    The ceiling appears in ``must`` as well as inside the ``should`` branches. That is
    deliberate redundancy: a bug in the branch construction can then only ever *narrow*
    the result, never widen it past the ceiling.
    """
    must: list[Match | In | Range | Or | And | IsNull | PrefixMatch] = [
        Match("mode", ctx.mode.value),
        Match("is_active", True),
        Range("visibility_level", lte=int(ctx.ceiling)),
        Or((IsNull("expires_at"), Range("expires_at", gt=_now_epoch()))),
    ]

    if ctx.collection_ids:
        must.append(In("collection_id", tuple(str(c) for c in ctx.collection_ids)))

    should: list[Match | In | Range | And] = [
        # Levels 0-2 need nothing beyond the ceiling.
        Range("visibility_level", lte=int(min(ctx.ceiling, VisibilityLevel.INTERNAL)))
    ]

    if ctx.ceiling >= VisibilityLevel.CONFIDENTIAL and ctx.department_path:
        should.append(
            And(
                (
                    Match("visibility_level", int(VisibilityLevel.CONFIDENTIAL)),
                    PrefixMatch("department_path", ctx.department_path),
                )
            )
        )

    if ctx.ceiling >= VisibilityLevel.RESTRICTED:
        should.append(Match("visibility_level", int(VisibilityLevel.RESTRICTED)))

    if ctx.granted_document_ids:
        # An explicit grant overrides level and department, which is the entire point of
        # having one. It stays inside `should`, so the mode and active constraints in
        # `must` still apply: a grant is permission to read a document, not permission to
        # pull an internal document into a customer-facing session.
        should.append(In("document_id", tuple(str(d) for d in ctx.granted_document_ids)))

    base = VectorFilter(must=tuple(must), should=tuple(should), min_should=1)
    return base.intersect(narrowing)


def _now_epoch() -> float:
    return datetime.now(UTC).timestamp()


def visibility_options(role: Role) -> tuple[Visibility, ...]:
    """Visibility levels this role may *assign* when uploading or editing.

    Assignment authority is narrower than read authority: an internal employee can read
    confidential documents in their department but cannot mark their own upload
    confidential, because classification is a management decision.
    """
    match role:
        case Role.ADMIN:
            return tuple(Visibility)
        case Role.MANAGER:
            return (
                Visibility.PUBLIC,
                Visibility.CUSTOMER,
                Visibility.INTERNAL,
                Visibility.CONFIDENTIAL,
            )
        case Role.INTERNAL_EMPLOYEE:
            return (Visibility.PUBLIC, Visibility.CUSTOMER, Visibility.INTERNAL)
        case _:
            return ()


def can_assign_visibility(role: Role, visibility: Visibility) -> bool:
    return visibility in visibility_options(role)


def can_administer_document(
    *,
    role: Role,
    actor_id: UUID | None,
    actor_department_path: str | None,
    owner_id: UUID | None,
    document_department_path: str | None,
) -> bool:
    """Whether the actor may edit, replace, reindex, or delete this document.

    Scope by role: an admin anywhere, a manager within their department subtree, an
    internal employee only their own uploads. Note this is *write* authority and is
    checked independently of read authority — being able to read a document has never
    implied being able to replace it.
    """
    match role:
        case Role.ADMIN:
            return True
        case Role.MANAGER:
            if (
                actor_department_path
                and document_department_path
                and is_in_subtree(document_department_path, actor_department_path)
            ):
                return True
            return owner_id is not None and owner_id == actor_id
        case Role.INTERNAL_EMPLOYEE:
            return owner_id is not None and owner_id == actor_id
        case _:
            return False
