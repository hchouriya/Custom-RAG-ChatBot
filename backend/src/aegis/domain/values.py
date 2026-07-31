"""Value objects: immutable, behaviour-carrying, and free of I/O.

The two that carry real weight:

:class:`SecurityContext`
    Everything the system knows about *who is asking*, derived server-side. It is the
    only input to ACL filter construction. No field of an HTTP request reaches it
    except through validated narrowing.

:class:`VectorFilter`
    A small filter algebra, translated per backend in ``rag.vector_stores.filters``.
    A domain-owned representation is what lets the ACL policy be tested exhaustively
    without a vector database, and what keeps a Qdrant type out of the security-critical
    code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from aegis.domain.enums import (
    AnswerStatus,
    ChunkType,
    Mode,
    Role,
    Visibility,
    VisibilityLevel,
)

# ── Filter algebra ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Match:
    """``field == value``"""

    field: str
    value: str | int | bool | float


@dataclass(frozen=True, slots=True)
class In:
    """``field IN values``. An empty value list matches nothing, never everything."""

    field: str
    values: tuple[str | int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True, slots=True)
class Range:
    """Half-open or closed numeric/temporal range. All bounds optional."""

    field: str
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """``field`` starts with ``prefix`` — department subtree containment."""

    field: str
    prefix: str


@dataclass(frozen=True, slots=True)
class IsNull:
    field: str


@dataclass(frozen=True, slots=True)
class HasAny:
    """Array field intersects the given values (tags)."""

    field: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True, slots=True)
class Or:
    clauses: tuple[Condition, ...]


@dataclass(frozen=True, slots=True)
class And:
    clauses: tuple[Condition, ...]


Condition = Match | In | Range | PrefixMatch | IsNull | HasAny | Or | And


@dataclass(frozen=True, slots=True)
class VectorFilter:
    """Conjunction of ``must``, disjunction of ``should``, negation of ``must_not``.

    ``min_should`` is required rather than implicit: with ``should`` clauses present and
    no minimum, most backends treat them as score boosts instead of constraints, which
    would turn an ACL restriction into a ranking preference. That distinction is the
    difference between a filter and a suggestion.
    """

    must: tuple[Condition, ...] = ()
    should: tuple[Condition, ...] = ()
    must_not: tuple[Condition, ...] = ()
    min_should: int = 1

    def intersect(self, other: VectorFilter | None) -> VectorFilter:
        """Combine with another filter so the result is never *wider* than either.

        A nested ``should`` group is folded into a single ``Or`` inside ``must`` rather
        than merged flat: merging two ``should`` lists would let a clause from one
        satisfy the minimum for the other, quietly widening access.
        """
        if other is None:
            return self
        must = list(self.must) + list(other.must)
        if other.should:
            must.append(Or(tuple(other.should)))
        if self.should:
            must.append(Or(tuple(self.should)))
        return VectorFilter(
            must=tuple(must),
            should=(),
            must_not=tuple(self.must_not) + tuple(other.must_not),
            min_should=0,
        )

    def with_must(self, *conditions: Condition) -> VectorFilter:
        return replace(self, must=tuple(self.must) + conditions)

    def describe(self) -> dict[str, Any]:
        """Serializable form, stored on ``query_traces.filter_applied``.

        Recording the exact filter is what makes "why could this user see that?"
        answerable after the fact instead of a reconstruction exercise.
        """

        def render(c: Condition) -> Any:
            match c:
                case Match(f, v):
                    return {"eq": {f: v}}
                case In(f, vs):
                    return {"in": {f: list(vs)[:20], "count": len(vs)}}
                case Range(f, gt, gte, lt, lte):
                    return {
                        "range": {
                            f: {
                                k: v
                                for k, v in (("gt", gt), ("gte", gte), ("lt", lt), ("lte", lte))
                                if v is not None
                            }
                        }
                    }
                case PrefixMatch(f, p):
                    return {"prefix": {f: p}}
                case IsNull(f):
                    return {"is_null": f}
                case HasAny(f, vs):
                    return {"has_any": {f: list(vs)}}
                case Or(clauses):
                    return {"or": [render(x) for x in clauses]}
                case And(clauses):
                    return {"and": [render(x) for x in clauses]}

        return {
            "must": [render(c) for c in self.must],
            "should": [render(c) for c in self.should],
            "must_not": [render(c) for c in self.must_not],
            "min_should": self.min_should,
        }


# ── Security ────────────────────────────────────────────────────────────────

MAX_EXPLICIT_GRANTS = 5000
"""Beyond this, an ACL should be expressed as a role or department grant.

Truncating silently would produce answers that look correct while missing documents the
user is entitled to, so the service raises instead and says so.
"""


@dataclass(frozen=True, slots=True)
class SecurityContext:
    """Who is asking, resolved server-side.

    ``ceiling`` is ``min(role ceiling, mode ceiling)``. Because customer mode caps at
    :attr:`VisibilityLevel.CUSTOMER` for every role including admin, "test the customer
    assistant with my own account" is a safe operation rather than a leak.
    """

    user_id: UUID | None
    role: Role
    mode: Mode
    ceiling: VisibilityLevel
    department_id: UUID | None = None
    department_path: str | None = None
    granted_document_ids: tuple[UUID, ...] = ()
    collection_ids: tuple[UUID, ...] = ()
    session_id: UUID | None = None
    permission_epoch: int = 0
    is_guest: bool = False

    @property
    def principal_key(self) -> str:
        """Stable identity for rate limiting and cache keys."""
        if self.user_id is not None:
            return f"user:{self.user_id}"
        return f"guest:{self.session_id}" if self.session_id else "anonymous"

    def can_read(self, visibility: Visibility) -> bool:
        """Level-only check.

        Deliberately *not* the whole answer: confidential additionally requires
        department containment and restricted requires an explicit grant, both of which
        need the document. Use :meth:`can_read_document` when you have one.
        """
        return visibility.level <= self.ceiling

    def can_read_document(
        self,
        *,
        document_id: UUID,
        visibility: Visibility,
        department_path: str | None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> bool:
        """The complete read decision for one document.

        This is the reference implementation of invariant I1 and the assertion used by
        the post-retrieval re-verification layer.
        """
        if expires_at is not None:
            moment = now or datetime.now(tz=expires_at.tzinfo)
            if expires_at <= moment:
                return False

        if document_id in self.granted_document_ids:
            return True

        if visibility.level > self.ceiling:
            return False

        if visibility.requires_explicit_grant:
            # Level 4 is reachable by ceiling only for admin; everyone else needs a grant,
            # which was checked above.
            return self.ceiling >= VisibilityLevel.RESTRICTED

        if visibility.requires_department_match:
            if self.department_path is None or department_path is None:
                return False
            return is_in_subtree(department_path, self.department_path)

        return True

    def narrowed_to(self, mode: Mode, ceiling: VisibilityLevel) -> Self:
        """Return a copy that can only be *less* permissive."""
        return replace(self, mode=mode, ceiling=min(self.ceiling, ceiling))


def is_in_subtree(document_path: str, principal_path: str) -> bool:
    """True when ``document_path`` is at or below ``principal_path``.

    Segment-aware so that ``company.hr`` does not match ``company.hr_contractors`` —
    a plain ``startswith`` here would silently widen access across departments whose
    names share a prefix.
    """
    if document_path == principal_path:
        return True
    return document_path.startswith(principal_path + ".")


# ── Retrieval ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ChunkLocator:
    """Everything needed to render a citation and to highlight its source."""

    document_id: UUID
    version_id: UUID
    document_title: str = ""
    version_no: int = 1
    page_from: int | None = None
    page_to: int | None = None
    heading_path: tuple[str, ...] = ()
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    @property
    def page(self) -> int | None:
        return self.page_from

    def display(self) -> str:
        parts = [self.document_title or "Untitled"]
        if self.version_no > 1:
            parts.append(f"v{self.version_no}")
        if self.page_from:
            span = (
                f"p.{self.page_from}"
                if not self.page_to or self.page_to == self.page_from
                else f"pp.{self.page_from}-{self.page_to}"
            )
            parts.append(span)
        if self.section:
            parts.append(f"§{self.section}")
        elif self.heading_path:
            parts.append(self.heading_path[-1])
        return " · ".join(parts)


@dataclass(slots=True)
class RetrievedChunk:
    """A candidate travelling through the retrieval pipeline.

    Mutable, unlike most value objects here, because scores are attached at successive
    stages (dense, sparse, fused, rerank) and copying the object at every stage would
    obscure that the *same* candidate is being re-scored.
    """

    chunk_id: UUID
    content: str
    locator: ChunkLocator
    chunk_type: ChunkType = ChunkType.TEXT
    token_count: int = 0
    visibility: Visibility = Visibility.INTERNAL
    department_path: str | None = None
    collection_id: UUID | None = None
    content_hash: bytes | None = None
    injection_flag: bool = False
    score_dense: float | None = None
    score_sparse: float | None = None
    score_fused: float | None = None
    score_rerank: float | None = None
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> float:
        """Most trustworthy available score.

        Rerank beats fused beats dense: each later stage strictly dominates the earlier
        one in ranking quality, so the confidence gate should read the best it has.
        """
        for score in (self.score_rerank, self.score_fused, self.score_dense, self.score_sparse):
            if score is not None:
                return score
        return 0.0


@dataclass(frozen=True, slots=True)
class Citation:
    """A validated pointer from an answer marker to a source span."""

    marker: int
    chunk_id: UUID
    locator: ChunkLocator
    quote: str
    quote_start: int | None = None
    quote_end: int | None = None
    score_rerank: float | None = None
    rank: int = 0
    was_used: bool = True


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Allocation of the prompt window across its segments."""

    system: int
    mode_rules: int
    summary: int
    history: int
    context: int
    question: int
    completion_reserve: int

    @property
    def total_prompt(self) -> int:
        return (
            self.system
            + self.mode_rules
            + self.summary
            + self.history
            + self.context
            + self.question
        )

    def fits(self, cap: int) -> bool:
        return self.total_prompt + self.completion_reserve <= cap


@dataclass(frozen=True, slots=True)
class Confidence:
    """Outcome of the pre-generation gate."""

    decision: Literal["answer", "clarify", "refuse"]
    score: float
    top_score: float
    supporting_chunks: int
    mean_top3: float
    entity_coverage: float
    reason: str = ""

    @property
    def answer_status(self) -> AnswerStatus:
        return {
            "answer": AnswerStatus.OK,
            "clarify": AnswerStatus.CLARIFY,
            "refuse": AnswerStatus.NO_ANSWER,
        }[self.decision]


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """A dense vector with the model that produced it.

    The model travels with the vector because comparing vectors from two models is
    meaningless, and a provider that silently changes dimension is a corruption event
    rather than a warning.
    """

    values: tuple[float, ...]
    model: str

    @property
    def dim(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Term-weighted sparse representation for keyword retrieval."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse vector indices and values must be the same length")

    @property
    def nnz(self) -> int:
        return len(self.indices)
