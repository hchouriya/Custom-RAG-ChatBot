"""Translating the domain filter algebra into each backend's dialect.

This module is security-critical. The ACL is expressed once, in the domain, as a
:class:`VectorFilter`; if a translation drops a clause the result is not a broken query but a
query that returns documents the principal may not read. Three rules follow from that:

1. **Fail closed on anything unrecognised.** An unmappable field or condition raises. The
   tempting alternative — skip the clause and carry on — converts an access control failure
   into a silent data leak.
2. **A missing payload value never satisfies a predicate.** Absence is not permission.
3. **Prefix matching is segment-aware.** ``company.hr`` must not match
   ``company.hr_contractors``. Postgres gets ``ltree`` containment; Qdrant gets an ancestor
   list materialised at index time, because Qdrant's text matching is tokenised substring
   search and would match the wrong things.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aegis.core.errors import VectorStoreError
from aegis.domain.values import (
    And,
    Condition,
    HasAny,
    In,
    IsNull,
    Match,
    Or,
    PrefixMatch,
    Range,
    VectorFilter,
    is_in_subtree,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

PATH_SEPARATOR = "."


def ancestor_paths(path: str | None) -> tuple[str, ...]:
    """Every prefix of a department path, including the path itself.

    ``"company.eng.platform"`` becomes ``("company", "company.eng", "company.eng.platform")``.

    Materialising this at index time is what makes subtree containment an exact keyword match
    in engines with no hierarchical type. The alternative, substring matching on the full path,
    is both slower and wrong: it matches across segment boundaries, and a filter that
    over-matches on a department path is a cross-department data leak.
    """
    if not path:
        return ()
    segments = [s for s in path.split(PATH_SEPARATOR) if s]
    return tuple(PATH_SEPARATOR.join(segments[: index + 1]) for index in range(len(segments)))


# ── In-memory evaluation ────────────────────────────────────────────────────


def matches(payload: Mapping[str, Any], vfilter: VectorFilter) -> bool:
    """Evaluate a filter against a payload.

    The reference implementation of the algebra's semantics. The in-memory store uses it
    directly, and the backend translations are tested against it so that "the same filter means
    the same thing in Qdrant, in Postgres, and in a unit test" is a property with tests behind
    it rather than an aspiration.
    """
    if not all(_holds(payload, c) for c in vfilter.must):
        return False
    if any(_holds(payload, c) for c in vfilter.must_not):
        return False
    if vfilter.should:
        satisfied = sum(1 for c in vfilter.should if _holds(payload, c))
        if satisfied < max(1, vfilter.min_should):
            return False
    return True


def _holds(payload: Mapping[str, Any], condition: Condition) -> bool:
    match condition:
        case Match(field=f, value=v):
            return _equals(payload.get(f), v)
        case In(field=f, values=vs):
            actual = payload.get(f)
            if actual is None:
                return False
            if isinstance(actual, list | tuple | set):
                return bool({_key(a) for a in actual} & {_key(v) for v in vs})
            return _key(actual) in {_key(v) for v in vs}
        case Range(field=f, gt=gt, gte=gte, lt=lt, lte=lte):
            value = _number(payload.get(f))
            if value is None:
                return False
            if gt is not None and not value > gt:
                return False
            if gte is not None and not value >= gte:
                return False
            if lt is not None and not value < lt:
                return False
            return not (lte is not None and not value <= lte)
        case PrefixMatch(field=f, prefix=p):
            return _path_matches(payload.get(f), p)
        case IsNull(field=f):
            return payload.get(f) is None
        case HasAny(field=f, values=vs):
            actual = payload.get(f) or []
            if isinstance(actual, str):
                actual = [actual]
            return bool({_key(a) for a in actual} & {_key(v) for v in vs})
        case Or(clauses=clauses):
            return any(_holds(payload, c) for c in clauses)
        case And(clauses=clauses):
            return all(_holds(payload, c) for c in clauses)
        case _:
            raise VectorStoreError(f"unsupported filter condition: {type(condition).__name__}")


def _equals(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    # A materialised ancestor list satisfies equality on any of its members, which is how
    # subtree containment reduces to an exact match. See `ancestor_paths`.
    if isinstance(actual, list | tuple | set):
        return any(_key(a) == _key(expected) for a in actual)
    return bool(_key(actual) == _key(expected))


def _key(value: Any) -> Any:
    """Normalise for comparison: UUIDs and enums arrive as objects or as their strings."""
    if isinstance(value, bool | int | float):
        return value
    return str(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _path_matches(actual: Any, prefix: str) -> bool:
    if actual is None:
        return False
    if isinstance(actual, list | tuple | set):
        return any(str(a) == prefix for a in actual)
    return is_in_subtree(str(actual), prefix)


# ── Qdrant ──────────────────────────────────────────────────────────────────


def to_qdrant(vfilter: VectorFilter) -> Any:
    """Build a ``qdrant_client.models.Filter``.

    Imported lazily so that a deployment on pgvector does not pay for the Qdrant client, and so
    this module stays importable in tests that have no vector database at all.
    """
    from qdrant_client import models as qm

    should = [_qdrant_condition(c) for c in vfilter.should]
    return qm.Filter(
        must=[_qdrant_condition(c) for c in vfilter.must] or None,
        should=should or None,
        must_not=[_qdrant_condition(c) for c in vfilter.must_not] or None,
        # Without this, Qdrant treats `should` as a score boost rather than a constraint,
        # which would turn the ACL's "at least one of these ways to qualify" into "rank these
        # higher" — an access control bypass, not a ranking bug.
        min_should=(
            qm.MinShould(conditions=should, min_count=max(1, vfilter.min_should))
            if should and vfilter.min_should
            else None
        ),
    )


def _qdrant_condition(condition: Condition) -> Any:
    from qdrant_client import models as qm

    match condition:
        case Match(field=f, value=v):
            return qm.FieldCondition(key=f, match=qm.MatchValue(value=v))
        case In(field=f, values=vs):
            return qm.FieldCondition(key=f, match=qm.MatchAny(any=list(vs)))
        case Range(field=f, gt=gt, gte=gte, lt=lt, lte=lte):
            return qm.FieldCondition(key=f, range=qm.Range(gt=gt, gte=gte, lt=lt, lte=lte))
        case PrefixMatch(field=f, prefix=p):
            # Exact match against the materialised ancestor list, not a text prefix search:
            # Qdrant's MatchText is tokenised and would match across segment boundaries.
            return qm.FieldCondition(key=f, match=qm.MatchValue(value=p))
        case IsNull(field=f):
            return qm.IsNullCondition(is_null=qm.PayloadField(key=f))
        case HasAny(field=f, values=vs):
            return qm.FieldCondition(key=f, match=qm.MatchAny(any=list(vs)))
        case Or(clauses=clauses):
            inner = [_qdrant_condition(c) for c in clauses]
            return qm.Filter(
                should=inner,
                min_should=qm.MinShould(conditions=inner, min_count=1),
            )
        case And(clauses=clauses):
            return qm.Filter(must=[_qdrant_condition(c) for c in clauses])
        case _:
            raise VectorStoreError(f"unsupported filter condition: {type(condition).__name__}")


# ── SQL (pgvector) ──────────────────────────────────────────────────────────

PAYLOAD_COLUMNS: dict[str, str] = {
    "visibility_level": "c.visibility_level",
    "collection_id": "c.collection_id",
    "document_id": "c.document_id",
    "version_id": "c.version_id",
    "chunk_type": "c.chunk_type::text",
    "language": "c.language",
    "injection_flag": "c.injection_flag",
    "mode": "col.mode::text",
    "department_path": "c.department_path",
    "expires_at": "EXTRACT(EPOCH FROM d.expires_at)",
    "effective_from": "EXTRACT(EPOCH FROM d.effective_from)",
}
"""Payload key to SQL expression, over the join in :mod:`aegis.rag.vector_stores.pgvector`.

Only these keys are translatable. Anything else raises rather than being ignored, so adding a
payload field to the ACL without teaching this map about it fails loudly at the first query
instead of quietly returning unfiltered rows.
"""

BOOLEAN_EXPRESSIONS: dict[str, str] = {
    # "Active" is a property of the whole chain, not a column: the collection is enabled, the
    # document is neither soft-deleted nor archived, and this chunk belongs to the version the
    # document currently points at. Indexing a superseded version's chunks and retrieving them
    # is the classic stale-answer bug, and this clause is what prevents it.
    "is_active": (
        "(col.is_active AND d.deleted_at IS NULL AND NOT d.is_archived "
        "AND d.active_version_id = c.version_id)"
    ),
}

TAGS_EXISTS = (
    "EXISTS (SELECT 1 FROM document_tags dt JOIN tags t ON t.id = dt.tag_id "
    "WHERE dt.document_id = c.document_id AND t.name = ANY(:{param}))"
)


@dataclass(slots=True)
class SqlFragment:
    """A WHERE fragment and its bind parameters."""

    sql: str
    params: dict[str, Any] = field(default_factory=dict)


class _SqlBuilder:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._counter = 0
        self.params: dict[str, Any] = {}

    def bind(self, value: Any) -> str:
        self._counter += 1
        name = f"{self._prefix}{self._counter}"
        self.params[name] = value
        return name

    def condition(self, condition: Condition) -> str:
        match condition:
            case Match(field="is_active", value=v):
                expression = BOOLEAN_EXPRESSIONS["is_active"]
                return expression if v else f"NOT {expression}"
            case Match(field=f, value=v):
                return f"{self.column(f)} = {self._typed(f, v)}"
            case In(field=f, values=vs):
                if not vs:
                    # An empty IN matches nothing. Emitting `IN ()` is a syntax error and
                    # omitting the clause would match everything, which is the dangerous one.
                    return "FALSE"
                return f"{self.column(f)} = ANY(:{self.bind(self._values(f, vs))})"
            case Range(field=f, gt=gt, gte=gte, lt=lt, lte=lte):
                column = self.column(f)
                parts = [
                    f"{column} {operator} :{self.bind(float(bound))}"
                    for operator, bound in ((">", gt), (">=", gte), ("<", lt), ("<=", lte))
                    if bound is not None
                ]
                return f"({' AND '.join(parts)})" if parts else "TRUE"
            case PrefixMatch(field=f, prefix=p):
                # ltree containment: exact subtree semantics, index-backed by the GiST index
                # on `department_path`.
                return f"{self.column(f)} <@ CAST(:{self.bind(p)} AS ltree)"
            case IsNull(field=f):
                return f"{self.column(f)} IS NULL"
            case HasAny(field="tags", values=vs):
                return TAGS_EXISTS.format(param=self.bind(list(vs)))
            case HasAny(field=f, values=vs):
                return f"{self.column(f)} && :{self.bind(list(vs))}"
            case Or(clauses=clauses):
                return f"({' OR '.join(self.condition(c) for c in clauses)})" if clauses else "TRUE"
            case And(clauses=clauses):
                return (
                    f"({' AND '.join(self.condition(c) for c in clauses)})" if clauses else "TRUE"
                )
            case _:
                raise VectorStoreError(f"unsupported filter condition: {type(condition).__name__}")

    def column(self, name: str) -> str:
        try:
            return PAYLOAD_COLUMNS[name]
        except KeyError:
            raise VectorStoreError(
                f"payload field {name!r} has no SQL mapping; refusing to run a query with a "
                "dropped filter clause"
            ) from None

    def _typed(self, name: str, value: Any) -> str:
        cast = "::uuid" if name.endswith("_id") else ""
        return f":{self.bind(str(value) if cast else value)}{cast}"

    def _values(self, name: str, values: tuple[Any, ...]) -> list[Any]:
        if name.endswith("_id"):
            return [str(v) for v in values]
        return list(values)


def to_sql(vfilter: VectorFilter, *, prefix: str = "f") -> SqlFragment:
    """Render a filter as a SQL boolean expression with bind parameters.

    Parameterised throughout: department paths and tag names come from user-controlled fields,
    and string interpolation here would be an injection point in the one code path that must
    not have one.
    """
    builder = _SqlBuilder(prefix)
    clauses = [builder.condition(c) for c in vfilter.must]
    clauses.extend(f"NOT {builder.condition(c)}" for c in vfilter.must_not)

    if vfilter.should:
        rendered = [builder.condition(c) for c in vfilter.should]
        minimum = max(1, vfilter.min_should)
        if minimum == 1:
            clauses.append(f"({' OR '.join(rendered)})")
        else:
            # Count satisfied branches rather than expanding combinations: the expansion is
            # exponential and this stays linear.
            summed = " + ".join(f"CASE WHEN {r} THEN 1 ELSE 0 END" for r in rendered)
            clauses.append(f"(({summed}) >= {minimum})")

    return SqlFragment(sql=" AND ".join(clauses) if clauses else "TRUE", params=builder.params)
