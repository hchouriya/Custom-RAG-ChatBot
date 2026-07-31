"""Vector store semantics, filter translation, and fusion.

The filter tests are security tests. They assert the two properties everything else depends on:
an ACL filter built by the domain excludes what it should in every backend's dialect, and a
clause that cannot be translated raises instead of being dropped.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from aegis.core.errors import VectorStoreError
from aegis.domain.enums import Mode, Role, VisibilityLevel
from aegis.domain.policies.acl import build_filter, build_security_context
from aegis.domain.ports.vector_store import CollectionSpec, VectorHit, VectorPoint
from aegis.domain.values import (
    And,
    EmbeddingVector,
    HasAny,
    In,
    IsNull,
    Match,
    Or,
    PrefixMatch,
    Range,
    SparseVector,
    VectorFilter,
)
from aegis.rag.vector_stores import (
    InMemoryVectorStore,
    acl_payload,
    ancestor_paths,
    build_payload,
    matches,
    reciprocal_rank_fusion,
    to_sql,
)

NAMESPACE = "test_ns"
DIM = 4

COLLECTION = uuid.uuid4()
DEPARTMENT = "company.eng.platform"


def _dense(*values: float) -> EmbeddingVector:
    padded = list(values) + [0.0] * (DIM - len(values))
    return EmbeddingVector(values=tuple(padded[:DIM]), model="test")


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "version_id": uuid.uuid4(),
        "collection_id": COLLECTION,
        "mode": Mode.INTERNAL.value,
        "visibility_level": int(VisibilityLevel.INTERNAL),
        "department_path": DEPARTMENT,
        "is_active": True,
    }
    base.update(overrides)
    return build_payload(**base)


def _point(
    payload: dict[str, Any],
    dense: EmbeddingVector | None = None,
    sparse: SparseVector | None = None,
) -> VectorPoint:
    chunk_id = uuid.UUID(payload["chunk_id"])
    return VectorPoint(
        id=uuid.uuid4(),
        chunk_id=chunk_id,
        dense=dense or _dense(1.0),
        sparse=sparse,
        payload=payload,
    )


class TestAncestorPaths:
    def test_expands_every_prefix(self) -> None:
        assert ancestor_paths("company.eng.platform") == (
            "company",
            "company.eng",
            "company.eng.platform",
        )

    def test_none_and_empty_are_empty(self) -> None:
        assert ancestor_paths(None) == ()
        assert ancestor_paths("") == ()


class TestPayload:
    def test_dates_become_epoch_seconds(self) -> None:
        payload = _payload(expires_at=date(2030, 1, 1))
        assert payload["expires_at"] == datetime(2030, 1, 1, tzinfo=UTC).timestamp()

    def test_absent_expiry_is_null_not_zero(self) -> None:
        """Zero would be 1970, which makes every unexpiring document expired."""
        assert _payload()["expires_at"] is None

    def test_department_is_stored_as_its_ancestors(self) -> None:
        assert _payload()["department_path"] == ["company", "company.eng", "company.eng.platform"]

    def test_acl_patch_covers_every_acl_key(self) -> None:
        from aegis.rag.vector_stores import ACL_KEYS

        patch = acl_payload(
            visibility_level=2, department_path="company", is_active=False, mode="internal"
        )
        assert set(patch) == set(ACL_KEYS)


class TestFilterSemantics:
    def test_missing_field_never_satisfies(self) -> None:
        """Absence is not permission."""
        assert not matches({}, VectorFilter(must=(Match("mode", "internal"),)))
        assert not matches({}, VectorFilter(must=(Range("visibility_level", lte=2),)))
        assert not matches({}, VectorFilter(must=(PrefixMatch("department_path", "company"),)))

    def test_empty_in_matches_nothing(self) -> None:
        assert not matches({"document_id": "x"}, VectorFilter(must=(In("document_id", ()),)))

    def test_should_requires_the_minimum(self) -> None:
        payload = {"a": 1, "b": 2}
        one_of = VectorFilter(should=(Match("a", 1), Match("b", 99)), min_should=1)
        both = VectorFilter(should=(Match("a", 1), Match("b", 99)), min_should=2)
        assert matches(payload, one_of)
        assert not matches(payload, both)

    def test_prefix_is_segment_aware(self) -> None:
        assert matches(
            {"department_path": ["company", "company.hr"]},
            VectorFilter(must=(PrefixMatch("department_path", "company.hr"),)),
        )
        assert not matches(
            {"department_path": list(ancestor_paths("company.hr_contractors"))},
            VectorFilter(must=(PrefixMatch("department_path", "company.hr"),)),
        )

    def test_is_null_and_range_alternative(self) -> None:
        never = VectorFilter(must=(Or((IsNull("expires_at"), Range("expires_at", gt=100.0))),))
        assert matches({"expires_at": None}, never)
        assert matches({"expires_at": 200.0}, never)
        assert not matches({"expires_at": 50.0}, never)

    def test_has_any_intersects_arrays(self) -> None:
        assert matches(
            {"tags": ["hr", "policy"]}, VectorFilter(must=(HasAny("tags", ("policy",)),))
        )
        assert not matches({"tags": ["hr"]}, VectorFilter(must=(HasAny("tags", ("policy",)),)))

    def test_must_not_excludes(self) -> None:
        f = VectorFilter(must_not=(Match("injection_flag", True),))
        assert matches({"injection_flag": False}, f)
        assert not matches({"injection_flag": True}, f)


class TestSqlTranslation:
    def test_unmappable_field_raises_rather_than_dropping_the_clause(self) -> None:
        with pytest.raises(VectorStoreError, match="no SQL mapping"):
            to_sql(VectorFilter(must=(Match("invented_field", 1),)))

    def test_values_are_parameterised(self) -> None:
        fragment = to_sql(VectorFilter(must=(PrefixMatch("department_path", "company.hr"),)))
        assert "company.hr" not in fragment.sql
        assert "company.hr" in fragment.params.values()

    def test_prefix_uses_ltree_containment(self) -> None:
        fragment = to_sql(VectorFilter(must=(PrefixMatch("department_path", "company"),)))
        assert "<@ CAST(" in fragment.sql

    def test_is_active_expands_to_the_whole_chain(self) -> None:
        fragment = to_sql(VectorFilter(must=(Match("is_active", True),)))
        assert "d.active_version_id = c.version_id" in fragment.sql
        assert "d.deleted_at IS NULL" in fragment.sql

    def test_empty_in_is_false_not_omitted(self) -> None:
        fragment = to_sql(VectorFilter(must=(In("document_id", ()),)))
        assert fragment.sql == "FALSE"

    def test_tags_become_an_exists_subquery(self) -> None:
        fragment = to_sql(VectorFilter(must=(HasAny("tags", ("policy",)),)))
        assert "EXISTS (SELECT 1 FROM document_tags" in fragment.sql

    def test_should_group_becomes_a_disjunction(self) -> None:
        fragment = to_sql(
            VectorFilter(
                must=(Match("mode", "internal"),),
                should=(Match("visibility_level", 1), Match("visibility_level", 2)),
                min_should=1,
            )
        )
        assert " OR " in fragment.sql

    def test_min_should_above_one_counts_branches(self) -> None:
        fragment = to_sql(
            VectorFilter(
                should=(Match("visibility_level", 1), Match("mode", "internal")), min_should=2
            )
        )
        assert "CASE WHEN" in fragment.sql
        assert ">= 2" in fragment.sql

    def test_real_acl_filter_translates(self) -> None:
        ctx = build_security_context(
            user_id=uuid.uuid4(),
            role=Role.MANAGER,
            requested_mode=Mode.INTERNAL,
            department_path=DEPARTMENT,
            collection_ids=(COLLECTION,),
        )
        fragment = to_sql(build_filter(ctx))
        assert "col.mode::text" in fragment.sql
        assert "c.visibility_level" in fragment.sql
        assert fragment.params


class TestAclEnforcement:
    """The same ACL filter, evaluated against the same payloads, in the reference engine."""

    async def _store(self) -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        await store.ensure_collection(CollectionSpec(namespace=NAMESPACE, dim=DIM))
        await store.upsert(
            NAMESPACE,
            [
                _point(_payload(visibility_level=int(VisibilityLevel.PUBLIC))),
                _point(_payload(visibility_level=int(VisibilityLevel.CUSTOMER))),
                _point(_payload(visibility_level=int(VisibilityLevel.INTERNAL))),
                _point(
                    _payload(
                        visibility_level=int(VisibilityLevel.CONFIDENTIAL),
                        department_path="company.finance",
                    )
                ),
                _point(
                    _payload(
                        visibility_level=int(VisibilityLevel.CONFIDENTIAL),
                        department_path=DEPARTMENT,
                    )
                ),
                _point(_payload(visibility_level=int(VisibilityLevel.RESTRICTED))),
                _point(
                    _payload(
                        mode=Mode.CUSTOMER.value,
                        visibility_level=int(VisibilityLevel.PUBLIC),
                    )
                ),
                _point(
                    _payload(
                        mode=Mode.CUSTOMER.value,
                        visibility_level=int(VisibilityLevel.CUSTOMER),
                    )
                ),
                _point(
                    _payload(
                        mode=Mode.CUSTOMER.value,
                        visibility_level=int(VisibilityLevel.INTERNAL),
                    )
                ),
                _point(_payload(is_active=False)),
                _point(_payload(expires_at=date(2000, 1, 1))),
            ],
        )
        return store

    async def _visible(self, role: Role, mode: Mode, **kwargs: Any) -> list[int]:
        store = await self._store()
        ctx = build_security_context(role=role, requested_mode=mode, user_id=uuid.uuid4(), **kwargs)
        hits = await store.search_dense(NAMESPACE, _dense(1.0), limit=50, vfilter=build_filter(ctx))
        return sorted(h.payload["visibility_level"] for h in hits)

    async def test_employee_sees_up_to_internal(self) -> None:
        levels = await self._visible(
            Role.INTERNAL_EMPLOYEE, Mode.INTERNAL, department_path=DEPARTMENT
        )
        assert levels == [0, 1, 2]

    async def test_manager_sees_confidential_only_in_their_subtree(self) -> None:
        levels = await self._visible(Role.MANAGER, Mode.INTERNAL, department_path=DEPARTMENT)
        assert levels == [0, 1, 2, 3]

    async def test_manager_without_department_sees_no_confidential(self) -> None:
        levels = await self._visible(Role.MANAGER, Mode.INTERNAL)
        assert 3 not in levels

    async def test_admin_sees_restricted(self) -> None:
        levels = await self._visible(Role.ADMIN, Mode.INTERNAL, department_path=DEPARTMENT)
        assert 4 in levels

    async def test_customer_mode_caps_at_customer_even_for_admins(self) -> None:
        store = await self._store()
        ctx = build_security_context(
            user_id=uuid.uuid4(), role=Role.ADMIN, requested_mode=Mode.CUSTOMER
        )
        hits = await store.search_dense(NAMESPACE, _dense(1.0), limit=50, vfilter=build_filter(ctx))
        assert sorted(h.payload["visibility_level"] for h in hits) == [0, 1]
        assert all(h.payload["mode"] == Mode.CUSTOMER.value for h in hits)

    async def test_guest_sees_public_only(self) -> None:
        store = await self._store()
        ctx = build_security_context(
            user_id=None, role=Role.GUEST, requested_mode=Mode.CUSTOMER, is_guest=True
        )
        hits = await store.search_dense(NAMESPACE, _dense(1.0), limit=50, vfilter=build_filter(ctx))
        assert [h.payload["visibility_level"] for h in hits] == [0]

    async def test_expired_and_inactive_are_never_returned(self) -> None:
        store = await self._store()
        ctx = build_security_context(
            user_id=uuid.uuid4(),
            role=Role.ADMIN,
            requested_mode=Mode.INTERNAL,
            department_path=DEPARTMENT,
        )
        hits = await store.search_dense(NAMESPACE, _dense(1.0), limit=50, vfilter=build_filter(ctx))
        assert all(h.payload["is_active"] for h in hits)
        assert all(
            h.payload["expires_at"] is None
            or h.payload["expires_at"] > datetime.now(UTC).timestamp()
            for h in hits
        )

    async def test_explicit_grant_crosses_departments_but_not_mode(self) -> None:
        store = InMemoryVectorStore()
        await store.ensure_collection(CollectionSpec(namespace=NAMESPACE, dim=DIM))
        granted = uuid.uuid4()
        foreign = _payload(
            document_id=granted,
            visibility_level=int(VisibilityLevel.CONFIDENTIAL),
            department_path="company.finance",
        )
        same_grant_wrong_mode = _payload(
            document_id=granted,
            visibility_level=int(VisibilityLevel.CONFIDENTIAL),
            department_path="company.finance",
            mode=Mode.CUSTOMER.value,
        )
        await store.upsert(NAMESPACE, [_point(foreign), _point(same_grant_wrong_mode)])

        ctx = build_security_context(
            user_id=uuid.uuid4(),
            role=Role.MANAGER,
            requested_mode=Mode.INTERNAL,
            department_path=DEPARTMENT,
            granted_document_ids=(granted,),
        )
        hits = await store.search_dense(NAMESPACE, _dense(1.0), limit=10, vfilter=build_filter(ctx))
        assert len(hits) == 1
        assert hits[0].payload["mode"] == Mode.INTERNAL.value

    async def test_a_grant_cannot_lift_a_document_above_the_ceiling(self) -> None:
        """The ceiling sits in ``must``, so no ``should`` branch can widen past it."""
        store = InMemoryVectorStore()
        await store.ensure_collection(CollectionSpec(namespace=NAMESPACE, dim=DIM))
        granted = uuid.uuid4()
        await store.upsert(
            NAMESPACE,
            [
                _point(
                    _payload(document_id=granted, visibility_level=int(VisibilityLevel.RESTRICTED))
                )
            ],
        )
        ctx = build_security_context(
            user_id=uuid.uuid4(),
            role=Role.INTERNAL_EMPLOYEE,
            requested_mode=Mode.INTERNAL,
            granted_document_ids=(granted,),
        )
        assert await store.count(NAMESPACE, build_filter(ctx)) == 0


class TestInMemoryStore:
    async def _store(self) -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        await store.ensure_collection(CollectionSpec(namespace=NAMESPACE, dim=DIM))
        return store

    async def test_upsert_is_idempotent_on_point_id(self) -> None:
        store = await self._store()
        point = _point(_payload())
        await store.upsert(NAMESPACE, [point])
        await store.upsert(NAMESPACE, [point])
        assert await store.count(NAMESPACE) == 1

    async def test_dimension_mismatch_is_rejected(self) -> None:
        store = await self._store()
        wrong = VectorPoint(
            id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            dense=EmbeddingVector(values=(1.0, 0.0), model="test"),
        )
        with pytest.raises(VectorStoreError, match="dimension"):
            await store.upsert(NAMESPACE, [wrong])

    async def test_dense_search_ranks_by_similarity(self) -> None:
        store = await self._store()
        near = _point(_payload(), dense=_dense(1.0, 0.1))
        far = _point(_payload(), dense=_dense(0.0, 1.0))
        await store.upsert(NAMESPACE, [far, near])
        hits = await store.search_dense(NAMESPACE, _dense(1.0), limit=10, vfilter=VectorFilter())
        assert hits[0].chunk_id == near.chunk_id

    async def test_score_threshold_excludes_weak_matches(self) -> None:
        store = await self._store()
        await store.upsert(NAMESPACE, [_point(_payload(), dense=_dense(0.0, 1.0))])
        hits = await store.search_dense(
            NAMESPACE, _dense(1.0), limit=10, vfilter=VectorFilter(), score_threshold=0.5
        )
        assert hits == []

    async def test_sparse_search_uses_term_overlap(self) -> None:
        store = await self._store()
        matching = _point(_payload(), sparse=SparseVector(indices=(1, 2), values=(1.0, 1.0)))
        unrelated = _point(_payload(), sparse=SparseVector(indices=(9,), values=(1.0,)))
        await store.upsert(NAMESPACE, [matching, unrelated])
        hits = await store.search_sparse(
            NAMESPACE,
            SparseVector(indices=(1,), values=(2.0,)),
            limit=10,
            vfilter=VectorFilter(),
        )
        assert [h.chunk_id for h in hits] == [matching.chunk_id]

    async def test_hybrid_promotes_agreement_between_arms(self) -> None:
        store = await self._store()
        both = _point(
            _payload(), dense=_dense(0.7, 0.7), sparse=SparseVector(indices=(1,), values=(1.0,))
        )
        dense_only = _point(_payload(), dense=_dense(1.0, 0.0))
        await store.upsert(NAMESPACE, [dense_only, both])
        hits = await store.search_hybrid(
            NAMESPACE,
            _dense(1.0, 0.0),
            SparseVector(indices=(1,), values=(1.0,)),
            limit=10,
            vfilter=VectorFilter(),
        )
        assert hits[0].chunk_id == both.chunk_id

    async def test_acl_patch_changes_visibility_without_reindexing(self) -> None:
        store = await self._store()
        point = _point(_payload(visibility_level=int(VisibilityLevel.INTERNAL)))
        await store.upsert(NAMESPACE, [point])
        customer_filter = VectorFilter(must=(Range("visibility_level", lte=1),))
        assert await store.count(NAMESPACE, customer_filter) == 0

        await store.set_payload(
            NAMESPACE,
            VectorFilter(must=(Match("document_id", point.payload["document_id"]),)),
            acl_payload(
                visibility_level=int(VisibilityLevel.CUSTOMER),
                department_path=None,
                is_active=True,
                mode=Mode.CUSTOMER.value,
            ),
        )
        assert await store.count(NAMESPACE, customer_filter) == 1

    async def test_delete_by_filter_removes_only_matching_points(self) -> None:
        store = await self._store()
        doomed = _point(_payload())
        kept = _point(_payload())
        await store.upsert(NAMESPACE, [doomed, kept])
        removed = await store.delete_by_filter(
            NAMESPACE, VectorFilter(must=(Match("document_id", doomed.payload["document_id"]),))
        )
        assert removed == 1
        assert await store.count(NAMESPACE) == 1

    async def test_delete_by_chunk_ids(self) -> None:
        store = await self._store()
        point = _point(_payload())
        await store.upsert(NAMESPACE, [point])
        assert await store.delete_by_chunk_ids(NAMESPACE, [point.chunk_id]) == 1
        assert await store.count(NAMESPACE) == 0

    async def test_changing_dimension_with_data_is_rejected(self) -> None:
        store = await self._store()
        await store.upsert(NAMESPACE, [_point(_payload())])
        with pytest.raises(VectorStoreError, match="cannot serve"):
            await store.ensure_collection(CollectionSpec(namespace=NAMESPACE, dim=DIM + 1))


class TestFusion:
    def _hit(self, score: float) -> VectorHit:
        return VectorHit(chunk_id=uuid.uuid4(), score=score, payload={})

    def test_agreement_beats_a_single_strong_arm(self) -> None:
        shared = self._hit(0.5)
        dense_only = self._hit(0.99)
        fused = reciprocal_rank_fusion([dense_only, shared], [shared])
        assert fused[0].chunk_id == shared.chunk_id

    def test_ranks_are_recorded_for_debugging(self) -> None:
        hit = self._hit(0.5)
        fused = reciprocal_rank_fusion([hit], [hit])
        assert fused[0].payload["_rrf"] == {"arm0": 1, "arm1": 1}

    def test_limit_is_applied_after_fusion(self) -> None:
        hits = [self._hit(1.0) for _ in range(5)]
        assert len(reciprocal_rank_fusion(hits, limit=2)) == 2

    def test_ties_break_on_chunk_id_regardless_of_arm_order(self) -> None:
        """Two chunks at rank 1 in different arms score identically; order must not drift."""
        a, b = self._hit(1.0), self._hit(1.0)
        forward = [h.chunk_id for h in reciprocal_rank_fusion([a], [b])]
        backward = [h.chunk_id for h in reciprocal_rank_fusion([b], [a])]
        assert forward == backward
        assert forward == sorted(forward, key=str)

    def test_empty_input_is_empty_output(self) -> None:
        assert reciprocal_rank_fusion() == []


class TestPgVectorTranslation:
    """No database here: these assert the SQL that would be sent, and the guards around it."""

    def test_namespace_is_validated_not_escaped(self) -> None:
        from aegis.rag.vector_stores.pgvector import _table_name

        assert _table_name("aegis_internal") == "emb_aegis_internal"
        for bad in ('a"b', "a;drop", "Aegis", "1abc", "a" * 60):
            with pytest.raises(VectorStoreError, match="invalid vector namespace"):
                _table_name(bad)

    def test_sparse_search_fails_loudly_instead_of_returning_nothing(self) -> None:
        from aegis.rag.vector_stores.pgvector import PgVectorStore

        store = PgVectorStore(engine=None)  # type: ignore[arg-type]
        assert store.supports_sparse is False
        with pytest.raises(VectorStoreError, match="does not host a sparse index"):
            import asyncio

            asyncio.run(
                store.search_sparse(
                    "ns", SparseVector(indices=(1,), values=(1.0,)), limit=5, vfilter=VectorFilter()
                )
            )

    def test_oversized_dimension_is_rejected_at_creation(self) -> None:
        import asyncio

        from aegis.rag.vector_stores.pgvector import PgVectorStore

        store = PgVectorStore(engine=None)  # type: ignore[arg-type]
        with pytest.raises(VectorStoreError, match="cannot index"):
            asyncio.run(store.ensure_collection(CollectionSpec(namespace="ns", dim=3072)))


class TestQdrantTranslation:
    def test_min_should_is_set_so_should_is_a_constraint(self) -> None:
        from aegis.rag.vector_stores import to_qdrant

        translated = to_qdrant(
            VectorFilter(
                must=(Match("mode", "internal"),),
                should=(Match("visibility_level", 1), Match("visibility_level", 2)),
                min_should=1,
            )
        )
        assert translated.min_should is not None
        assert translated.min_should.min_count == 1

    def test_nested_and_or_survive_translation(self) -> None:
        from aegis.rag.vector_stores import to_qdrant

        translated = to_qdrant(
            VectorFilter(
                must=(
                    Or(
                        (
                            IsNull("expires_at"),
                            And(
                                (Match("visibility_level", 3), PrefixMatch("department_path", "c"))
                            ),
                        )
                    ),
                )
            )
        )
        assert translated.must is not None
        assert len(translated.must) == 1

    def test_unknown_condition_type_raises(self) -> None:
        from aegis.rag.vector_stores.filters import _qdrant_condition

        with pytest.raises(VectorStoreError, match="unsupported filter condition"):
            _qdrant_condition(object())  # type: ignore[arg-type]


class TestExpiryBoundary:
    async def test_document_expiring_tomorrow_is_still_retrievable(self) -> None:
        store = InMemoryVectorStore()
        await store.ensure_collection(CollectionSpec(namespace=NAMESPACE, dim=DIM))
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
        await store.upsert(
            NAMESPACE,
            [
                _point(
                    _payload(
                        expires_at=tomorrow,
                        visibility_level=int(VisibilityLevel.PUBLIC),
                        mode=Mode.CUSTOMER.value,
                    )
                )
            ],
        )
        ctx = build_security_context(
            user_id=None, role=Role.GUEST, requested_mode=Mode.CUSTOMER, is_guest=True
        )
        assert await store.count(NAMESPACE, build_filter(ctx)) == 1
