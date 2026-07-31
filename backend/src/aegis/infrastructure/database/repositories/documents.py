"""Collection, document, version, chunk, ACL, job, and discrepancy repositories."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, and_, delete, func, insert, or_, select, text, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from aegis.core.errors import NotFoundError
from aegis.core.ids import new_id
from aegis.domain.entities import (
    Chunk,
    Collection,
    Document,
    DocumentAclGrant,
    DocumentVersion,
    IndexDiscrepancy,
    IngestJob,
)
from aegis.domain.enums import (
    IngestStatus,
    JobStatus,
    Mode,
    PrincipalType,
    Role,
    Visibility,
    VisibilityLevel,
)
from aegis.domain.ports.repositories import DocumentQuery
from aegis.infrastructure.database.models import (
    ChunkModel,
    CollectionModel,
    DepartmentModel,
    DocumentAclModel,
    DocumentModel,
    DocumentTagModel,
    DocumentVersionModel,
    IndexDiscrepancyModel,
    IngestJobModel,
    TagModel,
    UserModel,
)
from aegis.infrastructure.database.repositories.helpers import affected
from aegis.infrastructure.database.repositories.helpers import ltree as _ltree

# Columns that may appear in ORDER BY, checked against the client's `sort` parameter.
DOCUMENT_SORTABLE = {"created_at", "updated_at", "title", "visibility_level"}


def _to_document(row: DocumentModel) -> Document:
    return Document(
        id=row.id,
        collection_id=row.collection_id,
        title=row.title,
        description=row.description,
        source_type=row.source_type,
        source_ref=row.source_ref,
        visibility=row.visibility,
        department_id=row.department_id,
        department_path=row.department_path,
        language=row.language,
        owner_id=row.owner_id,
        active_version_id=row.active_version_id,
        effective_from=row.effective_from,
        expires_at=row.expires_at,
        is_archived=row.is_archived,
        tags=sorted(t.name for t in row.tags),
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
    )


def _to_chunk(row: ChunkModel) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        version_id=row.version_id,
        collection_id=row.collection_id,
        ordinal=row.ordinal,
        content=row.content,
        content_hash=row.content_hash,
        token_count=row.token_count,
        chunk_type=row.chunk_type,
        page_from=row.page_from,
        page_to=row.page_to,
        heading_path=list(row.heading_path or []),
        section=row.section,
        char_start=row.char_start,
        char_end=row.char_end,
        bbox=row.bbox,
        context_header=row.context_header,
        summary=row.summary,
        keywords=list(row.keywords or []),
        language=row.language,
        visibility_level=row.visibility_level,
        department_path=row.department_path,
        vector_point_id=row.vector_point_id,
        embedding_model=row.embedding_model,
        indexed_at=row.indexed_at,
        injection_flag=row.injection_flag,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
    )


class SqlCollectionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, collection_id: UUID) -> Collection | None:
        row = await self._s.get(CollectionModel, collection_id)
        return Collection.model_validate(row) if row else None

    async def get_by_slug(self, slug: str) -> Collection | None:
        stmt = select(CollectionModel).where(CollectionModel.slug == slug)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return Collection.model_validate(row) if row else None

    async def list_for_mode(
        self, mode: Mode, *, include_inactive: bool = False
    ) -> list[Collection]:
        stmt = select(CollectionModel).where(CollectionModel.mode == mode)
        if not include_inactive:
            stmt = stmt.where(CollectionModel.is_active.is_(True))
        stmt = stmt.order_by(CollectionModel.name)
        return [Collection.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()]

    async def list_all(self, *, include_inactive: bool = False) -> list[Collection]:
        stmt = select(CollectionModel)
        if not include_inactive:
            stmt = stmt.where(CollectionModel.is_active.is_(True))
        stmt = stmt.order_by(CollectionModel.name)
        return [Collection.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()]

    async def create(self, collection: Collection) -> Collection:
        row = CollectionModel(**collection.model_dump(exclude={"created_at", "updated_at"}))
        self._s.add(row)
        await self._s.flush()
        return Collection.model_validate(row)

    async def update(self, collection_id: UUID, **fields: Any) -> Collection:
        stmt = (
            update(CollectionModel)
            .where(CollectionModel.id == collection_id)
            .values(**fields)
            .returning(CollectionModel.id)
        )
        if (await self._s.execute(stmt)).scalar_one_or_none() is None:
            raise NotFoundError("Collection", collection_id)
        row = await self._s.get(CollectionModel, collection_id, populate_existing=True)
        assert row is not None
        return Collection.model_validate(row)

    async def document_count(self, collection_id: UUID) -> int:
        stmt = select(func.count()).where(
            DocumentModel.collection_id == collection_id, DocumentModel.deleted_at.is_(None)
        )
        return int((await self._s.execute(stmt)).scalar_one())

    async def chunk_count(self, collection_id: UUID) -> int:
        stmt = select(func.count()).where(ChunkModel.collection_id == collection_id)
        return int((await self._s.execute(stmt)).scalar_one())


class SqlDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, document_id: UUID, *, include_deleted: bool = False) -> Document | None:
        row = await self._s.get(DocumentModel, document_id)
        if row is None or (row.deleted_at is not None and not include_deleted):
            return None
        return _to_document(row)

    async def get_many(self, document_ids: Sequence[UUID]) -> list[Document]:
        if not document_ids:
            return []
        stmt = select(DocumentModel).where(
            DocumentModel.id.in_(list(document_ids)), DocumentModel.deleted_at.is_(None)
        )
        return [_to_document(r) for r in (await self._s.execute(stmt)).scalars().unique().all()]

    async def create(self, document: Document) -> Document:
        payload = document.model_dump(exclude={"tags", "created_at", "updated_at"})
        payload["visibility_level"] = int(document.visibility.level)
        row = DocumentModel(**payload)
        self._s.add(row)
        await self._s.flush()
        if document.tags:
            await self.set_tags(row.id, document.tags)
        await self._s.refresh(row, ["tags"])
        return _to_document(row)

    async def update(self, document_id: UUID, **fields: Any) -> Document:
        # visibility_level is trigger-maintained in the database, but it is also set here so
        # that a value read back inside the same transaction is already correct.
        if "visibility" in fields:
            visibility = fields["visibility"]
            if isinstance(visibility, str):
                visibility = Visibility(visibility)
            fields["visibility_level"] = int(visibility.level)
        tags = fields.pop("tags", None)
        if fields:
            stmt = (
                update(DocumentModel)
                .where(DocumentModel.id == document_id, DocumentModel.deleted_at.is_(None))
                .values(**fields)
                .returning(DocumentModel.id)
            )
            if (await self._s.execute(stmt)).scalar_one_or_none() is None:
                raise NotFoundError("Document", document_id)
        if tags is not None:
            await self.set_tags(document_id, tags)
        row = await self._s.get(DocumentModel, document_id, populate_existing=True)
        if row is None:
            raise NotFoundError("Document", document_id)
        await self._s.refresh(row, ["tags"])
        return _to_document(row)

    async def soft_delete(self, document_id: UUID, *, at: datetime) -> None:
        await self._s.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(deleted_at=at, active_version_id=None)
        )

    async def set_active_version(self, document_id: UUID, version_id: UUID) -> None:
        """Publish a version in one statement.

        This is the atomic flip that makes replacement zero-downtime: before it, the previous
        version keeps serving; after it, the new one does; there is no moment in between.
        """
        await self._s.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(active_version_id=version_id)
        )

    def _apply_query(self, stmt: Select[Any], q: DocumentQuery) -> Select[Any]:
        if not q.include_deleted:
            stmt = stmt.where(DocumentModel.deleted_at.is_(None))
        if not q.include_archived:
            stmt = stmt.where(DocumentModel.is_archived.is_(False))
        if q.collection_id is not None:
            stmt = stmt.where(DocumentModel.collection_id == q.collection_id)
        if q.visibility is not None:
            stmt = stmt.where(DocumentModel.visibility == q.visibility)
        if q.max_visibility_level is not None:
            # A document title is content: an internal employee browsing the admin table must
            # not see the titles of confidential documents they cannot read.
            stmt = stmt.where(DocumentModel.visibility_level <= q.max_visibility_level)
        if q.department_id is not None:
            stmt = stmt.where(DocumentModel.department_id == q.department_id)
        if q.department_path:
            stmt = stmt.where(DocumentModel.department_path.op("<@")(_ltree(q.department_path)))
        if q.owner_id is not None:
            stmt = stmt.where(DocumentModel.owner_id == q.owner_id)
        if q.created_after is not None:
            stmt = stmt.where(DocumentModel.created_at >= q.created_after)
        if q.created_before is not None:
            stmt = stmt.where(DocumentModel.created_at < q.created_before)
        if q.search:
            stmt = stmt.where(DocumentModel.title.ilike(f"%{q.search}%"))
        if q.tags:
            stmt = stmt.where(
                DocumentModel.id.in_(
                    select(DocumentTagModel.document_id)
                    .join(TagModel, TagModel.id == DocumentTagModel.tag_id)
                    .where(TagModel.name.in_(list(q.tags)))
                )
            )
        if q.status is not None:
            stmt = stmt.where(
                DocumentModel.active_version_id.in_(
                    select(DocumentVersionModel.id).where(DocumentVersionModel.status == q.status)
                )
                if q.status is IngestStatus.INDEXED
                else DocumentModel.id.in_(
                    select(DocumentVersionModel.document_id).where(
                        DocumentVersionModel.status == q.status
                    )
                )
            )
        return stmt

    async def list_documents(
        self,
        query: DocumentQuery,
        *,
        limit: int,
        sort_field: str = "created_at",
        descending: bool = True,
        cursor_value: Any = None,
        cursor_id: UUID | None = None,
    ) -> list[Document]:
        if sort_field not in DOCUMENT_SORTABLE:
            sort_field = "created_at"
        column = getattr(DocumentModel, sort_field)

        stmt = self._apply_query(select(DocumentModel), query)
        if cursor_value is not None and cursor_id is not None:
            if descending:
                stmt = stmt.where(
                    (column < cursor_value)
                    | ((column == cursor_value) & (DocumentModel.id < cursor_id))
                )
            else:
                stmt = stmt.where(
                    (column > cursor_value)
                    | ((column == cursor_value) & (DocumentModel.id > cursor_id))
                )
        order = (
            (column.desc(), DocumentModel.id.desc())
            if descending
            else (column.asc(), DocumentModel.id.asc())
        )
        stmt = stmt.order_by(*order).limit(limit)
        rows = (await self._s.execute(stmt)).scalars().unique().all()
        return [_to_document(r) for r in rows]

    async def estimate_count(self, query: DocumentQuery) -> int:
        """Planner estimate for the filtered set.

        ``EXPLAIN`` on the real query is used rather than ``COUNT(*)`` because an exact count
        over a filtered half-million-row table costs more than the page it decorates, and the
        UI shows "about 1,284" regardless.
        """
        stmt = self._apply_query(select(DocumentModel.id), query)
        # `postgresql.dialect` is untyped in SQLAlchemy's stubs.
        compiled = stmt.compile(
            dialect=postgresql.dialect(),  # type: ignore[no-untyped-call]
            compile_kwargs={"literal_binds": True},
        )
        try:
            result = await self._s.execute(text(f"EXPLAIN (FORMAT JSON) {compiled}"))
            plan = result.scalar_one()
            if isinstance(plan, list) and plan:
                return int(plan[0]["Plan"]["Plan Rows"])
            return 0
        except Exception:
            return 0

    async def set_tags(self, document_id: UUID, tags: Sequence[str]) -> None:
        """Replace the tag set, creating tags that do not yet exist."""
        normalized = sorted({t.strip().lower() for t in tags if t.strip()})
        await self._s.execute(
            delete(DocumentTagModel).where(DocumentTagModel.document_id == document_id)
        )
        if not normalized:
            return

        stmt = pg_insert(TagModel).values([{"id": new_id(), "name": n} for n in normalized])
        await self._s.execute(stmt.on_conflict_do_nothing(index_elements=["name"]))

        tag_ids = (
            (await self._s.execute(select(TagModel.id).where(TagModel.name.in_(normalized))))
            .scalars()
            .all()
        )
        if tag_ids:
            await self._s.execute(
                insert(DocumentTagModel),
                [{"document_id": document_id, "tag_id": tid} for tid in tag_ids],
            )

    async def find_by_checksum(self, collection_id: UUID, checksum: bytes) -> Document | None:
        stmt = (
            select(DocumentModel)
            .join(DocumentVersionModel, DocumentVersionModel.document_id == DocumentModel.id)
            .where(
                DocumentModel.collection_id == collection_id,
                DocumentVersionModel.checksum_sha256 == checksum,
                DocumentModel.deleted_at.is_(None),
            )
            .limit(1)
        )
        row = (await self._s.execute(stmt)).scalars().unique().one_or_none()
        return _to_document(row) if row else None


class SqlDocumentVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, version_id: UUID) -> DocumentVersion | None:
        row = await self._s.get(DocumentVersionModel, version_id)
        return DocumentVersion.model_validate(row) if row else None

    async def list_for_document(self, document_id: UUID) -> list[DocumentVersion]:
        stmt = (
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document_id)
            .order_by(DocumentVersionModel.version_no.desc())
        )
        return [
            DocumentVersion.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()
        ]

    async def create(self, version: DocumentVersion) -> DocumentVersion:
        row = DocumentVersionModel(**version.model_dump(exclude={"created_at"}))
        self._s.add(row)
        await self._s.flush()
        return DocumentVersion.model_validate(row)

    async def update_status(
        self,
        version_id: UUID,
        *,
        status: IngestStatus,
        error_message: str | None = None,
        indexed_at: datetime | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status, "error_message": error_message}
        if indexed_at is not None:
            values["indexed_at"] = indexed_at
        await self._s.execute(
            update(DocumentVersionModel)
            .where(DocumentVersionModel.id == version_id)
            .values(**values)
        )

    async def update_stats(self, version_id: UUID, **stats: Any) -> None:
        values = {k: v for k, v in stats.items() if v is not None}
        if not values:
            return
        await self._s.execute(
            update(DocumentVersionModel)
            .where(DocumentVersionModel.id == version_id)
            .values(**values)
        )

    async def mark_superseded(self, version_id: UUID, *, at: datetime) -> None:
        await self._s.execute(
            update(DocumentVersionModel)
            .where(DocumentVersionModel.id == version_id)
            .values(status=IngestStatus.SUPERSEDED, superseded_at=at)
        )

    async def next_version_no(self, document_id: UUID) -> int:
        stmt = select(func.coalesce(func.max(DocumentVersionModel.version_no), 0) + 1).where(
            DocumentVersionModel.document_id == document_id
        )
        return int((await self._s.execute(stmt)).scalar_one())

    async def find_by_checksum(self, document_id: UUID, checksum: bytes) -> DocumentVersion | None:
        stmt = select(DocumentVersionModel).where(
            DocumentVersionModel.document_id == document_id,
            DocumentVersionModel.checksum_sha256 == checksum,
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return DocumentVersion.model_validate(row) if row else None

    async def find_stale_pending(self, *, older_than: datetime) -> list[DocumentVersion]:
        stmt = select(DocumentVersionModel).where(
            DocumentVersionModel.status.in_(
                [
                    IngestStatus.PENDING,
                    IngestStatus.SCANNING,
                    IngestStatus.PARSING,
                    IngestStatus.CHUNKING,
                    IngestStatus.EMBEDDING,
                    IngestStatus.INDEXING,
                ]
            ),
            DocumentVersionModel.created_at < older_than,
        )
        return [
            DocumentVersion.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()
        ]


class SqlChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def bulk_create(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        rows = []
        for c in chunks:
            payload = c.model_dump(exclude={"created_at", "metadata"})
            payload["metadata_"] = c.metadata
            rows.append(payload)
        # Core insert rather than ORM objects: for a 500-chunk document this is one statement
        # instead of 500 unit-of-work identity operations.
        await self._s.execute(insert(ChunkModel), rows)
        return len(rows)

    async def get(self, chunk_id: UUID) -> Chunk | None:
        row = await self._s.get(ChunkModel, chunk_id)
        return _to_chunk(row) if row else None

    async def get_many(self, chunk_ids: Sequence[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        stmt = select(ChunkModel).where(ChunkModel.id.in_(list(chunk_ids)))
        rows = (await self._s.execute(stmt)).scalars().all()
        by_id = {r.id: r for r in rows}
        # Preserve the caller's order, which is retrieval rank — reordering it here would
        # silently discard the ranking the pipeline just computed.
        return [_to_chunk(by_id[cid]) for cid in chunk_ids if cid in by_id]

    async def list_for_version(
        self, version_id: UUID, *, limit: int = 100, offset: int = 0
    ) -> list[Chunk]:
        stmt = (
            select(ChunkModel)
            .where(ChunkModel.version_id == version_id)
            .order_by(ChunkModel.ordinal)
            .limit(limit)
            .offset(offset)
        )
        return [_to_chunk(r) for r in (await self._s.execute(stmt)).scalars().all()]

    async def list_for_reindex(
        self, collection_id: UUID, *, batch_size: int = 500, after_ordinal: int = -1
    ) -> list[Chunk]:
        stmt = (
            select(ChunkModel)
            .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
            .where(
                ChunkModel.collection_id == collection_id,
                DocumentModel.deleted_at.is_(None),
                DocumentModel.active_version_id == ChunkModel.version_id,
                ChunkModel.ordinal > after_ordinal,
            )
            .order_by(ChunkModel.ordinal, ChunkModel.id)
            .limit(batch_size)
        )
        return [_to_chunk(r) for r in (await self._s.execute(stmt)).scalars().all()]

    async def mark_indexed(
        self, chunk_ids: Sequence[UUID], *, at: datetime, model: str, point_ids: dict[UUID, UUID]
    ) -> None:
        if not chunk_ids:
            return
        await self._s.execute(
            update(ChunkModel)
            .where(ChunkModel.id.in_(list(chunk_ids)))
            .values(indexed_at=at, embedding_model=model)
        )
        for chunk_id, point_id in point_ids.items():
            await self._s.execute(
                update(ChunkModel).where(ChunkModel.id == chunk_id).values(vector_point_id=point_id)
            )

    async def delete_for_version(self, version_id: UUID) -> int:
        result = await self._s.execute(
            delete(ChunkModel).where(ChunkModel.version_id == version_id)
        )
        return affected(result)

    async def count_for_version(self, version_id: UUID) -> int:
        stmt = select(func.count()).where(ChunkModel.version_id == version_id)
        return int((await self._s.execute(stmt)).scalar_one())

    def _acl_clause(
        self,
        *,
        max_visibility_level: int,
        department_path: str | None,
        granted_document_ids: Sequence[UUID],
        mode: Mode | None,
        now: datetime,
    ) -> Any:
        """The SQL twin of ``domain.policies.acl.build_filter``.

        Deliberately kept structurally parallel to the vector filter — the ceiling as an
        outer bound, then the qualifying branches as an ``OR``. Two implementations of one
        rule diverge unless they are written to look the same, and the RBAC matrix test
        exercises both against the same expectations.
        """
        branches: list[Any] = [
            ChunkModel.visibility_level <= min(max_visibility_level, VisibilityLevel.INTERNAL)
        ]

        if max_visibility_level >= VisibilityLevel.CONFIDENTIAL and department_path:
            branches.append(
                and_(
                    ChunkModel.visibility_level == int(VisibilityLevel.CONFIDENTIAL),
                    ChunkModel.department_path.op("<@")(_ltree(department_path)),
                )
            )
        if max_visibility_level >= VisibilityLevel.RESTRICTED:
            branches.append(ChunkModel.visibility_level == int(VisibilityLevel.RESTRICTED))
        if granted_document_ids:
            branches.append(ChunkModel.document_id.in_(list(granted_document_ids)))

        clauses: list[Any] = [
            # Redundant with the branches by design: a bug in branch construction can then
            # only ever narrow the result, never widen it past the ceiling.
            ChunkModel.visibility_level <= max_visibility_level,
            DocumentModel.deleted_at.is_(None),
            DocumentModel.is_archived.is_(False),
            DocumentModel.active_version_id == ChunkModel.version_id,
            or_(DocumentModel.expires_at.is_(None), DocumentModel.expires_at > now.date()),
            or_(
                DocumentModel.effective_from.is_(None),
                DocumentModel.effective_from <= now.date(),
            ),
            or_(*branches),
        ]
        if mode is not None:
            clauses.append(
                ChunkModel.collection_id.in_(
                    select(CollectionModel.id).where(CollectionModel.mode == mode)
                )
            )
        return and_(*clauses)

    async def keyword_search(
        self,
        query: str,
        *,
        limit: int,
        collection_ids: Sequence[UUID] = (),
        max_visibility_level: int = 0,
        department_path: str | None = None,
        granted_document_ids: Sequence[UUID] = (),
    ) -> list[tuple[UUID, float]]:
        """BM25-style ranking over the generated ``tsvector``.

        ``websearch_to_tsquery`` is used rather than ``plainto_tsquery`` because it accepts
        quoted phrases and ``or``/``-`` operators that users actually type, and it never raises
        on malformed input — a syntax error from a search box is not an acceptable outcome.

        Mode is not a parameter here: ``collection_ids`` already comes from the security
        context, and every collection belongs to exactly one mode.
        """
        now = datetime.now(UTC)
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(ChunkModel.tsv, tsquery, 32)

        stmt = (
            select(ChunkModel.id, rank.label("rank"))
            .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
            .where(
                ChunkModel.tsv.op("@@")(tsquery),
                self._acl_clause(
                    max_visibility_level=max_visibility_level,
                    department_path=department_path,
                    granted_document_ids=granted_document_ids,
                    mode=None,
                    now=now,
                ),
            )
            .order_by(rank.desc())
            .limit(limit)
        )
        if collection_ids:
            stmt = stmt.where(ChunkModel.collection_id.in_(list(collection_ids)))
        return [(row[0], float(row[1])) for row in (await self._s.execute(stmt)).all()]

    async def verify_readable(
        self,
        chunk_ids: Sequence[UUID],
        *,
        max_visibility_level: int,
        department_path: str | None,
        granted_document_ids: Sequence[UUID],
        mode: Mode,
        now: datetime,
    ) -> set[UUID]:
        """Enforcement layer 2: re-check survivors against the source of truth.

        The vector payload is a replica. Between "a manager tightens a document to restricted"
        and "reindex finishes", the payload still says internal. One indexed query over the
        eight surviving chunks closes that window for about 30 ms.
        """
        if not chunk_ids:
            return set()
        stmt = (
            select(ChunkModel.id)
            .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
            .where(
                ChunkModel.id.in_(list(chunk_ids)),
                self._acl_clause(
                    max_visibility_level=max_visibility_level,
                    department_path=department_path,
                    granted_document_ids=granted_document_ids,
                    mode=mode,
                    now=now,
                ),
            )
        )
        return set((await self._s.execute(stmt)).scalars().all())


class SqlAclRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def list_for_document(self, document_id: UUID) -> list[DocumentAclGrant]:
        stmt = select(DocumentAclModel).where(DocumentAclModel.document_id == document_id)
        return [
            DocumentAclGrant.model_validate(r)
            for r in (await self._s.execute(stmt)).scalars().all()
        ]

    async def replace_for_document(self, document_id: UUID, grants: list[DocumentAclGrant]) -> None:
        await self._s.execute(
            delete(DocumentAclModel).where(DocumentAclModel.document_id == document_id)
        )
        for g in grants:
            self._s.add(
                DocumentAclModel(
                    id=new_id(),
                    document_id=document_id,
                    principal_type=g.principal_type,
                    principal_role=g.principal_role,
                    principal_id=g.principal_id,
                    include_subtree=g.include_subtree,
                    granted_by=g.granted_by,
                    expires_at=g.expires_at,
                )
            )
        await self._s.flush()

    async def granted_document_ids(
        self,
        *,
        user_id: UUID | None,
        role: Role,
        department_paths: Sequence[str],
        limit: int,
    ) -> list[UUID]:
        conditions = [
            and_(
                DocumentAclModel.principal_type == PrincipalType.ROLE,
                DocumentAclModel.principal_role == role,
            )
        ]
        if user_id is not None:
            conditions.append(
                and_(
                    DocumentAclModel.principal_type == PrincipalType.USER,
                    DocumentAclModel.principal_id == user_id,
                )
            )
        if department_paths:
            # Grants are stored against a department id; the caller passes the principal's
            # own path plus its ancestors, so a grant on a parent department is inherited.
            conditions.append(
                and_(
                    DocumentAclModel.principal_type == PrincipalType.DEPARTMENT,
                    DocumentAclModel.principal_id.in_(
                        select(DepartmentModel.id).where(
                            DepartmentModel.path.in_(list(department_paths))
                        )
                    ),
                )
            )
        stmt = (
            select(DocumentAclModel.document_id)
            .where(
                or_(*conditions),
                or_(
                    DocumentAclModel.expires_at.is_(None), DocumentAclModel.expires_at > func.now()
                ),
            )
            .distinct()
            .limit(limit + 1)
        )
        return list((await self._s.execute(stmt)).scalars().all())

    async def effective_reader_count(self, document_id: UUID) -> int:
        """How many active users can read this document.

        Derived from the visibility level plus explicit user grants, inverting the ceiling
        table. Approximate by design — it is a sanity check for an administrator ("this HR
        salary band is readable by 4 people, not 900"), not an authorization decision.
        """
        document = await self._s.get(DocumentModel, document_id)
        if document is None:
            return 0

        base = (
            select(func.count())
            .select_from(UserModel)
            .where(UserModel.deleted_at.is_(None), UserModel.is_active.is_(True))
        )
        level = document.visibility_level

        if level <= VisibilityLevel.CUSTOMER:
            stmt = base
        elif level == VisibilityLevel.INTERNAL:
            stmt = base.where(
                UserModel.role.in_([Role.ADMIN, Role.MANAGER, Role.INTERNAL_EMPLOYEE])
            )
        elif level == VisibilityLevel.CONFIDENTIAL:
            stmt = base.where(UserModel.role.in_([Role.ADMIN, Role.MANAGER]))
            if document.department_path:
                stmt = stmt.where(
                    UserModel.department_id.in_(
                        select(DepartmentModel.id).where(
                            DepartmentModel.path.op("<@")(_ltree(document.department_path))
                        )
                    )
                )
        else:
            stmt = base.where(UserModel.role == Role.ADMIN)

        count = int((await self._s.execute(stmt)).scalar_one())
        granted = int(
            (
                await self._s.execute(
                    select(func.count(func.distinct(DocumentAclModel.principal_id))).where(
                        DocumentAclModel.document_id == document_id,
                        DocumentAclModel.principal_type == PrincipalType.USER,
                        or_(
                            DocumentAclModel.expires_at.is_(None),
                            DocumentAclModel.expires_at > func.now(),
                        ),
                    )
                )
            ).scalar_one()
        )
        return count + granted


class SqlIngestJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, job: IngestJob) -> IngestJob:
        row = IngestJobModel(**job.model_dump(exclude={"queued_at"}))
        self._s.add(row)
        await self._s.flush()
        return IngestJob.model_validate(row)

    async def get(self, job_id: UUID) -> IngestJob | None:
        row = await self._s.get(IngestJobModel, job_id)
        return IngestJob.model_validate(row) if row else None

    async def find_by_idempotency_key(self, key: str) -> IngestJob | None:
        stmt = select(IngestJobModel).where(IngestJobModel.idempotency_key == key)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return IngestJob.model_validate(row) if row else None

    async def start(self, job_id: UUID, *, worker_id: str, at: datetime) -> None:
        await self._s.execute(
            update(IngestJobModel)
            .where(IngestJobModel.id == job_id)
            .values(
                status=JobStatus.RUNNING,
                worker_id=worker_id,
                started_at=at,
                attempts=IngestJobModel.attempts + 1,
            )
        )

    async def update_stage(
        self, job_id: UUID, *, stage: IngestStatus, metrics: dict[str, Any]
    ) -> None:
        await self._s.execute(
            update(IngestJobModel)
            .where(IngestJobModel.id == job_id)
            .values(stage=stage, metrics=IngestJobModel.metrics.op("||")(metrics))
        )

    async def finish(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        at: datetime,
        error_message: str | None = None,
        error_class: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "finished_at": at,
            "error_message": error_message,
            "error_class": error_class,
        }
        if metrics:
            values["metrics"] = IngestJobModel.metrics.op("||")(metrics)
        await self._s.execute(
            update(IngestJobModel).where(IngestJobModel.id == job_id).values(**values)
        )

    async def list_recent(
        self, *, status: JobStatus | None = None, limit: int = 50
    ) -> list[IngestJob]:
        stmt = select(IngestJobModel).order_by(IngestJobModel.queued_at.desc()).limit(limit)
        if status is not None:
            stmt = stmt.where(IngestJobModel.status == status)
        return [IngestJob.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()]

    async def queue_stats(self) -> dict[str, int]:
        stmt = select(IngestJobModel.status, func.count()).group_by(IngestJobModel.status)
        return {str(row[0].value): int(row[1]) for row in (await self._s.execute(stmt)).all()}


class SqlDiscrepancyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def record(self, discrepancy: IndexDiscrepancy) -> None:
        self._s.add(IndexDiscrepancyModel(**discrepancy.model_dump(exclude={"detected_at"})))
        await self._s.flush()

    async def list_open(self, *, limit: int = 100) -> list[IndexDiscrepancy]:
        stmt = (
            select(IndexDiscrepancyModel)
            .where(IndexDiscrepancyModel.repaired_at.is_(None))
            .order_by(IndexDiscrepancyModel.detected_at.desc())
            .limit(limit)
        )
        return [
            IndexDiscrepancy.model_validate(r)
            for r in (await self._s.execute(stmt)).scalars().all()
        ]

    async def mark_repaired(self, discrepancy_id: UUID, *, at: datetime) -> None:
        await self._s.execute(
            update(IndexDiscrepancyModel)
            .where(IndexDiscrepancyModel.id == discrepancy_id)
            .values(repaired_at=at)
        )

    async def counts_by_kind(self) -> dict[str, int]:
        stmt = (
            select(IndexDiscrepancyModel.kind, func.count())
            .where(IndexDiscrepancyModel.repaired_at.is_(None))
            .group_by(IndexDiscrepancyModel.kind)
        )
        return {str(row[0]): int(row[1]) for row in (await self._s.execute(stmt)).all()}
