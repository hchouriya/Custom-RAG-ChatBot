"""The ingestion pipeline.

Fetch → scan → sniff → parse → clean → chunk → embed → index → activate. Runs in a worker,
never in a request, because a 300-page scanned PDF is minutes of OCR and tying that to an HTTP
connection caps document size at the proxy timeout.

Properties that make this safe to retry, which is the only property that matters for a
pipeline that will certainly be interrupted:

* **Every stage is recorded on the version row.** The row, not the queue message, is the
  authoritative record of "needs indexing", so a lost Redis message costs a delay rather than
  a document.
* **Chunk ids and vector point ids are deterministic.** A retried batch overwrites instead of
  duplicating. A duplicated vector is invisible until it quietly distorts a ranking.
* **The version is activated last.** Until the flip, readers keep seeing the previous version;
  a failure at any earlier stage is invisible to them.
* **Failure is classified.** A corrupt PDF is permanent and must not be retried three times; a
  provider 503 is transient and must be. Retrying a permanent failure wastes a queue slot and
  buries the real error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from aegis.core.errors import (
    IngestionError,
    MalwareDetectedError,
    NotFoundError,
    ParserError,
)
from aegis.core.ids import new_id, vector_point_id
from aegis.core.logging import get_logger
from aegis.core.telemetry import chunks_produced, ingest_stage_duration, timed_stage
from aegis.domain.entities import Chunk
from aegis.domain.enums import IngestStatus
from aegis.domain.ports.chunker import ChunkingConfig
from aegis.domain.ports.vector_store import CollectionSpec, VectorPoint
from aegis.rag.vector_stores.payload import build_payload

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.domain.entities import Collection, Document, DocumentVersion
    from aegis.domain.ports.embeddings import EmbeddingProvider, SparseEmbeddingProvider
    from aegis.domain.ports.infrastructure import MalwareScanner, ObjectStore
    from aegis.domain.ports.parser import ParsedDocument
    from aegis.domain.ports.repositories import Repositories
    from aegis.domain.ports.vector_store import VectorStore
    from aegis.rag.chunking.router import DefaultChunkRouter
    from aegis.rag.guardrails import Guardrails
    from aegis.rag.parsing.registry import ParserRegistry

logger = get_logger(__name__)

EMBED_BATCH = 64


@dataclass(slots=True)
class IngestReport:
    version_id: UUID
    status: IngestStatus
    pages: int = 0
    characters: int = 0
    chunks: int = 0
    tokens: int = 0
    used_ocr: bool = False
    parser: str = ""
    injection_flags: int = 0
    duplicate_of: UUID | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class IngestionService:
    def __init__(
        self,
        repos: Repositories,
        *,
        settings: Settings,
        storage: ObjectStore,
        parsers: ParserRegistry,
        chunker: DefaultChunkRouter,
        embedder: EmbeddingProvider,
        sparse_encoder: SparseEmbeddingProvider | None,
        vector_store: VectorStore,
        scanner: MalwareScanner,
        guardrails: Guardrails,
    ) -> None:
        self._repos = repos
        self._settings = settings
        self._storage = storage
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._vectors = vector_store
        self._scanner = scanner
        self._guardrails = guardrails

    async def ingest(self, version_id: UUID, *, worker_id: str = "worker") -> IngestReport:
        """Run the full pipeline for one version."""
        version = await self._repos.versions.get(version_id)
        if version is None:
            raise NotFoundError("Version", version_id)
        document = await self._repos.documents.get(version.document_id)
        if document is None:
            raise NotFoundError("Document", version.document_id)
        collection = await self._repos.collections.get(document.collection_id)
        if collection is None:
            raise NotFoundError("Collection", document.collection_id)

        report = IngestReport(version_id=version_id, status=IngestStatus.PENDING)
        logger.info(
            "ingest.start",
            version_id=str(version_id),
            document_id=str(document.id),
            filename=version.original_filename,
            size_bytes=version.size_bytes,
            worker=worker_id,
        )

        try:
            data = await self._fetch(version, report)
            await self._scan(data, version, report)
            parsed = await self._parse(data, version, report)
            checksum = hashlib.sha256(data).digest()

            duplicate = await self._repos.versions.find_by_checksum(document.id, checksum)
            if duplicate is not None and duplicate.id != version_id:
                # Identical bytes already indexed for this document: re-embedding them would
                # cost real money to produce the same vectors.
                report.duplicate_of = duplicate.id
                report.status = IngestStatus.SUPERSEDED
                await self._repos.versions.update_status(
                    version_id,
                    status=IngestStatus.SUPERSEDED,
                    error_message=f"identical to version {duplicate.version_no}",
                )
                await self._repos.uow.commit()
                logger.info("ingest.duplicate", version_id=str(version_id))
                return report

            await self._repos.versions.update_stats(version.id, checksum_sha256=checksum)
            chunks = await self._chunk(parsed, document, version, collection, report)
            await self._embed_and_index(chunks, collection, document, version, report)
            await self._activate(document, version, report)
            report.status = IngestStatus.INDEXED
        except (IngestionError, MalwareDetectedError) as exc:
            status = (
                IngestStatus.QUARANTINED
                if isinstance(exc, MalwareDetectedError)
                else IngestStatus.FAILED
            )
            report.status = status
            report.error = str(exc.detail)
            await self._repos.versions.update_status(
                version_id, status=status, error_message=str(exc.detail)[:500]
            )
            await self._repos.uow.commit()
            logger.error(
                "ingest.failed",
                version_id=str(version_id),
                stage=getattr(exc, "stage", "scan"),
                error=str(exc.detail),
            )
            raise

        logger.info(
            "ingest.done",
            version_id=str(version_id),
            chunks=report.chunks,
            tokens=report.tokens,
            pages=report.pages,
            used_ocr=report.used_ocr,
            timings_ms=report.timings_ms,
        )
        return report

    # ── stages ──────────────────────────────────────────────────────────────

    async def _stage(self, version_id: UUID, status: IngestStatus) -> None:
        await self._repos.versions.update_status(version_id, status=status)
        await self._repos.uow.commit()

    async def _fetch(self, version: DocumentVersion, report: IngestReport) -> bytes:
        with timed_stage("ingest_fetch") as span:
            try:
                data = await self._storage.get(version.storage_uri)
            # Storage adapters raise provider-specific types; all of them mean the same thing
            # here, and the stage name is what the operator needs.
            except Exception as exc:
                raise IngestionError("fetch", f"could not read the stored object: {exc}") from exc
            span["bytes"] = len(data)
        report.timings_ms["fetch"] = span["duration_ms"]
        if not data:
            raise IngestionError("fetch", "the stored object is empty", retryable=False)
        return data

    async def _scan(self, data: bytes, version: DocumentVersion, report: IngestReport) -> None:
        if not self._settings.malware_scan_enabled:
            return
        await self._stage(version.id, IngestStatus.SCANNING)
        with timed_stage("ingest_scan") as span:
            result = await self._scanner.scan(data, filename=version.original_filename)
            span["clean"] = result.clean
        report.timings_ms["scan"] = span["duration_ms"]
        if not result.clean:
            raise MalwareDetectedError(
                f"{version.original_filename} was rejected by the malware scanner."
            )

    async def _parse(
        self, data: bytes, version: DocumentVersion, report: IngestReport
    ) -> ParsedDocument:
        await self._stage(version.id, IngestStatus.PARSING)
        with timed_stage("ingest_parse") as span:
            try:
                parsed, detected = await self._parsers.parse(
                    data, filename=version.original_filename, declared_mime=version.mime_type
                )
            except ParserError:
                raise
            # Third-party parsers raise anything at all on a malformed file.
            except Exception as exc:
                raise ParserError(f"could not parse {version.original_filename}: {exc}") from exc
            span["blocks"] = len(parsed.blocks)
        report.timings_ms["parse"] = span["duration_ms"]
        # Labelled with the *sniffed* type, not the declared one: a metric grouped by what the
        # client claimed would not tell us which real formats are slow.
        ingest_stage_duration.labels(stage="parse", mime=detected.mime_type).observe(
            report.timings_ms["parse"] / 1000
        )
        if detected.mime_type != version.mime_type:
            await self._repos.versions.update_stats(version.id, parser=parsed.parser)
            logger.info(
                "ingest.mime_corrected",
                version_id=str(version.id),
                declared=version.mime_type,
                detected=detected.mime_type,
            )

        characters = sum(len(b.text) for b in parsed.blocks)
        if characters == 0:
            raise ParserError(
                "no text could be extracted; the file may be an image-only scan with OCR disabled"
            )

        report.pages = len(parsed.pages) or parsed.metadata.get("page_count", 0) or 0
        report.characters = characters
        report.used_ocr = bool(parsed.used_ocr)
        report.parser = parsed.parser
        await self._repos.versions.update_stats(
            version.id,
            page_count=report.pages or None,
            extracted_chars=characters,
            used_ocr=report.used_ocr,
            parser=parsed.parser,
        )
        await self._repos.uow.commit()
        return parsed

    async def _chunk(
        self,
        parsed: ParsedDocument,
        document: Document,
        version: DocumentVersion,
        collection: Collection,
        report: IngestReport,
    ) -> list[Chunk]:
        await self._stage(version.id, IngestStatus.CHUNKING)
        target = collection.chunk_size or self._settings.chunk_size
        config = ChunkingConfig(
            target_tokens=target,
            overlap_pct=round(100 * (collection.chunk_overlap or 0) / target)
            if collection.chunk_overlap
            else self._settings.chunk_overlap_pct,
            min_tokens=self._settings.chunk_min_tokens,
            max_tokens=self._settings.chunk_max_tokens,
            contextual_headers=self._settings.contextual_headers,
            document_title=document.title,
        )
        with timed_stage("ingest_chunk") as span:
            protos = await self._chunker.chunk(parsed, config)
            span["chunks"] = len(protos)
        report.timings_ms["chunk"] = span["duration_ms"]
        if not protos:
            raise IngestionError("chunk", "chunking produced nothing", retryable=False)

        chunks: list[Chunk] = []
        flags = 0
        for proto in protos:
            categories, confidence = self._guardrails.scan_document(proto.content)
            flagged = bool(categories) and confidence >= 0.8
            flags += 1 if flagged else 0
            chunk_id = new_id()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    version_id=version.id,
                    collection_id=collection.id,
                    ordinal=proto.ordinal,
                    content=proto.content,
                    content_hash=hashlib.sha256(proto.content.encode("utf-8")).digest(),
                    token_count=proto.token_count,
                    chunk_type=proto.chunk_type,
                    page_from=proto.page_from,
                    page_to=proto.page_to,
                    heading_path=list(proto.heading_path),
                    section=proto.section,
                    char_start=proto.char_start,
                    char_end=proto.char_end,
                    context_header=proto.context_header,
                    keywords=list(proto.keywords),
                    language=document.language,
                    visibility_level=int(document.visibility.level),
                    department_path=document.department_path,
                    vector_point_id=vector_point_id(collection.id, chunk_id),
                    embedding_model=collection.embedding_model,
                    injection_flag=flagged,
                    metadata={"injection_categories": list(categories)} if categories else {},
                )
            )

        # Replacing a version's chunks is idempotent by construction: delete then insert,
        # inside the transaction, so a retry cannot leave two generations of chunks behind.
        await self._repos.chunks.delete_for_version(version.id)
        await self._repos.chunks.bulk_create(chunks)
        report.chunks = len(chunks)
        report.tokens = sum(c.token_count for c in chunks)
        report.injection_flags = flags
        await self._repos.versions.update_stats(
            version.id,
            chunk_count=len(chunks),
            token_count=report.tokens,
            chunk_strategy=collection.chunk_strategy,
            embedding_model=collection.embedding_model,
            injection_flags=flags,
        )
        await self._repos.uow.commit()
        chunks_produced.observe(len(chunks))
        return chunks

    async def _embed_and_index(
        self,
        chunks: list[Chunk],
        collection: Collection,
        document: Document,
        version: DocumentVersion,
        report: IngestReport,
    ) -> None:
        await self._stage(version.id, IngestStatus.EMBEDDING)
        await self._vectors.ensure_collection(
            CollectionSpec(
                namespace=collection.vector_namespace,
                dim=collection.embedding_dim,
                with_sparse=self._sparse is not None,
                quantization=self._settings.qdrant_quantization,
                on_disk_payload=self._settings.qdrant_on_disk_payload,
            )
        )

        indexed = 0
        with timed_stage("ingest_embed") as span:
            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start : start + EMBED_BATCH]
                # `embed_text` prepends the contextual header, which is the only place that
                # happens — so the embedded text and the displayed text cannot drift.
                texts = [c.embed_text for c in batch]
                dense = await self._embedder.embed_documents(texts)
                sparse = (
                    await self._sparse.embed_documents(texts)
                    if self._sparse
                    else [None] * len(batch)
                )

                points = [
                    VectorPoint(
                        id=chunk.vector_point_id or vector_point_id(collection.id, chunk.id),
                        chunk_id=chunk.id,
                        dense=vector,
                        sparse=sparse_vector,
                        payload=build_payload(
                            chunk_id=chunk.id,
                            document_id=document.id,
                            version_id=version.id,
                            collection_id=collection.id,
                            mode=collection.mode.value,
                            visibility_level=chunk.visibility_level,
                            department_path=chunk.department_path,
                            # Not active until the version is: indexing a not-yet-published
                            # version must not make it retrievable.
                            is_active=False,
                            expires_at=document.expires_at,
                            effective_from=document.effective_from,
                            chunk_type=chunk.chunk_type.value,
                            language=chunk.language,
                            tags=tuple(document.tags),
                            ordinal=chunk.ordinal,
                            page_from=chunk.page_from,
                            page_to=chunk.page_to,
                            section=chunk.section,
                            injection_flag=chunk.injection_flag,
                        ),
                    )
                    for chunk, vector, sparse_vector in zip(batch, dense, sparse, strict=True)
                ]
                await self._stage(version.id, IngestStatus.INDEXING)
                indexed += await self._vectors.upsert(collection.vector_namespace, points)
                await self._repos.chunks.mark_indexed(
                    [c.id for c in batch],
                    at=datetime.now(UTC),
                    model=collection.embedding_model,
                    point_ids={c.id: p.id for c, p in zip(batch, points, strict=True)},
                )
                await self._repos.uow.commit()
            span["indexed"] = indexed
        report.timings_ms["embed"] = span["duration_ms"]
        if indexed != len(chunks):
            logger.warning(
                "ingest.partial_index",
                expected=len(chunks),
                indexed=indexed,
                version_id=str(version.id),
            )

    async def _activate(
        self, document: Document, version: DocumentVersion, report: IngestReport
    ) -> None:
        """Publish the version: flip ``is_active`` in the index, then in PostgreSQL.

        Index first. If the process dies between the two, retrieval briefly returns chunks of
        a version PostgreSQL does not consider active — and layer-2 ACL re-verification drops
        them, so the failure mode is a missing answer rather than an unauthorized one. The
        reverse order would have the opposite, unacceptable, failure mode.
        """
        from aegis.domain.values import Match, VectorFilter

        collection = await self._repos.collections.get(document.collection_id)
        if collection is None:  # pragma: no cover - checked by the caller
            raise NotFoundError("Collection", document.collection_id)

        previous = document.active_version_id
        await self._vectors.set_payload(
            collection.vector_namespace,
            VectorFilter(must=(Match("version_id", str(version.id)),), min_should=0),
            {"is_active": True},
        )
        if previous and previous != version.id:
            # Deactivate then delete: deactivation is instant and stops serving, deletion is
            # the housekeeping that follows.
            await self._vectors.set_payload(
                collection.vector_namespace,
                VectorFilter(must=(Match("version_id", str(previous)),), min_should=0),
                {"is_active": False},
            )

        await self._repos.versions.update_status(
            version.id, status=IngestStatus.INDEXED, indexed_at=datetime.now(UTC)
        )
        await self._repos.documents.set_active_version(document.id, version.id)
        if previous and previous != version.id:
            await self._repos.versions.mark_superseded(previous, at=datetime.now(UTC))
        await self._repos.uow.commit()
        _ = report

    # ── maintenance ─────────────────────────────────────────────────────────

    async def purge_document(self, document_id: UUID) -> int:
        """Remove a document's vectors. Called on delete, before the row is soft-deleted."""
        from aegis.domain.values import Match, VectorFilter

        document = await self._repos.documents.get(document_id, include_deleted=True)
        if document is None:
            return 0
        collection = await self._repos.collections.get(document.collection_id)
        if collection is None:
            return 0
        removed = await self._vectors.delete_by_filter(
            collection.vector_namespace,
            VectorFilter(must=(Match("document_id", str(document_id)),), min_should=0),
        )
        logger.info("purge.done", document_id=str(document_id), vectors_removed=removed)
        return removed

    async def refresh_acl(self, document_id: UUID) -> int:
        """Patch the vector payload after an ACL change, without re-embedding.

        This is the operation that makes correct ACL maintenance cheap. Re-embedding a
        300-chunk document because its visibility changed would take minutes and cost money,
        and anything that expensive gets avoided — which is how stale ACLs happen.
        """
        from aegis.domain.values import Match, VectorFilter
        from aegis.rag.vector_stores.payload import acl_payload

        document = await self._repos.documents.get(document_id)
        if document is None:
            return 0
        collection = await self._repos.collections.get(document.collection_id)
        if collection is None:
            return 0
        patched = await self._vectors.set_payload(
            collection.vector_namespace,
            VectorFilter(must=(Match("document_id", str(document_id)),), min_should=0),
            acl_payload(
                visibility_level=int(document.visibility.level),
                department_path=document.department_path,
                is_active=document.is_retrievable,
                mode=collection.mode.value,
                expires_at=document.expires_at,
                effective_from=document.effective_from,
            ),
        )
        logger.info("acl.payload_patched", document_id=str(document_id), points=patched)
        return patched

    async def reindex_document(
        self, document_id: UUID, *, force_reparse: bool = False
    ) -> IngestReport:
        """Re-embed a document's stored chunks, or re-run the whole pipeline.

        Re-embedding from stored chunk text — rather than re-parsing the original — is what
        makes an embedding-model migration hours instead of days, and possible at all when the
        original files have been archived.
        """
        document = await self._repos.documents.get(document_id)
        if document is None or document.active_version_id is None:
            raise NotFoundError("Document", document_id)
        if force_reparse:
            return await self.ingest(document.active_version_id)

        version = await self._repos.versions.get(document.active_version_id)
        collection = await self._repos.collections.get(document.collection_id)
        if version is None or collection is None:  # pragma: no cover
            raise NotFoundError("Version", document.active_version_id)

        chunks = await self._repos.chunks.list_for_version(version.id, limit=10_000)
        report = IngestReport(
            version_id=version.id, status=IngestStatus.EMBEDDING, chunks=len(chunks)
        )
        await self._embed_and_index(list(chunks), collection, document, version, report)
        await self._activate(document, version, report)
        report.status = IngestStatus.INDEXED
        return report


def stage_of(exc: BaseException) -> str:
    return str(getattr(exc, "stage", "unknown"))


def is_retryable(exc: BaseException) -> bool:
    """Whether the worker should try this job again.

    Defaults to *not* retrying an unknown error. A permanent failure retried three times
    occupies a queue slot for a quarter of an hour and buries the real cause under two
    duplicate stack traces.
    """
    return bool(getattr(exc, "retryable", False))
