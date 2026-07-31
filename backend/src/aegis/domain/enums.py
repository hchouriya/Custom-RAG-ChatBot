"""Domain enumerations.

``StrEnum`` throughout so values serialize as their PostgreSQL enum labels with no
conversion layer, and so a log line or an API payload reads as ``"internal_employee"``
rather than ``"Role.INTERNAL_EMPLOYEE"``.

The important type here is :class:`Visibility`. It is a *total order*, and that ordering
is what makes ACL filtering a single integer range comparison inside the vector search
instead of set membership over N labels. The ``level`` property is the only place the
mapping from label to integer exists.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    MANAGER = "manager"
    INTERNAL_EMPLOYEE = "internal_employee"
    CUSTOMER = "customer"
    GUEST = "guest"

    @property
    def is_internal(self) -> bool:
        return self in {Role.ADMIN, Role.MANAGER, Role.INTERNAL_EMPLOYEE}

    @property
    def rank(self) -> int:
        """Administrative seniority. Not an access level — see ``Visibility``."""
        return {
            Role.GUEST: 0,
            Role.CUSTOMER: 1,
            Role.INTERNAL_EMPLOYEE: 2,
            Role.MANAGER: 3,
            Role.ADMIN: 4,
        }[self]


class VisibilityLevel(IntEnum):
    """Numeric form of :class:`Visibility`, stored denormalized for range filtering."""

    PUBLIC = 0
    CUSTOMER = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    RESTRICTED = 4


class Visibility(StrEnum):
    PUBLIC = "public"
    CUSTOMER = "customer"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def level(self) -> VisibilityLevel:
        return {
            Visibility.PUBLIC: VisibilityLevel.PUBLIC,
            Visibility.CUSTOMER: VisibilityLevel.CUSTOMER,
            Visibility.INTERNAL: VisibilityLevel.INTERNAL,
            Visibility.CONFIDENTIAL: VisibilityLevel.CONFIDENTIAL,
            Visibility.RESTRICTED: VisibilityLevel.RESTRICTED,
        }[self]

    @classmethod
    def from_level(cls, level: int) -> Visibility:
        for member in cls:
            if member.level == level:
                return member
        raise ValueError(f"no visibility with level {level}")

    @property
    def requires_department_match(self) -> bool:
        """Confidential content is scoped to the owning department subtree."""
        return self is Visibility.CONFIDENTIAL

    @property
    def requires_explicit_grant(self) -> bool:
        """Restricted content is reachable only through ``document_acl``."""
        return self is Visibility.RESTRICTED


class Mode(StrEnum):
    """Which assistant is being used. A hard ceiling, not a preference."""

    INTERNAL = "internal"
    CUSTOMER = "customer"


class ChunkType(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    CODE = "code"
    LIST = "list"
    CAPTION = "caption"
    FORM = "form"
    OCR = "ocr"

    @property
    def is_structured(self) -> bool:
        """Structured chunks must never be sentence-compressed or split mid-block.

        A table with rows silently removed is worse than no table at all.
        """
        return self in {ChunkType.TABLE, ChunkType.CODE, ChunkType.FORM}


class IngestStatus(StrEnum):
    PENDING = "pending"
    SCANNING = "scanning"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"

    @property
    def is_terminal(self) -> bool:
        return self in {
            IngestStatus.INDEXED,
            IngestStatus.FAILED,
            IngestStatus.QUARANTINED,
            IngestStatus.SUPERSEDED,
        }

    @property
    def is_searchable(self) -> bool:
        return self is IngestStatus.INDEXED


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"


class JobName(StrEnum):
    """Background job names.

    One vocabulary for three things that must agree: the queue's function name, the
    ``ingest_jobs.job_type`` column, and the ``job_type`` metric label. Keeping them as one
    enum means a renamed worker function cannot silently orphan queued jobs.
    """

    INGEST = "ingest"
    REINDEX = "reindex"
    PURGE = "purge"
    RECONCILE = "reconcile"
    RETENTION = "retention"
    ROLLUP = "rollup"
    EVALUATE = "evaluate"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AnswerStatus(StrEnum):
    OK = "ok"
    NO_ANSWER = "no_answer"
    REFUSED = "refused"
    CLARIFY = "clarify"
    ESCALATED = "escalated"
    ERROR = "error"

    @property
    def is_grounded_answer(self) -> bool:
        return self is AnswerStatus.OK


class PrincipalType(StrEnum):
    ROLE = "role"
    USER = "user"
    DEPARTMENT = "department"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"


class EvalMetric(StrEnum):
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CITATION_CORRECTNESS = "citation_correctness"
    ANSWER_CORRECTNESS = "answer_correctness"


class Intent(StrEnum):
    """Query intent. Drives whether retrieval runs at all."""

    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    SUMMARIZATION = "summarization"
    AGGREGATION = "aggregation"
    FOLLOWUP = "followup"
    GREETING = "greeting"
    FEEDBACK = "feedback"
    OUT_OF_SCOPE = "out_of_scope"
    UNSAFE = "unsafe"

    @property
    def needs_retrieval(self) -> bool:
        """Roughly a sixth of production traffic is small talk.

        Answering "hi" should not cost an embedding call, a hybrid search, a rerank, and
        a six-thousand-token prompt.
        """
        return self not in {
            Intent.GREETING,
            Intent.FEEDBACK,
            Intent.OUT_OF_SCOPE,
            Intent.UNSAFE,
        }


class Permission(StrEnum):
    """Administrative capabilities, as ``resource:action``.

    Distinct from ``Role``/``Visibility``, which govern *retrieval*. Permissions govern
    what a principal may do to the platform; visibility governs what they may read from
    it. Conflating the two produces either an unbounded set in the hot retrieval filter
    or an admin panel that cannot express real organisations.
    """

    CHAT_CUSTOMER = "chat:customer"
    CHAT_INTERNAL = "chat:internal"
    TICKET_CREATE = "ticket:create"
    TICKET_MANAGE = "ticket:manage"

    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_DOWNLOAD = "document:download"
    DOCUMENT_REINDEX = "document:reindex"
    ACL_MANAGE = "acl:manage"

    COLLECTION_READ = "collection:read"
    COLLECTION_MANAGE = "collection:manage"

    USER_READ = "user:read"
    USER_MANAGE = "user:manage"
    ROLE_MANAGE = "role:manage"
    APIKEY_MANAGE = "apikey:manage"

    ANALYTICS_READ = "analytics:read"
    ANALYTICS_READ_ALL = "analytics:read_all"
    AUDIT_READ = "audit:read"
    TRACE_READ = "trace:read"
    RETRIEVAL_DEBUG = "retrieval:debug"

    PROMPT_MANAGE = "prompt:manage"
    SETTINGS_MANAGE = "settings:manage"
    EVAL_RUN = "eval:run"
    INDEX_MANAGE = "index:manage"
