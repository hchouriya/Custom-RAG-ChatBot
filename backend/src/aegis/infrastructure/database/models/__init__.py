"""SQLAlchemy models.

Every model must be imported here so that ``Base.metadata`` is complete before Alembic
autogenerate or ``create_all`` runs. A model that is only imported by the module that uses it
is invisible to migrations — and the symptom is a missing table in production, not an error
in development.
"""

from aegis.infrastructure.database.models.base import Base
from aegis.infrastructure.database.models.conversations import (
    ConversationModel,
    FeedbackModel,
    MessageCitationModel,
    MessageModel,
    OutboxEventModel,
    SupportTicketModel,
)
from aegis.infrastructure.database.models.documents import (
    ChunkModel,
    CollectionModel,
    DocumentAclModel,
    DocumentModel,
    DocumentTagModel,
    DocumentVersionModel,
    IndexDiscrepancyModel,
    IngestJobModel,
    TagModel,
)
from aegis.infrastructure.database.models.identity import (
    ApiKeyModel,
    DepartmentModel,
    PermissionModel,
    RefreshTokenModel,
    RolePermissionModel,
    UserModel,
    UserPermissionOverrideModel,
)
from aegis.infrastructure.database.models.telemetry import (
    AuditLogModel,
    EvalCaseModel,
    EvalDatasetModel,
    EvalResultModel,
    EvalRunModel,
    PromptTemplateModel,
    QueryTraceModel,
    SettingModel,
)
from aegis.infrastructure.database.models.vectors import ChunkEmbeddingModel

__all__ = [
    "ApiKeyModel",
    "AuditLogModel",
    "Base",
    "ChunkEmbeddingModel",
    "ChunkModel",
    "CollectionModel",
    "ConversationModel",
    "DepartmentModel",
    "DocumentAclModel",
    "DocumentModel",
    "DocumentTagModel",
    "DocumentVersionModel",
    "EvalCaseModel",
    "EvalDatasetModel",
    "EvalResultModel",
    "EvalRunModel",
    "FeedbackModel",
    "IndexDiscrepancyModel",
    "IngestJobModel",
    "MessageCitationModel",
    "MessageModel",
    "OutboxEventModel",
    "PermissionModel",
    "PromptTemplateModel",
    "QueryTraceModel",
    "RefreshTokenModel",
    "RolePermissionModel",
    "SettingModel",
    "SupportTicketModel",
    "TagModel",
    "UserModel",
    "UserPermissionOverrideModel",
]
