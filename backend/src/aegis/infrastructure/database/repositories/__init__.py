"""SQLAlchemy repository implementations and the per-request bundle factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from aegis.domain.ports.repositories import Repositories
from aegis.infrastructure.database.engine import SqlAlchemyUnitOfWork
from aegis.infrastructure.database.repositories.conversations import SqlConversationRepository
from aegis.infrastructure.database.repositories.documents import (
    SqlAclRepository,
    SqlChunkRepository,
    SqlCollectionRepository,
    SqlDiscrepancyRepository,
    SqlDocumentRepository,
    SqlDocumentVersionRepository,
    SqlIngestJobRepository,
)
from aegis.infrastructure.database.repositories.identity import (
    SqlApiKeyRepository,
    SqlDepartmentRepository,
    SqlPermissionRepository,
    SqlRefreshTokenRepository,
    SqlUserRepository,
)
from aegis.infrastructure.database.repositories.misc import (
    SqlAuditRepository,
    SqlSettingsRepository,
)


def build_repositories(session: AsyncSession) -> Repositories:
    """Assemble the repository bundle over one session.

    All repositories share a single session, which is what makes a service able to write a
    document, its version, and its job inside one transaction. Constructing them is cheap —
    each is a thin object holding the session — so there is no lazy-initialisation
    machinery here to save work that does not exist.
    """
    return Repositories(
        uow=SqlAlchemyUnitOfWork(session),
        users=SqlUserRepository(session),
        departments=SqlDepartmentRepository(session),
        refresh_tokens=SqlRefreshTokenRepository(session),
        api_keys=SqlApiKeyRepository(session),
        permissions=SqlPermissionRepository(session),
        collections=SqlCollectionRepository(session),
        documents=SqlDocumentRepository(session),
        versions=SqlDocumentVersionRepository(session),
        chunks=SqlChunkRepository(session),
        acl=SqlAclRepository(session),
        conversations=SqlConversationRepository(session),
        jobs=SqlIngestJobRepository(session),
        discrepancies=SqlDiscrepancyRepository(session),
        audit=SqlAuditRepository(session),
        settings=SqlSettingsRepository(session),
    )


__all__ = [
    "SqlAclRepository",
    "SqlApiKeyRepository",
    "SqlAuditRepository",
    "SqlChunkRepository",
    "SqlCollectionRepository",
    "SqlConversationRepository",
    "SqlDepartmentRepository",
    "SqlDiscrepancyRepository",
    "SqlDocumentRepository",
    "SqlDocumentVersionRepository",
    "SqlIngestJobRepository",
    "SqlPermissionRepository",
    "SqlRefreshTokenRepository",
    "SqlSettingsRepository",
    "SqlUserRepository",
    "build_repositories",
]
