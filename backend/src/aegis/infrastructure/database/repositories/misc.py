"""Audit log and runtime settings repositories."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from aegis.domain.entities import AuditEntry
from aegis.infrastructure.database.models import AuditLogModel, SettingModel


class SqlAuditRepository:
    """Append-only audit trail.

    There is deliberately no ``update`` or ``delete`` here, and the migration grants the
    application role only ``INSERT`` and ``SELECT`` on the table. An audit log the
    application can rewrite is not an audit log — the absence of those methods is the
    control, not a gap.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def append(self, entry: AuditEntry) -> None:
        self._s.add(AuditLogModel(**entry.model_dump(exclude={"created_at"})))
        await self._s.flush()

    async def list_entries(
        self,
        *,
        actor_id: UUID | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        cursor_created_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[AuditEntry]:
        stmt = select(AuditLogModel)
        if actor_id is not None:
            stmt = stmt.where(AuditLogModel.actor_id == actor_id)
        if action is not None:
            stmt = stmt.where(AuditLogModel.action == action)
        if resource_type is not None:
            stmt = stmt.where(AuditLogModel.resource_type == resource_type)
        if resource_id is not None:
            stmt = stmt.where(AuditLogModel.resource_id == resource_id)
        # `since`/`until` bound the partition scan as well as the result: without them a
        # query touches every monthly partition that exists.
        if since is not None:
            stmt = stmt.where(AuditLogModel.created_at >= since)
        if until is not None:
            stmt = stmt.where(AuditLogModel.created_at < until)
        if cursor_created_at is not None and cursor_id is not None:
            stmt = stmt.where(
                (AuditLogModel.created_at < cursor_created_at)
                | ((AuditLogModel.created_at == cursor_created_at) & (AuditLogModel.id < cursor_id))
            )
        stmt = stmt.order_by(AuditLogModel.created_at.desc(), AuditLogModel.id.desc()).limit(limit)
        return [AuditEntry.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()]


class SqlSettingsRepository:
    """Runtime overrides for tunables that would otherwise need a redeploy.

    Values are stored as ``jsonb`` rather than text so a threshold stays a number and a
    list of enabled providers stays a list; the alternative is a parsing rule per key,
    duplicated between the writer and the reader.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, key: str) -> Any | None:
        stmt = select(SettingModel.value).where(SettingModel.key == key)
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def get_all(self) -> dict[str, Any]:
        rows = (await self._s.execute(select(SettingModel.key, SettingModel.value))).all()
        return {row[0]: row[1] for row in rows}

    async def set(self, key: str, value: Any, *, updated_by: UUID | None) -> None:
        stmt = pg_insert(SettingModel).values(key=key, value=value, updated_by=updated_by)
        await self._s.execute(
            stmt.on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "value": stmt.excluded.value,
                    "updated_by": stmt.excluded.updated_by,
                    "updated_at": func.now(),
                },
            )
        )

    async def delete(self, key: str) -> None:
        """Remove an override, restoring the environment default.

        Deleting rather than writing the default back matters: the admin UI distinguishes
        "operator chose this value" from "inherited from configuration", and only the
        absence of a row can express the second.
        """
        await self._s.execute(delete(SettingModel).where(SettingModel.key == key))
