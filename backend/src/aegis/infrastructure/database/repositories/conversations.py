"""Conversation, message, citation, and feedback persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from aegis.core.errors import NotFoundError
from aegis.domain.entities import Conversation, Message, MessageCitation
from aegis.infrastructure.database.models import (
    ConversationModel,
    FeedbackModel,
    MessageCitationModel,
    MessageModel,
)


class SqlConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    # ── conversations ───────────────────────────────────────────────────────

    async def create(self, conversation: Conversation) -> Conversation:
        row = ConversationModel(
            **conversation.model_dump(
                exclude={
                    "created_at",
                    "last_message_at",
                    "deleted_at",
                    "message_count",
                    "total_tokens",
                }
            )
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return Conversation.model_validate(row)

    async def get(self, conversation_id: UUID) -> Conversation | None:
        row = await self._s.get(ConversationModel, conversation_id)
        if row is None or row.deleted_at is not None:
            return None
        return Conversation.model_validate(row)

    async def update(self, conversation_id: UUID, **fields: Any) -> Conversation:
        if not fields:
            existing = await self.get(conversation_id)
            if existing is None:
                raise NotFoundError("conversation", conversation_id)
            return existing
        stmt = (
            update(ConversationModel)
            .where(ConversationModel.id == conversation_id, ConversationModel.deleted_at.is_(None))
            .values(**fields)
            .returning(ConversationModel)
        )
        row = (await self._s.execute(stmt)).scalars().one_or_none()
        if row is None:
            raise NotFoundError("conversation", conversation_id)
        return Conversation.model_validate(row)

    async def soft_delete(self, conversation_id: UUID, *, at: datetime) -> None:
        await self._s.execute(
            update(ConversationModel)
            .where(ConversationModel.id == conversation_id)
            .values(deleted_at=at)
        )

    async def list_for_principal(
        self,
        *,
        user_id: UUID | None,
        guest_session_id: str | None,
        search: str | None = None,
        archived: bool | None = False,
        pinned: bool | None = None,
        limit: int = 30,
        cursor_last_message_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[Conversation]:
        stmt = select(ConversationModel).where(ConversationModel.deleted_at.is_(None))
        # Ownership is a `WHERE`, not a post-filter: a listing that fetches then filters is
        # one refactor away from returning someone else's conversations.
        if user_id is not None:
            stmt = stmt.where(ConversationModel.user_id == user_id)
        elif guest_session_id is not None:
            stmt = stmt.where(ConversationModel.guest_session_id == guest_session_id)
        else:
            return []

        if archived is not None:
            stmt = stmt.where(ConversationModel.is_archived.is_(archived))
        if pinned is not None:
            stmt = stmt.where(ConversationModel.is_pinned.is_(pinned))
        if search:
            stmt = stmt.where(ConversationModel.title.ilike(f"%{search}%"))
        if cursor_last_message_at is not None and cursor_id is not None:
            stmt = stmt.where(
                (ConversationModel.last_message_at < cursor_last_message_at)
                | (
                    (ConversationModel.last_message_at == cursor_last_message_at)
                    & (ConversationModel.id < cursor_id)
                )
            )
        stmt = stmt.order_by(
            ConversationModel.is_pinned.desc(),
            ConversationModel.last_message_at.desc().nullslast(),
            ConversationModel.id.desc(),
        ).limit(limit)
        rows = (await self._s.execute(stmt)).scalars().all()
        return [Conversation.model_validate(r) for r in rows]

    # ── messages ────────────────────────────────────────────────────────────

    async def add_message(self, message: Message) -> Message:
        row = MessageModel(**message.model_dump(exclude={"created_at", "edited_at"}))
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return Message.model_validate(row)

    async def update_message(self, message_id: UUID, **fields: Any) -> Message:
        stmt = (
            update(MessageModel)
            .where(MessageModel.id == message_id)
            .values(**fields)
            .returning(MessageModel)
        )
        row = (await self._s.execute(stmt)).scalars().one_or_none()
        if row is None:
            raise NotFoundError("message", message_id)
        return Message.model_validate(row)

    async def get_message(self, message_id: UUID) -> Message | None:
        row = await self._s.get(MessageModel, message_id)
        return None if row is None else Message.model_validate(row)

    async def list_messages(
        self, conversation_id: UUID, *, limit: int = 50, before: datetime | None = None
    ) -> list[Message]:
        stmt = select(MessageModel).where(MessageModel.conversation_id == conversation_id)
        if before is not None:
            stmt = stmt.where(MessageModel.created_at < before)
        stmt = stmt.order_by(MessageModel.created_at.asc(), MessageModel.id.asc()).limit(limit)
        rows = (await self._s.execute(stmt)).scalars().all()
        return [Message.model_validate(r) for r in rows]

    async def recent_turns(self, conversation_id: UUID, *, limit: int = 6) -> list[Message]:
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.desc(), MessageModel.id.desc())
            .limit(limit)
        )
        rows = list((await self._s.execute(stmt)).scalars().all())
        return [Message.model_validate(r) for r in reversed(rows)]

    # ── citations ───────────────────────────────────────────────────────────

    async def add_citations(self, citations: list[MessageCitation]) -> int:
        if not citations:
            return 0
        self._s.add_all([MessageCitationModel(**c.model_dump()) for c in citations])
        await self._s.flush()
        return len(citations)

    async def citations_for(self, message_ids: Any) -> dict[UUID, list[MessageCitation]]:
        ids = list(message_ids)
        if not ids:
            return {}
        stmt = (
            select(MessageCitationModel)
            .where(MessageCitationModel.message_id.in_(ids))
            .order_by(MessageCitationModel.marker.asc(), MessageCitationModel.rank.asc())
        )
        grouped: dict[UUID, list[MessageCitation]] = {}
        for row in (await self._s.execute(stmt)).scalars().all():
            grouped.setdefault(row.message_id, []).append(MessageCitation.model_validate(row))
        return grouped

    # ── counters and feedback ───────────────────────────────────────────────

    async def record_turn(
        self, conversation_id: UUID, *, at: datetime, tokens: int, title: str | None = None
    ) -> None:
        values: dict[str, Any] = {
            "message_count": ConversationModel.message_count + 1,
            "total_tokens": ConversationModel.total_tokens + tokens,
            "last_message_at": at,
        }
        if title:
            # Only fill an absent title: a user who renamed their conversation should not have
            # it overwritten by the next turn's generated title.
            values["title"] = func.coalesce(ConversationModel.title, title)
        await self._s.execute(
            update(ConversationModel)
            .where(ConversationModel.id == conversation_id)
            .values(**values)
        )

    async def set_feedback(
        self,
        *,
        message_id: UUID,
        user_id: UUID | None,
        rating: str,
        reason: str | None,
        comment: str | None,
    ) -> None:
        stmt = pg_insert(FeedbackModel).values(
            message_id=message_id,
            user_id=user_id,
            rating=rating,
            reason=reason,
            comment=comment,
        )
        # Changing a thumbs-down to a thumbs-up is an edit, not a second opinion.
        await self._s.execute(
            stmt.on_conflict_do_update(
                index_elements=["message_id", "user_id"],
                set_={
                    "rating": stmt.excluded.rating,
                    "reason": stmt.excluded.reason,
                    "comment": stmt.excluded.comment,
                },
            )
        )
