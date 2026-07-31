"""Conversations, and the persistence around one answer.

The service owns three things the pipeline deliberately does not: conversation ownership,
what gets written to the database, and the stream-cancellation flag.

Persistence happens **after** the stream finishes, in one transaction, including on failure.
A partial answer is stored with ``status='error'`` and ``finish_reason='cancelled'`` rather
than discarded, because history that omits what the user actually saw is worse than no
history — it turns "the assistant told me X" into an unresolvable dispute.

Citations are stored for retrieved-but-unused sources too. "What did the model ignore" is the
question that explains most bad answers, and it is unanswerable after the fact unless the
whole evidence set was recorded at the time.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aegis.agents.state import AnswerRequest
from aegis.core.errors import AuthorizationError, ConflictError, NotFoundError, ValidationError
from aegis.core.ids import new_id
from aegis.core.logging import get_logger
from aegis.core.telemetry import feedback_total
from aegis.domain.entities import Conversation, Message, MessageCitation
from aegis.domain.enums import AnswerStatus, FeedbackRating, MessageRole, Mode, Permission
from aegis.domain.ports.infrastructure import ChatMessage
from aegis.rag.prompts.templates import TITLE_PROMPT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from aegis.agents.pipeline import AnswerPipeline
    from aegis.agents.state import AnswerState, StreamEvent
    from aegis.core.config import Settings
    from aegis.domain.entities import Collection
    from aegis.domain.ports.infrastructure import Cache
    from aegis.domain.ports.repositories import Repositories
    from aegis.rag.llm.router import LLMRouter
    from aegis.services.principal import Principal

logger = get_logger(__name__)

CANCEL_TTL = 300
MAX_HISTORY_TURNS = 6
STARTER_QUESTIONS: dict[Mode, tuple[str, ...]] = {
    Mode.INTERNAL: (
        "What is our parental leave entitlement?",
        "How do I request access to a production system?",
        "What is the expense approval limit for a manager?",
        "Which security training is mandatory this year?",
    ),
    Mode.CUSTOMER: (
        "What is your refund policy?",
        "How do I reset my password?",
        "Which plans include priority support?",
        "How do I export my data?",
    ),
}


class ChatService:
    def __init__(
        self,
        repos: Repositories,
        *,
        pipeline: AnswerPipeline,
        settings: Settings,
        cache: Cache,
    ) -> None:
        self._repos = repos
        self._pipeline = pipeline
        self._settings = settings
        self._cache = cache
        self._cancelled_ids: set[UUID] = set()

    # ── conversations ───────────────────────────────────────────────────────

    async def create_conversation(
        self,
        principal: Principal,
        *,
        collection_ids: list[UUID] | None = None,
        title: str | None = None,
    ) -> Conversation:
        return await self._repos.conversations.create(
            Conversation(
                id=new_id(),
                user_id=principal.user_id,
                guest_session_id=str(principal.ctx.session_id) if principal.is_guest else None,
                mode=principal.ctx.mode,
                collection_ids=collection_ids or [],
                title=title.strip() if title else None,
            )
        )

    async def list_conversations(
        self,
        principal: Principal,
        *,
        search: str | None = None,
        archived: bool | None = False,
        pinned: bool | None = None,
        limit: int = 30,
        cursor_last_message_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[Conversation]:
        return await self._repos.conversations.list_for_principal(
            user_id=principal.user_id,
            guest_session_id=str(principal.ctx.session_id) if principal.is_guest else None,
            search=search,
            archived=archived,
            pinned=pinned,
            limit=limit,
            cursor_last_message_at=cursor_last_message_at,
            cursor_id=cursor_id,
        )

    async def get_conversation(self, conversation_id: UUID, principal: Principal) -> Conversation:
        conversation = await self._repos.conversations.get(conversation_id)
        if conversation is None or not self._owns(conversation, principal):
            # 404 rather than 403: whether a conversation id exists is not information a
            # stranger is entitled to.
            raise NotFoundError("Conversation", conversation_id)
        return conversation

    async def messages(
        self, conversation_id: UUID, principal: Principal, *, limit: int = 100
    ) -> tuple[list[Message], dict[UUID, list[MessageCitation]]]:
        await self.get_conversation(conversation_id, principal)
        messages = await self._repos.conversations.list_messages(conversation_id, limit=limit)
        citations = await self._repos.conversations.citations_for([m.id for m in messages])
        return messages, citations

    async def update_conversation(
        self, conversation_id: UUID, principal: Principal, **fields: Any
    ) -> Conversation:
        await self.get_conversation(conversation_id, principal)
        allowed = {k: v for k, v in fields.items() if k in {"title", "is_pinned", "is_archived"}}
        if title := allowed.get("title"):
            allowed["title"] = str(title).strip()[:200]
        return await self._repos.conversations.update(conversation_id, **allowed)

    async def delete_conversation(self, conversation_id: UUID, principal: Principal) -> None:
        await self.get_conversation(conversation_id, principal)
        await self._repos.conversations.soft_delete(conversation_id, at=datetime.now(UTC))
        await self._repos.uow.commit()

    def suggestions(self, principal: Principal) -> tuple[str, ...]:
        return STARTER_QUESTIONS[principal.ctx.mode]

    # ── one turn ────────────────────────────────────────────────────────────

    async def ask(
        self,
        conversation_id: UUID,
        principal: Principal,
        *,
        question: str,
        collections: Sequence[Collection],
        model: str | None = None,
        temperature: float | None = None,
        max_citations: int = 8,
        narrowing: Any = None,
        parent_id: UUID | None = None,
    ) -> AsyncIterator[tuple[StreamEvent, AnswerState]]:
        """Run one turn and stream its events, persisting the result at the end.

        The user's message is written *before* generation so that a crash mid-answer leaves a
        conversation whose last turn is a question — recoverable and honest — rather than
        losing the question entirely.
        """
        conversation = await self.get_conversation(conversation_id, principal)
        if conversation.mode is not principal.ctx.mode:
            # A conversation carries its mode. Continuing an internal thread from a customer
            # session would put internal content next to customer-visible content in one
            # transcript.
            raise ConflictError(
                f"This conversation belongs to the {conversation.mode.value} assistant."
            )
        if not question.strip():
            raise ValidationError("A question is required.")

        user_message = await self._repos.conversations.add_message(
            Message(
                id=new_id(),
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=question.strip(),
                parent_id=parent_id,
            )
        )
        await self._repos.uow.commit()

        history = await self._history(conversation_id, exclude=user_message.id)
        selected = [
            c
            for c in collections
            if not conversation.collection_ids or c.id in set(conversation.collection_ids)
        ]

        message_id = new_id()
        request = AnswerRequest(
            question=question.strip(),
            ctx=principal.ctx,
            collections=list(selected),
            conversation_id=conversation_id,
            message_id=message_id,
            history=history,
            summary=conversation.summary,
            narrowing=narrowing,
            model=model,
            temperature=temperature,
            max_citations=max_citations,
            stream=True,
        )
        await self._clear_cancel(message_id)
        watcher = asyncio.create_task(self._watch_cancel(message_id))

        final: AnswerState | None = None
        try:
            async for event, state in self._pipeline.run(
                request, is_cancelled=lambda: self._cancelled(message_id)
            ):
                final = state
                yield event, state
        finally:
            watcher.cancel()
            if final is not None:
                # In `finally` so that a client disconnect — which raises through the
                # generator — still records what was generated and what it cost.
                await self._persist(conversation, user_message, request.message_id, final)

    async def cancel(self, message_id: UUID) -> None:
        """Set the flag the generator polls between deltas."""
        await self._cache.set(f"cancel:{message_id}", b"1", ttl_seconds=CANCEL_TTL)

    async def regenerate(
        self,
        conversation_id: UUID,
        message_id: UUID,
        principal: Principal,
        *,
        collections: Sequence[Collection],
        model: str | None = None,
    ) -> AsyncIterator[tuple[StreamEvent, AnswerState]]:
        """Re-answer a question as a sibling branch.

        The previous answer is kept. Overwriting it would make the conversation a lie about
        what the user saw, and comparing two answers to the same question is one of the more
        useful signals we get about retrieval quality.
        """
        await self.get_conversation(conversation_id, principal)
        target = await self._repos.conversations.get_message(message_id)
        if target is None or target.conversation_id != conversation_id:
            raise NotFoundError("Message", message_id)

        question = target.content
        if target.role is MessageRole.ASSISTANT:
            previous = await self._repos.conversations.list_messages(conversation_id, limit=200)
            user_turn = next(
                (
                    m
                    for m in reversed(previous)
                    if m.role is MessageRole.USER
                    and m.created_at
                    and target.created_at
                    and m.created_at < target.created_at
                ),
                None,
            )
            if user_turn is None:
                raise ConflictError("There is no question to regenerate an answer for.")
            question = user_turn.content

        async for event, state in self.ask(
            conversation_id,
            principal,
            question=question,
            collections=collections,
            model=model,
            parent_id=message_id,
        ):
            yield event, state

    async def feedback(
        self,
        message_id: UUID,
        principal: Principal,
        *,
        rating: FeedbackRating,
        reason: str | None = None,
        comment: str | None = None,
    ) -> None:
        message = await self._repos.conversations.get_message(message_id)
        if message is None:
            raise NotFoundError("Message", message_id)
        conversation = await self._repos.conversations.get(message.conversation_id)
        if conversation is None or not self._owns(conversation, principal):
            raise NotFoundError("Message", message_id)
        await self._repos.conversations.set_feedback(
            message_id=message_id,
            user_id=principal.user_id,
            rating=rating.value,
            reason=reason,
            comment=(comment or "")[:2000] or None,
        )
        await self._repos.uow.commit()
        feedback_total.labels(rating=rating.value, mode=conversation.mode.value).inc()

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _owns(conversation: Conversation, principal: Principal) -> bool:
        return conversation.belongs_to(
            user_id=principal.user_id,
            guest_session_id=str(principal.ctx.session_id) if principal.is_guest else None,
        )

    async def _history(self, conversation_id: UUID, *, exclude: UUID) -> list[tuple[str, str]]:
        turns = await self._repos.conversations.recent_turns(
            conversation_id, limit=MAX_HISTORY_TURNS + 1
        )
        return [
            (m.role.value, m.content)
            for m in turns
            if m.id != exclude and m.role in {MessageRole.USER, MessageRole.ASSISTANT}
        ][-MAX_HISTORY_TURNS:]

    def _cancelled(self, message_id: UUID) -> bool:
        """Synchronous view of the cancel flag.

        The pipeline polls this between token deltas, so it must not await: one Redis read
        per token would add a round trip to every few characters of output. A background task
        refreshes it instead, which trades up to a second of cancellation latency for no
        per-token cost.
        """
        return message_id in self._cancelled_ids

    async def _clear_cancel(self, message_id: UUID) -> None:
        self._cancelled_ids.discard(message_id)
        await self._cache.delete(f"cancel:{message_id}")

    async def _watch_cancel(self, message_id: UUID, *, interval: float = 1.0) -> None:
        """Poll the distributed cancel flag until it is set or the task is cancelled.

        The flag lives in Redis because the ``DELETE .../stream`` request that sets it may
        land on a different API replica than the one holding the stream.
        """
        while True:
            if await self._cache.get(f"cancel:{message_id}") is not None:
                self._cancelled_ids.add(message_id)
                return
            await asyncio.sleep(interval)

    async def _persist(
        self,
        conversation: Conversation,
        user_message: Message,
        message_id: UUID,
        state: AnswerState,
    ) -> None:
        usage = state.usage
        status = state.status
        assistant = await self._repos.conversations.add_message(
            Message(
                id=message_id,
                conversation_id=conversation.id,
                parent_id=user_message.id,
                role=MessageRole.ASSISTANT,
                content=state.answer,
                status=status,
                refusal_reason=state.refusal_reason,
                model=usage.model if usage else None,
                provider=usage.provider if usage else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                cost_usd=Decimal(str(round(usage.cost_usd, 6))) if usage else None,
                latency_ms=state.elapsed_ms,
                ttft_ms=usage.ttft_ms if usage else None,
                confidence=state.confidence.score if state.confidence else None,
                is_grounded=bool(state.citations and status is AnswerStatus.OK),
                finish_reason=usage.finish_reason if usage else None,
            )
        )

        if state.citations:
            await self._repos.conversations.add_citations(
                [
                    MessageCitation(
                        id=new_id(),
                        message_id=assistant.id,
                        chunk_id=citation.chunk_id,
                        document_id=citation.locator.document_id,
                        version_id=citation.locator.version_id,
                        marker=citation.marker,
                        rank=citation.rank,
                        quote=citation.quote or None,
                        quote_start=citation.quote_start,
                        quote_end=citation.quote_end,
                        page=citation.locator.page_from,
                        score_rerank=citation.score_rerank,
                        was_used=citation.was_used,
                    )
                    for citation in state.citations
                ]
            )

        tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
        title = None
        if not conversation.title:
            title = self._quick_title(user_message.content)
        await self._repos.conversations.record_turn(
            conversation.id, at=datetime.now(UTC), tokens=tokens, title=title
        )
        await self._repos.uow.commit()
        logger.info(
            "chat.turn_persisted",
            conversation_id=str(conversation.id),
            message_id=str(assistant.id),
            status=status.value,
            citations=len([c for c in state.citations if c.was_used]),
            tokens=tokens,
            latency_ms=state.elapsed_ms,
        )

    @staticmethod
    def _quick_title(question: str) -> str:
        """Derive a title without an LLM call.

        A generated title is nicer, but it costs a completion on the first turn of every
        conversation — the exact moment the user is waiting for their answer. Truncating the
        question is good enough, and the user can rename.
        """
        cleaned = " ".join(question.split())
        if len(cleaned) <= 60:
            return cleaned
        return cleaned[:57].rsplit(" ", 1)[0] + "…"

    async def generate_title(
        self, conversation_id: UUID, question: str, llm: LLMRouter
    ) -> str | None:
        """Optional LLM-generated title, for callers that want one out of band."""
        try:
            completion = await llm.complete(
                [ChatMessage(role="user", content=TITLE_PROMPT.format(question=question))],
                model=llm.fast_model,
                temperature=0.0,
                max_tokens=24,
            )
        # A title is cosmetic; no provider failure here is worth surfacing to the user.
        except Exception as exc:
            logger.info("chat.title_generation_failed", error=str(exc))
            return None
        title = completion.text.strip().strip('"').strip()[:80]
        if title:
            await self._repos.conversations.update(conversation_id, title=title)
            await self._repos.uow.commit()
        return title or None


def require_chat_permission(principal: Principal) -> None:
    """Assert the principal may use the assistant they asked for."""
    needed = (
        Permission.CHAT_INTERNAL
        if principal.ctx.mode is Mode.INTERNAL
        else Permission.CHAT_CUSTOMER
    )
    if not principal.has(needed):
        raise AuthorizationError(
            f"This session cannot use the {principal.ctx.mode.value} assistant."
        )
