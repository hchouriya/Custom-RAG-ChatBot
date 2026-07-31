"""Chat endpoints, including SSE message streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter

from aegis.agents.state import (
    CitationsEvent,
    ClarifyEvent,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    RefusalEvent,
    StreamEvent,
    TokenEvent,
    UsageEvent,
)
from aegis.api.deps import (
    ChatServiceDep,
    ContainerDep,
    PrincipalDep,
    ReposDep,
    enforce_rate_limit,
)
from aegis.api.schemas import (
    CitationOut,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationUpdate,
    FeedbackRequest,
    MessageCreate,
    MessageOut,
    PageOut,
    SuggestionsResponse,
)
from aegis.core.errors import ValidationError
from aegis.core.pagination import Cursor, Page
from aegis.core.ratelimit import LimitBucket
from aegis.domain.values import Citation
from aegis.services.chat import require_chat_permission
from aegis.services.documents import visible_collections

router = APIRouter(prefix="/chat", tags=["chat"])

_EVENT_NAME: dict[type, str] = {
    MetaEvent: "meta",
    CitationsEvent: "citations",
    TokenEvent: "token",
    UsageEvent: "usage",
    RefusalEvent: "refusal",
    ClarifyEvent: "clarify",
    DoneEvent: "done",
    ErrorEvent: "error",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Citation):
        return {
            "marker": value.marker,
            "document_id": str(value.locator.document_id),
            "document_title": value.locator.document_title,
            "version_id": str(value.locator.version_id),
            "version_no": value.locator.version_no,
            "page": value.locator.page,
            "section": value.locator.section,
            "heading_path": list(value.locator.heading_path),
            "quote": value.quote,
            "score_rerank": value.score_rerank,
            "was_used": value.was_used,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _sse(event: StreamEvent) -> str:
    name = _EVENT_NAME[type(event)]
    payload = _jsonable(event)
    return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


@router.get("/conversations", response_model=PageOut)
async def list_conversations(
    principal: PrincipalDep,
    chat: ChatServiceDep,
    *,
    q: str | None = None,
    archived: bool | None = False,
    pinned: bool | None = None,
    limit: int = Query(default=30, ge=1, le=100),
    cursor: str | None = None,
) -> PageOut:
    require_chat_permission(principal)
    decoded = Cursor.decode(cursor) if cursor else None
    rows = await chat.list_conversations(
        principal,
        search=q,
        archived=archived,
        pinned=pinned,
        limit=limit + 1,
        cursor_last_message_at=decoded.last_value if decoded else None,
        cursor_id=decoded.last_id if decoded else None,
    )
    page = Page[ConversationOut].build(
        TypeAdapter(list[ConversationOut]).validate_python(rows),
        limit=limit,
        cursor_of=lambda c: Cursor(last_value=c.last_message_at or c.created_at, last_id=c.id),
    )
    return PageOut(
        items=page.items,
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    principal: PrincipalDep,
    chat: ChatServiceDep,
    repos: ReposDep,
) -> ConversationOut:
    require_chat_permission(principal)
    conversation = await chat.create_conversation(
        principal, collection_ids=body.collection_ids or None, title=body.title
    )
    await repos.uow.commit()
    return ConversationOut.model_validate(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID,
    principal: PrincipalDep,
    chat: ChatServiceDep,
) -> ConversationDetail:
    require_chat_permission(principal)
    conversation = await chat.get_conversation(conversation_id, principal)
    messages, citations = await chat.messages(conversation_id, principal)
    citation_map: dict[str, list[CitationOut]] = {}
    for mid, entries in citations.items():
        citation_map[str(mid)] = [
            CitationOut(
                marker=c.marker,
                document_id=c.document_id,
                version_id=c.version_id,
                page=c.page,
                quote=c.quote,
                score_rerank=c.score_rerank,
                was_used=c.was_used,
            )
            for c in entries
        ]
    return ConversationDetail(
        **ConversationOut.model_validate(conversation).model_dump(),
        messages=TypeAdapter(list[MessageOut]).validate_python(messages),
        citations=citation_map,
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    conversation_id: UUID,
    body: ConversationUpdate,
    principal: PrincipalDep,
    chat: ChatServiceDep,
    repos: ReposDep,
) -> ConversationOut:
    require_chat_permission(principal)
    fields = body.model_dump(exclude_unset=True)
    updated = await chat.update_conversation(conversation_id, principal, **fields)
    await repos.uow.commit()
    return ConversationOut.model_validate(updated)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    principal: PrincipalDep,
    chat: ChatServiceDep,
) -> None:
    require_chat_permission(principal)
    await chat.delete_conversation(conversation_id, principal)


@router.post("/conversations/{conversation_id}/messages")
async def post_message(
    conversation_id: UUID,
    body: MessageCreate,
    request: Request,
    principal: PrincipalDep,
    chat: ChatServiceDep,
    repos: ReposDep,
    container: ContainerDep,
) -> StreamingResponse:
    require_chat_permission(principal)
    await enforce_rate_limit(
        request,
        container,
        LimitBucket.CHAT,
        principal_key=str(principal.user_id or principal.ctx.session_id),
        role=principal.role.value,
    )
    if not body.stream:
        raise ValidationError("Non-streaming responses are not supported yet; set stream=true.")

    all_collections = await repos.collections.list_for_mode(principal.ctx.mode)
    if body.collection_ids:
        wanted = set(body.collection_ids)
        collections = [c for c in all_collections if c.id in wanted]
    else:
        collections = visible_collections(all_collections, principal.ctx.mode)

    options = body.options or None

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event, state in chat.ask(
                conversation_id,
                principal,
                question=body.content,
                collections=collections,
                model=options.model if options else None,
                temperature=options.temperature if options else None,
                max_citations=options.max_citations if options else 8,
            ):
                mid = state.request.message_id
                if mid is not None:
                    await container.cache.set(
                        f"active_stream:{conversation_id}",
                        str(mid).encode(),
                        ttl_seconds=300,
                    )
                yield _sse(event)
        finally:
            await container.cache.delete(f"active_stream:{conversation_id}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/conversations/{conversation_id}/stream", status_code=204)
async def cancel_stream(
    conversation_id: UUID,
    principal: PrincipalDep,
    chat: ChatServiceDep,
    container: ContainerDep,
) -> None:
    require_chat_permission(principal)
    await chat.get_conversation(conversation_id, principal)
    raw = await container.cache.get(f"active_stream:{conversation_id}")
    if raw is not None:
        await chat.cancel(UUID(raw.decode()))


@router.post("/messages/{message_id}/feedback", status_code=204)
async def feedback(
    message_id: UUID,
    body: FeedbackRequest,
    principal: PrincipalDep,
    chat: ChatServiceDep,
) -> None:
    require_chat_permission(principal)
    await chat.feedback(
        message_id,
        principal,
        rating=body.rating,
        reason=body.reason,
        comment=body.comment,
    )


@router.get("/suggestions", response_model=SuggestionsResponse)
async def suggestions(principal: PrincipalDep, chat: ChatServiceDep) -> SuggestionsResponse:
    require_chat_permission(principal)
    return SuggestionsResponse(suggestions=list(chat.suggestions(principal)))
