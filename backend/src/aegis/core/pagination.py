"""Keyset (cursor) pagination.

``OFFSET`` degrades linearly with depth and shifts pages under concurrent inserts, both
of which matter on tables that take tens of thousands of rows a day. Keyset pagination
is O(log n) at any depth and stable while the caller walks it.

The cursor is opaque to clients — base64 of the last sort value plus the last id — so
its shape stays an implementation detail we can change. It is *not* signed: it encodes
only values the caller already received, so tampering can reposition their own scan and
nothing else. Authorization is applied to the query regardless of the cursor.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, Field

from aegis.core.errors import BadRequestError

MAX_LIMIT = 200
DEFAULT_LIMIT = 50


@dataclass(frozen=True, slots=True)
class Cursor:
    """Position in a stably ordered scan.

    ``last_id`` is the tiebreaker: sort values are rarely unique (two documents created
    in the same millisecond) and without a unique tiebreaker keyset pagination can skip
    or repeat rows.
    """

    last_value: Any
    last_id: UUID

    def encode(self) -> str:
        raw = json.dumps(
            {"v": _serialize(self.last_value), "i": str(self.last_id)},
            separators=(",", ":"),
        )
        return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, token: str) -> Self:
        try:
            padded = token + "=" * (-len(token) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            return cls(last_value=payload["v"], last_id=UUID(payload["i"]))
        except (
            binascii.Error,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            raise BadRequestError("Malformed pagination cursor", code="INVALID_CURSOR") from exc


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


class PageParams(BaseModel):
    """Query parameters shared by every list endpoint."""

    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    cursor: str | None = None
    sort: str = "-created_at"

    @property
    def descending(self) -> bool:
        return self.sort.startswith("-")

    @property
    def sort_field(self) -> str:
        return self.sort.lstrip("-+")

    def decoded_cursor(self) -> Cursor | None:
        return Cursor.decode(self.cursor) if self.cursor else None

    def resolve_sort(self, allowed: set[str]) -> str:
        """Validate the sort field against an allowlist.

        Interpolating a client string into ``ORDER BY`` is SQL injection with extra
        steps, so the caller declares which columns are sortable and anything else is a
        400 rather than a surprise.
        """
        field = self.sort_field
        if field not in allowed:
            raise BadRequestError(
                f"Cannot sort by {field!r}. Allowed: {', '.join(sorted(allowed))}",
                code="INVALID_SORT",
            )
        return field


class Page[T](BaseModel):
    """One page of results.

    ``total_estimate`` is the planner's row estimate, not ``COUNT(*)``: an exact count
    over a filtered half-million-row table on every keystroke costs more than the page
    itself, and "about 1,284" is what the UI shows anyway.
    """

    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False
    total_estimate: int | None = None

    @classmethod
    def build(
        cls,
        rows: list[T],
        *,
        limit: int,
        cursor_of: Any,
        total_estimate: int | None = None,
    ) -> Page[T]:
        """Build a page from ``limit + 1`` fetched rows.

        Fetching one extra row is how ``has_more`` is determined without a second query.
        """
        has_more = len(rows) > limit
        items = rows[:limit]
        next_cursor = cursor_of(items[-1]).encode() if has_more and items else None
        return cls(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
            total_estimate=total_estimate,
        )
