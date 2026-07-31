"""Small shared pieces for the SQL repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func

if TYPE_CHECKING:
    from sqlalchemy import CursorResult, Result


def affected(result: Result[Any]) -> int:
    """Rows touched by a DML statement.

    ``AsyncSession.execute`` is typed as returning ``Result``, which has no ``rowcount``;
    every ``UPDATE`` and ``DELETE`` in fact returns a ``CursorResult``. The cast is narrowing
    a type that is already correct at runtime rather than asserting anything new.
    """
    return int(cast("CursorResult[Any]", result).rowcount or 0)


def ltree(value: str) -> Any:
    """Cast a Python string to ``ltree`` so the containment operators can be used with it.

    ``text2ltree`` rather than ``CAST(:x AS ltree)`` because it keeps the value a bound
    parameter, which a cast in the SQL text would not.
    """
    return func.text2ltree(value)
