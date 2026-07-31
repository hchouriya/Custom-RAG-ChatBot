"""Table serialisation.

A table has to become text to be embedded, and the encoding chosen determines whether a
question about a cell can ever retrieve the row containing it. Two encodings are produced
here for different purposes:

* **Markdown** — compact and familiar to every model, used when the table is small enough to
  fit in one chunk. Column headers stay adjacent to their values.
* **Row records** — ``"Region: EMEA | Q3 revenue: 4.2M"`` per row, used when a table is too
  large for one chunk. This is the encoding that actually retrieves: a chunk containing only
  ``"EMEA | 4.2M | 12% | 2026-03-31"`` shares no vocabulary with "what was EMEA revenue in
  Q3", while the record form shares most of it.

The row-record form costs roughly 2.5 times the tokens of raw rows. That is the price of a
retrievable table, and it is why the choice is made on size rather than always going one way.
"""

from __future__ import annotations

from collections.abc import Sequence

MAX_CELL_CHARS = 300
_MD_ESCAPE = str.maketrans({"|": "\\|", "\n": " ", "\r": " "})


def normalize_cell(value: object) -> str:
    """Stringify a cell, bounding pathological content.

    A single cell holding a 10 kB note would otherwise dominate its chunk and push out every
    other row.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > MAX_CELL_CHARS:
        return text[: MAX_CELL_CHARS - 1] + "…"
    return text


def to_markdown(rows: Sequence[Sequence[object]], *, header: Sequence[object] | None = None) -> str:
    """Render as a Markdown table.

    When no header is supplied the first row is used, because a headerless table is far more
    often a table whose header the extractor failed to mark than a genuinely headerless one.
    """
    materialised = [[normalize_cell(c).translate(_MD_ESCAPE) for c in row] for row in rows]
    if header is None:
        if not materialised:
            return ""
        head, body = materialised[0], materialised[1:]
    else:
        head = [normalize_cell(c).translate(_MD_ESCAPE) for c in header]
        body = materialised

    width = max(len(head), *(len(r) for r in body)) if body else len(head)
    if width == 0:
        return ""

    def pad(row: list[str]) -> list[str]:
        return row + [""] * (width - len(row))

    lines = [
        "| " + " | ".join(pad(head)) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(pad(row)) + " |" for row in body)
    return "\n".join(lines)


def to_row_records(
    rows: Sequence[Sequence[object]],
    *,
    header: Sequence[object],
    separator: str = " | ",
) -> list[str]:
    """Render each row as ``"Header: value"`` pairs.

    Empty cells are dropped rather than rendered as ``"Header: "``: a sparse table would
    otherwise spend most of its tokens on labels for absent values.
    """
    labels = [normalize_cell(c) or f"column_{i + 1}" for i, c in enumerate(header)]
    records: list[str] = []
    for row in rows:
        parts = [
            f"{labels[i] if i < len(labels) else f'column_{i + 1}'}: {value}"
            for i, value in ((i, normalize_cell(c)) for i, c in enumerate(row))
            if value
        ]
        if parts:
            records.append(separator.join(parts))
    return records


def looks_like_header(row: Sequence[object]) -> bool:
    """Whether a row is plausibly a header.

    Short, non-numeric, non-empty cells. Used when an extractor gives rows without saying
    which one is the header — every CSV and most PDF tables.
    """
    cells = [normalize_cell(c) for c in row]
    if not cells or not any(cells):
        return False
    filled = [c for c in cells if c]
    if any(len(c) > 60 for c in filled):
        return False
    numeric = sum(1 for c in filled if _is_numeric(c))
    return numeric <= len(filled) // 3


def _is_numeric(value: str) -> bool:
    cleaned = value.replace(",", "").replace("%", "").replace("$", "").replace("€", "").strip()
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def table_dimensions(rows: Sequence[Sequence[object]]) -> tuple[int, int]:
    return len(rows), max((len(r) for r in rows), default=0)
