"""The machinery every chunking strategy shares.

The strategies differ in one thing only: where a chunk is *allowed* to end. Sentence
boundaries for prose, heading boundaries for Markdown, embedding-distance minima for semantic,
row boundaries for tables, definition boundaries for code. Everything after that decision —
packing to a token budget, carrying overlap, merging an undersized tail, attaching heading
context, mapping page ranges — is identical, and is implemented once here.

That is why this module holds :class:`Segment` and :func:`pack`: a new strategy is a function
that emits segments with break hints, not a new splitter with its own budget arithmetic and
its own off-by-one bugs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.domain.enums import ChunkType
from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk

if TYPE_CHECKING:
    from aegis.domain.ports.parser import DocumentBlock
    from aegis.rag.chunking.tokenizer import TokenCounter

HEADING_SEPARATOR = " > "
CONTEXT_MAX_TOKENS = 64
"""Cap on the contextual header.

The header is prepended to every chunk before embedding, so its cost is paid once per chunk
across the whole corpus. A deep heading path from a long document would otherwise consume a
tenth of the chunk's budget with breadcrumbs.
"""

_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "inc",
        "ltd",
        "llc",
        "co",
        "corp",
        "dept",
        "est",
        "fig",
        "no",
        "sec",
        "art",
        "para",
        "approx",
        "cf",
        "al",
    }
)

STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "may",
        "more",
        "most",
        "must",
        "no",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "out",
        "over",
        "own",
        "same",
        "shall",
        "she",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
"""Function words, excluded from every lexical measure in this package.

Deliberately short and English-only. A long list starts removing terms that matter in a
policy corpus — "no", "not", and "shall" carry the meaning of a compliance clause — and the
measures here only need enough filtering that two sentences are not judged similar because
they both contain "the".
"""

_TERM = re.compile(r"[^\W_]{2,}", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])[\"'\u201d\u2019)\]]*\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_WORD_BREAK = re.compile(r"\s+")


@dataclass(slots=True)
class Segment:
    """The smallest unit a chunk boundary may fall between.

    ``break_before`` is a hard instruction from the strategy: start a new chunk here even if
    the current one is under budget. ``atomic`` is the opposite guarantee — this text must
    never be split internally, whatever the budget says, because a half table row or a
    truncated function signature is worse than an oversized chunk.
    """

    text: str
    tokens: int
    char_start: int
    char_end: int
    page: int | None = None
    block_type: ChunkType = ChunkType.TEXT
    heading_path: tuple[str, ...] = ()
    separator: str = " "
    """Joined onto the *preceding* segment, so reassembled content resembles the source."""
    break_before: bool = False
    atomic: bool = False
    bbox: dict[str, float] | None = None
    language: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def term_counts(text: str, *, min_length: int = 2) -> dict[str, int]:
    """Content-word frequencies, shared by the semantic measure and keyword extraction."""
    counts: dict[str, int] = {}
    for match in _TERM.finditer(text.lower()):
        term = match.group(0)
        if len(term) < min_length or term in STOPWORDS:
            continue
        counts[term] = counts.get(term, 0) + 1
    return counts


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences with offsets, keeping abbreviations intact.

    Offsets are returned rather than just strings because they are the only way a citation can
    highlight the supporting sentence inside its paragraph. Recomputing them later with a
    substring search breaks on any text that repeats — which, in a policy document, is most of
    it.

    A regex is used rather than a sentence-boundary model: the model would be another
    downloaded artefact in the ingest path, and its wins are on informal prose that enterprise
    corpora contain little of. The abbreviation guard covers the failure that actually matters,
    which is "Rev. 3 of the policy" becoming two sentences.
    """
    if not text.strip():
        return []

    sentences: list[tuple[str, int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        candidate = text[start:end]
        if not _is_boundary(candidate, text[match.end() : match.end() + 1]):
            continue
        stripped = candidate.strip()
        if stripped:
            offset = candidate.index(stripped[0]) if stripped else 0
            sentences.append((stripped, start + offset, start + offset + len(stripped)))
        start = match.end()

    tail = text[start:]
    stripped = tail.strip()
    if stripped:
        offset = tail.index(stripped[0])
        sentences.append((stripped, start + offset, start + offset + len(stripped)))
    return sentences


def _is_boundary(candidate: str, following: str) -> bool:
    """Whether the period ending ``candidate`` really ends a sentence.

    Two signals, and both are needed. The abbreviation list catches "Dr. Smith", where the
    next word is capitalised and looks like a new sentence. The capitalisation of ``following``
    catches everything the list cannot enumerate — "Jan. 5", "Rev. 3", "no. 4" — because a
    sentence that starts with a lowercase letter or a digit is far rarer than an abbreviation
    followed by one.
    """
    if not candidate.strip():
        return False
    if _ends_with_abbreviation(candidate):
        return False
    return not following or following.isupper() or following in "\"'\u201c\u2018([\u00bf\u00a1"


def _ends_with_abbreviation(text: str) -> bool:
    if not text.endswith("."):
        return False
    last = text[:-1].rsplit(" ", 1)[-1].lower()
    # Single initials ("J. Smith") are mid-name far more often than terminal.
    return last in _ABBREVIATIONS or len(last) == 1


def split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Split on blank lines, with offsets."""
    result: list[tuple[str, int, int]] = []
    position = 0
    for piece in _PARAGRAPH_BREAK.split(text):
        start = text.find(piece, position) if piece else position
        stripped = piece.strip()
        if stripped:
            offset = piece.index(stripped[0])
            result.append((stripped, start + offset, start + offset + len(stripped)))
        position = start + len(piece)
    return result


def segments_from_block(
    block: DocumentBlock,
    counter: TokenCounter,
    *,
    break_before: bool = False,
) -> list[Segment]:
    """Turn a prose block into sentence segments carrying the block's locators."""
    segments: list[Segment] = []
    for para_index, (paragraph, para_start, _) in enumerate(split_paragraphs(block.text)):
        for sent_index, (sentence, rel_start, rel_end) in enumerate(split_sentences(paragraph)):
            absolute = block.char_start + para_start
            first = para_index == 0 and sent_index == 0
            segments.append(
                Segment(
                    text=sentence,
                    tokens=counter.count(sentence),
                    char_start=absolute + rel_start,
                    char_end=absolute + rel_end,
                    page=block.page,
                    block_type=block.block_type,
                    heading_path=block.heading_path,
                    separator="\n\n" if sent_index == 0 and para_index else " ",
                    break_before=break_before and first,
                    bbox=block.bbox if first else None,
                    language=block.language,
                )
            )
    return segments


def link_heading_layout(segments: list[Segment]) -> None:
    """Put headings on their own line in reassembled content.

    Cosmetic in isolation, load-bearing in aggregate: the LLM reads chunk content verbatim,
    and "4.2 Termination Employees may terminate" is a sentence that says something different
    from a heading followed by its clause.
    """
    for index, segment in enumerate(segments):
        if segment.block_type is not ChunkType.HEADING:
            continue
        segment.separator = "\n\n"
        if index + 1 < len(segments):
            segments[index + 1].separator = "\n"


def pack(
    segments: list[Segment],
    config: ChunkingConfig,
    counter: TokenCounter,
    *,
    chunk_type: ChunkType | None = None,
    overlap: bool = True,
) -> list[ProtoChunk]:
    """Greedily fill chunks to ``target_tokens``, respecting breaks, atoms, and ``max_tokens``.

    Greedy rather than optimal on purpose. An optimal packer minimises the number of chunks,
    which drifts boundaries away from the semantic breaks the strategy just computed; greedy
    packing keeps them, and the number of chunks is not what retrieval quality depends on.

    Overlap is carried as whole trailing segments of the previous chunk, never a fixed
    character count. A chunk that begins mid-sentence embeds as a fragment, and the fragment
    is what gets shown as a citation.
    """
    if not segments:
        return []

    chunks: list[ProtoChunk] = []
    current: list[Segment] = []
    used = 0
    carried = 0
    """Leading segments of ``current`` that are overlap, not new content.

    Tracked so the final flush can tell "a chunk still being filled" from "nothing left but
    the overlap tail", which would otherwise be emitted as a duplicate of the previous chunk.
    """

    def flush(*, carry_overlap: bool = True) -> None:
        nonlocal current, used, carried
        if len(current) <= carried:
            return
        chunks.append(_to_chunk(current, config, counter, chunk_type))
        current = _overlap_tail(current, config) if overlap and carry_overlap else []
        used = sum(s.tokens for s in current)
        carried = len(current)

    for segment in segments:
        pieces = (
            [segment]
            if segment.atomic or segment.tokens <= config.max_tokens
            else _hard_split(segment, config, counter)
        )
        for piece in pieces:
            wants_break = piece.break_before and used >= config.min_tokens
            if wants_break or used + piece.tokens > config.max_tokens:
                # Overlap exists to avoid cutting mid-thought. At a boundary the strategy
                # chose — a new section, a topic change — there is no thought to preserve, and
                # carrying text across it would put the previous section's content into this
                # chunk and defeat the boundary that was just detected.
                flush(carry_overlap=not wants_break)
                # An atom larger than the budget starts a chunk of its own; carrying overlap
                # into it would push it further over.
                if piece.atomic and piece.tokens > config.target_tokens:
                    current = []
                    used = 0
                    carried = 0
            current.append(piece)
            used += piece.tokens
            if used >= config.target_tokens:
                flush()

    if len(current) > carried:
        chunks.append(_to_chunk(current, config, counter, chunk_type))

    return _merge_undersized(chunks, config, counter)


def _hard_split(segment: Segment, config: ChunkingConfig, counter: TokenCounter) -> list[Segment]:
    """Split a single oversized non-atomic segment on whitespace.

    Reached by a sentence longer than ``max_tokens``: a wall-of-text paragraph with no
    punctuation, or a URL list. Splitting on words is lossless; refusing to split would push
    the chunk past the embedding model's input limit, where the provider truncates it silently.
    """
    words = _WORD_BREAK.split(segment.text)
    if len(words) <= 1:
        return [segment]

    pieces: list[Segment] = []
    buffer: list[str] = []
    tokens = 0
    cursor = segment.char_start
    for word in words:
        cost = counter.count(word) or 1
        if buffer and tokens + cost > config.target_tokens:
            text = " ".join(buffer)
            pieces.append(
                _clone(
                    segment, text=text, tokens=tokens, char_start=cursor, break_before=not pieces
                )
            )
            cursor += len(text) + 1
            buffer, tokens = [], 0
        buffer.append(word)
        tokens += cost
    if buffer:
        text = " ".join(buffer)
        pieces.append(
            _clone(segment, text=text, tokens=tokens, char_start=cursor, break_before=not pieces)
        )
    return pieces


def _clone(
    segment: Segment, *, text: str, tokens: int, char_start: int, break_before: bool
) -> Segment:
    return Segment(
        text=text,
        tokens=tokens,
        char_start=char_start,
        char_end=char_start + len(text),
        page=segment.page,
        block_type=segment.block_type,
        heading_path=segment.heading_path,
        separator=segment.separator,
        break_before=break_before or segment.break_before,
        atomic=segment.atomic,
        bbox=segment.bbox,
        language=segment.language,
        metadata=dict(segment.metadata),
    )


def _overlap_tail(segments: list[Segment], config: ChunkingConfig) -> list[Segment]:
    """Trailing segments worth at most ``overlap_tokens``, for the next chunk to start with."""
    budget = config.overlap_tokens
    if budget <= 0:
        return []
    tail: list[Segment] = []
    total = 0
    for segment in reversed(segments):
        if segment.atomic or total + segment.tokens > budget:
            break
        tail.append(segment)
        total += segment.tokens
    tail.reverse()
    # Overlap that is the entire previous chunk would produce a duplicate, which then
    # competes with the original in the ranking.
    return [] if len(tail) == len(segments) else tail


def _to_chunk(
    segments: list[Segment],
    config: ChunkingConfig,
    counter: TokenCounter,
    chunk_type: ChunkType | None,
) -> ProtoChunk:
    content = segments[0].text
    for segment in segments[1:]:
        content += segment.separator + segment.text

    heading_path = _heading_path(segments)
    pages = [s.page for s in segments if s.page is not None]
    resolved_type = chunk_type or _dominant_type(segments)

    return ProtoChunk(
        content=content,
        chunk_type=resolved_type,
        token_count=counter.count(content),
        page_from=min(pages) if pages else None,
        page_to=max(pages) if pages else None,
        heading_path=heading_path,
        section=heading_path[-1] if heading_path else None,
        char_start=segments[0].char_start,
        char_end=segments[-1].char_end,
        bbox=next((s.bbox for s in segments if s.bbox), None),
        context_header=(
            build_context_header(config.document_title, heading_path, counter)
            if config.contextual_headers
            else None
        ),
        language=next((s.language for s in segments if s.language), None),
        metadata={"segments": len(segments)},
    )


def _heading_path(segments: list[Segment]) -> tuple[str, ...]:
    """The section this chunk belongs to.

    The first segment is often the heading itself, and parsers do not agree on whether a
    heading's own path includes it. Taking the first non-empty path in order gives the same
    answer either way, and "the section this chunk starts in" is the answer a citation needs.
    """
    for segment in segments:
        if segment.heading_path:
            return segment.heading_path
    return ()


def _dominant_type(segments: list[Segment]) -> ChunkType:
    """The structured type wins if any segment has one.

    A chunk that is half prose and half table must be treated as a table downstream, because
    the compression and reranking paths are what would otherwise mangle the rows.
    """
    for segment in segments:
        if segment.block_type.is_structured:
            return segment.block_type
    return segments[0].block_type


def _merge_undersized(
    chunks: list[ProtoChunk], config: ChunkingConfig, counter: TokenCounter
) -> list[ProtoChunk]:
    """Fold chunks below ``min_tokens`` into their neighbour where the budget allows.

    Short chunks are the quiet killer of hybrid retrieval: a 40-token orphan scores highly on
    a two-word query by sheer term density, then contributes nothing to the answer. Structured
    chunks are exempt — a small table is complete, not a fragment.
    """
    if not chunks:
        return []

    merged: list[ProtoChunk] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        mergeable = (
            previous is not None
            and chunk.token_count < config.min_tokens
            and not chunk.chunk_type.is_structured
            and not previous.chunk_type.is_structured
            and previous.token_count + chunk.token_count <= config.max_tokens
            and previous.heading_path == chunk.heading_path
        )
        if mergeable and previous is not None:
            previous.content = f"{previous.content}\n\n{chunk.content}"
            previous.token_count = counter.count(previous.content)
            previous.char_end = max(previous.char_end, chunk.char_end)
            if chunk.page_to is not None:
                previous.page_to = max(previous.page_to or chunk.page_to, chunk.page_to)
            continue
        merged.append(chunk)

    for ordinal, chunk in enumerate(merged):
        chunk.ordinal = ordinal
    return merged


def build_context_header(
    document_title: str, heading_path: tuple[str, ...], counter: TokenCounter
) -> str | None:
    """Build the breadcrumb prepended to a chunk before embedding.

    Without it, a chunk reading "This does not apply to contractors" is unretrievable: the
    subject it refers to is in a heading two levels up. With it, the heading's terms are in
    the embedded text and in the BM25 index, which is where the query's terms will be.
    """
    parts = [p for p in (document_title, *heading_path) if p]
    if not parts:
        return None
    header = HEADING_SEPARATOR.join(parts)
    if counter.count(header) <= CONTEXT_MAX_TOKENS:
        return header
    # Drop from the middle: the document title and the immediate section are the two parts
    # that disambiguate, and the levels between them are the least informative.
    while len(parts) > 2 and counter.count(header) > CONTEXT_MAX_TOKENS:
        del parts[len(parts) // 2]
        header = HEADING_SEPARATOR.join(parts)
    return counter.truncate(header, CONTEXT_MAX_TOKENS)
