"""Token counting.

Chunk budgets are expressed in tokens rather than characters because both limits that matter
are token limits: the embedding model's input window and the prompt's context budget. Counting
characters and dividing by four is how a "800 token" chunk silently becomes 1,300 tokens of
dense tabular text and gets truncated by the provider mid-row.

``tiktoken`` is the counter when it can load its BPE table, and a structural estimator when it
cannot. The fallback is not optional politeness: ``tiktoken`` downloads its vocabulary on first
use, so an air-gapped deployment would otherwise fail at ingest time rather than at startup.
The estimator overshoots slightly by design — a chunk a little under budget costs recall, a
chunk over budget gets truncated.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol, runtime_checkable

from aegis.core.logging import get_logger

logger = get_logger(__name__)

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_SYMBOLS = re.compile(r"[^\w\s]", re.UNICODE)

_SUBWORD_CHARS = 6
"""Characters per subword piece for long words, from BPE behaviour on English and code."""

DEFAULT_ENCODING = "cl100k_base"

_MODEL_ENCODINGS: dict[str, str] = {
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
}
"""Explicit map for the models in use.

Models outside it fall back to ``cl100k_base``. Sentence-piece models (BGE, Voyage) have no
tiktoken equivalent, and cl100k is within a few percent of them on prose — close enough for a
budget with a max-token guard behind it, and the guard is what actually prevents truncation.
"""


@runtime_checkable
class TokenCounter(Protocol):
    """Counts and truncates by token, for whichever tokenizer the deployment can load."""

    name: str

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Cut ``text`` to at most ``max_tokens``, on a token boundary."""
        ...


class TiktokenCounter:
    """Exact counts from the model's own BPE table."""

    __slots__ = ("_encoding", "name")

    def __init__(self, encoding: object, name: str) -> None:
        self._encoding = encoding
        self.name = name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))  # type: ignore[attr-defined]

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        tokens = self._encoding.encode(text, disallowed_special=())  # type: ignore[attr-defined]
        if len(tokens) <= max_tokens:
            return text
        return str(self._encoding.decode(tokens[:max_tokens]))  # type: ignore[attr-defined]


class HeuristicCounter:
    """Structural estimate: words, punctuation, and subword splits for long words.

    Deliberately not ``len(text) / 4``. That ratio holds for English prose and breaks on
    exactly the content whose budget matters most — code, tables of numbers, and CJK text,
    where it underestimates by two to four times.
    """

    __slots__ = ("name",)

    def __init__(self) -> None:
        self.name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = _WORDS.findall(text)
        subwords = sum((len(w) - 1) // _SUBWORD_CHARS for w in words)
        symbols = len(_SYMBOLS.findall(text))
        # CJK has no spaces, so the word count above collapses a whole clause into one
        # "word"; roughly one token per character is the honest floor there.
        cjk = sum(1 for ch in text if "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef")
        return max(len(words) + subwords + symbols, cjk)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        if self.count(text) <= max_tokens:
            return text
        # Walk words, not characters: cutting mid-word produces a fragment that tokenizes
        # to more pieces than the word it came from.
        kept: list[str] = []
        used = 0
        for piece in re.split(r"(\s+)", text):
            cost = self.count(piece)
            if used + cost > max_tokens:
                break
            kept.append(piece)
            used += cost
        return "".join(kept).rstrip()


@lru_cache(maxsize=8)
def get_token_counter(model: str = DEFAULT_ENCODING) -> TokenCounter:
    """Return a counter for ``model``, cached because loading a BPE table is not cheap.

    Never raises: an unavailable tokenizer degrades to estimation with a warning, since a
    document that cannot be counted exactly is still a document that must be ingested.
    """
    encoding_name = _MODEL_ENCODINGS.get(model)
    if encoding_name is None:
        # A model with no tiktoken equivalent (BGE, Voyage) still gets a real tokenizer
        # rather than the estimator: cl100k is far closer to sentence-piece than counting
        # words is.
        encoding_name = model if model.endswith("_base") else DEFAULT_ENCODING
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        logger.warning(
            "tokenizer_fallback",
            model=model,
            error=str(exc),
            detail="Token budgets will be estimated; set TIKTOKEN_CACHE_DIR for exact counts.",
        )
        return HeuristicCounter()
    return TiktokenCounter(encoding, encoding_name)
