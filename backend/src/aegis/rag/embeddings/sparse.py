"""BM25 sparse vectors for the keyword half of hybrid retrieval.

Dense retrieval fails on exactly the queries an enterprise assistant gets most: part numbers,
error codes, policy identifiers, surnames. "SKU-44921" and "SKU-44922" are near-identical
vectors and completely different products. Lexical matching is what saves those queries, which
is why the platform runs both and fuses the results rather than trusting embeddings alone.

BM25 term weights are computed here and stored as sparse vectors so the *vector store* can do
the keyword search with the same filters and the same ACL payload as the dense search. The
alternative — Postgres full-text search alongside Qdrant — means two indexes with two ACL
implementations that must agree, and the day they disagree is a data leak rather than a bug.

Term indices are hashes, not dictionary positions. A learned vocabulary would need to be built
before the first document could be indexed, versioned with the corpus, and rebuilt whenever it
drifted. Hashing removes all three problems; collisions at 2^32 buckets are rare enough to be
noise next to the ranking signal.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from aegis.domain.values import SparseVector

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*", re.IGNORECASE)
"""Keeps identifiers intact.

``SKU-44921``, ``CVE-2024-1234``, ``v2.3.1``, and ``policy_hr_014`` are single terms. Splitting
them on punctuation — which a naive ``\\w+`` does — turns the most distinctive token in the
query into three common ones, and the exact-match query becomes the one that fails.
"""

_INDEX_SPACE = 2**32

K1 = 1.5
"""Term-frequency saturation. The standard value; above ~2 repetition stops being discounted."""

B = 0.75
"""Length normalisation strength. The standard value."""

DEFAULT_AVG_LENGTH = 220.0
"""Assumed average document length in tokens, until the corpus supplies its own.

Roughly a 800-token chunk after stopword removal. It only affects length normalisation, and
being wrong by a factor of two changes rankings by very little — but leaving it unset until
statistics exist would mean the first documents indexed are scored differently from the rest.
"""

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
"""Only the most frequent function words.

Aggressive stopword removal breaks phrase-like queries ("right to be forgotten", "no fault
termination") where the function words carry the meaning. IDF already discounts common terms;
this list exists to keep vectors from being dominated by "the".
"""


@dataclass(slots=True)
class CorpusStatistics:
    """Document frequencies for IDF, accumulated as documents are indexed.

    Kept per collection rather than globally. IDF is only meaningful relative to a corpus, and
    a term that is distinctive in the HR handbook ("leave") is background noise in a corpus of
    leave requests.
    """

    document_count: int = 0
    document_frequency: dict[int, int] = field(default_factory=dict)
    total_length: int = 0

    @property
    def average_length(self) -> float:
        if not self.document_count:
            return DEFAULT_AVG_LENGTH
        return self.total_length / self.document_count

    def observe(self, indices: set[int], length: int) -> None:
        self.document_count += 1
        self.total_length += length
        for index in indices:
            self.document_frequency[index] = self.document_frequency.get(index, 0) + 1

    def idf(self, index: int) -> float:
        """Robertson-Spärck-Jones IDF with the standard +0.5 smoothing.

        The ``max`` floor matters: without it a term appearing in more than half the documents
        gets a negative weight, and a chunk can be *penalised* for containing a query term.
        """
        if not self.document_count:
            return 1.0
        df = self.document_frequency.get(index, 0)
        raw = math.log(1 + (self.document_count - df + 0.5) / (df + 0.5))
        return max(raw, 0.0)


class Bm25SparseEncoder:
    """Turns text into a BM25-weighted sparse vector.

    Document and query encodings are deliberately different, exactly as in BM25 itself:
    documents carry the saturated, length-normalised term frequencies, and queries carry the
    IDF weights. Their dot product is then the BM25 score, which is what lets a vector store's
    sparse index reproduce a real BM25 ranking rather than an approximation of one.
    """

    name = "bm25"

    def __init__(
        self,
        statistics: CorpusStatistics | None = None,
        *,
        k1: float = K1,
        b: float = B,
    ) -> None:
        self._stats = statistics or CorpusStatistics()
        self._k1 = k1
        self._b = b

    @property
    def statistics(self) -> CorpusStatistics:
        return self._stats

    async def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.encode_document(text) for text in texts]

    async def embed_query(self, text: str) -> SparseVector:
        return self.encode_query(text)

    def encode_document(self, text: str, *, observe: bool = True) -> SparseVector:
        counts = _term_frequencies(text)
        if not counts:
            return SparseVector(indices=(), values=())

        length = sum(counts.values())
        if observe:
            self._stats.observe(set(counts), length)

        normaliser = self._k1 * (
            1 - self._b + self._b * length / max(1.0, self._stats.average_length)
        )
        indices: list[int] = []
        values: list[float] = []
        for index, frequency in sorted(counts.items()):
            indices.append(index)
            values.append(frequency * (self._k1 + 1) / (frequency + normaliser))
        return SparseVector(indices=tuple(indices), values=tuple(values))

    def encode_query(self, text: str) -> SparseVector:
        counts = _term_frequencies(text)
        if not counts:
            return SparseVector(indices=(), values=())
        indices: list[int] = []
        values: list[float] = []
        for index in sorted(counts):
            weight = self._stats.idf(index)
            if weight <= 0.0:
                continue
            indices.append(index)
            values.append(weight)
        if not indices:
            # Every query term is either unseen or corpus-wide. Falling back to uniform
            # weights keeps the sparse arm from returning nothing at all, which would silently
            # turn hybrid retrieval into dense-only for that query.
            indices = sorted(counts)
            values = [1.0] * len(indices)
        return SparseVector(indices=tuple(indices), values=tuple(values))


def term_index(term: str) -> int:
    """Stable bucket for a term.

    blake2b rather than Python's ``hash``: ``hash`` is randomised per process by PYTHONHASHSEED,
    so indices written by one worker would not match those written by another, and the index
    would quietly stop matching anything after a restart.
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "big") % _INDEX_SPACE


def tokenize(text: str) -> list[str]:
    """Lowercase, keep identifiers whole, drop only the most common function words."""
    return [
        token
        for raw in _TOKEN.findall(text.lower())
        if (token := raw.strip("-_./")) and token not in STOPWORDS
    ]


def _term_frequencies(text: str) -> Counter[int]:
    return Counter(term_index(token) for token in tokenize(text))
