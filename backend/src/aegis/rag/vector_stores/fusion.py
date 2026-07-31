"""Reciprocal rank fusion.

Dense and sparse scores are not comparable. Cosine similarity lives in [-1, 1] with most real
results bunched between 0.6 and 0.9; BM25 is unbounded and depends on corpus statistics. Any
attempt to combine them by weighted sum requires normalising two distributions that shift with
the query, and the weights end up tuned to one corpus.

RRF sidesteps the problem by throwing the scores away and keeping only the ranks. A document at
rank 1 in either arm contributes ``1/(k+1)``, and agreement between arms is what accumulates.
It is the fusion method used by production hybrid search for the same reason it is used here:
no tuning, no normalisation, and no way for one arm's scale to swamp the other's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from aegis.domain.ports.vector_store import VectorHit

DEFAULT_K = 60
"""The constant from the original RRF paper.

It controls how quickly rank contributions decay. Small values make the top rank dominate;
large values flatten everything toward equality. 60 is the published default and the value the
retrieval evaluation is calibrated against, so it is configuration rather than a constant.
"""


def reciprocal_rank_fusion(
    *rankings: Sequence[VectorHit],
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[VectorHit]:
    """Fuse ranked lists into one, best first.

    The returned hits carry the fused score, not the original similarity, and the per-arm ranks
    are recorded in the payload under ``_rrf``. That detail is what makes a bad ranking
    debuggable after the fact: "the answer was in the sparse arm at rank 2 and the dense arm
    never returned it" is a diagnosis, whereas a fused score alone is not.
    """
    fused: dict[UUID, VectorHit] = {}
    scores: dict[UUID, float] = {}
    ranks: dict[UUID, dict[str, int]] = {}

    for arm, ranking in enumerate(rankings):
        for position, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(hit.chunk_id, {})[f"arm{arm}"] = position
            # Keep the first arm's payload: arms return the same payload for the same chunk,
            # and preferring the earlier one makes fusion deterministic.
            fused.setdefault(hit.chunk_id, hit)

    results: list[VectorHit] = []
    for chunk_id, hit in fused.items():
        hit.score = scores[chunk_id]
        hit.payload = {**hit.payload, "_rrf": ranks[chunk_id]}
        results.append(hit)

    # Tie-break on chunk id so equal scores produce a stable order. Without it, pagination and
    # test assertions both become flaky for no visible reason.
    results.sort(key=lambda h: (-h.score, str(h.chunk_id)))
    return results[:limit] if limit is not None else results
