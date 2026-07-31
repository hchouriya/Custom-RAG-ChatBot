"""The confidence gate — refuse before generating, not after.

This is the mechanism that turns "never hallucinate" from an instruction in a prompt into
a property of the system. It runs after reranking and before any tokens are generated,
because a refusal decided after generation has already paid for, and exposed the model to,
content that did not support an answer.

Four signals, all cheap, none sufficient alone:

``top_score``
    The best reranked score. Catches "nothing relevant was found".
``supporting``
    How many chunks clear a lower bar. Catches the single-lucky-match case, where one
    chunk scores well by coincidence and no other evidence exists.
``mean_top3``
    Catches a sharp cliff: one strong chunk followed by noise, which usually means the
    corpus covers the topic but not the question.
``entity_coverage``
    Do the specific identifiers in the question — numbers, codes, capitalized names —
    actually appear in the context? This is what catches the most dangerous failure mode,
    where retrieval returns text about the right *topic* but the wrong *entity*, and the
    model fluently answers about the wrong product or region.

The defaults lean towards refusing. A false refusal costs one escalation; a confident
fabrication about pricing or leave entitlement can cost a legal dispute. Thresholds are
configuration because that tradeoff belongs to the business, and every refusal is recorded
so the cost of the choice is visible on the content-gap dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aegis.domain.values import Confidence, RetrievedChunk

_ENTITY_PATTERN = re.compile(
    r"""(?:
        \b\d+(?:[.,]\d+)*%?\b            # numbers, percentages, amounts
      | \b[A-Z]{2,}(?:-\d+)?\b           # acronyms and codes: EU, SKU-42, ISO
      | \b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b   # proper nouns
    )""",
    re.VERBOSE,
)

_STOP_ENTITIES = frozenset(
    {
        "How",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
        "Can",
        "Does",
        "Do",
        "Is",
        "Are",
        "The",
        "This",
        "That",
        "Please",
        "Tell",
        "Explain",
        "Show",
        "List",
        "Give",
    }
)


def extract_entities(text: str) -> set[str]:
    """Pull identifier-like tokens out of a question.

    Intentionally lexical rather than a NER model: it must add no latency to the hot path,
    and the tokens that matter in enterprise corpora — ``SKU-42``, ``ISO 27001``, ``20
    weeks``, ``EU`` — are exactly the ones a regex catches reliably and a general-purpose
    NER model often misses.
    """
    found = {m.group(0).strip() for m in _ENTITY_PATTERN.finditer(text)}
    return {e for e in found if e not in _STOP_ENTITIES and len(e) > 1}


def entity_coverage(question: str, context: str) -> float:
    """Fraction of the question's entities that appear in the context.

    Returns 1.0 when the question has no identifiable entities — a generic question
    ("what is our leave policy?") should not be penalised for lacking specifics, and the
    score thresholds already cover that case.
    """
    entities = extract_entities(question)
    if not entities:
        return 1.0
    haystack = context.lower()
    hits = sum(1 for e in entities if e.lower() in haystack)
    return hits / len(entities)


@dataclass(frozen=True, slots=True)
class ConfidenceGate:
    """Thresholds for the pre-generation decision."""

    min_top_score: float = 0.35
    min_supporting: int = 2
    supporting_threshold: float = 0.25
    min_mean_top3: float = 0.28
    min_entity_coverage: float = 0.5
    clarify_band: float = 0.05
    """How far below ``min_top_score`` still warrants asking a clarifying question.

    A near-miss is usually an under-specified question rather than a missing document, and
    one clarifying question is far more useful to the user than a flat refusal.
    """

    def evaluate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        degraded_rerank: bool = False,
    ) -> Confidence:
        """Decide whether to answer, clarify, or refuse.

        ``degraded_rerank`` raises the bar. Without a cross-encoder the ranking is fused
        rank order, which is a weaker signal, so the same nominal score means less and the
        gate should be more conservative rather than pretending nothing changed.
        """
        if not chunks:
            return Confidence(
                decision="refuse",
                score=0.0,
                top_score=0.0,
                supporting_chunks=0,
                mean_top3=0.0,
                entity_coverage=0.0,
                reason="no_results",
            )

        penalty = 0.05 if degraded_rerank else 0.0
        scores = sorted((c.best_score for c in chunks), reverse=True)
        top = scores[0]
        top3 = scores[:3]
        mean_top3 = sum(top3) / len(top3)
        supporting = sum(1 for s in scores if s >= self.supporting_threshold + penalty)

        context = "\n".join(c.content for c in chunks)
        coverage = entity_coverage(question, context)

        failures: list[str] = []
        if top < self.min_top_score + penalty:
            failures.append("low_top_score")
        if supporting < self.min_supporting:
            failures.append("insufficient_supporting_chunks")
        if mean_top3 < self.min_mean_top3 + penalty:
            failures.append("weak_score_distribution")
        if coverage < self.min_entity_coverage:
            failures.append("entity_not_covered")

        # A composite score for logging and threshold tuning. Weighted towards top score
        # and coverage because those two correlate best with human judgement of whether an
        # answer was actually supported.
        composite = min(1.0, 0.45 * top + 0.2 * mean_top3 + 0.35 * coverage)

        if not failures:
            return Confidence(
                decision="answer",
                score=composite,
                top_score=top,
                supporting_chunks=supporting,
                mean_top3=mean_top3,
                entity_coverage=coverage,
            )

        # Close enough to answerable, and the shortfall is about specificity rather than
        # absence of evidence: ask instead of refusing.
        near_miss = top >= self.min_top_score - self.clarify_band and supporting >= 1
        if near_miss and failures == ["entity_not_covered"]:
            return Confidence(
                decision="clarify",
                score=composite,
                top_score=top,
                supporting_chunks=supporting,
                mean_top3=mean_top3,
                entity_coverage=coverage,
                reason="ambiguous_entity",
            )

        return Confidence(
            decision="refuse",
            score=composite,
            top_score=top,
            supporting_chunks=supporting,
            mean_top3=mean_top3,
            entity_coverage=coverage,
            reason=",".join(failures),
        )


REFUSAL_MESSAGE = "I could not find enough information in the available documents."
"""The verbatim refusal text.

Fixed rather than generated so it is detectable in logs and analytics by exact match, with
no classifier and no ambiguity about whether a given answer was a refusal.
"""
