"""Query understanding: intent, then rewriting.

Both stages exist because the raw question a user types is usually a poor retrieval query.
"What about managers?" as a follow-up has no retrievable content at all; "how many days do I
get" omits the word "leave" that the policy document actually uses.

Rewriting produces *several* queries rather than one improved query, and the pipeline searches
with all of them. Recall is what a retrieval system cannot recover from later — a reranker can
demote a bad candidate but cannot conjure a missing one — so paying for three searches to
widen the net is the right trade at this stage of the funnel.

Both stages have deterministic fallbacks. When the fast model is unavailable, intent falls back
to pattern matching and rewriting falls back to the original question plus a coreference-resolved
variant. A retrieval system that stops working because a *classifier* is down would be a poor
design; these stages improve results, they do not gate them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.core.logging import get_logger
from aegis.core.telemetry import timed_stage
from aegis.domain.enums import Intent, Mode
from aegis.domain.ports.infrastructure import ChatMessage

if TYPE_CHECKING:
    from aegis.rag.llm.router import LLMRouter

logger = get_logger(__name__)

MAX_QUERY_CHARS = 2000
MAX_REWRITES = 3

_GREETING = re.compile(
    r"^\s*(hi|hey|hello|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|"
    r"bye|goodbye|ok(ay)?|cool|great|nice)\b[\s!.?]*$",
    re.IGNORECASE,
)
_FEEDBACK = re.compile(
    r"^\s*(that('s| is)?\s+(wrong|incorrect|helpful|great|perfect)|not\s+what\s+i\s+asked)",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(vs\.?|versus|compare[ds]?|difference between|better than)\b", re.IGNORECASE
)
_PROCEDURAL = re.compile(
    r"\b(how (do|can|to)|steps?|process|procedure|walk me through|set ?up|configure)\b",
    re.IGNORECASE,
)
_SUMMARY = re.compile(r"\b(summar(y|ise|ize)|overview|tl;?dr|key points|in short)\b", re.IGNORECASE)
_AGGREGATION = re.compile(
    r"\b(how many|how much|count|total|average|list all|all of the)\b", re.IGNORECASE
)
# A follow-up is a question that cannot stand alone: it refers out of itself and has no
# nouns of its own to search with.
_FOLLOWUP = re.compile(
    r"^\s*(and |but |what about|how about|why( not)?\??$|really\??$|"
    r"(what|who|when|where|which|how)\b[^?]{0,40}\b(it|its|they|them|their|that|those|this|these|he|she)\b)",
    re.IGNORECASE,
)
_PRONOUN = re.compile(
    r"\b(it|its|they|them|their|that|those|this|these|the same|there)\b", re.IGNORECASE
)


@dataclass(slots=True)
class QueryPlan:
    """Everything downstream retrieval needs about the question."""

    original: str
    intent: Intent
    queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    used_llm: bool = False
    clarification: str | None = None

    @property
    def primary(self) -> str:
        return self.queries[0] if self.queries else self.original


def classify_heuristically(question: str, *, has_history: bool) -> Intent:
    """Pattern-based intent.

    Order matters: greeting and feedback are checked first because they are the cases where
    skipping retrieval saves the most, and they are unambiguous.
    """
    text = question.strip()
    if not text:
        return Intent.GREETING
    if _GREETING.match(text):
        return Intent.GREETING
    if _FEEDBACK.match(text):
        return Intent.FEEDBACK
    if has_history and _FOLLOWUP.match(text):
        return Intent.FOLLOWUP
    if _COMPARISON.search(text):
        return Intent.COMPARISON
    if _SUMMARY.search(text):
        return Intent.SUMMARIZATION
    if _AGGREGATION.search(text):
        return Intent.AGGREGATION
    if _PROCEDURAL.search(text):
        return Intent.PROCEDURAL
    return Intent.FACTUAL_LOOKUP


_CLASSIFY_PROMPT = """You classify a user's question for a document-grounded assistant.

Reply with JSON only: {{"intent": "...", "queries": ["..."], "keywords": ["..."]}}

intent is exactly one of: factual_lookup, comparison, procedural, summarization,
aggregation, followup, greeting, feedback, out_of_scope.

queries: 1-3 standalone search queries in the language of the question. Resolve every
pronoun using the conversation. Use the vocabulary a formal policy or manual would use, not
the user's paraphrase. Do not invent facts, names, numbers, or constraints that are not in
the question or the conversation. If the question is a greeting or feedback, return [].

keywords: up to 6 distinctive terms for keyword search — identifiers, codes, proper nouns,
numbers. No stopwords.

Conversation so far:
{history}

Question: {question}"""


class QueryPlanner:
    """Intent classification and query rewriting in one fast-model call.

    One call rather than two: they need the same context, the output is small, and two
    sequential round trips to a hosted model before retrieval even starts would add 400 ms
    to every question for no additional information.
    """

    def __init__(self, router: LLMRouter, *, enabled: bool = True) -> None:
        self._router = router
        self._enabled = enabled

    async def plan(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        mode: Mode = Mode.INTERNAL,
    ) -> QueryPlan:
        question = question.strip()[:MAX_QUERY_CHARS]
        turns = history or []
        fallback = self._fallback(question, turns)

        if not self._enabled or not question:
            return fallback

        with timed_stage("plan_query", mode.value) as span:
            span["intent_source"] = "llm"
            rendered = (
                "\n".join(f"{role}: {content[:300]}" for role, content in turns[-4:]) or "(none)"
            )
            prompt = _CLASSIFY_PROMPT.format(history=rendered, question=question)
            try:
                completion = await self._router.complete(
                    [ChatMessage(role="user", content=prompt)],
                    model=self._router.fast_model,
                    temperature=0.0,
                    max_tokens=300,
                )
            # Any failure here degrades to the heuristic plan rather than failing the turn.
            except Exception as exc:
                span["intent_source"] = "heuristic"
                logger.warning("query_plan.llm_failed", error=str(exc))
                return fallback

            parsed = _parse(completion.text)
            if parsed is None:
                span["intent_source"] = "heuristic"
                return fallback

            intent_value, queries, keywords = parsed
            try:
                intent = Intent(intent_value)
            except ValueError:
                intent = fallback.intent

            # The original question is always searched, even when the model produced better
            # phrasings. A rewrite that drifts is a recall regression the reranker cannot fix,
            # and keeping the original bounds that risk at zero.
            merged = [question, *(q for q in queries if q and q.lower() != question.lower())]
            span["intent"] = intent.value
            span["queries"] = len(merged[:MAX_REWRITES])
            return QueryPlan(
                original=question,
                intent=intent,
                queries=merged[:MAX_REWRITES] if intent.needs_retrieval else [],
                keywords=keywords[:6] or fallback.keywords,
                used_llm=True,
            )

    def _fallback(self, question: str, turns: list[tuple[str, str]]) -> QueryPlan:
        intent = classify_heuristically(question, has_history=bool(turns))
        queries: list[str] = []
        if intent.needs_retrieval:
            queries.append(question)
            if intent is Intent.FOLLOWUP or _PRONOUN.search(question):
                stitched = _stitch(question, turns)
                if stitched:
                    queries.append(stitched)
        return QueryPlan(
            original=question,
            intent=intent,
            queries=queries,
            keywords=_keywords(question),
        )


def _stitch(question: str, turns: list[tuple[str, str]]) -> str | None:
    """Coreference resolution for the fallback path, by concatenation.

    Crude but effective and free: prefixing the previous user question puts its nouns into
    the embedded text, which is most of what pronoun resolution buys for retrieval. The
    result is never shown to the user, only searched with.
    """
    previous = next((content for role, content in reversed(turns) if role == "user"), None)
    if not previous:
        return None
    return f"{previous.strip()[:200]} {question}"


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/]{1,}")
_STOP = frozenset(
    """a an the and or but if then than that this these those for from with without within
    into onto over under about above below between during is are was were be been being do does
    did doing have has had having i me my we our you your he she it its they them their what
    which who whom whose when where why how can could should would may might must will shall
    of in on at to by as not no yes please tell explain show list give many much
    """.split()
)


def _keywords(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text):
        token = match.group(0)
        if token.lower() in _STOP or len(token) < 2:
            continue
        seen.setdefault(token, None)
    return list(seen)[:6]


def _parse(text: str) -> tuple[str, list[str], list[str]] | None:
    """Pull the JSON object out of a model reply, tolerating fences and prose."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = (
            candidate.partition("\n")[2] if candidate.lower().startswith("json") else candidate
        )
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
    keywords = [str(k).strip() for k in payload.get("keywords", []) if str(k).strip()]
    return str(payload.get("intent", "")), queries, keywords
