"""Input and output guardrails.

Two checks with deliberately different postures.

**Input** is checked before anything expensive runs. A high-confidence prompt injection is
blocked outright; a low-confidence one is allowed through and recorded, because "ignore the
previous instructions in section 4" is a legitimate question about a document. Blocking on
suspicion would make the assistant useless for exactly the security and compliance corpora
where it is most valuable.

**Output** is checked after generation and before persistence. Two things are looked for:
credentials that a document contained and the model repeated, and the system prompt leaking
verbatim. Secrets are redacted rather than blocked — the surrounding answer is usually
correct and useful, and destroying it punishes the user for a problem in the corpus.

Neither check is a substitute for the architecture around it. Injection is mitigated
structurally: retrieved text is fenced as data, the prompt says so, tool use is absent, and
the ACL filter means a successful injection still cannot reach content the user could not
already read. These scanners are the last layer, not the only one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aegis.core.errors import GuardrailViolationError
from aegis.core.logging import get_logger
from aegis.core.telemetry import guardrail_blocks
from aegis.domain.enums import Mode

if TYPE_CHECKING:
    from aegis.domain.ports.infrastructure import InjectionScanner, SecretScanner

logger = get_logger(__name__)

MAX_QUESTION_CHARS = 4000
BLOCK_CONFIDENCE = 0.8

# Phrases that only appear when a model is reciting its own instructions. Matched on the
# output, where their presence is a prompt leak rather than a user's turn of phrase.
_LEAK_MARKERS = (
    re.compile(r"<<<SOURCE\s+\d+>>>"),
    re.compile(r"^\s*##\s*(Grounding|Citations|Untrusted content)\b", re.MULTILINE),
    re.compile(r"\byou are a document-grounded assistant\b", re.IGNORECASE),
)

# Customer-mode answers must not name internal classification. This is a backstop: the ACL
# filter already prevents internal *content* from reaching the prompt, so a hit here means
# the model editorialised about access control, which is confusing rather than dangerous.
_INTERNAL_TELLS = re.compile(
    r"\b(internal[- ]only|confidential document|restricted document|employee[- ]only|"
    r"internal knowledge base|you are not authori[sz]ed)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class InputVerdict:
    allowed: bool
    question: str
    categories: tuple[str, ...] = ()
    confidence: float = 0.0
    reason: str = ""

    @property
    def suspicious(self) -> bool:
        return bool(self.categories)


@dataclass(slots=True)
class OutputVerdict:
    answer: str
    redacted: bool = False
    secret_categories: tuple[str, ...] = ()
    leaked_prompt: bool = False
    notes: list[str] = field(default_factory=list)


class Guardrails:
    def __init__(
        self,
        *,
        injection: InjectionScanner,
        secrets: SecretScanner,
        block_confidence: float = BLOCK_CONFIDENCE,
    ) -> None:
        self._injection = injection
        self._secrets = secrets
        self._block_confidence = block_confidence

    def check_input(self, question: str, *, mode: Mode) -> InputVerdict:
        """Validate a user turn. Raises :class:`GuardrailViolationError` when blocked.

        Length is checked first and cheaply. A 200 kB "question" is either a mistake or an
        attempt to push the real instructions out of the context window, and neither
        deserves an embedding call.
        """
        text = question.strip()
        if not text:
            raise GuardrailViolationError("empty_question", "A question is required.")
        if len(text) > MAX_QUESTION_CHARS:
            guardrail_blocks.labels(layer="input", category="too_long").inc()
            raise GuardrailViolationError(
                "question_too_long",
                f"Questions are limited to {MAX_QUESTION_CHARS} characters.",
            )

        result = self._injection.scan(text)
        if result.flagged and result.max_confidence >= self._block_confidence:
            for category in result.categories:
                guardrail_blocks.labels(layer="input", category=category).inc()
            logger.warning(
                "guardrail.input_blocked",
                categories=list(result.categories),
                confidence=result.max_confidence,
                mode=mode.value,
            )
            raise GuardrailViolationError(
                "prompt_injection",
                "That request looks like an attempt to change how I work rather than a "
                "question about the documents.",
            )

        if result.flagged:
            # Allowed, but recorded: a principal who accumulates these is worth looking at,
            # and the aggregate is what tells us whether the threshold is set sensibly.
            logger.info(
                "guardrail.input_suspicious",
                categories=list(result.categories),
                confidence=result.max_confidence,
            )

        return InputVerdict(
            allowed=True,
            question=text,
            categories=result.categories,
            confidence=result.max_confidence,
        )

    def check_output(self, answer: str, *, mode: Mode) -> OutputVerdict:
        verdict = OutputVerdict(answer=answer)

        if categories := self._secrets.scan(answer):
            verdict.answer = self._secrets.redact(verdict.answer)
            verdict.redacted = True
            verdict.secret_categories = categories
            for category in categories:
                guardrail_blocks.labels(layer="output", category=f"secret:{category}").inc()
            logger.warning("guardrail.output_redacted", categories=list(categories))

        if any(pattern.search(answer) for pattern in _LEAK_MARKERS):
            verdict.leaked_prompt = True
            guardrail_blocks.labels(layer="output", category="prompt_leak").inc()
            logger.warning("guardrail.prompt_leak_detected", mode=mode.value)
            raise GuardrailViolationError(
                "prompt_leak",
                "I ran into a problem answering that. Please rephrase and try again.",
            )

        if mode is Mode.CUSTOMER and _INTERNAL_TELLS.search(answer):
            verdict.notes.append("internal_reference_in_customer_mode")
            guardrail_blocks.labels(layer="output", category="internal_reference").inc()
            logger.warning("guardrail.internal_reference_in_customer_mode")

        return verdict

    def scan_document(self, text: str) -> tuple[tuple[str, ...], float]:
        """Flag, never block, at ingest time.

        Returns the categories found and the highest confidence. A security policy that
        quotes attack strings must stay searchable, and letting an upload be rejected because
        it *discusses* prompt injection would hand anyone with upload rights a way to
        suppress documents.
        """
        result = self._injection.scan(text)
        return result.categories, result.max_confidence
