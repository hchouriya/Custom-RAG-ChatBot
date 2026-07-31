"""Prompt-injection detection.

Pattern matching, and honest about being pattern matching. It catches the published attacks
and the copy-paste variants that make up most real traffic; it does not catch a paraphrase, an
encoding, or a novel phrasing, and no regex will. The architectural defences are what actually
contain injection: the retrieved context is delimited and labelled untrusted, citations are
validated against the retrieved set, ACLs are enforced in the retrieval filter rather than by
asking the model to be discreet, and the model has no tools. This scanner is the cheap first
layer, not the control.

The same detector is used at two boundaries with deliberately different consequences:

* **User input.** A high-confidence match blocks the request.
* **Document content.** A match only *flags* the chunk. A security policy that legitimately
  contains the phrase "ignore previous instructions" must stay findable, and blocking on
  ingest would hand anyone with upload rights a censorship tool.
"""

from __future__ import annotations

import re
from typing import Final

from aegis.domain.ports.infrastructure import InjectionFinding, InjectionScanResult

EXCERPT_CHARS: Final = 120

# (category, confidence, pattern). Confidence is calibrated so that the API's blocking
# threshold (0.8) admits only phrasings with no plausible innocent reading.
_PATTERNS: Final[tuple[tuple[str, float, re.Pattern[str]], ...]] = (
    (
        "instruction_override",
        0.9,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|above|earlier|all)\b[^.\n]{0,20}?"
            r"\b(?:instruction|prompt|rule|direction|context|message)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_extraction",
        0.9,
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|tell me|"
            r"what (?:are|is|was|were))\b"
            r"[^.\n]{0,40}?\b(?:system|initial|original|hidden|secret)\b[^.\n]{0,20}?"
            r"\b(?:prompt|instruction|message|rule)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        0.85,
        re.compile(
            r"\b(?:you are now|from now on you|act as|pretend to be|roleplay as|"
            r"behave (?:as|like))\b[^.\n]{0,60}?"
            r"\b(?:dan|developer mode|jailbroken|unrestricted|without (?:any )?"
            r"(?:restriction|filter|limit)s?|no longer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        0.85,
        re.compile(
            r"(?:^|\n)\s*(?:\[/?(?:SYSTEM|INST|ASSISTANT)\]|<\|?(?:im_start|im_end|system|"
            r"endoftext)\|?>|###\s*(?:system|instruction)s?\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "guardrail_bypass",
        0.8,
        re.compile(
            r"\b(?:bypass|circumvent|disable|turn off|get around)\b[^.\n]{0,30}?"
            r"\b(?:safety|guardrail|filter|restriction|policy|content polic(?:y|ies)|"
            r"security)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration",
        0.8,
        re.compile(
            r"\b(?:send|post|upload|transmit|leak|exfiltrate)\b[^.\n]{0,40}?"
            r"(?:https?://|\bto\b[^.\n]{0,20}?\b(?:webhook|attacker|external)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "acl_bypass",
        0.75,
        re.compile(
            r"\b(?:you (?:are|have)|assume|grant me|as)\b[^.\n]{0,20}?"
            r"\b(?:admin(?:istrator)?|root|superuser|full|unrestricted)\b[^.\n]{0,20}?"
            r"\b(?:access|privilege|permission|role|right)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "encoded_payload",
        0.6,
        re.compile(
            # A long base64 run next to decode-and-execute language. Either alone is common
            # and innocent; together they are the standard obfuscation.
            r"(?:base64|b64decode|atob|fromCharCode|\\u00|%[0-9a-f]{2}){1,}"
            r"[^.\n]{0,60}?\b(?:decode|execute|run|eval|then (?:do|follow))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_text",
        0.7,
        # Bidi overrides and zero-width characters: invisible to a reviewer reading the
        # document, fully visible to the model reading the extracted text.
        re.compile(r"[\u202a-\u202e\u2066-\u2069]|[\u200b-\u200f]{3,}"),
    ),
)

_MAX_SCAN_CHARS: Final = 50_000
"""Only the head of a long document is scanned. Injection payloads are placed where a model
will read them — the beginning — and scanning a 400-page PDF end to end with eight regexes on
every ingest is time better spent embedding it."""


class PatternInjectionScanner:
    """Regex-based detector. Stateless, sub-millisecond, safe to call on every request."""

    name = "pattern"

    def scan(self, text: str) -> InjectionScanResult:
        if not text:
            return InjectionScanResult()

        window = text[:_MAX_SCAN_CHARS]
        findings: list[InjectionFinding] = []
        seen: set[str] = set()

        for category, confidence, pattern in _PATTERNS:
            match = pattern.search(window)
            if match is None or category in seen:
                continue
            seen.add(category)
            findings.append(
                InjectionFinding(
                    category=category,
                    pattern=pattern.pattern[:60],
                    excerpt=_excerpt(window, match.start(), match.end()),
                    confidence=confidence,
                )
            )
        return InjectionScanResult(findings=tuple(findings))


def _excerpt(text: str, start: int, end: int) -> str:
    """A bounded window around the match, for the audit log.

    Bounded because the excerpt is stored and displayed: an unbounded one would let an
    attacker use our own audit log as a delivery channel for a much longer payload.
    """
    left = max(0, start - 20)
    right = min(len(text), end + 20)
    fragment = text[left:right].replace("\n", " ").strip()
    return fragment[:EXCERPT_CHARS]
