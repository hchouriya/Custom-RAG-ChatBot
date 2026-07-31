"""Credential detection and redaction.

Run in two places, both of which are real leaks that RAG makes easy:

* **On ingested content.** Someone uploads a runbook with a live AWS key in a code block. The
  chunk is indexed, and from then on the assistant will happily quote it to anyone whose ACL
  covers that document — which is a wider audience than the runbook ever had.
* **On model output.** Defence in depth against the model reproducing a credential that
  reached the context anyway.

``scan`` returns categories, never the matched value. A scanner that logs what it found has
copied the secret into a second system, usually one with weaker access controls and longer
retention than the first.
"""

from __future__ import annotations

import re
from typing import Final

REDACTION: Final = "[REDACTED]"

# Anchored on issuer-defined prefixes and lengths wherever one exists. Entropy-only
# detection produces false positives on checksums, minified assets, and UUIDs — and a
# redactor that mangles legitimate content is one that gets switched off.
_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    (
        "aws_secret_access_key",
        re.compile(
            r"(?i)\baws_?secret_?access_?key\b\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?",
        ),
    ),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{24,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{24,}\b")),
    ("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b")),
    ("twilio_key", re.compile(r"\bSK[0-9a-fA-F]{32}\b")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
            r"[\s\S]{0,4000}?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
        ),
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "connection_string",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://"
            # A password must actually be present: `postgres://host/db` in documentation is
            # not a secret, and flagging it trains people to ignore the scanner.
            r"[^\s:/@]+:[^\s:/@]{3,}@[^\s/]+",
            re.IGNORECASE,
        ),
    ),
    (
        "generic_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|"
            r"passwd|client[_-]?secret)\b\s*[:=]\s*[\"']([^\"'\s]{12,})[\"']"
        ),
    ),
)

_PLACEHOLDERS: Final = re.compile(
    r"(?i)^(?:x{3,}|\*{3,}|\.{3,}|<[^>]+>|\$\{[^}]+\}|change[_-]?me|your[_-]?\w+|"
    r"placeholder|example|redacted|dummy|test|todo|none|null|secret|password)$"
)


class PatternSecretScanner:
    """Detects and redacts credentials in text."""

    name = "pattern"

    def scan(self, text: str) -> tuple[str, ...]:
        """Categories of secret present, in a stable order, without the values."""
        if not text:
            return ()
        found: list[str] = []
        for category, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                if _is_placeholder(match):
                    continue
                found.append(category)
                break
        return tuple(dict.fromkeys(found))

    def redact(self, text: str) -> str:
        """Replace credential values with ``[REDACTED]``.

        Where a pattern captures a group, only the group is replaced, so
        ``api_key = "..."`` keeps its key name. That readability is not cosmetic: a redacted
        document should still be usable for the reason it was uploaded.
        """
        if not text:
            return text
        redacted = text
        for _, pattern in _PATTERNS:
            redacted = pattern.sub(_replace, redacted)
        return redacted


def _replace(match: re.Match[str]) -> str:
    if _is_placeholder(match):
        return match.group(0)
    if match.groups() and match.group(1):
        return match.group(0).replace(match.group(1), REDACTION)
    return REDACTION


def _is_placeholder(match: re.Match[str]) -> bool:
    """Whether the matched value is obviously a stand-in.

    Documentation and ``.env.example`` files are full of ``password = "changeme"``. Redacting
    those is noise, and noise is what makes people stop reading the findings.
    """
    value = match.group(1) if match.groups() and match.group(1) else match.group(0)
    return bool(_PLACEHOLDERS.match(value.strip()))
