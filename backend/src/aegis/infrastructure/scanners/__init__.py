"""Content scanners: malware, prompt injection, credentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis.infrastructure.scanners.clamav import ClamAvScanner, NoopScanner
from aegis.infrastructure.scanners.injection import PatternInjectionScanner
from aegis.infrastructure.scanners.secrets import PatternSecretScanner

if TYPE_CHECKING:
    from aegis.core.config import Settings
    from aegis.domain.ports.infrastructure import (
        InjectionScanner,
        MalwareScanner,
        SecretScanner,
    )

__all__ = [
    "ClamAvScanner",
    "NoopScanner",
    "PatternInjectionScanner",
    "PatternSecretScanner",
    "build_injection_scanner",
    "build_malware_scanner",
    "build_secret_scanner",
]


def build_malware_scanner(settings: Settings) -> MalwareScanner:
    """ClamAV when enabled, otherwise a scanner that admits it did nothing."""
    if settings.malware_scan_enabled:
        return ClamAvScanner.from_settings(settings)
    return NoopScanner()


def build_injection_scanner(settings: Settings) -> InjectionScanner:
    return PatternInjectionScanner()


def build_secret_scanner(settings: Settings) -> SecretScanner:
    return PatternSecretScanner()
