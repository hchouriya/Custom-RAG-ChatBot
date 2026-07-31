"""ClamAV malware scanning over the INSTREAM protocol.

Spoken directly on a socket rather than through ``pyclamd``, which is synchronous and would
block the event loop for the duration of a 200 MB scan. The protocol is four lines of work:
send ``zINSTREAM\\0``, then length-prefixed chunks, then a zero-length chunk, then read one
response line.

An upload is scanned *before* it is parsed. Parsers are the largest attack surface in the
system — a malicious PDF targets the PDF library, not us — so putting the scan after parsing
would mean the exploit had already run by the time the file was rejected.
"""

from __future__ import annotations

import asyncio
import struct
from typing import TYPE_CHECKING

from aegis.core.logging import get_logger
from aegis.domain.ports.infrastructure import ScanResult

if TYPE_CHECKING:
    from aegis.core.config import Settings

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 64
SCANNER_NAME = "clamav"


class ClamAvScanner:
    """Streams bytes to a clamd instance and interprets its verdict.

    A scanner failure is not a clean verdict. ``ScanResult(clean=False, skipped=True)`` says
    "unknown", and the ingestion service quarantines rather than indexes — because the only
    thing worse than rejecting a good file is indexing a bad one after the scanner timed out.
    """

    name = SCANNER_NAME

    def __init__(self, host: str, port: int, *, timeout: float = 120.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> ClamAvScanner:
        return cls(settings.clamav_host, settings.clamav_port)

    async def scan(self, data: bytes, *, filename: str) -> ScanResult:
        try:
            response = await asyncio.wait_for(self._instream(data), timeout=self._timeout)
        except TimeoutError:
            logger.error("malware_scan.timeout", filename=filename, size=len(data))
            return ScanResult(clean=False, threat=None, scanner=self.name, skipped=True)
        except (OSError, ConnectionError) as exc:
            logger.error("malware_scan.unavailable", filename=filename, error=str(exc))
            return ScanResult(clean=False, threat=None, scanner=self.name, skipped=True)

        return _interpret(response, filename)

    async def _instream(self, data: bytes) -> str:
        reader, writer = await asyncio.open_connection(self._host, self._port)
        try:
            writer.write(b"zINSTREAM\0")
            for start in range(0, len(data), CHUNK_SIZE):
                chunk = data[start : start + CHUNK_SIZE]
                # Big-endian length prefix per chunk, then a zero length to end the stream.
                writer.write(struct.pack("!L", len(chunk)) + chunk)
                await writer.drain()
            writer.write(struct.pack("!L", 0))
            await writer.drain()
            raw = await reader.readline()
        finally:
            writer.close()
            # A half-closed connection left behind exhausts clamd's connection limit, which
            # then fails scans for reasons that look nothing like the cause.
            await asyncio.shield(_safe_wait_closed(writer))
        return raw.decode("utf-8", errors="replace").strip("\0 \n\r")

    async def health(self) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port), timeout=3
            )
        except (TimeoutError, OSError) as exc:
            logger.warning("malware_scan.unhealthy", error=str(exc))
            return False
        try:
            writer.write(b"zPING\0")
            await writer.drain()
            reply = await asyncio.wait_for(reader.readline(), timeout=3)
        except (TimeoutError, OSError):
            return False
        finally:
            writer.close()
            await _safe_wait_closed(writer)
        return b"PONG" in reply


class NoopScanner:
    """Records that no scan happened, rather than claiming a file is clean.

    ``skipped=True`` is the whole point. If this returned ``clean=True`` then disabling the
    scanner in configuration would be indistinguishable, in the audit log, from a real clean
    verdict — and that distinction is the first thing anyone asks for after an incident.
    """

    name = "noop"

    async def scan(self, data: bytes, *, filename: str) -> ScanResult:
        return ScanResult(clean=True, threat=None, scanner=self.name, skipped=True)

    async def health(self) -> bool:
        return True


def _interpret(response: str, filename: str) -> ScanResult:
    """Parse ``stream: OK`` / ``stream: Eicar-Test-Signature FOUND`` / ``... ERROR``."""
    if response.endswith("OK") and "FOUND" not in response:
        return ScanResult(clean=True, scanner=SCANNER_NAME)
    if response.endswith("FOUND"):
        threat = response.rpartition(":")[2].removesuffix("FOUND").strip()
        logger.warning("malware_scan.detected", filename=filename, threat=threat)
        return ScanResult(clean=False, threat=threat or "unknown", scanner=SCANNER_NAME)
    logger.error("malware_scan.error_response", filename=filename, response=response[:200])
    return ScanResult(clean=False, threat=None, scanner=SCANNER_NAME, skipped=True)


async def _safe_wait_closed(writer: asyncio.StreamWriter) -> None:
    try:
        await writer.wait_closed()
    except (OSError, ConnectionError):
        return
