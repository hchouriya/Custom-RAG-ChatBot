"""UUIDv7 generation and deterministic point identifiers.

UUIDv7 is time-ordered: the first 48 bits are a Unix millisecond timestamp. That keeps
B-tree inserts at the right edge of the index instead of scattering them like v4,
which on a table taking 50k rows a day is the difference between a compact index and a
bloated one. It also lets a worker mint an id before insert and still get locality.

Implemented here rather than taken from a dependency because it is 20 lines and
``uuid.uuid7`` is not in the standard library for the versions we target.
"""

from __future__ import annotations

import time
import uuid
from secrets import randbits

_UUID7_VERSION = 0x7000
_VARIANT_RFC4122 = 0x8000


def new_id() -> uuid.UUID:
    """Return a fresh UUIDv7 (RFC 9562 §5.7)."""
    unix_ms = time.time_ns() // 1_000_000

    # 48 bits of timestamp | 4 bits version | 12 bits rand_a | 2 bits variant | 62 bits rand_b
    time_high = (unix_ms >> 16) & 0xFFFF_FFFF
    time_low = unix_ms & 0xFFFF
    ver_rand_a = _UUID7_VERSION | randbits(12)
    var_rand_b_hi = _VARIANT_RFC4122 | randbits(14)
    rand_b_lo = randbits(48)

    value = (
        (time_high << 96)
        | (time_low << 80)
        | (ver_rand_a << 64)
        | (var_rand_b_hi << 48)
        | rand_b_lo
    )
    return uuid.UUID(int=value)


def timestamp_of(value: uuid.UUID) -> float:
    """Extract the embedded creation time (Unix seconds) from a UUIDv7.

    Useful for sanity-checking imported data and for range-pruning without a join.
    """
    if value.version != 7:
        raise ValueError(f"not a UUIDv7: version={value.version}")
    return ((value.int >> 80) & 0xFFFF_FFFF_FFFF) / 1000.0


_POINT_NAMESPACE = uuid.UUID("6f3c8d1a-2b47-5e9a-8c1d-0f2e3a4b5c6d")


def vector_point_id(collection_id: uuid.UUID, chunk_id: uuid.UUID) -> uuid.UUID:
    """Deterministic Qdrant point id for a chunk.

    Derived rather than random so that a retried or duplicated upsert overwrites the
    same point instead of creating a second copy. A duplicate vector is invisible until
    it quietly distorts a ranking, which makes non-determinism here expensive to debug.
    """
    return uuid.uuid5(_POINT_NAMESPACE, f"{collection_id}:{chunk_id}")
