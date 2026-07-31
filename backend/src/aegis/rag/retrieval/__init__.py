"""Query understanding, hybrid retrieval, compression, context assembly, citations."""

from __future__ import annotations

from aegis.rag.retrieval.citations import CitationResult, extract, find_markers, mark_unused
from aegis.rag.retrieval.compression import CompressionConfig, CompressionStats, compress
from aegis.rag.retrieval.context import AssembledContext, allocate_budget, assemble
from aegis.rag.retrieval.hybrid import HybridRetriever, RetrievalConfig, RetrievalResult
from aegis.rag.retrieval.query import QueryPlan, QueryPlanner, classify_heuristically

__all__ = [
    "AssembledContext",
    "CitationResult",
    "CompressionConfig",
    "CompressionStats",
    "HybridRetriever",
    "QueryPlan",
    "QueryPlanner",
    "RetrievalConfig",
    "RetrievalResult",
    "allocate_budget",
    "assemble",
    "classify_heuristically",
    "compress",
    "extract",
    "find_markers",
    "mark_unused",
]
