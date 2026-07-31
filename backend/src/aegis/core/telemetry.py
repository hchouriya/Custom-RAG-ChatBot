"""Prometheus metrics and optional OpenTelemetry tracing.

The metric set is the one from ``docs/architecture/08`` §3, defined in one place so that
names and labels cannot drift between call sites. Two rules are enforced by construction:

* **Label cardinality is bounded.** No metric is labelled by user id, document id, or
  question text. Each is unbounded and would multiply series until the scrape falls over.
* **Quality metrics exist alongside latency metrics.** A RAG system degrades silently;
  ``aegis_retrieval_top_score`` and ``aegis_citation_validation_failures_total`` are the
  signals that catch it while availability dashboards stay green.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from aegis.core.config import Settings
from aegis.core.logging import get_logger

logger = get_logger(__name__)

REGISTRY = CollectorRegistry(auto_describe=True)

# Buckets chosen around the SLOs in docs/architecture/01: sub-second stages need
# resolution below 1 s, and the LLM tail needs headroom to 30 s.
_STAGE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_SCORE_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# ── Latency ─────────────────────────────────────────────────────────────────
stage_duration = Histogram(
    "aegis_stage_duration_seconds",
    "Duration of a pipeline stage",
    ["stage", "mode", "status"],
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)
request_duration = Histogram(
    "aegis_request_duration_seconds",
    "HTTP request duration",
    ["route", "method", "status"],
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)
llm_ttft = Histogram(
    "aegis_llm_ttft_seconds",
    "LLM time to first token",
    ["provider", "model"],
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)

# ── Quality ─────────────────────────────────────────────────────────────────
retrieval_top_score = Histogram(
    "aegis_retrieval_top_score",
    "Top reranked score of a retrieval",
    ["mode"],
    buckets=_SCORE_BUCKETS,
    registry=REGISTRY,
)
retrieval_candidates = Histogram(
    "aegis_retrieval_candidates",
    "Candidate count at a pipeline stage",
    ["stage"],
    buckets=(0, 1, 2, 5, 8, 10, 20, 40, 80, 160),
    registry=REGISTRY,
)
answers_total = Counter(
    "aegis_answers_total", "Answers by outcome", ["status", "mode"], registry=REGISTRY
)
citations_per_answer = Histogram(
    "aegis_citations_per_answer",
    "Valid citations attached to an answer",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10),
    registry=REGISTRY,
)
citation_validation_failures = Counter(
    "aegis_citation_validation_failures_total",
    "Citation markers rejected by the validator",
    ["reason"],
    registry=REGISTRY,
)
confidence_gate = Counter(
    "aegis_confidence_gate_total", "Confidence gate decisions", ["decision"], registry=REGISTRY
)
feedback_total = Counter(
    "aegis_feedback_total", "User feedback", ["rating", "mode"], registry=REGISTRY
)

# ── Cost ────────────────────────────────────────────────────────────────────
tokens_total = Counter(
    "aegis_tokens_total", "Tokens consumed", ["provider", "model", "kind"], registry=REGISTRY
)
cost_usd_total = Counter(
    "aegis_cost_usd_total", "Estimated spend", ["provider", "model"], registry=REGISTRY
)
embedding_cache_hits = Counter(
    "aegis_embedding_cache_hits_total", "Embedding cache hits", registry=REGISTRY
)
embedding_cache_misses = Counter(
    "aegis_embedding_cache_misses_total", "Embedding cache misses", registry=REGISTRY
)

# ── Security ────────────────────────────────────────────────────────────────
guardrail_blocks = Counter(
    "aegis_guardrail_blocks_total",
    "Guardrail blocks",
    ["layer", "category"],
    registry=REGISTRY,
)
authz_denials = Counter(
    "aegis_authz_denials_total", "Authorization denials", ["role", "resource"], registry=REGISTRY
)
acl_layer2_drops = Counter(
    "aegis_acl_layer2_drops_total",
    "Chunks dropped by post-retrieval ACL re-verification (expected to be ~0)",
    registry=REGISTRY,
)
login_failures = Counter(
    "aegis_login_failures_total", "Failed logins", ["reason"], registry=REGISTRY
)

# ── Ingestion ───────────────────────────────────────────────────────────────
ingest_jobs_total = Counter(
    "aegis_ingest_jobs_total", "Ingestion jobs", ["status", "job_type"], registry=REGISTRY
)
ingest_stage_duration = Histogram(
    "aegis_ingest_stage_duration_seconds",
    "Ingestion stage duration",
    ["stage", "mime"],
    buckets=(0.05, 0.25, 1.0, 5.0, 15.0, 60.0, 300.0, 1800.0),
    registry=REGISTRY,
)
chunks_produced = Histogram(
    "aegis_chunks_produced",
    "Chunks produced per document version",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
    registry=REGISTRY,
)
queue_depth = Gauge("aegis_queue_depth", "Jobs waiting", ["queue"], registry=REGISTRY)
queue_oldest_job = Gauge(
    "aegis_queue_oldest_job_seconds", "Age of the oldest waiting job", ["queue"], registry=REGISTRY
)
index_discrepancies = Gauge(
    "aegis_index_discrepancies", "Open PostgreSQL/vector discrepancies", ["kind"], registry=REGISTRY
)

# ── Dependencies ────────────────────────────────────────────────────────────
provider_errors = Counter(
    "aegis_provider_errors_total",
    "Upstream provider errors",
    ["provider", "code"],
    registry=REGISTRY,
)
circuit_breaker_state = Gauge(
    "aegis_circuit_breaker_state",
    "0=closed 1=half-open 2=open",
    ["provider"],
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Prometheus exposition payload for ``GET /metrics``."""
    return generate_latest(REGISTRY)


@contextmanager
def timed_stage(stage: str, mode: str = "-") -> Iterator[dict[str, Any]]:
    """Time a pipeline stage and record its outcome.

    Yields a mutable dict so the body can add detail (counts, scores) that the caller
    logs; the metric only ever sees the bounded labels.

    >>> with timed_stage("retrieve", "internal") as span:
    ...     span["candidates"] = 37
    """
    started = time.perf_counter()
    span: dict[str, Any] = {}
    status = "ok"
    try:
        yield span
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - started
        span["duration_ms"] = round(elapsed * 1000, 2)
        stage_duration.labels(stage=stage, mode=mode, status=status).observe(elapsed)


def setup_tracing(settings: Settings, app: Any = None) -> None:
    """Install OpenTelemetry if enabled and installed.

    Optional and import-guarded: the OTel packages are an extra, so a deployment that
    does not want them should not have to install them, and their absence must not stop
    the process from booting.
    """
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError:
        logger.warning(
            "otel.unavailable",
            hint="install the 'otel' extra or set OTEL_ENABLED=false",
        )
        return

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.otel_service_name, "deployment.environment": settings.app_env}
        ),
        sampler=ParentBased(TraceIdRatioBased(settings.trace_sample_ratio)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
    )
    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(
                app, excluded_urls="/health/live,/health/ready,/metrics"
            )
        except ImportError:
            pass
    logger.info("otel.enabled", endpoint=settings.otel_exporter_otlp_endpoint)


def current_trace_id() -> str | None:
    """Active trace id as hex, for storing on ``query_traces``."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")
