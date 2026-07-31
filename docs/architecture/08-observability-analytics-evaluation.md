# 08 — Observability, Analytics, and Evaluation

A RAG system fails silently. A broken API returns 500s and pages someone; a degraded retriever
returns fluent, plausible, wrong answers and nobody notices for a month. Everything in this
document exists to make quality failures as loud as availability failures.

Three separate concerns, deliberately not conflated:

| Layer | Question it answers | Storage | Consumer |
|---|---|---|---|
| Observability | Is the system healthy right now? | Logs, traces, metrics (external) | On-call engineer |
| Analytics | Is the product working for users? | `query_traces` → materialized views | Product owner, content owner |
| Evaluation | Is answer quality regressing? | `eval_runs` / `eval_results` | CI gate, AI engineer |

---

## 1. Structured logging

`structlog` in JSON, one event per pipeline stage. Every line carries `request_id`, `trace_id`,
`user_id`, `role`, `mode`, and `stage`, bound once into a `contextvar` and inherited by everything
downstream, including worker jobs (the ids travel in the job payload).

```json
{"ts":"2026-07-28T07:41:02.118Z","level":"info","event":"retrieval.completed",
 "request_id":"01J8Z9X2QK","trace_id":"7f3c…","user_id":"0190f3a2","role":"manager",
 "mode":"internal","stage":"retrieve","sub_queries":3,"dense_hits":120,"sparse_hits":118,
 "after_dedupe":37,"top_score":0.81,"duration_ms":246,"collection_ids":["019…"]}
```

Rules that matter more than the format: **no PII in logs** (a redaction processor strips by key
name and value pattern, and question text is logged only as a length plus a hash — the text itself
lives in `query_traces`, which is access-controlled and retention-managed); **no secrets, ever**
(`SecretStr` plus the same processor); **one event per stage, not per iteration** (a log line per
candidate at 40 candidates × 50k queries/day is 2M lines/day of noise).

Log levels are used for their intended meaning: `INFO` for lifecycle, `WARNING` for degradation
that the user notices (reranker unavailable, fallback model used), `ERROR` only for things a human
must act on. A `WARNING` that fires 10k times a day trains everyone to ignore warnings.

## 2. Tracing

OpenTelemetry, auto-instrumented for FastAPI, SQLAlchemy, httpx, and Redis, plus one manual span
per graph node. Span attributes follow OTel GenAI semantic conventions where they exist
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`) so vendor dashboards work
without custom mapping.

```
POST /chat/…/messages                                  4,310 ms
├── auth.verify                                            6 ms
├── guardrail.input                                        8 ms
├── graph.invoke                                       4,280 ms
│   ├── node.classify_intent            118 ms  → llm.chat (gpt-5-mini)
│   ├── node.rewrite_query              287 ms  → llm.chat
│   ├── node.build_filter                14 ms  → db.select (acl)
│   ├── node.retrieve                   246 ms
│   │   ├── embed.query (cache hit)       2 ms
│   │   ├── qdrant.query_points (q1)    131 ms
│   │   ├── qdrant.query_points (q2)    128 ms   ← parallel
│   │   └── qdrant.query_points (q3)    142 ms
│   ├── node.rerank                     331 ms  → tei.rerank (37 pairs)
│   ├── node.verify_acl                  27 ms  → db.select
│   ├── node.compress                    88 ms
│   └── node.generate                 2,401 ms  → llm.stream (ttft 1,180 ms)
└── persist.message                      21 ms
```

`trace_id` is stored on `query_traces`, and the admin trace viewer links straight to Tempo/Jaeger.
The path from "user says the answer was wrong" to the exact span tree and the exact prompt is two
clicks — that path being short is what makes quality debugging routine instead of heroic.

Sampling: 100 % of errors, refusals, and thumbs-down; 10 % of successful requests; 100 % when the
`X-Debug-Trace` header is sent by an admin. Full sampling of successes at 50k/day is affordable but
buys nothing over the aggregate metrics.

## 3. Metrics (Prometheus)

```
# Latency, per stage — the SLO surface
aegis_stage_duration_seconds{stage,mode,status}                    histogram
aegis_request_duration_seconds{route,method,status}                 histogram
aegis_llm_ttft_seconds{provider,model}                             histogram

# Quality — the signals that reveal silent degradation
aegis_retrieval_top_score{mode}                                    histogram
aegis_retrieval_candidates{stage="fused"|"reranked"|"acl_passed"}  histogram
aegis_answers_total{status="ok|no_answer|refused|escalated|error"}  counter
aegis_citations_per_answer                                         histogram
aegis_citation_validation_failures_total{reason}                   counter
aegis_confidence_gate_total{decision}                              counter
aegis_feedback_total{rating,mode}                                  counter

# Cost
aegis_tokens_total{provider,model,kind="prompt|completion"}        counter
aegis_cost_usd_total{provider,model}                               counter
aegis_embedding_cache_hits_total / _misses_total                   counter

# Security
aegis_guardrail_blocks_total{layer,category}                       counter
aegis_authz_denials_total{role,resource}                           counter
aegis_acl_layer2_drops_total                                       counter   ← should be ~0

# Ingestion
aegis_ingest_jobs_total{status,job_type}                           counter
aegis_ingest_stage_duration_seconds{stage,mime}                    histogram
aegis_queue_depth{queue} / aegis_queue_oldest_job_seconds{queue}   gauge
aegis_index_discrepancies{kind}                                    gauge

# Dependencies
aegis_provider_errors_total{provider,code}                         counter
aegis_circuit_breaker_state{provider}                              gauge
```

### Alerts

| Alert | Condition | Severity | Why it matters |
|---|---|---|---|
| No-answer rate spike | `> 2×` 7-day baseline for 15 min | page | Usually a broken retriever or an embedding-model mismatch, not a content gap |
| Citation validation failures | `> 1 %` of answers for 10 min | page | The grounding contract is breaking; answers may be ungrounded |
| ACL layer-2 drops | `> 0` sustained 5 min | page | Vector payloads are stale — a would-be leak was caught |
| Pre-LLM latency | p95 `> 1.8 s` for 10 min | ticket | Regression in retrieval or rerank |
| Retrieval top-score drift | median drops `> 20 %` week-over-week | ticket | Corpus or model drift; the classic silent failure |
| Queue oldest job | `> 15 min` | ticket | Ingestion is backing up; documents are invisible |
| Cost burn | daily spend `> 1.5×` 7-day mean | ticket | Loop, abuse, or a prompt-size regression |
| Guardrail blocks | `> 20×` baseline from one principal | ticket | Active probing |
| Provider errors | `> 5 %` for 5 min | page | Fallback chain is carrying load |

The two alerts that do not exist in a typical service — citation-validation failures and
top-score drift — are the ones that catch RAG-specific decay. Availability alerts would stay green
through both.

---

## 4. Analytics

Dashboard queries read materialized views (schema §11) refreshed every 5 minutes
`CONCURRENTLY`, never `query_traces` directly. Raw traces are ~1.5 GB/month; a dashboard that
scans them turns every page load into a sequential scan across partitions.

| Panel | Source | Decision it drives |
|---|---|---|
| Questions / day, by mode and role | `mv_query_daily` | Adoption, capacity |
| No-answer rate trend | `mv_query_daily` | Is the corpus improving |
| **Content gaps** | `query_traces` where `answer_status IN ('no_answer','refused')`, clustered, with nearest-document scores | *Which documents to write next* |
| p50/p95 latency + stage breakdown | `mv_query_daily` + stage histogram | Where to optimize |
| Token and cost by model, role, department | `mv_query_daily` | Chargeback, model selection |
| Top questions with answer rate | `mv_top_questions` | FAQ candidates, prompt tuning |
| Most-cited documents | `mv_document_usage` | Which content earns its keep |
| **Never-cited documents** | `documents` LEFT JOIN `mv_document_usage` | Index bloat to remove — smaller index, better precision |
| Retrieval quality vs feedback | `query_traces` ⋈ `feedback` | Empirical threshold tuning |
| User activity | `mv_query_daily` | Enablement targets |

Analytics respects the caller's scope: an internal employee sees only their own usage, a manager
their department subtree, an admin everything. Analytics is a read of question text, which is
sensitive — "what is everyone asking about layoffs" is not a query a manager should be able to run
across departments.

**Top-question clustering** starts as normalized-text grouping (schema §11) and upgrades in Phase 5
to embedding-based clustering, because "how do I claim expenses" and "expense reimbursement
process" are the same question and text normalization will never merge them.

---

## 5. Evaluation

### Datasets

| Dataset | Size target | Purpose |
|---|---|---|
| `golden_internal.jsonl` | 150 cases | Answerable internal questions with expected document + page |
| `golden_customer.jsonl` | 100 cases | Customer-facing questions, no internal leakage permitted |
| `adversarial.jsonl` | 120 cases | Injection, jailbreak, escalation, exfiltration, secret extraction |
| `unanswerable.jsonl` | 60 cases | Plausible questions **not** in the corpus — `must_refuse: true` |
| `regressions.jsonl` | grows | Every production bad answer, added from the trace viewer in one click |

The `unanswerable` set is the one most teams omit and the one that matters most here: without it,
every metric rewards answering, and the model learns (via prompt tuning) to always answer. Its
metric is refusal rate, and the target is 100 %.

Each case carries `as_role`, so the same question is evaluated under different principals — which
is how the permission boundary is tested as a *quality* property, not only a security property.

### Metrics

| Metric | Tool | Gate | Measures |
|---|---|---|---|
| Faithfulness | Ragas | ≥ 0.92 | Are claims supported by retrieved context (the hallucination metric) |
| Answer relevancy | Ragas | ≥ 0.85 | Does the answer address the question |
| Context precision | Ragas | ≥ 0.75 | Is retrieved context mostly relevant (reranker quality) |
| Context recall | Ragas | ≥ 0.85 | Did retrieval find what was needed (retriever quality) |
| Answer correctness | Ragas / DeepEval | ≥ 0.80 | Against ground truth where available |
| **Citation correctness** | custom | ≥ 0.95 | Does every marker resolve to a real chunk, and does the quote exist in it, and is the cited document the expected one |
| Refusal accuracy | custom | 1.00 on `unanswerable`, ≥ 0.98 on adversarial | Refuses when it should |
| Retrieval recall@k | custom | ≥ 0.90 @20 | Expected document in the candidate set |
| MRR / nDCG@10 | custom | tracked | Ranking quality, for reranker comparisons |

Citation correctness is custom because no framework checks the property this product actually
sells: that `[^1]` points at a chunk which really is in the cited document, on the cited page, and
contains the quoted span. Ragas measures whether the *answer* is supported; it does not verify the
*pointer*.

DeepEval is used for its assertion ergonomics inside pytest (`assert_test` with per-case thresholds
in `tests/retrieval/`), Ragas for the aggregate metric suite, and LangSmith (optional, feature-
flagged) for trace-level dataset curation and side-by-side experiment comparison. All three are
behind an `EvaluationBackend` port — none of them is in the request path, and none is a hard
dependency of the runtime.

### Running evaluations

```
Local        make eval                      full suite against docker-compose corpus
CI (PR)      retrieval-only + adversarial   ~4 min, no generation cost, blocking
CI (main)    full suite with generation     ~15 min, blocking on the gates above
Nightly      full suite + production sample judged for unsupported claims
Ad hoc       POST /evals/runs               from the admin UI, results persisted
```

Every run stores `git_sha` and a `config_snapshot` (chunk size, top-k, rerank depth, thresholds,
model, **prompt version**). Without the config snapshot, comparing two runs is meaningless — a
metric moved, and you cannot tell whether the cause was code, config, or a prompt edit.

The PR gate deliberately skips generation: retrieval and refusal metrics catch most regressions,
run in minutes, and cost nothing. Full generation eval runs on merge, where a 15-minute wall and a
few dollars are acceptable.

### Regression discipline

A gate failure blocks the merge and prints a per-case diff (question, previous score, new score,
which chunks changed rank). Aggregate-only reporting hides the case where two cases improved and
one collapsed — which is precisely the change you must not ship.

---

## 6. Tradeoffs

**A domain `query_traces` table in addition to OTel traces.** Duplication, and roughly 1.5 GB/month.
Justified because observability backends have 7–30 day retention and are optimized for spans, not
for "show every question in the last quarter whose top score was below 0.4, grouped by department".
Product analytics and quality forensics are SQL workloads. OTel answers "why is this request slow";
`query_traces` answers "is the product any good".

**Sampling successful traces at 10 %.** Cheaper, and aggregates come from metrics rather than
traces. The cost is that a specific successful-but-wrong request may have no span tree — mitigated
by the `X-Debug-Trace` header and by always sampling refusals and negative feedback, which is where
investigations start anyway.

**LLM-as-judge for faithfulness.** It is imperfect and it costs money per eval run. The alternatives
are human review (accurate, unscalable, cannot gate CI) and n-gram overlap (cheap, useless for
paraphrase). Mitigations: a fixed judge model and prompt version pinned per run so scores are
comparable, and thresholds calibrated once against ~50 human-labelled cases so the absolute numbers
mean something rather than being tracked as a bare trend.

**Materialized views instead of a streaming aggregation pipeline.** Five-minute staleness on the
dashboard. A real-time pipeline (Kafka → ClickHouse) would give sub-second freshness and far better
analytical performance, but adds two stateful services for a dashboard nobody watches by the second.
The upgrade path is clean: `query_traces` is already an append-only event stream, so it can be
tailed into ClickHouse later without changing a single write path.

**Evaluation gates block deploys.** This will occasionally block a legitimate release on a flaky
judge score. Accepted, with an admin-only documented override that records who overrode which gate
and why — because a quality gate that anyone can silently skip is decoration.
