# Architecture Decision Log

Decisions that would be expensive to reverse. Each record states what was decided, what was
rejected, and what it costs us. Records are immutable — a changed decision is a new record that
supersedes an old one.

New decisions from Phase 2 onward get their own file (`NNNN-title.md`) using the template at the
bottom. The Phase 1 set is recorded inline here because they were taken together as one coherent
design.

| # | Decision | Status |
|---|---|---|
| [0001](#adr-0001) | Hexagonal architecture with ports and adapters | Accepted |
| [0002](#adr-0002) | LangGraph for orchestration, LlamaIndex for ingestion | Accepted |
| [0003](#adr-0003) | Qdrant as default vector store, pgvector as alternative | Accepted |
| [0004](#adr-0004) | Hybrid retrieval with RRF fusion, not vector-only | Accepted |
| [0005](#adr-0005) | Mandatory cross-encoder reranking | Accepted |
| [0006](#adr-0006) | ACL enforced pre-retrieval, verified twice | Accepted |
| [0007](#adr-0007) | Immutable document versions with atomic activation | Accepted |
| [0008](#adr-0008) | Refuse below a confidence gate; validate citations post-generation | Accepted |
| [0009](#adr-0009) | SSE for streaming, not WebSocket | Accepted |
| [0010](#adr-0010) | Tokens in httpOnly cookies via a Next.js BFF | Accepted |
| [0011](#adr-0011) | Asynchronous ingestion with arq, presigned direct upload | Accepted |
| [0012](#adr-0012) | Domain `query_traces` table alongside OpenTelemetry | Accepted |
| [0013](#adr-0013) | Local inference sidecars (TEI) for embedding and reranking | Accepted |
| [0014](#adr-0014) | Prompts versioned in the database | Accepted |

---

## ADR-0001 {#adr-0001}
### Hexagonal architecture with ports and adapters

**Context.** The platform must swap LLMs, embedding models, rerankers, and vector stores without
touching business logic, and its security-critical logic (ACL filter construction, token budgeting,
confidence gating) must be testable exhaustively and fast. Model and vendor churn in this space is
measured in months.

**Decision.** Four layers — `api → services → domain ← infrastructure/rag/agents` — with `domain`
depending on nothing, all external capabilities expressed as `Protocol` ports, and wiring in one
explicit DI container built at application startup. Layering is enforced in CI by `import-linter`.

**Alternatives.** A conventional FastAPI layout (`routers/`, `models/`, `crud/`) is faster to start
and familiar, but business logic diffuses into routers and ORM models, and swapping a provider means
editing call sites everywhere. A framework-driven design (LlamaIndex `QueryEngine` as the core
abstraction) would delete a lot of code, at the price of coupling the domain to a library whose API
changes frequently. Full DDD with aggregates and domain events was considered overweight for a
system whose core domain is a retrieval pipeline rather than complex business invariants.

**Consequences.** More files and one indirection per capability. Unit tests for services and
policies run without Docker in milliseconds, which is what makes a 200-case RBAC matrix and a
120-case injection corpus practical on every commit. Provider swaps are container-level edits.

---

## ADR-0002 {#adr-0002}
### LangGraph for orchestration, LlamaIndex for ingestion

**Context.** The retrieval flow has real branches: block, clarify, short-circuit small talk,
refuse, escalate. It must be inspectable stage by stage, and each stage must be independently
testable. Separately, we need excellent document readers for nine formats.

**Decision.** LangGraph owns control flow as an explicit typed state machine. LlamaIndex is used
only for readers and node parsers, behind the `DocumentParser` and `Chunker` ports. LangChain is not
in the request path.

**Alternatives.** Hand-rolled orchestration (total control, but we reimplement state reduction,
streaming, and checkpointing, and lose graph visualization). LlamaIndex end-to-end (fastest demo;
branch logic ends up inside response synthesizers and callbacks, which destroys explainability). A
LangChain agent with tools (the LLM decides control flow — unacceptable non-determinism for an
audited enterprise flow, and it would also hand an injected instruction a tool to call).

**Consequences.** Two AI frameworks in the dependency set, mitigated by strict import confinement —
neither type appears in `domain` or `services`. Graph state must be serializable for checkpointing.
The payoff is that "why did it answer that?" is answered by reading a per-node trace.

---

## ADR-0003 {#adr-0003}
### Qdrant as default vector store, pgvector as alternative

**Context.** Target scale is 10M chunks with mandatory ACL pre-filtering on every query and hybrid
dense+sparse retrieval.

**Decision.** Qdrant by default: named dense and sparse vectors on one point, server-side fusion,
payload indexes for filtering, int8 quantization with rescoring, on-disk payloads. A `PgVectorStore`
adapter satisfies the same port for deployments that refuse a second stateful service or run below
~2M chunks.

**Alternatives.** pgvector only (one datastore, one backup, transactional consistency — but slow
HNSW builds, degraded filtered ANN, and BM25 must be bolted on via `tsvector` with client-side
fusion). Elasticsearch/OpenSearch (best BM25, heavier ops, weaker ANN ergonomics). Pinecone
(managed, but data residency and no self-host option). Weaviate (comparable, no decisive advantage).

**Consequences.** Two stores can diverge. Mitigated structurally: PostgreSQL is authoritative,
Qdrant is a rebuildable derived index, `chunks.vector_point_id`/`indexed_at` track sync state, point
IDs are deterministic (`uuid5`) so upserts are idempotent, and a nightly reconciler reports and
repairs drift into `index_discrepancies`. Qdrant is deliberately excluded from the backup strategy.

---

## ADR-0004 {#adr-0004}
### Hybrid retrieval with RRF fusion, not vector-only

**Context.** Enterprise corpora are dense with exact tokens that embeddings blur: product SKUs,
error codes, policy numbers, acronyms, person names. Pure dense retrieval misses them; pure BM25
misses paraphrase.

**Decision.** Every query runs dense and sparse search under the same ACL filter and merges with
Reciprocal Rank Fusion (`k=60`), then dedupes by content hash.

**Alternatives.** Dense-only (simpler and ~100 ms faster; measurably worse on identifier queries).
Weighted score normalization instead of RRF (requires calibrating incommensurable score scales, and
the calibration drifts with corpus and model). A learned fusion model (better ceiling, needs
training data and a serving path we do not yet have).

**Consequences.** Two searches and a sparse representation per chunk — more ingestion work and more
index memory. RRF needs no tuning beyond `k`, which is its main virtue. Fusion weights are
configurable for corpora that are genuinely one-sided.

---

## ADR-0005 {#adr-0005}
### Mandatory cross-encoder reranking

**Context.** Bi-encoder retrieval optimizes a symmetric embedding space; it does not read the query
and the passage together. Sending 40 loosely-relevant chunks to the LLM both dilutes attention and
costs tokens.

**Decision.** Rerank the fused top ~40 with a cross-encoder and pass only the top 8 forward. Local
BGE-reranker-v2-m3 on TEI by default; Cohere Rerank and a `sentence-transformers` cross-encoder are
alternative adapters; `NoopReranker` exists for load tests and degraded mode.

**Alternatives.** No reranking (~350 ms faster, materially worse context precision). LLM-as-reranker
(higher quality ceiling, an order of magnitude more latency and cost). MMR only (cheap diversity, no
relevance re-estimation — used *in addition*, during compression).

**Consequences.** ~350 ms on the critical path and one more service to run. Failure degrades to
fused order with a flag and a *raised* confidence threshold, since rank order is then less
trustworthy. This is the single largest quality lever in the pipeline and it is why "top 8 good
chunks" beats "top 40 mediocre chunks" on both accuracy and cost.

---

## ADR-0006 {#adr-0006}
### ACL enforced pre-retrieval, verified twice

**Context.** Invariant I1: a principal must never receive content above their entitlement. Vector
payloads carrying ACL data are replicas of PostgreSQL and can be stale between an ACL change and a
reindex.

**Decision.** A single `build_filter(SecurityContext, narrowing)` function derives the filter
server-side and pushes it into the ANN search (layer 1); survivors are re-verified against
PostgreSQL before context assembly (layer 2). Client-supplied filters may only intersect. Visibility
is a total order stored as an integer so filtering is a range predicate. Customer mode is a hard
ceiling that no role, including admin, can raise.

**Alternatives.** Post-filtering after unfiltered retrieval (destroys recall — the top 40 may be
entirely unauthorized — and puts unauthorized text through the reranker). Separate collections per
role (combinatorial explosion with departments and per-document grants). An external policy engine
such as OPA (more expressive and externally auditable, but adds a per-query round trip and the
policy still has to compile into a vector filter, so the hard part remains).

**Consequences.** ACL data is denormalized into `chunks` and the vector payload, requiring triggers,
an update job on ACL change, and nightly verification. Layer 2 costs ~30 ms. Every layer-2 drop is
counted and alerted, turning a would-be leak into a monitored event. `build_filter` is the seam where
OPA would later plug in.

---

## ADR-0007 {#adr-0007}
### Immutable document versions with atomic activation

**Context.** Documents are replaced regularly, and stored answers cite specific text. Reindexing
takes minutes for large files. A citation that silently resolves to different text than the model
actually read is worse than no citation.

**Decision.** `documents` is a stable identity; `document_versions` is immutable content. A
replacement ingests fully into a new version, then flips `documents.active_version_id` in one
statement. `chunks` are never hard-deleted while a `message_citations` row references them
(`ON DELETE RESTRICT`). Superseded vectors are purged with a short delay; superseded rows are
retained until unreferenced and past retention.

**Alternatives.** Delete-and-reingest (a window with no searchable document, and every prior
citation breaks). Mutate chunks in place (no history, no rollback, corrupt audit trail). Full
temporal tables (more machinery than the two-level identity/version model needs).

**Consequences.** More storage and a purge job with retention rules. In exchange: zero-downtime
replacement, one-click rollback to any indexed version, failed ingests that cannot affect
production, and citations that remain honest years later. The admin UI surfaces "cited by N answers"
so the retention rule is legible rather than mysterious.

---

## ADR-0008 {#adr-0008}
### Refuse below a confidence gate; validate citations post-generation

**Context.** The product requirement is "never hallucinate". Prompt instructions alone do not
achieve it.

**Decision.** Two independent mechanisms. Before generation, a confidence gate (top rerank score,
count of supporting chunks, mean of top 3, query-entity coverage) refuses with a fixed verbatim
message plus nearest documents and an escalation offer. After generation, every `[^n]` marker is
resolved against the chunks actually in context and every quoted span is fuzzy-matched against its
chunk; invalid markers are dropped, and a factual answer left with zero valid citations is discarded
and converted to a refusal.

**Alternatives.** Prompt-only instruction (cheapest, unreliable). A separate NLI entailment model
per sentence (stronger, adds latency and another model to the hot path — used offline for nightly
sampling instead). Always answering with a confidence score shown to the user (shifts the judgement
onto a reader who cannot verify it).

**Consequences.** Some answerable questions get refused. That is the accepted asymmetry: a false
refusal costs one escalation, a confident fabrication about pricing or leave entitlement can cost a
legal dispute. Thresholds are configurable, tuned against a golden set that includes a dedicated
unanswerable set, and every refusal feeds the content-gap dashboard so the cost is visible and
fixable by writing documents.

---

## ADR-0009 {#adr-0009}
### SSE for streaming, not WebSocket

**Context.** Answers must stream token by token to keep perceived latency low, through corporate
proxies and cloud load balancers.

**Decision.** Server-Sent Events over a normal authenticated `POST`, with named events
(`meta`, `citations`, `token`, `usage`, `done`, `error`, `refusal`, `clarify`) and a 15 s comment
heartbeat. Cancellation is a separate `DELETE`.

**Alternatives.** WebSocket (bidirectional, needed for nothing here; adds a second auth path, a
second rate-limit implementation, stateful connections, and sticky-session concerns). Polling
(simple, terrible perceived latency). Long-lived HTTP chunked responses without event framing (works,
but adding a new stage later breaks clients).

**Consequences.** Proxies must not buffer the chat route and idle timeouts must exceed the longest
stream — both documented in the deployment guide because both fail only in production. Client
disconnects are detected through the ASGI receive channel and abort generation, so an abandoned tab
stops costing money.

---

## ADR-0010 {#adr-0010}
### Tokens in httpOnly cookies via a Next.js BFF

**Context.** Access and refresh tokens grant access to confidential corpora. Frontend XSS is the
realistic threat.

**Decision.** The browser holds no bearer token. Next.js route handlers keep the refresh token in an
httpOnly `Secure` `SameSite=Lax` cookie and an encrypted session cookie for the access token, and
attach `Authorization` server-side when proxying to FastAPI — including streaming SSE through.

**Alternatives.** `localStorage` (trivial for any XSS to exfiltrate a 14-day credential). In-memory
tokens with a silent-refresh iframe (loses the session on reload; refresh token still exposed).
Direct browser→API calls with cookies and no BFF (needs cross-site cookies plus CSRF defenses and
gives up server-side rendering with data).

**Consequences.** One extra hop for API calls, and the BFF must correctly stream SSE rather than
buffering it. `SameSite=Lax` plus an origin check covers CSRF for the cookie-authenticated proxy
routes. Server components can render lists and history with data on first paint.

---

## ADR-0011 {#adr-0011}
### Asynchronous ingestion with arq, presigned direct upload

**Context.** A 300-page scanned PDF is minutes of OCR. Upload must not block a request, occupy API
memory, or degrade chat latency.

**Decision.** Three-step upload (validate intent → presigned PUT direct to object storage →
register), then `202 Accepted` and an arq job. Workers share the API codebase and run in separate
containers. Jobs are idempotent, keyed on content checksum, with per-stage checkpoints.

**Alternatives.** Synchronous processing (request lifetime tied to document size; a hard size cap
equal to the proxy timeout). Celery (mature, huge ecosystem, but sync-first — it would force a
duplicate synchronous data layer in an otherwise async codebase). SQS/Cloud Tasks (better durability,
another managed dependency, and Redis is already present for cache and rate limiting).

**Consequences.** The UI must handle pending states and show progress, which the admin screens do
explicitly. Redis-grade durability is acceptable because the authoritative "needs indexing" record
is a `document_versions` row with `status='pending'` and a startup reconciler re-enqueues stranded
work — a lost message costs a delay, never a document.

---

## ADR-0012 {#adr-0012}
### Domain `query_traces` table alongside OpenTelemetry

**Context.** Two different consumers: an on-call engineer asking "why is this slow" and a product
owner asking "which questions failed last quarter and what was retrieved". Observability backends
retain 7–30 days and are optimized for spans, not SQL analytics.

**Decision.** Keep both. OTel spans for live debugging; a partitioned `query_traces` table
(monthly, 400-day retention) recording the rewritten queries, the exact filter, every candidate with
all four scores, the context, latency per stage, tokens, cost, and guardrail flags. Written off the
critical path through a bounded queue; the `trace_id` links the two systems.

**Alternatives.** OTel only (loses long-horizon product analytics and cheap SQL forensics). A
separate analytics pipeline into ClickHouse (better analytical performance; two more stateful
services for a workload of ~3.5 QPS). Log-only (no aggregation without a log platform, and question
text in logs is a PII problem).

**Consequences.** ~1.5 GB/month and a table nobody may query without a time predicate. Dashboards
read materialized views, never raw traces. Because the table is already an append-only event stream,
tailing it into ClickHouse later requires no change to any write path.

---

## ADR-0013 {#adr-0013}
### Local inference sidecars (TEI) for embedding and reranking

**Context.** Confidential documents should not be sent to third-party APIs for embedding, and
reranking every query through a hosted API adds cost and latency. Running the models in-process is
worse than it looks.

**Decision.** Two HuggingFace Text Embeddings Inference sidecars — BGE-M3 for embeddings,
bge-reranker-v2-m3 for reranking — reached over HTTP behind the `EmbeddingProvider` and `Reranker`
ports. Hosted providers (OpenAI, Voyage, Cohere) remain first-class adapters and are the default in
the `.env.example` for teams without local capacity.

**Alternatives.** In-process `sentence-transformers` (adds ~4 GB RSS per API replica, ~40 s cold
start, ML wheels in the API image, and CPU inference inside the asyncio event loop — the fastest way
to destroy tail latency for 200 concurrent streams). Hosted-only (simplest, but content leaves the
network and per-query cost scales with traffic).

**Consequences.** Two more services and capacity planning; GPU is optional but changes rerank
latency substantially. Sidecars batch across API replicas, scale independently, and move to GPU
nodes with a URL change. The API image stays small and its cold start fast.

---

## ADR-0014 {#adr-0014}
### Prompts versioned in the database

**Context.** Prompt edits change answer quality more than most code changes, are frequently made by
non-engineers, and are the least reviewed change in a RAG system.

**Decision.** `prompt_templates` holds every version with exactly one active row per
`(key, mode)`. Files under `rag/prompts/` are the seed defaults and the fallback if the table is
empty. Every eval run snapshots the prompt version alongside the git SHA.

**Alternatives.** Files only (reviewable in PRs, but every tweak needs a deploy and prompt-vs-code
regressions are indistinguishable in history). A hosted prompt-management SaaS (nice UI, another
vendor in the request path, and prompts are as sensitive as code).

**Consequences.** Prompts become runtime state that must be backed up, audited, and permission-gated
(`admin` only). In exchange: instant rollback without a deploy, A/B comparison against the golden
set, and an eval report that can attribute a metric move to a prompt change rather than to code.

---

## Template for new records

```markdown
# ADR-00NN — Title

**Status.** Proposed | Accepted | Superseded by ADR-00MM
**Date.** YYYY-MM-DD
**Deciders.** roles

## Context
The forces at play, the constraints, and what makes this decision necessary now.

## Decision
What we will do, stated so that a reader can tell whether the code complies.

## Alternatives considered
Each option with its genuine advantage and the specific reason it was not chosen.

## Consequences
What this costs, what it enables, what must now be monitored or maintained, and
what would have to be true for us to revisit it.
```
