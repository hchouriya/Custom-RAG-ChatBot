# 02 — Project Structure

## 1. The dependency rule

Four layers, and imports may only point inward:

```
api  →  services  →  domain  ←  infrastructure / rag / agents
```

`domain` imports nothing from the project. `infrastructure`, `rag`, and `agents` implement
`domain` ports and may import `domain`, but no `service` may import a concrete adapter — it
receives the port via the container. This is enforced mechanically in CI with
[`import-linter`](https://import-linter.readthedocs.io) contracts, because a layering rule that
only lives in a document is a layering rule that is already broken.

```ini
# backend/.importlinter (Phase 2)
[importlinter:contract:layers]
name = Aegis layering
type = layers
layers =
    aegis.api
    aegis.services
    aegis.rag | aegis.agents | aegis.infrastructure
    aegis.domain
```

## 2. Backend tree

`src` layout, so tests import the installed package and cannot accidentally rely on CWD.

```
backend/
├── pyproject.toml                  deps, ruff, mypy, pytest, coverage config
├── alembic.ini
├── Dockerfile                      multi-stage; api + worker share it
├── .importlinter
├── migrations/                     Alembic
│   ├── env.py                      async engine, autogenerate target
│   └── versions/
├── src/aegis/
│   ├── main.py                     ASGI app factory, lifespan, middleware order
│   ├── worker.py                   arq WorkerSettings, cron definitions
│   ├── cli.py                      typer: seed, reindex, eval, create-admin, reconcile
│   │
│   ├── core/                       ── cross-cutting, no business logic
│   │   ├── config.py               Settings (pydantic-settings), one source of truth
│   │   ├── container.py            DI container; builds and closes all clients
│   │   ├── errors.py               domain exception hierarchy → RFC 9457
│   │   ├── logging.py              structlog setup, redaction, request-id contextvar
│   │   ├── telemetry.py            OTel tracer/meter, Prometheus registry
│   │   ├── security.py             JWT encode/verify, Argon2id, constant-time compare
│   │   ├── ratelimit.py            Redis sliding-window limiter, per-role policies
│   │   ├── pagination.py           cursor pagination primitives
│   │   └── secrets.py              SecretsProvider port + env/AWS/Vault adapters
│   │
│   ├── domain/                     ── pure; stdlib + pydantic only
│   │   ├── enums.py                Role, Visibility, Mode, ChunkType, IngestStatus, …
│   │   ├── entities.py             User, Document, DocumentVersion, Chunk, Conversation…
│   │   ├── values.py               SecurityContext, RetrievalPlan, RetrievedChunk,
│   │   │                           Citation, TokenBudget, Confidence
│   │   ├── policies/
│   │   │   ├── acl.py              principal + mode → VectorFilter (the G2 guarantee)
│   │   │   ├── budget.py           token allocation across summary/history/context
│   │   │   └── confidence.py       score+coverage → answer | clarify | refuse
│   │   └── ports/                  Protocols only, no implementations
│   │       ├── llm.py              LLMProvider: complete, stream, count_tokens
│   │       ├── embeddings.py       EmbeddingProvider: embed_documents, embed_query, dim
│   │       ├── reranker.py         Reranker: rerank(query, docs, top_n)
│   │       ├── vector_store.py     VectorStore: upsert, search, hybrid_search, delete
│   │       ├── parser.py           DocumentParser: supports(mime), parse → ParsedDoc
│   │       ├── chunker.py          Chunker: split(ParsedDoc) → list[ProtoChunk]
│   │       ├── object_store.py     ObjectStore: presign, get, head, delete
│   │       ├── cache.py            Cache, RateLimiter
│   │       ├── queue.py            JobQueue: enqueue, enqueue_in
│   │       ├── scanner.py          MalwareScanner, InjectionScanner
│   │       ├── crm.py              TicketSink
│   │       └── repositories.py     one Protocol per aggregate repository
│   │
│   ├── services/                   ── use cases; orchestrate ports, own transactions
│   │   ├── auth_service.py         login, refresh rotation, revocation, guest sessions
│   │   ├── user_service.py         users, roles, permission overrides
│   │   ├── collection_service.py
│   │   ├── document_service.py     metadata, ACL, versions, replace, delete, reindex
│   │   ├── ingestion_service.py    the pipeline coordinator (runs in worker)
│   │   ├── chat_service.py         conversations, messages, streaming, persistence
│   │   ├── retrieval_service.py    thin facade over the LangGraph app
│   │   ├── ticket_service.py       escalation + outbox
│   │   ├── analytics_service.py    dashboard queries against MVs
│   │   ├── audit_service.py        append-only audit writes
│   │   └── evaluation_service.py   dataset runs, metric persistence
│   │
│   ├── rag/                        ── the RAG subsystem (adapters + algorithms)
│   │   ├── parsing/                pdf, ocr, docx, pptx, xlsx, csv, html, markdown, txt
│   │   │   ├── registry.py         mime → parser resolution
│   │   │   ├── cleaning.py         de-hyphenation, header/footer strip, normalization
│   │   │   └── structure.py        heading tree, page map, table/code region detection
│   │   ├── chunking/               recursive, semantic, markdown, table, code, adaptive
│   │   │   └── router.py           picks the strategy per document region
│   │   ├── embeddings/             openai, voyage, cohere, bge_tei, cached, fake
│   │   ├── retrievers/             dense, sparse_bm25, hybrid, fusion (RRF), mmr
│   │   ├── reranker/               bge_tei, cross_encoder, cohere, noop
│   │   ├── compression/            sentence selection, dedupe, context packing
│   │   ├── citations/              marker mapping, quote extraction, validation
│   │   ├── guardrails/             input, document, output scanners + policies
│   │   ├── prompts/                versioned templates + renderer
│   │   ├── llm/                    openai, anthropic, gemini, router, fallback chain
│   │   └── vector_stores/          qdrant, pgvector, in_memory (tests)
│   │
│   ├── agents/                     ── LangGraph
│   │   ├── state.py                typed graph state + reducers
│   │   ├── nodes/                  one file per node, each independently testable
│   │   ├── edges.py                routing predicates
│   │   ├── graph.py                assembly + compile
│   │   └── checkpointer.py         Redis/Postgres checkpoint saver
│   │
│   ├── infrastructure/             ── everything that touches the outside world
│   │   ├── database/
│   │   │   ├── engine.py           async engine, session factory, unit of work
│   │   │   ├── models/             SQLAlchemy 2.0 declarative (mapped_column)
│   │   │   ├── repositories/       Protocol implementations
│   │   │   └── seed.py
│   │   ├── storage/                s3 (aioboto3), local
│   │   ├── cache/                  redis cache + limiter
│   │   ├── queue/                  arq adapter
│   │   ├── scanners/               clamav, regex injection scanner
│   │   └── crm/                    local_outbox, webhook
│   │
│   ├── api/                        ── HTTP edge; thin, no business logic
│   │   ├── router.py               /api/v1 aggregation
│   │   ├── deps.py                 principal, security context, container, idempotency
│   │   ├── middleware/             request-id, timing, error handler, security headers
│   │   ├── schemas/                request/response models (never SQLAlchemy models)
│   │   ├── sse.py                  event framing, heartbeats, disconnect handling
│   │   └── v1/                     auth, chat, documents, collections, users, acl,
│   │                               analytics, tickets, evals, admin, health
│   └── workers/
│       ├── jobs/                   ingest, reindex, purge, summarize, refresh_mv, eval
│       └── retry.py                backoff policy, dead-letter handling
└── tests/
    ├── conftest.py                 testcontainers (pg, qdrant, redis), factories
    ├── unit/                       domain + services with fakes; no I/O
    ├── integration/                repositories, vector stores, ingestion, migrations
    ├── api/                        httpx ASGI transport, full contract coverage
    ├── security/                   rbac matrix, prompt injection corpus, authz fuzzing
    ├── retrieval/                  recall/precision on the golden set, citation accuracy
    └── load/                       locust scenarios + k6 smoke
```

### Mapping to the requested layout

Every module named in the brief exists; some are nested for cohesion. Nothing was dropped.

| Requested | Here | Note |
|---|---|---|
| `api/` | `api/` | + versioned `v1/`, middleware, SSE framing |
| `services/` | `services/` | use-case layer |
| `rag/` | `rag/` | parent of the RAG adapters below |
| `embeddings/` | `rag/embeddings/` | grouped under `rag` — all four providers |
| `retrievers/` | `rag/retrievers/` | dense, BM25, hybrid, fusion, MMR |
| `reranker/` | `rag/reranker/` | BGE, cross-encoder, Cohere, noop |
| `chunking/` | `rag/chunking/` | five strategies + router |
| `prompts/` | `rag/prompts/` | versioned, DB-overridable templates |
| `agents/` | `agents/` | LangGraph nodes/edges/state |
| `auth/` | `core/security.py` + `services/auth_service.py` + `api/v1/auth.py` | split by layer rather than by feature, so JWT crypto is not importable from a router |
| `database/` | `infrastructure/database/` | models, repositories, engine, seed |
| `logging/` | `core/logging.py` + `core/telemetry.py` | renamed to avoid a package that shadows stdlib `logging`, which is a real hazard when a module inside it does `import logging` |
| `analytics/` | `services/analytics_service.py` + `api/v1/analytics.py` + MVs in migrations | analytics is a query concern, not a subsystem |
| `tests/` | `tests/` | split by test *type*, since each type has a different runtime and gate |

## 3. Frontend tree

Feature-sliced. A feature owns its components, hooks, and API calls; `shared/` owns primitives
and must not import from a feature.

```
frontend/
├── package.json  tsconfig.json (strict)  next.config.ts  tailwind.config.ts
├── Dockerfile
├── src/
│   ├── app/
│   │   ├── layout.tsx              html shell, theme provider, fonts
│   │   ├── (auth)/login/page.tsx
│   │   ├── (app)/layout.tsx        authenticated shell: sidebar, mode switch, user menu
│   │   ├── (app)/chat/page.tsx                 new conversation
│   │   ├── (app)/chat/[id]/page.tsx            existing conversation (RSC-loaded history)
│   │   ├── (app)/dashboard/page.tsx            analytics
│   │   ├── (app)/admin/documents/page.tsx      list, filters, bulk actions
│   │   ├── (app)/admin/documents/[id]/page.tsx detail, versions, chunk inspector
│   │   ├── (app)/admin/collections/page.tsx
│   │   ├── (app)/admin/users/page.tsx
│   │   ├── (app)/admin/roles/page.tsx          permission matrix editor
│   │   ├── (app)/admin/index-status/page.tsx   queue depth, failures, reindex
│   │   ├── (app)/admin/logs/page.tsx           audit log + query trace viewer
│   │   ├── (app)/admin/tickets/page.tsx
│   │   ├── (app)/admin/evaluations/page.tsx
│   │   └── api/[...proxy]/route.ts  BFF: httpOnly cookie → Bearer, streams SSE through
│   ├── features/
│   │   ├── auth/                   login form, session hook, role guards
│   │   ├── chat/
│   │   │   ├── components/         MessageList, Composer, MessageBubble, Markdown,
│   │   │   │                       CodeBlock, CitationChip, CitationDrawer,
│   │   │   │                       SourcePanel, TypingIndicator, SuggestedQuestions,
│   │   │   │                       ConversationSidebar, FeedbackButtons, EscalationForm
│   │   │   ├── hooks/              useChatStream (SSE reader), useConversations
│   │   │   └── lib/                event parsing, marker → citation resolution
│   │   ├── admin/                  DocumentTable, UploadDropzone, MetadataForm,
│   │   │                           AclEditor, VersionTimeline, ChunkInspector,
│   │   │                           ReindexButton, PermissionMatrix, TraceViewer
│   │   └── dashboard/              StatCard, TimeSeries, TopQuestionsTable,
│   │                               NoAnswerPanel, LatencyBreakdown, TokenSpend
│   └── shared/
│       ├── ui/                     Button, Input, Select, Dialog, Drawer, Table, Toast,
│       │                           Badge, Tabs, Skeleton, EmptyState, ThemeToggle
│       ├── api/                    typed fetch client, generated types from OpenAPI
│       ├── lib/                    cn(), formatters, date/token/bytes helpers
│       └── hooks/                  useTheme, useDebounce, useMediaQuery, useCopy
└── tests/                          vitest unit + playwright e2e
```

Types are **generated from the backend's OpenAPI schema** into `shared/api/types.gen.ts` by a
script, not hand-written. Hand-maintained duplicate types are the most common source of
frontend/backend drift, and here the contract is large.

## 4. Supporting trees

```
infra/
├── docker-compose.yml              full stack
├── docker-compose.override.yml     dev: hot reload, exposed ports
├── docker-compose.prod.yml         resource limits, no ports, healthcheck gates
├── nginx/nginx.conf                TLS, SSE-safe buffering off, body limits
├── postgres/init/                  extensions: vector, pg_trgm, citext, uuid-ossp
├── qdrant/config.yaml              quantization, on-disk payload
├── observability/                  prometheus.yml, grafana dashboards, otel-collector.yaml
└── seeds/                          users, collections, prompt templates, permissions

evals/
├── datasets/golden_internal.jsonl  question, expected answer, expected doc/page, role
├── datasets/golden_customer.jsonl
├── datasets/adversarial.jsonl      injection, jailbreak, escalation, exfiltration
├── harness/                        ragas + deepeval runners, CI reporter
└── reports/

samples/                            licence-clean corpus: handbook.pdf, policy.docx,
                                    pricing.xlsx, faq.md, onboarding.pptx, scanned_invoice.pdf,
                                    api_guide.html, employees.csv

scripts/                            bootstrap.ps1/.sh, gen-openapi-types, reindex, load-test,
                                    reconcile-vectors, rotate-keys
```

## 5. Conventions

- **Naming.** `snake_case` modules, `PascalCase` classes, verb-first services
  (`DocumentService.replace_version`), adapters named `<Tech><Port>` (`QdrantVectorStore`).
- **Typing.** `mypy --strict` on `domain` and `services`; `ruff` with `ANN`, `ASYNC`, `S`
  (bandit), and `TID` (banned imports) enabled repo-wide.
- **Every port has a fake.** `tests/fakes/` holds an in-memory implementation of each protocol.
  Unit tests never reach the network; if a unit test needs Docker, the layering is wrong.
- **Migrations are reviewed as code.** Autogenerate is a starting point; every revision gets a
  hand-checked `downgrade`, and destructive steps are split across releases.
- **No business logic in `api/`.** A router may validate, authorize, call one service, and shape
  a response. If a router contains an `if` about domain rules, it belongs in a service or policy.
