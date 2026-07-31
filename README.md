# Aegis RAG Platform

An enterprise Retrieval-Augmented Generation platform that answers questions **only** from
company documents, cites every claim, enforces permissions **before** retrieval, and refuses
to answer when the corpus does not contain the answer.

It is not a prompt proxy. A user question travels through authentication, role resolution,
intent classification, query rewriting, ACL-constrained hybrid retrieval, cross-encoder
reranking, context compression, citation mapping, a grounding gate, and output guardrails
before a single token reaches the user.

---

## Quick start

### Full stack with Docker

```bash
cp .env.example .env
# set SECRET_KEY and SESSION_SECRET (32+ chars each)
docker compose up -d postgres redis qdrant minio minio-init
# wait until healthy (or: docker compose ps)

# Migrate + admin — either inside the image (uses service DNS from .env):
docker compose run --rm api alembic upgrade head
docker compose run --rm api aegis create-admin --email admin@example.com --password '<strong-password>'
# optional: docker compose run --rm api aegis seed
# Or from the host venv: point DATABASE_URL at localhost:5432, then alembic / aegis CLI.

docker compose up -d api worker frontend
```

Or with Make: `make infra`, then `make migrate` / `make seed`, then `make up`.

| Service | Host port |
|---|---|
| API | http://localhost:8000 |
| Frontend | http://localhost:3000 |
| Postgres | localhost:5432 |
| Redis | localhost:6379 |
| Qdrant | http://localhost:6333 |
| MinIO API / console | http://localhost:9000 / http://localhost:9001 |

ClamAV is opt-in: `docker compose --profile malware up -d` (default `MALWARE_SCAN_ENABLED=false`).

**Assumptions:** API CMD is `uvicorn aegis.main:app --host 0.0.0.0 --port 8000`. Worker CMD is `python -m arq aegis.worker.WorkerSettings` (module lands with Phase 2 workers). Frontend image expects a Next.js `package.json` with `build` / `start` scripts.

### Dev without Docker for app processes

Run only datastores in Compose; run API and Next on the host:

```bash
cp .env.example .env
# Point datastores at localhost (compose still uses service DNS inside containers):
#   DATABASE_URL=postgresql+asyncpg://aegis:aegis@localhost:5432/aegis
#   REDIS_URL=redis://localhost:6379/0
#   QDRANT_URL=http://localhost:6333
#   S3_ENDPOINT_URL=http://localhost:9000
#   API_INTERNAL_URL=http://localhost:8000

docker compose up -d postgres redis qdrant minio minio-init

cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,ocr]"
alembic upgrade head
aegis create-admin --email admin@example.com --password '<strong-password>'
uvicorn aegis.main:app --reload --host 0.0.0.0 --port 8000

# other terminal — worker (when aegis.worker exists)
python -m arq aegis.worker.WorkerSettings

# other terminal
cd frontend && npm install && npm run dev
```

---

## Two isolated assistants, one platform

| | Internal AI Assistant | Customer Support AI |
|---|---|---|
| Audience | Admin, Manager, Internal Employee | Customer, Guest |
| Corpus | Public + customer + internal + confidential (subject to role, department, per-document ACL) | Public + customer-facing only, enforced by a hard ceiling that no role can lift |
| Escalation | Links to document owner | Creates a support ticket |
| Tone/prompt profile | Dense, technical, assumes context | Plain language, empathetic, never mentions internal systems |
| Rate limits | Generous | Strict, per-IP and per-session |

The isolation is structural, not prompt-based: the two modes resolve to different collections,
different retrieval filters, different prompt templates, and different guardrail profiles.
A `customer` principal cannot construct a request that reaches an internal chunk, because the
visibility ceiling is derived server-side from the authenticated principal and the mode, and it
is enforced twice — once as a pre-filter pushed into the vector store, and again as a
post-retrieval re-check against PostgreSQL before context assembly.

---

## Stack

| Layer | Choice | Why (short) |
|---|---|---|
| API | Python 3.12, FastAPI, fully async | Streaming-native, high I/O concurrency per core |
| Orchestration | LangGraph | Explicit, inspectable state machine with conditional branches; replaces hidden agent loops |
| Ingestion/parsing | LlamaIndex readers + node parsers | Best-in-class document readers and structure-aware parsers |
| LLM | OpenAI GPT-5.x, Anthropic Claude, Google Gemini | Swappable behind one `LLMProvider` port |
| Embeddings | OpenAI `text-embedding-3-large`, Voyage, Cohere, BGE (local via TEI) | Swappable behind one `EmbeddingProvider` port; dimension recorded per collection |
| Reranker | BGE cross-encoder (local, TEI), `ms-marco` cross-encoder, Cohere Rerank | Swappable behind one `Reranker` port |
| Vector DB | Qdrant (default), PostgreSQL + pgvector (alternative) | Native hybrid dense+sparse with server-side fusion; pgvector for single-datastore deployments |
| Relational DB | PostgreSQL 16, SQLAlchemy 2.0 async, Alembic | Source of truth for documents, ACL, traces, analytics |
| Cache / queue | Redis 7 + arq | Async-native worker pool, rate limiting, idempotency keys |
| Object storage | S3-compatible (MinIO locally) | Originals never live on the API container's disk |
| Frontend | Next.js 15 App Router, React 19, TypeScript (strict), Tailwind CSS v4 | Server components for shells, client components for streaming |
| Deploy | Docker, Docker Compose, cloud-ready 12-factor | One command locally, portable to ECS/GKE/AKS |

---

## Repository map

```
.
├── backend/            FastAPI service, RAG subsystem, workers, Alembic migrations, tests
├── frontend/           Next.js app: login, chat, admin, dashboard
├── docs/               Architecture documentation (start here)
│   ├── architecture/   Numbered design documents
│   └── adr/            Architecture Decision Records
├── infra/              Dockerfiles, compose files, Nginx, observability config, seeds
├── evals/              Golden datasets and Ragas/DeepEval harness
├── samples/            Sample corpus used by seed data and integration tests
└── scripts/            Developer and operational scripts
```

## Documentation index

Read in order; each document is self-contained and ends with tradeoffs.

| # | Document | Contents |
|---|---|---|
| 01 | [System architecture](docs/architecture/01-system-architecture.md) | Context/container/component views, request lifecycles, SLOs, capacity model |
| 02 | [Project structure](docs/architecture/02-project-structure.md) | Every directory, its responsibility, and its dependency rule |
| 03 | [Database schema](docs/architecture/03-database-schema.md) | 30 tables, indexes, partitioning, document versioning model |
| 04 | [API design](docs/architecture/04-api-design.md) | Full REST contract, SSE streaming protocol, error envelope |
| 05 | [RAG pipeline](docs/architecture/05-rag-pipeline.md) | Ingestion pipeline, retrieval graph, grounding gate, token budget |
| 06 | [Security, RBAC, guardrails](docs/architecture/06-security-rbac-guardrails.md) | Threat model, permission matrix, injection defenses |
| 07 | [UI wireframes](docs/architecture/07-ui-wireframes.md) | Screen-by-screen wireframes and interaction states |
| 08 | [Observability & evaluation](docs/architecture/08-observability-analytics-evaluation.md) | Logs, traces, metrics, analytics queries, Ragas gates |
| 09 | [Deployment topology](docs/architecture/09-deployment-topology.md) | Compose topology, cloud mapping, scaling and DR |
| — | [ADR index](docs/adr/README.md) | Decisions, alternatives rejected, consequences |

---

## Delivery phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Structure, architecture, schema, API design, wireframes, documentation | **In review** |
| 2 | Backend: auth, database, migrations, upload pipeline, chunking, embeddings, vector store, tests | Pending approval |
| 3 | Retrieval: hybrid search, metadata filtering, reranking, LangGraph workflow, evaluation | Pending |
| 4 | Frontend: admin panel, chat UI, streaming, citations, history | Pending |
| 5 | Analytics, logging, Docker deployment, documentation, load tests, optimization | Pending |

Nothing in Phase 1 is placeholder code — it is the contract that Phases 2–5 implement. The
schema, API surface, and pipeline stages defined here are what the code will be reviewed against.

---

## Assumptions taken (correct any before Phase 2)

These were unspecified in the brief. Defaults were chosen for the enterprise case and are all
reversible via configuration or an ADR revision.

1. **Single tenant, many departments.** One company, hierarchical departments, no cross-tenant
   isolation requirement. Multi-tenant would add a `tenant_id` to every table and a row-level
   security policy — the schema leaves room for it but does not pay for it now.
2. **Identity is local-first.** Email/password with Argon2id, plus a documented OIDC hook for
   Entra ID/Okta in Phase 5. No SCIM provisioning.
3. **English-primary corpus.** Language is detected and stored per chunk, and the embedding
   providers chosen are multilingual, but no per-language index routing.
4. **Cloud LLM APIs are permitted.** If document content may not leave the network, the
   provider ports allow a self-hosted vLLM adapter; embeddings and reranking already run locally
   via TEI in the default compose file.
5. **Scale target:** 10M chunks, ~500k documents, 5,000 internal users, 200 concurrent chat
   sessions, 50k questions/day. Sizing in document 01 follows from these numbers.
6. **Retention:** query traces 400 days (monthly partitions), audit logs 7 years, document
   versions keep the last 5 plus all versions still referenced by a stored citation.
7. **CRM integration is stubbed behind a port.** Tickets are stored locally with an outbox row
   so a Salesforce/Zendesk adapter can be added without touching the domain.

## Explicit non-goals

Agentic tool use against production systems, fine-tuning, real-time document sync connectors
(SharePoint/Drive crawlers), speech I/O, and multi-region active-active writes. Each is
architecturally anticipated (ports exist) but out of scope.
