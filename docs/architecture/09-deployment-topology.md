# 09 — Deployment Topology

One command locally, the same images in production. Twelve-factor throughout: configuration from
the environment, no state in containers, logs to stdout, and processes that can be killed at any
moment without data loss.

---

## 1. Local topology (Docker Compose)

```
                          ┌──────────────────────────────┐
                          │ nginx :80/:443               │
                          │ TLS · rate limit · body cap  │
                          │ proxy_buffering off on /sse   │
                          └──────┬───────────────┬───────┘
                                 │               │
                    ┌────────────▼──┐     ┌──────▼────────────┐
                    │ frontend :3000│     │ api :8000         │
                    │ Next.js (BFF) ├────►│ FastAPI + Uvicorn │
                    └───────────────┘     └──┬──┬──┬──┬───┬───┘
                                             │  │  │  │   │
   ┌──────────────┐  ┌──────────────┐        │  │  │  │   │
   │ worker (×2)  │  │ scheduler    │        │  │  │  │   │
   │ arq pool     │  │ arq cron     │        │  │  │  │   │
   └──┬──┬──┬──┬──┘  └──────┬───────┘        │  │  │  │   │
      │  │  │  │            │                │  │  │  │   │
      ▼  ▼  ▼  ▼            ▼                ▼  ▼  ▼  ▼   ▼
   ┌────────┐ ┌────────┐ ┌───────┐ ┌───────┐ ┌──────────┐ ┌──────────┐
   │postgres│ │ qdrant │ │ redis │ │ minio │ │tei-embed │ │tei-rerank│
   │ :5432  │ │ :6333  │ │ :6379 │ │ :9000 │ │  :8080   │ │  :8081   │
   │+pgvector│ │        │ │       │ │       │ │  BGE-M3  │ │bge-rerank│
   └────────┘ └────────┘ └───────┘ └───────┘ └──────────┘ └──────────┘
        │
   ┌────▼──────────────────────────────────────┐
   │ profile: observability                    │
   │ otel-collector · prometheus · grafana     │
   │ tempo · clamav                            │
   └───────────────────────────────────────────┘
```

Compose profiles keep the default `up` lean: `core` (postgres, qdrant, redis, minio, api, worker,
scheduler, frontend), `local-models` (the two TEI sidecars — omit them if using hosted embeddings),
`observability`, and `security` (ClamAV). A developer on a laptop runs `core` and points embeddings
at OpenAI; CI runs `core` + `local-models` so tests never depend on an external API key.

Startup ordering is expressed with `depends_on: condition: service_healthy` and real healthchecks
(`pg_isready`, Qdrant `/readyz`, Redis `PING`, TEI `/health`). `api` runs migrations through a
one-shot `migrate` service that must exit 0 first, so two API replicas never race Alembic.

### Files

| File | Purpose |
|---|---|
| `infra/docker-compose.yml` | Base topology, profiles, healthchecks, named volumes |
| `infra/docker-compose.override.yml` | Dev: source bind mounts, `--reload`, exposed ports, debug logging |
| `infra/docker-compose.prod.yml` | Resource limits, replicas, no host ports except the edge, read-only rootfs |
| `backend/Dockerfile` | Multi-stage; `api` and `worker` differ only by entrypoint |
| `frontend/Dockerfile` | Multi-stage; Next.js `standalone` output |
| `.env.example` | Every variable, documented, with safe defaults and clearly marked secrets |

### Image construction

```dockerfile
# backend/Dockerfile — shape, not the final file (Phase 5)
FROM python:3.12-slim@sha256:… AS builder
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev            # hash-pinned, reproducible

FROM python:3.12-slim@sha256:… AS runtime
RUN apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng poppler-utils libmagic1 ghostscript
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
USER 10001:10001
HEALTHCHECK CMD python -m aegis.healthcheck
ENTRYPOINT ["/app/.venv/bin/python", "-m", "uvicorn", "aegis.main:app", …]
```

Digest-pinned bases, no build toolchain in the runtime layer, non-root UID, read-only root
filesystem with `tmpfs` on `/tmp` (parsers need scratch space), and `trivy` + SBOM generation in CI.
The OCR and PDF system packages are the reason a `slim` base is used rather than `alpine` — musl
breaks several wheels in this dependency set, and debugging that is not a good use of anyone's time.

---

## 2. Configuration surface

`.env.example` groups variables by concern; `Settings` validates them at boot and the process
**refuses to start** on a bad or missing required value. A service that boots with a broken
configuration and fails on the first user request is strictly worse than one that never boots.

```ini
# ── Core
APP_ENV=development            # development | staging | production
LOG_LEVEL=INFO
SECRET_KEY=                    # required, ≥32 bytes; JWT signing
CORS_ORIGINS=http://localhost:3000

# ── Datastores
DATABASE_URL=postgresql+asyncpg://aegis:aegis@postgres:5432/aegis
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
VECTOR_BACKEND=qdrant          # qdrant | pgvector
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET=aegis-documents
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=

# ── Models
LLM_PROVIDER=openai            # openai | anthropic | gemini
LLM_MODEL=gpt-5.1
LLM_FALLBACK_CHAIN=anthropic:claude-sonnet-4.5,gemini:gemini-2.5-pro
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=2000
INTENT_MODEL=gpt-5-mini
EMBEDDING_PROVIDER=openai      # openai | voyage | cohere | bge
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
RERANKER_PROVIDER=bge_tei      # bge_tei | cross_encoder | cohere | noop
TEI_EMBED_URL=http://tei-embed:8080
TEI_RERANK_URL=http://tei-rerank:8081
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
VOYAGE_API_KEY=
COHERE_API_KEY=

# ── Retrieval (tunable without a deploy via /settings)
RETRIEVAL_TOP_K_DENSE=40
RETRIEVAL_TOP_K_SPARSE=40
RETRIEVAL_RRF_K=60
RERANK_TOP_N=8
CONFIDENCE_MIN_TOP_SCORE=0.35
CONFIDENCE_MIN_SUPPORTING=2
CONTEXT_TOKEN_BUDGET=10000
PROMPT_TOKEN_CAP=16000

# ── Ingestion
MAX_UPLOAD_BYTES=209715200
OCR_ENABLED=true
OCR_DPI=300
CHUNK_SIZE=800
CHUNK_OVERLAP_PCT=15
CONTEXTUAL_HEADERS=true
MALWARE_SCAN_ENABLED=false     # true in staging/production

# ── Security
ACCESS_TOKEN_TTL_MINUTES=15
REFRESH_TOKEN_TTL_DAYS=14
GUEST_ACCESS_ENABLED=true
RATE_LIMIT_ENABLED=true
SECRETS_PROVIDER=env           # env | aws | vault

# ── Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
TRACE_SAMPLE_RATIO=0.1
LANGSMITH_TRACING=false
```

Retrieval thresholds appear both here and in the `settings` table on purpose: the env value is the
boot default, and the database row (when present) wins, so an operator can retune a threshold in the
admin UI during an incident without a redeploy. Which layer supplied each effective value is shown
in the admin settings screen, because "why is the threshold 0.4 when the env says 0.35" is otherwise
an hour of confusion.

---

## 3. Cloud mapping

Nothing here is cloud-specific; the compose topology maps directly.

| Component | AWS | GCP | Azure |
|---|---|---|---|
| api / worker / frontend | ECS Fargate or EKS | Cloud Run or GKE | Container Apps or AKS |
| PostgreSQL | RDS/Aurora PostgreSQL 16 | Cloud SQL | Flexible Server |
| Redis | ElastiCache | Memorystore | Cache for Redis |
| Object storage | S3 + KMS | GCS + CMEK | Blob + CMK |
| Vector DB | Qdrant Cloud or self-managed on EKS with EBS gp3 | Qdrant Cloud / GKE | Qdrant Cloud / AKS |
| Secrets | Secrets Manager | Secret Manager | Key Vault |
| Edge | ALB + WAF + CloudFront | Cloud LB + Armor | App Gateway + WAF |
| Observability | Managed Prometheus + X-Ray, or Grafana Cloud | Cloud Ops | Monitor |
| Queue | Redis (arq); SQS if durability > Redis is needed | Redis / Pub-Sub | Redis / Service Bus |

Two things need explicit attention at the edge and are easy to get wrong: **response buffering must
be off** on the chat route (an ALB or Nginx buffering the SSE stream turns a 1.2 s time-to-first-
token into a 9 s wait), and the **idle timeout must exceed the longest stream** (60 s default on an
ALB will cut long answers; set 180 s and keep the 15 s SSE heartbeat so intermediaries see traffic).

### Environments

| | Local | Staging | Production |
|---|---|---|---|
| Data | seeded samples | anonymized subset | real |
| Models | local TEI + one hosted LLM | production models | production models |
| Malware scan | off | on | on |
| Guest access | on | on | policy decision |
| Eval gates | on demand | every deploy | blocking pre-promotion |
| Replicas | 1 | 2 api / 1 worker | ≥ 3 api / ≥ 3 worker, multi-AZ |
| PgBouncer | no | yes | yes |
| Backups | none | daily | PITR + cross-region |

---

## 4. Operations

**Deploy.** Build → test → eval gates → push image → migrate (one-shot, forward-only) → rolling
restart with readiness gating → smoke test → observe. Migrations are always backward-compatible with
the previous image (expand/contract), so a rollback never needs a down-migration: add nullable
column and backfill in release *n*, start writing in *n*, stop reading the old column in *n+1*, drop
in *n+2*.

**Backups and recovery.** PostgreSQL is the only irreplaceable store: PITR with 30-day retention,
nightly logical dumps, restore rehearsed quarterly. Object storage is versioned with
cross-region replication. **Qdrant is not backed up** — it is a derived index; snapshots exist for
fast recovery, but the authoritative recovery path is `aegis reindex --collection all`, which reads
`chunks` from PostgreSQL and re-embeds using the cache. Recovery targets: RPO 5 min (PostgreSQL
WAL), RTO 1 h for API/workers, 4 h for a full vector rebuild of 10M chunks.

**Runbooks** (`docs/runbooks/`, Phase 5): ingestion backlog, no-answer-rate spike, vector drift
detected, provider outage, cost spike, suspected data leak, key rotation, and index rebuild.

**Zero-downtime index migration** (embedding model upgrade) is a first-class procedure, because it
will happen at least once a year:

```
1. create collection v2 (new model, new namespace)      — no traffic
2. worker backfills from `chunks` (no re-parsing)        — hours, throttled
3. shadow-evaluate v2 against the golden set             — compare metrics
4. flip `collections.vector_namespace` in one statement  — instant
5. keep v1 for 7 days, then drop
```

---

## 5. Tradeoffs

**Docker Compose as the primary artifact, not Kubernetes manifests or Helm.** The brief asks for
Compose, and it is genuinely the right first artifact: one file a developer can read, one command,
identical images in production. The cost is that production HA needs an orchestrator, and Compose
does not describe autoscaling, pod disruption budgets, or rolling strategies. Mitigated by keeping
every service stateless and configured purely by environment, so the translation is mechanical
(Kompose gets 80 % of the way, and the remaining 20 % is HPA and secrets wiring).

**Self-hosted Qdrant + TEI rather than fully managed everything.** More to operate (two extra
stateful/inference services, GPU capacity planning if reranking moves to GPU). In exchange:
document content never leaves the network for embedding or reranking — which is what makes the
platform viable for confidential corpora — and per-query cost drops to compute. Both are behind
ports, so a switch to Qdrant Cloud or Cohere Rerank is a config change, not a rewrite.

**MinIO locally, S3 in production, one adapter.** Divergence between dev and prod storage is a
classic source of "works on my machine" bugs (multipart uploads, presigned URL signature versions,
CORS on PUT). One `ObjectStore` implementation speaking the S3 API to both keeps the code path
identical.

**arq on Redis instead of Celery or SQS.** arq is asyncio-native, so ingestion shares the same async
repositories and HTTP clients as the API — no sync/async duplication of the data layer, which is the
real cost of Celery in an async codebase. The tradeoff is a smaller ecosystem, no built-in
monitoring UI (we surface queue metrics ourselves at `/admin/index-status`), and Redis-grade
durability rather than SQS-grade. Acceptable because jobs are idempotent and re-derivable: the
authoritative record of "this version needs indexing" is a `document_versions` row with
`status='pending'`, and a startup reconciler re-enqueues anything stranded — so a lost Redis message
costs a delay, never a document.

**Nginx in front even locally.** Adds a hop, but TLS, buffering behaviour, body limits, and header
handling are exactly the things that break only in production. Exercising the same edge locally is
cheap insurance.
