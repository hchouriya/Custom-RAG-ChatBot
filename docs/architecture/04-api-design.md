# 04 — API Design

Base path `/api/v1`. JSON in, JSON out, `application/problem+json` on error. All timestamps are
RFC 3339 UTC. All list endpoints are cursor-paginated. OpenAPI 3.1 is generated from the FastAPI
app and is the source for the frontend's TypeScript types.

## 1. Conventions

### Authentication

`Authorization: Bearer <access_jwt>` on every endpoint except `/health/*`, `/auth/login`,
`/auth/refresh`, `/auth/guest`, and `POST /support/tickets`.

The browser never holds the token. The Next.js BFF stores the refresh token in an httpOnly,
`Secure`, `SameSite=Lax` cookie and the access token in an encrypted session cookie, then attaches
the `Authorization` header server-side. Consequence: an XSS in the frontend cannot exfiltrate a
long-lived credential, which `localStorage` token storage cannot promise.

Access token claims — deliberately minimal, because a JWT is a cache, not a database:

```json
{
  "sub": "0190f3a2-...", "role": "manager", "dept": "0190f3b1-...",
  "mode": "internal", "sid": "0190f3c0-...", "ver": 3,
  "iat": 1753600000, "exp": 1753600900, "iss": "aegis", "aud": "aegis-api"
}
```

`ver` is the user's permission epoch, bumped whenever role, department, or overrides change. A
token whose `ver` is stale is rejected even if unexpired, which is how a role revocation takes
effect in under a second instead of after a 15-minute TTL. This costs one Redis lookup per
request (`user:{sub}:ver`, cached 60 s) and is the difference between RBAC that is merely
documented and RBAC that is enforced.

Access TTL 15 min, refresh TTL 14 days with rotation and reuse detection (schema §3).

### Mode

`mode` is a claim on the token, chosen at login (or forced to `customer` for `customer`/`guest`
roles). Requests may also send `X-Assistant-Mode`, but it is only honoured when it *narrows*
access — an internal user may explore customer mode, never the reverse. This is checked in one
place, `SecurityContext.resolve()`, and covered by the RBAC test matrix.

### Errors — RFC 9457

```json
{
  "type": "https://aegis.local/errors/document-not-indexed",
  "title": "Document is not searchable yet",
  "status": 409,
  "detail": "Version 3 is at stage 'embedding'.",
  "instance": "/api/v1/documents/0190.../reindex",
  "request_id": "01J8Z9X2QK...",
  "code": "DOCUMENT_NOT_INDEXED",
  "errors": [{ "field": "version_id", "message": "must be an indexed version" }]
}
```

`code` is the stable machine identifier the frontend switches on; `type`/`title`/`detail` are for
humans and logs. Validation failures return `422` with a populated `errors` array. Authorization
failures return **`404` rather than `403` when revealing existence itself is a leak** — asking for
a confidential document you cannot see must not confirm that it exists.

### Pagination, filtering, sorting

```
GET /api/v1/documents?limit=50&cursor=eyJ...&sort=-created_at
    &q=leave%20policy&collection_id=…&visibility=internal&tag=hr&status=indexed
    &department_id=…&created_after=2026-01-01
```

Keyset (cursor) pagination, not `OFFSET`: `OFFSET 10000` degrades linearly and pages shift under
concurrent inserts. The opaque cursor is base64 of `{last_sort_value, last_id}`.

```json
{ "items": [ ... ], "next_cursor": "eyJ...", "has_more": true, "total_estimate": 1284 }
```

`total_estimate` comes from the planner's row estimate; an exact `COUNT(*)` over a 500k-row
filtered set on every keystroke is not worth its latency, and the UI shows "about 1,284".

### Rate limits

Redis sliding window, keyed by principal and by IP, applied before any expensive work. `429`
carries `Retry-After` plus `X-RateLimit-{Limit,Remaining,Reset}`.

| Bucket | Guest | Customer | Internal | Manager | Admin |
|---|---|---|---|---|---|
| Chat messages | 5/min | 20/min | 60/min | 60/min | 120/min |
| Document upload | — | — | 10/hour | 60/hour | 300/hour |
| Reindex | — | — | — | 5/hour | 20/hour |
| Login attempts | 10/15 min per IP + per email, then exponential lockout | | | | |
| Ticket creation | 3/hour per IP | | | | |

### Other headers

Request: `X-Request-ID` (echoed), `Idempotency-Key` (required on `POST` that creates billable or
non-idempotent work: messages, uploads, reindex, tickets), `If-Match` (ETag concurrency on
document/user updates). Response: `X-Request-ID`, `X-Response-Time-Ms`, and on chat responses
`X-Trace-Id` so a user-reported bad answer maps to a trace.

---

## 2. Endpoint catalogue

### Auth — `/auth`

| Method | Path | Purpose | Roles |
|---|---|---|---|
| POST | `/auth/login` | email+password (+ TOTP) → access + refresh cookie | public |
| POST | `/auth/refresh` | rotate refresh, mint access | cookie |
| POST | `/auth/logout` | revoke current refresh family | any |
| POST | `/auth/logout-all` | revoke every session for the user | any |
| POST | `/auth/guest` | anonymous customer-mode session (CAPTCHA-gated) | public |
| GET | `/auth/me` | principal, role, permissions, allowed modes, limits | any |
| POST | `/auth/password/change` | old + new, revokes other sessions | any |
| POST | `/auth/password/reset-request` · `/auth/password/reset` | emailed single-use token | public |
| GET | `/auth/sso/login` · `/auth/sso/callback` | OIDC hook (Phase 5) | public |

### Chat — `/chat`

| Method | Path | Purpose |
|---|---|---|
| GET | `/chat/conversations` | list (cursor, `q`, `archived`, `pinned`) |
| POST | `/chat/conversations` | create; optional `collection_ids`, first message |
| GET | `/chat/conversations/{id}` | metadata + messages + citations, hydrated |
| PATCH | `/chat/conversations/{id}` | rename, pin, archive |
| DELETE | `/chat/conversations/{id}` | soft delete |
| POST | `/chat/conversations/{id}/messages` | **ask a question → SSE stream** |
| POST | `/chat/conversations/{id}/messages/{mid}/regenerate` | re-answer, new branch via `parent_id` |
| DELETE | `/chat/conversations/{id}/stream` | cancel the in-flight generation |
| POST | `/chat/messages/{id}/feedback` | thumb + reason |
| GET | `/chat/messages/{id}/citations` | full citation objects with quotes |
| GET | `/chat/suggestions` | starter questions for the mode/role |
| POST | `/chat/escalate` | conversation → support ticket |

### Documents — `/documents`

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents/uploads` | validate intent → presigned PUT + `upload_id` |
| POST | `/documents` | register document + first version from `upload_id` → `202` |
| GET | `/documents` | list with the full filter set |
| GET | `/documents/{id}` | detail: metadata, ACL, active version, usage stats |
| PATCH | `/documents/{id}` | title, description, visibility, department, tags, dates |
| DELETE | `/documents/{id}` | soft delete + immediate vector purge |
| GET | `/documents/{id}/versions` | version timeline |
| POST | `/documents/{id}/versions` | **replace**: new version, atomic flip on success → `202` |
| GET | `/documents/{id}/versions/{vid}/download` | short-lived presigned GET (audited) |
| POST | `/documents/{id}/versions/{vid}/activate` | roll back to an earlier indexed version |
| POST | `/documents/{id}/reindex` | re-chunk and/or re-embed → `202` |
| GET | `/documents/{id}/chunks` | paginated chunk inspector |
| GET | `/chunks/{id}` | chunk with neighbours, for the citation drawer |
| PUT | `/documents/{id}/acl` | replace ACL grants (audited, atomic) |
| POST | `/documents/bulk` | bulk retag / revisibility / reindex / delete |

Upload is a three-step handshake so bytes never transit the API process:

```
POST /documents/uploads          { filename, size_bytes, declared_mime, collection_id }
  ← 201 { upload_id, url, fields, expires_at, max_bytes }
PUT  <presigned url>             (browser → S3, direct)
POST /documents                  { upload_id, title, visibility, department_id, tags[], … }
  ← 202 { document_id, version_id, status: "pending", poll: "/documents/{id}/versions" }
```

The server then `HEAD`s the object, verifies size and checksum, and **sniffs magic bytes** —
`declared_mime` is a hint used only for the early rejection, never for parser selection.

### Collections, users, roles, admin

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/collections` | list, create (embedding provider/model/dim are immutable after data) |
| GET/PATCH/DELETE | `/collections/{id}` | detail, update, delete (blocked while documents exist) |
| POST | `/collections/{id}/reindex` | full rebuild, optionally into a new namespace |
| GET | `/collections/{id}/stats` | chunk/vector counts, drift, last index time |
| GET/POST | `/users` | list, create (invite email) |
| GET/PATCH/DELETE | `/users/{id}` | detail, role/department/status, deactivate |
| POST | `/users/{id}/permissions` | allow/deny overrides |
| GET | `/roles` | roles with resolved permission sets |
| PUT | `/roles/{role}/permissions` | edit the matrix |
| GET | `/permissions` | catalogue grouped by category |
| GET/POST/DELETE | `/api-keys` | service credentials (hash shown once) |
| GET/POST | `/tags`, `/departments` | taxonomy management |
| GET/POST/PATCH | `/prompts` | template versions, activate |
| GET/PUT | `/settings` | runtime settings (thresholds, top-k, models) |
| GET | `/admin/index-status` | queue depth, oldest job, failures, discrepancies |
| POST | `/admin/retrieval/debug` | run the pipeline read-only and return every stage |
| POST | `/admin/reconcile` | trigger PG↔vector reconciliation |

`POST /admin/retrieval/debug` is the highest-value operational endpoint in the system: given a
question and an impersonated role, it returns the rewritten queries, the exact filter, all
candidates with all four scores, the compressed context, and the assembled prompt — **without**
calling the LLM. Nearly every "why did it answer that?" investigation ends here. Impersonation is
capped at the caller's own ceiling and is audited.

### Analytics, logs, tickets, evaluation

| Method | Path | Purpose |
|---|---|---|
| GET | `/analytics/overview` | KPI cards for a date range |
| GET | `/analytics/timeseries` | queries, latency, tokens, cost by bucket |
| GET | `/analytics/top-questions` | clustered, with answer rate |
| GET | `/analytics/no-answer` | refusals and near-misses, the content-gap worklist |
| GET | `/analytics/documents` | most cited, never cited, stale |
| GET | `/analytics/users` | activity, per-role adoption |
| GET | `/analytics/retrieval-quality` | score distributions, feedback correlation |
| GET | `/analytics/export` | CSV/JSON export (audited) |
| GET | `/audit-logs` | filter by actor, action, resource, range |
| GET | `/query-traces` · `/query-traces/{id}` | trace search and full detail |
| GET/PATCH | `/support/tickets`, `/support/tickets/{id}` | queue management |
| POST | `/support/tickets` | public creation (rate-limited, CAPTCHA) |
| GET/POST | `/evals/datasets`, `/evals/datasets/{id}/cases` | golden set management |
| POST | `/evals/runs` | trigger a run → `202` |
| GET | `/evals/runs`, `/evals/runs/{id}` | results, per-metric, per-case diffs |

### Health

`GET /health/live` (process up, no dependencies — Kubernetes liveness),
`GET /health/ready` (PostgreSQL, Redis, Qdrant, TEI probed with 500 ms budgets — readiness),
`GET /health/deep` (admin: provider reachability, migration head, queue lag),
`GET /metrics` (Prometheus, internal network only), `GET /version` (git sha, migration revision).

Liveness and readiness are deliberately different. A liveness check that touches PostgreSQL turns
a brief database blip into a rolling restart of every API pod — an outage manufactured by its own
health check.

---

## 3. The chat request

```http
POST /api/v1/chat/conversations/0190.../messages
Authorization: Bearer …
Idempotency-Key: 4f2c…
Accept: text/event-stream

{
  "content": "How many days of parental leave can a manager in the EU take?",
  "stream": true,
  "collection_ids": [],
  "filters": { "tags": ["hr"], "document_ids": null },
  "options": { "model": null, "temperature": 0.1, "max_citations": 8 }
}
```

`filters` may only **narrow** what the principal can already see; it is intersected with the
server-derived ACL filter and can never widen it. `options.model` is validated against a
role-scoped allowlist so a customer session cannot select an expensive model. `temperature` is
clamped to `[0, 0.3]` — this is a grounded-answer product, and sampling entropy is hallucination
surface.

### SSE protocol

Named events, one JSON object per event, `\n\n`-terminated. A comment heartbeat (`: ping`) every
15 s keeps proxies from idling the connection out.

```
event: meta
data: {"message_id":"019...","trace_id":"7f3...","model":"gpt-5.1","mode":"internal",
       "intent":"factual_lookup","rewritten":["parental leave entitlement EU managers",
       "paternity maternity leave policy Europe"],"retrieved":37,"reranked":8}

event: citations
data: {"citations":[
  {"marker":1,"document_id":"019...","document_title":"Global Leave Policy 2026",
   "version_no":4,"page":12,"section":"4.2 Parental Leave","heading_path":["Leave","Parental"],
   "quote":"Employees in EU entities are entitled to 20 weeks of paid parental leave…",
   "score_rerank":0.94,"url":"/admin/documents/019...?page=12&chunk=019..."}]}

event: token
data: {"delta":"Managers based in EU entities are entitled to "}

event: token
data: {"delta":"**20 weeks** of paid parental leave"}

event: usage
data: {"prompt_tokens":5218,"completion_tokens":214,"cost_usd":0.0271,
       "ttft_ms":1180,"total_ms":4310,"confidence":0.91,"grounded":true}

event: done
data: {"status":"ok","message_id":"019...","citations_used":[1,3]}
```

Additional event types: `error` (`{code, message, retryable}`), `refusal`
(`{reason, suggestion, escalation_available}`), `clarify` (`{question, options}`), and
`heartbeat`. A stream always terminates with exactly one of `done` or `error`, so the client's
state machine has no ambiguous end state.

Why events rather than raw token text: the client needs structure (citations before prose, usage
after), and a self-describing envelope means adding a stage later — say `event: reasoning` —
does not break existing clients. Ordering is guaranteed: `meta` → `citations` → `token`* →
`usage` → `done`.

Cancellation: `DELETE /chat/conversations/{id}/stream` sets a Redis flag the generator polls
between chunks; the partial message is persisted with `status='error'`, `finish_reason='cancelled'`
so history is never a lie about what the user saw. Client disconnects are detected via the ASGI
receive channel and treated identically — an abandoned tab must not keep paying an LLM.

### Fallback and refusal shape

When the confidence gate fails, the API returns `200` with a `refusal` event, not an error status —
"I could not find this" is a successful, correct outcome of a working system:

```json
{ "reason": "insufficient_context",
  "message": "I could not find enough information in the available documents.",
  "detail": "The closest matches were about general leave policy but did not cover EU parental leave for managers.",
  "nearest_documents": [{"title":"Global Leave Policy 2026","score":0.41}],
  "suggestions": ["Ask about standard annual leave entitlement"],
  "escalation_available": true,
  "escalation_hint": {"endpoint":"/api/v1/chat/escalate","conversation_id":"019..."} }
```

Returning the nearest documents and their scores is what turns a refusal into a usable signal: the
user learns what *is* covered, and the no-answer dashboard gets a ranked content-gap backlog.

---

## 4. Tradeoffs

**REST + SSE over GraphQL or gRPC.** The surface is resource-shaped and the one hard requirement is
token streaming through corporate proxies. GraphQL would give the admin panel flexible queries but
brings N+1 risk on ACL-filtered lists, subscription complexity for streaming, and a harder security
story (per-field authorization on a 30-table schema). gRPC-web needs a proxy and loses
browser-native streaming ergonomics. FastAPI additionally gives OpenAPI for free, which is what
generates the frontend types.

**Explicit `/api/v1` prefix.** Breaking changes ship as `/v2` with `v1` maintained for a
deprecation window; `Sunset` headers announce it. Header-based versioning is invisible in logs and
in a browser address bar, which makes incident triage harder.

**`202 Accepted` for all ingestion.** The client polls or subscribes to job status. Synchronous
upload processing would tie request lifetime to document size — a 300-page scanned PDF is minutes
of OCR — and would put a hard cap on document size equal to the proxy timeout.

**Idempotency keys on message creation.** A retried `POST` after a network blip must not produce
two answers and two LLM charges. The key maps to the created `message_id` in Redis for 24 h, and a
replay returns the original result.

**Performance.** Every list endpoint is keyset-paginated and backed by a matching composite index;
`/auth/me` and permission resolution are Redis-cached with epoch invalidation; analytics reads hit
materialized views; `ETag`/`If-None-Match` on document detail cuts admin-panel refetching; response
`gzip` above 1 KB but explicitly **disabled** for `text/event-stream` (buffering would destroy
time-to-first-token, and Nginx needs `proxy_buffering off` on the same route).

**Security.** Every mutating endpoint writes an audit row including denied attempts; request bodies
are size-capped per route (1 MB JSON, 200 MB presigned upload); all IDs are UUIDs so nothing is
enumerable; `403` becomes `404` where existence is sensitive; CORS is an explicit origin allowlist
with credentials; `Content-Security-Policy`, `HSTS`, `X-Content-Type-Options`, and
`Referrer-Policy` are set at the edge; and the OpenAPI docs endpoints are admin-gated in production.
