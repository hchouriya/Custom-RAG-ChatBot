# 07 — UI Wireframes

Layout, information hierarchy, and interaction states for every screen. Visual style is defined by
the token set in §9; these wireframes fix *what is on the screen and why*, which is the part that
is expensive to change later.

Design principle throughout: **the citation is a first-class object, not a footnote.** In an
enterprise RAG product the user's real question is "can I trust this?", so provenance is always
one click away and never hidden behind a hover.

---

## 1. Login

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                        ┌──────────────────────┐                          │
│                        │   ◆  Aegis Assistant │                          │
│                        └──────────────────────┘                          │
│                     Grounded answers from your documents                  │
│                                                                          │
│        ┌────────────────────────────────────────────────────┐            │
│        │  Work email                                        │            │
│        │  ┌──────────────────────────────────────────────┐  │            │
│        │  │ you@company.com                              │  │            │
│        │  └──────────────────────────────────────────────┘  │            │
│        │  Password                            [Forgot?]     │            │
│        │  ┌──────────────────────────────────────────────┐  │            │
│        │  │ ••••••••••••                            [👁] │  │            │
│        │  └──────────────────────────────────────────────┘  │            │
│        │  ☐ Keep me signed in                               │            │
│        │  ┌──────────────────────────────────────────────┐  │            │
│        │  │                Sign in                       │  │            │
│        │  └──────────────────────────────────────────────┘  │            │
│        │  ──────────────── or ────────────────              │            │
│        │  ┌──────────────────────────────────────────────┐  │            │
│        │  │   Continue with company SSO                  │  │            │
│        │  └──────────────────────────────────────────────┘  │            │
│        │                                                    │            │
│        │   Customer? Ask our support assistant →            │            │
│        └────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘

States: idle · submitting (button spinner, inputs locked) · invalid credentials
        (inline, generic "Email or password is incorrect" — never "no such user")
        · locked out (countdown) · TOTP step (6-digit input, autofocus, paste-aware)
        · must-change-password (redirect) · SSO redirect
```

The guest path is a link, not a form: customer support must be reachable in one click with no
account. It leads straight to a customer-mode chat with a CAPTCHA on first message rather than a
gate before the page.

---

## 2. Chat — the primary surface

```
┌──────────────┬───────────────────────────────────────────────────┬─────────────────┐
│ ◆ Aegis      │  Parental leave policy            [⋮] [☾] [Share] │ SOURCES     [×] │
│ [+ New chat] │───────────────────────────────────────────────────│─────────────────│
│ ┌──────────┐ │                                                   │ 1 ▸ Global Leave│
│ │🔍 Search │ │  ┌─────────────────────────────────────────────┐  │   Policy 2026   │
│ └──────────┘ │  │ How many days of parental leave can a  [You]│  │   p.12 · §4.2   │
│              │  │ manager in the EU take?                     │  │   ▓▓▓▓▓▓▓░ 0.94 │
│ MODE         │  └─────────────────────────────────────────────┘  │                 │
│ ◉ Internal   │                                                   │ 2 ▸ EU Employment│
│ ○ Customer   │  ┌─────────────────────────────────────────────┐  │   Addendum v2   │
│              │  │ ◆ Managers in EU entities are entitled to   │  │   p.3 · §1.1    │
│ TODAY        │  │   **20 weeks** of paid parental leave, which│  │   ▓▓▓▓▓░░░ 0.71 │
│ ▸ Parental…◀ │  │   may be split into up to three blocks [^1].│  │                 │
│ ▸ Expense p… │  │                                             │  │ 3 ▸ Manager     │
│              │  │   An additional 4 weeks is available where  │  │   Handbook 2026 │
│ YESTERDAY    │  │   local law requires it [^2].               │  │   p.44 · §7     │
│ ▸ VPN setup  │  │                                             │  │   ▓▓▓░░░░░ 0.52 │
│ ▸ Q3 targets │  │   | Region | Weeks | Paid |                 │  │                 │
│              │  │   |--------|-------|------|                 │  │ ─────────────── │
│ LAST 7 DAYS  │  │   | EU     | 20    | 100% |                 │  │ Retrieved 37    │
│ ▸ Onboarding │  │   | UK     | 18    | 90%  |                 │  │ Used 3 · 4.3 s  │
│              │  │                                             │  │ gpt-5.1 · 5.4k  │
│              │  │   [^1] Global Leave Policy 2026 · p.12      │  │ tok             │
│              │  │   [^2] EU Employment Addendum · p.3         │  │                 │
│              │  │                                             │  │                 │
│──────────────│  │  [⧉ Copy] [↻ Retry] [👍] [👎] [⚑ Report]    │  │                 │
│ ⚙ Admin      │  └─────────────────────────────────────────────┘  │                 │
│ 📊 Dashboard │                                                   │                 │
│──────────────│  ┌─────────────────────────────────────────────┐  │                 │
│ AM  A. Mehta │  │ Ask about your documents…              [↑]  │  │                 │
│     Manager  │  └─────────────────────────────────────────────┘  │                 │
│              │   Grounded in company documents · 12 collections  │                 │
└──────────────┴───────────────────────────────────────────────────┴─────────────────┘
```

Reasoning behind the three-pane layout: conversations on the left (navigation), the thread in the
middle (content), sources on the right (verification). The source panel is open by default on
screens ≥ 1280 px because trust is the product, and collapsed below that with a badge showing the
citation count.

The mode switch is in the sidebar, is visible at all times, and repaints the header accent colour.
A user must never be uncertain which assistant they are talking to — that ambiguity is how someone
pastes internal information into a customer-facing session.

### Streaming states

```
Waiting for retrieval   ◆ ▣ Searching 12 collections…              (skeleton, 0–1.5 s)
Citations arrived       ◆ Found 3 sources  ▸▸▸                     (source panel populates first)
Generating              ◆ Managers in EU entities are entitled to ▊ (caret, [■ Stop] replaces send)
Complete                ◆ …full answer + citations + action row
Refused                 ⚠ I could not find enough information in the available documents.
                          Closest matches: Global Leave Policy (0.41), Benefits FAQ (0.38)
                          Try: "standard annual leave entitlement"
                          [Create a support ticket]
Error                   ⊘ The assistant is temporarily unavailable. [Retry]
Degraded                ℹ Answering with reduced search quality (reranking unavailable).
```

Citations appearing before prose is a designed sequence, not an implementation artifact: the user
sees *what will be used* while the answer types itself, which builds trust and gives them something
to read during the slowest part of the request.

### Citation drawer — opens on clicking a `[^n]` or a source card

```
┌──────────────────────────────────────────────────────────────────────┐
│ Global Leave Policy 2026 · v4 · page 12                        [×]   │
│ Benefits ▸ Leave ▸ 4.2 Parental Leave        [Open document] [⇩ PDF] │
│──────────────────────────────────────────────────────────────────────│
│  …preceding paragraph for context, dimmed…                           │
│                                                                      │
│  ███ Employees in EU entities are entitled to 20 weeks of paid  ███  │
│  ███ parental leave, which may be taken in up to three separate ███  │
│  ███ blocks within 24 months of the birth or adoption.          ███  │
│                                                                      │
│  …following paragraph, dimmed…                                       │
│──────────────────────────────────────────────────────────────────────│
│ Relevance 0.94 (reranked) · dense 0.81 · keyword 0.66 · rank 1 of 37 │
│ Updated 14 Jan 2026 by HR Ops · Visibility: Internal · Tags: hr      │
│ [◂ Previous chunk]                                  [Next chunk ▸]   │
└──────────────────────────────────────────────────────────────────────┘
```

The exact supporting sentences are highlighted inside their surrounding paragraphs, because a quote
without its neighbourhood is not verifiable. Score breakdown is shown to internal users only —
customers get the source and the quote without the retrieval internals.

### Empty state (new conversation)

```
                          ◆  What can I help you find?
              Answers come only from documents you have access to.

  ┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
  │ 📋 Policies            │ │ 🔧 IT & Systems        │ │ 💼 HR & Benefits       │
  │ "What is our expense   │ │ "How do I request VPN  │ │ "How many annual leave │
  │  approval limit?"      │ │  access?"              │ │  days do I get?"       │
  └────────────────────────┘ └────────────────────────┘ └────────────────────────┘
        Suggestions are drawn from your permitted collections and popular questions.
```

Suggestions are generated from `mv_top_questions` filtered to the caller's own ceiling, so a guest
is never shown an internal question — a subtle leak that suggestion features commonly introduce.

### Mobile (< 768 px)

```
┌─────────────────────┐   Sidebar → slide-over drawer (☰)
│ ☰  Parental leave ⋮ │   Source panel → bottom sheet, opened by the citation count chip
│─────────────────────│   Composer pinned above the keyboard, auto-growing to 5 lines
│ ┌─────────────────┐ │   Tables → horizontally scrollable with an edge fade affordance
│ │ How many days…  │ │   Code blocks → scroll + copy, never wrapped
│ └─────────────────┘ │   Action row → icon-only, 44 px minimum touch targets
│ ┌─────────────────┐ │
│ │ ◆ Managers in…  │ │
│ │ [3 sources ▾]   │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ Ask…        [↑] │ │
│ └─────────────────┘ │
└─────────────────────┘
```

---

## 3. Admin — documents

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Documents                                            [⟲ Reconcile] [+ Upload]    │
│──────────────────────────────────────────────────────────────────────────────────│
│ 🔍 search title/content   Collection ▾  Visibility ▾  Dept ▾  Tag ▾  Status ▾    │
│ 1,284 documents · 48,912 chunks · 3 failed · 2 indexing        [Clear filters]    │
│──────────────────────────────────────────────────────────────────────────────────│
│ ☐ │ Title                    │ Coll.   │ Vis.      │ Ver │ Chunks │ Status  │    │
│───┼──────────────────────────┼─────────┼───────────┼─────┼────────┼─────────┼────│
│ ☐ │ 📕 Global Leave Policy…  │ HR      │ ● Internal│ v4  │ 142    │ ✓ Indexed│ ⋮ │
│ ☐ │ 📗 Pricing Sheet 2026    │ Sales   │ ◐ Customer│ v2  │  38    │ ✓ Indexed│ ⋮ │
│ ☐ │ 📘 Compensation Bands    │ HR      │ ▲ Confid. │ v1  │  56    │ ✓ Indexed│ ⋮ │
│ ☐ │ 📙 M&A Due Diligence     │ Legal   │ ■ Restr.  │ v3  │ 310    │ ✓ Indexed│ ⋮ │
│ ☐ │ 📕 Scanned Invoices Q1   │ Finance │ ▲ Confid. │ v1  │   0    │ ⟳ OCR 62%│ ⋮ │
│ ☐ │ 📕 Vendor Contract ACME  │ Legal   │ ■ Restr.  │ v1  │   0    │ ✗ Failed │ ⋮ │
│───┴──────────────────────────┴─────────┴───────────┴─────┴────────┴─────────┴────│
│ 2 selected: [Retag] [Change visibility] [Reindex] [Delete]      ‹ 1 2 3 … 26 ›   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Visibility is shown with both a glyph and a word — colour alone fails for colour-blind users and
this is the field where a mistake leaks data. Failed rows expose the error inline on expand rather
than requiring a log dive; index progress is a live percentage, since OCR on a long PDF is minutes
and a spinner with no number reads as "hung".

### Upload

```
┌──────────────────────────────────────────────────────────────────┐
│ Upload documents                                          [×]    │
│──────────────────────────────────────────────────────────────────│
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            ⬆  Drop files or click to browse                │  │
│  │   PDF · DOCX · PPTX · XLSX · CSV · TXT · MD · HTML         │  │
│  │   Max 200 MB per file · scanned PDFs are OCR'd             │  │
│  └────────────────────────────────────────────────────────────┘  │
│  handbook.pdf         12.4 MB  ▓▓▓▓▓▓▓▓░░ 78%   [×]              │
│  pricing.xlsx          0.8 MB  ✓ uploaded       [×]              │
│  contract.zip                  ⊘ type not allowed                │
│──────────────────────────────────────────────────────────────────│
│  Collection *      [HR Knowledge Base            ▾]              │
│  Visibility *      [● Internal                   ▾] ⓘ            │
│  Department        [Human Resources              ▾]              │
│  Tags              [hr ×] [policy ×] [+ add]                     │
│  Effective from    [2026-02-01]  Expires [ — ]                   │
│  Chunking          [Adaptive (recommended)       ▾] ⌄ advanced    │
│    ⌄ target 800 tokens · overlap 15% · contextual headers ☑       │
│      LLM context sentences ☐  (adds ~$0.42 for this document)     │
│──────────────────────────────────────────────────────────────────│
│  Applies to all files.               [Cancel]  [Upload & index]   │
└──────────────────────────────────────────────────────────────────┘
```

Visibility has **no default** on the form (the collection's default is pre-selected but visually
marked as inherited) and the tooltip states plainly who will be able to read the document. Cost of
the optional enrichment is shown before the checkbox is ticked, not on the invoice.

### Document detail

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ ‹ Documents   Global Leave Policy 2026            [⇩] [↻ Reindex] [⇅ Replace] [⋮]│
│──────────────────────────────────────────────────────────────────────────────────│
│ [Overview] [Chunks (142)] [Versions (4)] [Permissions] [Usage] [Activity]        │
│──────────────────────────────────────────────────────────────────────────────────│
│ OVERVIEW                                     │ INDEX HEALTH                      │
│ Collection    HR Knowledge Base              │ Status      ✓ Indexed             │
│ Visibility    ● Internal              [edit] │ Chunks      142 / 142 vectors     │
│ Department    Human Resources                │ Model       text-embedding-3-large│
│ Tags          hr, policy, benefits    [edit] │ Indexed     14 Jan 2026 09:12     │
│ Owner         HR Ops (hrops@company.com)     │ Drift       none                  │
│ Pages · Size  88 · 12.4 MB · OCR: no         │ Injection   0 flagged chunks      │
│ Effective     01 Feb 2026 → —                │                                   │
│──────────────────────────────────────────────┴───────────────────────────────────│
│ VERSIONS                                                                          │
│ ● v4  14 Jan 2026  HR Ops   142 chunks  "Updated EU parental leave"     ACTIVE   │
│ ○ v3  02 Oct 2025  HR Ops   139 chunks  "Annual refresh"      [Activate] [⇩]     │
│ ○ v2  11 Mar 2025  A. Mehta 131 chunks  cited by 14 answers   [Activate] [⇩]     │
│ ○ v1  04 Jan 2025  A. Mehta 128 chunks                        [Activate] [⇩]     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

"Cited by 14 answers" on an old version is the visible face of the retention rule in doc 03 §4 —
it explains to an admin why a superseded version cannot simply be purged.

### Chunk inspector

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Chunks · Global Leave Policy 2026 v4      🔍 filter   Type ▾   [Test retrieval]  │
│──────────────────────────────────────────────────────────────────────────────────│
│ #012 │ text  │ p.12 │ Benefits ▸ Leave ▸ 4.2 Parental Leave │ 712 tok │ ✓ vector │
│      │ [Global Leave Policy 2026 › Leave › 4.2 › p.12]  ← embedded header        │
│      │ Employees in EU entities are entitled to 20 weeks of paid parental…       │
│      │                                                     [Expand] [Re-embed]   │
│ #013 │ table │ p.13 │ Benefits ▸ Leave ▸ 4.3 Entitlements   │ 388 tok │ ✓ vector │
│      │ | Region | Weeks | Paid |  …rendered as a table…                          │
│ #014 │ text  │ p.13 │ …                                     │ 654 tok │ ⚠ stale  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

`[Test retrieval]` runs `POST /admin/retrieval/debug` scoped to this document and shows which
chunk a given question would surface. This is how a content owner debugs "the bot can't find our
policy" without involving engineering — the single most requested capability in every RAG
deployment.

### Permissions tab (ACL editor)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Who can read this document                                        [+ Add grant]  │
│──────────────────────────────────────────────────────────────────────────────────│
│ By visibility   ● Internal → all internal employees, managers, admins            │
│──────────────────────────────────────────────────────────────────────────────────│
│ Explicit grants                                                                  │
│ 👤 j.patel@company.com          user        expires 2026-12-31   [Revoke]         │
│ 🏢 Legal (incl. sub-teams)      department  no expiry            [Revoke]         │
│ 🎭 Manager                      role        no expiry            [Revoke]         │
│──────────────────────────────────────────────────────────────────────────────────│
│ ⓘ Effective readers: 412 users.  [Preview as user…]                              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

"Effective readers: 412" and "Preview as user" convert an abstract policy into a checkable fact,
which is the only way ACL mistakes get caught before an audit.

---

## 4. Admin — users, roles, index

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Users                                        🔍  Role ▾  Dept ▾   [+ Invite user] │
│──────────────────────────────────────────────────────────────────────────────────│
│ Name          │ Email                │ Role      │ Department │ Last active │     │
│ Aarav Mehta   │ a.mehta@company.com  │ Manager   │ Sales      │ 2 min ago   │ ⋮   │
│ Jaya Patel    │ j.patel@company.com  │ Internal  │ Legal      │ 1 hour ago  │ ⋮   │
│ Ops Bot       │ (api key: ak_7f3c…)  │ Internal  │ —          │ 5 min ago   │ ⋮   │
└──────────────────────────────────────────────────────────────────────────────────┘

Permission matrix (roles)                          Index status
┌───────────────────────┬─────┬────┬────┬────┬───┐ ┌──────────────────────────────┐
│ Capability            │Guest│Cust│Int │Mgr │Adm│ │ Queue        4 jobs · 0 dead │
├───────────────────────┼─────┼────┼────┼────┼───┤ │ Oldest job   38 s            │
│ chat:internal         │  ·  │ ·  │ ✓  │ ✓  │ ✓ │ │ Workers      3 online        │
│ document:write        │  ·  │ ·  │ ✓  │ ✓  │ ✓ │ │ Discrepancy  0 chunks        │
│ document:delete       │  ·  │ ·  │ ◐  │ ✓  │ ✓ │ │ Last recon.  02:00 today     │
│ acl:manage            │  ·  │ ·  │ ·  │ ·  │ ✓ │ │──────────────────────────────│
│ analytics:read        │  ·  │ ·  │ ◐  │ ✓  │ ✓ │ │ ✗ Vendor Contract ACME       │
│ audit:read            │  ·  │ ·  │ ·  │ ◐  │ ✓ │ │   parse: encrypted PDF       │
└───────────────────────┴─────┴────┴────┴────┴───┘ │   [Retry] [View log]         │
   ◐ = scoped (own uploads / own department)        └──────────────────────────────┘
```

---

## 5. Analytics dashboard

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Analytics            [Last 30 days ▾] [Mode: All ▾] [Dept: All ▾]   [⇩ Export]   │
│──────────────────────────────────────────────────────────────────────────────────│
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐      │
│ │ Questions  │ │ No-answer  │ │ p95 latency│ │ Tokens     │ │ Active     │      │
│ │  48,210    │ │   6.4%     │ │   4.2 s    │ │  128.4 M   │ │  1,842     │      │
│ │  ▲ 12%     │ │  ▼ 1.8% ✓  │ │  ▲ 0.3 s ⚠ │ │  $412 ▲ 8% │ │  ▲ 6%      │      │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘      │
│──────────────────────────────────────────────────────────────────────────────────│
│ Questions & no-answer rate                    │ Latency breakdown (p95, ms)      │
│  ▁▂▄▆█▇▅▃▄▆█▇▆▅▄▃▂▄▆█▇▆▅ ── questions        │ retrieve  ▓▓▓▓ 240               │
│  ─────────────────────── ·· no-answer %      │ rerank    ▓▓▓▓▓ 320              │
│                                               │ compress  ▓▓ 90                  │
│──────────────────────────────────────────────│ llm ttft  ▓▓▓▓▓▓▓▓▓▓▓▓ 1180      │
│ Top questions                    Asks  Ans.% │ llm total ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 2400  │
│ How do I submit an expense?      1,204  98%  │──────────────────────────────────│
│ What is the parental leave…        842  94%  │ Most cited documents             │
│ How do I reset my VPN?             713  89%  │ Global Leave Policy      2,104   │
│──────────────────────────────────────────────│ Expense Policy 2026      1,880   │
│ ⚠ Content gaps (unanswered, ranked)          │ IT Onboarding Guide      1,241   │
│ "2026 holiday calendar"       142 asks       │──────────────────────────────────│
│ "Sabbatical eligibility"       98 asks       │ Never cited (candidates for       │
│ "EV charging reimbursement"    64 asks       │ removal)                   38 →  │
│ [Create document request]                    │                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The content-gap panel is the dashboard's most valuable element and the reason the refusal path
records nearest documents and scores: it converts every refusal into a prioritized backlog of
documents to write. A no-answer rate on its own is a complaint; a ranked list of missing topics is
a work item.

---

## 6. Query trace viewer (admin)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Trace 01J8Z9…  "parental leave EU managers"   a.mehta · Manager · internal        │
│ ✓ ok · 4.31 s · gpt-5.1 · 5,432 tok · $0.027 · confidence 0.91   [Open in Tempo] │
│──────────────────────────────────────────────────────────────────────────────────│
│ guardrail  8 ms   ✓ clean                                                        │
│ intent    118 ms   factual_lookup (0.96)                                         │
│ rewrite   287 ms   ["parental leave entitlement EU managers", "paternity…", orig] │
│ filter     14 ms   mode=internal · visibility_level ≤ 3 · dept ⊂ company.sales    │
│ retrieve  246 ms   dense 40 + sparse 40 × 3 queries → 174 → dedupe 37            │
│ rerank    331 ms   37 → 8   top 0.94 · mean top3 0.79                            │
│ acl        27 ms   8 → 8 verified, 0 dropped                                      │
│ gate        3 ms   ✓ pass (top 0.94 ≥ 0.35, coverage 0.83)                        │
│ compress   88 ms   8 chunks · 6,204 → 4,118 tokens                                │
│ generate 2,401 ms  ttft 1,180 ms · 214 completion tokens                          │
│ validate    9 ms   2 markers, both valid                                          │
│──────────────────────────────────────────────────────────────────────────────────│
│ Candidates                                    dense  sparse  fused  rerank  used │
│ #012 Global Leave Policy p.12                 0.81   0.66    0.031  0.94    ✓    │
│ #003 EU Employment Addendum p.3               0.77   0.71    0.029  0.71    ✓    │
│ #044 Manager Handbook p.44                    0.74   0.12    0.021  0.52    ·    │
│──────────────────────────────────────────────────────────────────────────────────│
│ [View assembled prompt] [Re-run with current config] [Add to eval dataset]        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

"Add to eval dataset" closes the loop: a reported bad answer becomes a permanent regression test
in one click, which is how retrieval quality ratchets up instead of oscillating.

---

## 7. Escalation form

```
┌──────────────────────────────────────────────────────────────────┐
│ Talk to a person                                          [×]    │
│ We could not find this in our documents. A specialist will reply. │
│──────────────────────────────────────────────────────────────────│
│ Name *   [                    ]   Email * [                    ] │
│ Phone    [                    ]   Priority [Normal          ▾]   │
│ Your question *                                                  │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ How many days of parental leave can a manager in the EU take?│ │  ← prefilled
│ └──────────────────────────────────────────────────────────────┘ │
│ ☑ Include this conversation (6 messages) to give context         │
│                                        [Cancel]  [Submit]        │
│──────────────────────────────────────────────────────────────────│
│ ✓ Ticket #4821 created. We usually reply within one business day.│
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Component inventory

| Group | Components |
|---|---|
| Chat | `MessageList`, `MessageBubble`, `MarkdownRenderer`, `CodeBlock`, `DataTable`, `CitationChip`, `CitationDrawer`, `SourcePanel`, `SourceCard`, `TypingIndicator`, `Composer`, `SuggestedQuestions`, `ConversationSidebar`, `ModeSwitch`, `FeedbackButtons`, `RefusalCard`, `EscalationDialog`, `StreamErrorCard` |
| Admin | `DocumentTable`, `UploadDropzone`, `MetadataForm`, `VisibilitySelect`, `TagInput`, `AclEditor`, `VersionTimeline`, `ChunkInspector`, `RetrievalTester`, `ReindexDialog`, `PermissionMatrix`, `UserTable`, `InviteDialog`, `IndexHealthPanel`, `JobFailureList`, `TraceViewer`, `TicketQueue`, `EvalRunTable` |
| Dashboard | `StatCard`, `TimeSeriesChart`, `StageLatencyBars`, `TopQuestionsTable`, `ContentGapPanel`, `DocumentUsageTable`, `TokenSpendChart`, `DateRangePicker` |
| Shared | `Button`, `Input`, `Textarea`, `Select`, `Combobox`, `Checkbox`, `Switch`, `Dialog`, `Drawer`, `Sheet`, `Tabs`, `Table`, `Badge`, `Tooltip`, `Toast`, `Skeleton`, `EmptyState`, `ErrorBoundary`, `Pagination`, `ThemeToggle`, `Avatar`, `ProgressBar` |

## 9. Design tokens and interaction rules

**Colour.** Neutral slate surfaces; one indigo accent for internal mode and teal for customer mode
(the mode is legible at a glance from the header alone). Semantic: emerald success, amber warning,
rose danger. Visibility glyphs: `○ public`, `◐ customer`, `● internal`, `▲ confidential`,
`■ restricted` — always glyph **and** label.

**Type.** Inter for UI, JetBrains Mono for code and identifiers. Body 15 px / 1.6 in the chat pane
with a 68-character measure; long answers are read, not scanned, and an unconstrained line length
is the fastest way to make them unreadable.

**Dark mode.** Class strategy on `<html>`, system-preference default, persisted per user, no flash
on load (theme resolved in a blocking inline script before paint).

**Motion.** 150 ms ease-out for state changes, 250 ms for drawers, none for streaming text (a
fade-in per token is nauseating at 40 tokens/s). All animation respects
`prefers-reduced-motion`.

**Accessibility (WCAG 2.2 AA).** Full keyboard operation; `⌘K` command palette, `⌘↵` send,
`Esc` closes drawers, `j/k` moves between messages; the streaming region is
`aria-live="polite"` and announces "answer complete" rather than every token; citation chips are
real buttons with `aria-describedby` pointing at the source card; focus is trapped in dialogs and
restored on close; 4.5:1 contrast minimum in both themes; every icon-only control has a label.

**Loading and error discipline.** Skeletons for known shapes (never spinners for lists), optimistic
UI for renames and pins with rollback on failure, inline retry on every failed request, and
progress *percentages* for anything longer than 5 s.

## 10. Tradeoffs

**Three panes instead of two.** Costs horizontal space and a breakpoint strategy. Bought:
provenance is visible without interaction, which is the difference between a demo and a tool people
trust with policy questions. Below 1280 px it collapses to a chip, and the citation drawer carries
the same information.

**Mode as an explicit switch rather than automatic inference.** Inferring from role would be fewer
clicks, but an internal user testing the customer bot is a routine need, and an ambiguous mode is a
data-handling hazard. Explicit, always visible, colour-coded.

**Server components for shells, client components only for streaming and interaction.** Conversation
history, document lists, and dashboards render on the server with the session cookie, so no token
touches JS and initial paint carries data. The chat thread is a client island because it holds an
SSE reader.

**Scores exposed to internal users, hidden from customers.** Retrieval scores make internal users
better at debugging their own corpus; they would only erode confidence for a customer who cannot
act on them.

**Custom primitives instead of a component library.** A dependency like MUI would be faster
initially but fights Tailwind, ships a large runtime, and makes the streaming-specific components
(citation chips inside a Markdown renderer, live regions) awkward. The primitive set here is small,
owned, and shaped to this product.
