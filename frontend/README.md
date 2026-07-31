# Aegis Frontend

Next.js 15 App Router UI for the Aegis RAG platform.

## Auth approach

**Next.js BFF (chosen).** The browser never holds JWTs.

- `POST /api/auth/login`, `/api/auth/guest`, `/api/auth/refresh`, `/api/auth/logout` manage sessions.
- Access + refresh tokens are stored in **httpOnly** cookies (`aegis_access`, `aegis_refresh`).
- UI calls `/api/bff/*`, which proxies to `http://localhost:8000/api/v1` and attaches `Authorization: Bearer …`.
- SSE chat streams are proxied through the same BFF without buffering.

## Run

```bash
cp .env.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Ensure the FastAPI backend is running on port 8000.

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server |
| `npm run build` | Production build |
| `npm run typecheck` | `tsc --noEmit` |

## Screens

- `/login` — email/password, MFA step, guest continue
- `/` → `/chat` — conversation list, streaming answers, citation sources panel
- `/documents` — list + upload (ticket → PUT storage → register)

## Stack

Next.js 15 · React 19 · TypeScript strict · Tailwind CSS v4 · Fraunces + DM Sans
