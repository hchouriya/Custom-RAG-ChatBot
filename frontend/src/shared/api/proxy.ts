import { NextRequest, NextResponse } from "next/server";

import { cookies } from "next/headers";

import {
  backendBaseUrl,
  clearSessionCookies,
  getAccessToken,
  getRefreshToken,
  MODE_COOKIE,
  setSessionCookies,
} from "@/shared/api/session";

async function tryRefresh(): Promise<string | null> {
  const refresh = await getRefreshToken();
  if (!refresh) return null;

  const response = await fetch(`${backendBaseUrl()}/auth/refresh`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ refresh_token: refresh }),
    cache: "no-store",
  });

  if (!response.ok) {
    await clearSessionCookies();
    return null;
  }

  const tokens = (await response.json()) as {
    access_token: string;
    access_expires_at?: string;
    refresh_token?: string | null;
    refresh_expires_at?: string | null;
    mode?: string;
  };
  await setSessionCookies(tokens);
  return tokens.access_token;
}

/**
 * Proxies browser requests to the FastAPI backend, attaching the access token
 * from httpOnly cookies. Supports SSE by streaming the body through.
 */
export async function proxyToBackend(
  request: NextRequest,
  backendPath: string,
): Promise<Response> {
  let access = await getAccessToken();
  if (!access) {
    access = (await tryRefresh()) ?? undefined;
  }

  const url = new URL(request.url);
  const target = `${backendBaseUrl()}${backendPath}${url.search}`;

  const headers = new Headers();
  const accept = request.headers.get("accept");
  const contentType = request.headers.get("content-type");
  const idempotency = request.headers.get("idempotency-key");
  const requestId = request.headers.get("x-request-id");
  const jar = await cookies();
  const mode =
    request.headers.get("x-assistant-mode") ?? jar.get(MODE_COOKIE)?.value ?? undefined;

  if (accept) headers.set("Accept", accept);
  if (contentType) headers.set("Content-Type", contentType);
  if (idempotency) headers.set("Idempotency-Key", idempotency);
  if (mode) headers.set("X-Assistant-Mode", mode);
  if (requestId) headers.set("X-Request-ID", requestId);
  if (access) headers.set("Authorization", `Bearer ${access}`);

  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD" && method !== "DELETE";
  const body = hasBody ? await request.arrayBuffer() : undefined;

  let upstream = await fetch(target, {
    method,
    headers,
    body,
    cache: "no-store",
  });

  if (upstream.status === 401) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      headers.set("Authorization", `Bearer ${refreshed}`);
      upstream = await fetch(target, {
        method,
        headers,
        body,
        cache: "no-store",
      });
    }
  }

  const responseHeaders = new Headers();
  const passThrough = [
    "content-type",
    "x-request-id",
    "x-trace-id",
    "x-response-time-ms",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
  ];
  for (const key of passThrough) {
    const value = upstream.headers.get(key);
    if (value) responseHeaders.set(key, value);
  }

  // Prevent intermediary buffering of SSE.
  if ((upstream.headers.get("content-type") ?? "").includes("text/event-stream")) {
    responseHeaders.set("Cache-Control", "no-cache, no-transform");
    responseHeaders.set("Connection", "keep-alive");
    responseHeaders.set("X-Accel-Buffering", "no");
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}
