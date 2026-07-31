import { NextRequest, NextResponse } from "next/server";

import { backendBaseUrl, setSessionCookies } from "@/shared/api/session";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const upstream = await fetch(`${backendBaseUrl()}/auth/login`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-Forwarded-For": request.headers.get("x-forwarded-for") ?? "127.0.0.1",
      "User-Agent": request.headers.get("user-agent") ?? "aegis-frontend",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  const payload = await upstream.json().catch(() => ({
    status: upstream.status,
    title: "Login failed",
    detail: "Unable to parse auth response",
    code: "AUTH_ERROR",
  }));

  if (!upstream.ok) {
    // MFA challenge: surface challenge_token to the client (short-lived, not a session).
    if (
      typeof payload === "object" &&
      payload !== null &&
      "code" in payload &&
      (payload as { code?: string }).code === "MFA_REQUIRED"
    ) {
      return NextResponse.json(payload, { status: upstream.status });
    }
    return NextResponse.json(payload, { status: upstream.status });
  }

  await setSessionCookies(payload);

  const { access_token: _a, refresh_token: _r, ...safe } = payload as Record<
    string,
    unknown
  >;
  return NextResponse.json({
    ...safe,
    authenticated: true,
  });
}
