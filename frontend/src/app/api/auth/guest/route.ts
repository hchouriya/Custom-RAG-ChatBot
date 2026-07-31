import { NextRequest, NextResponse } from "next/server";

import { backendBaseUrl, setSessionCookies } from "@/shared/api/session";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));

  const upstream = await fetch(`${backendBaseUrl()}/auth/guest`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-Forwarded-For": request.headers.get("x-forwarded-for") ?? "127.0.0.1",
      "User-Agent": request.headers.get("user-agent") ?? "aegis-frontend",
    },
    body: JSON.stringify(body ?? {}),
    cache: "no-store",
  });

  const payload = await upstream.json().catch(() => ({
    status: upstream.status,
    title: "Guest session failed",
    detail: "Unable to start a guest session",
    code: "AUTH_ERROR",
  }));

  if (!upstream.ok) {
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
    is_guest: true,
  });
}
