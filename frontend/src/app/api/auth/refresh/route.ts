import { NextResponse } from "next/server";

import {
  backendBaseUrl,
  clearSessionCookies,
  getRefreshToken,
  setSessionCookies,
} from "@/shared/api/session";

export async function POST() {
  const refresh = await getRefreshToken();
  if (!refresh) {
    await clearSessionCookies();
    return NextResponse.json(
      {
        status: 401,
        title: "Not authenticated",
        detail: "No refresh session",
        code: "AUTHENTICATION_FAILED",
      },
      { status: 401 },
    );
  }

  const upstream = await fetch(`${backendBaseUrl()}/auth/refresh`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ refresh_token: refresh }),
    cache: "no-store",
  });

  const payload = await upstream.json().catch(() => ({
    status: upstream.status,
    title: "Refresh failed",
    detail: "Unable to refresh session",
    code: "AUTHENTICATION_FAILED",
  }));

  if (!upstream.ok) {
    await clearSessionCookies();
    return NextResponse.json(payload, { status: upstream.status });
  }

  await setSessionCookies(payload);
  return NextResponse.json({ ok: true, mode: (payload as { mode?: string }).mode });
}
