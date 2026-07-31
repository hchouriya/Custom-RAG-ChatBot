import { NextResponse } from "next/server";

import { backendFetch, clearSessionCookies, getAccessToken } from "@/shared/api/session";

export async function GET() {
  const access = await getAccessToken();
  if (!access) {
    return NextResponse.json(
      {
        status: 401,
        title: "Not authenticated",
        detail: "Sign in to continue",
        code: "AUTHENTICATION_FAILED",
      },
      { status: 401 },
    );
  }

  const upstream = await backendFetch("/auth/me", { method: "GET" });
  const payload = await upstream.json().catch(() => ({
    status: upstream.status,
    title: "Failed to load profile",
    detail: "Unable to parse /auth/me",
    code: "AUTH_ERROR",
  }));

  if (!upstream.ok) {
    if (upstream.status === 401) {
      await clearSessionCookies();
    }
    return NextResponse.json(payload, { status: upstream.status });
  }

  return NextResponse.json(payload);
}
