import { NextResponse } from "next/server";

import {
  backendBaseUrl,
  clearSessionCookies,
  getAccessToken,
  getRefreshToken,
} from "@/shared/api/session";

export async function POST() {
  const refresh = await getRefreshToken();
  const access = await getAccessToken();

  try {
    if (refresh || access) {
      await fetch(`${backendBaseUrl()}/auth/logout`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          ...(access ? { Authorization: `Bearer ${access}` } : {}),
        },
        body: JSON.stringify({ refresh_token: refresh ?? null }),
        cache: "no-store",
      });
    }
  } finally {
    await clearSessionCookies();
  }

  return NextResponse.json({ ok: true });
}
