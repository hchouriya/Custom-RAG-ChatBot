import { cookies } from "next/headers";

export const ACCESS_COOKIE = "aegis_access";
export const REFRESH_COOKIE = "aegis_refresh";
export const MODE_COOKIE = "aegis_mode";

export function backendBaseUrl(): string {
  return (
    process.env.API_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    "http://localhost:8000/api/v1"
  );
}

export function cookieSecure(): boolean {
  return process.env.NODE_ENV === "production";
}

export async function getAccessToken(): Promise<string | undefined> {
  const jar = await cookies();
  return jar.get(ACCESS_COOKIE)?.value;
}

export async function getRefreshToken(): Promise<string | undefined> {
  const jar = await cookies();
  return jar.get(REFRESH_COOKIE)?.value;
}

export function accessCookieOptions(maxAgeSeconds = 60 * 15) {
  return {
    httpOnly: true,
    secure: cookieSecure(),
    sameSite: "lax" as const,
    path: "/",
    maxAge: maxAgeSeconds,
  };
}

export function refreshCookieOptions(maxAgeSeconds = 60 * 60 * 24 * 14) {
  return {
    httpOnly: true,
    secure: cookieSecure(),
    sameSite: "lax" as const,
    path: "/",
    maxAge: maxAgeSeconds,
  };
}

export function modeCookieOptions() {
  return {
    httpOnly: false,
    secure: cookieSecure(),
    sameSite: "lax" as const,
    path: "/",
    maxAge: 60 * 60 * 24 * 14,
  };
}

export interface TokenPayload {
  access_token: string;
  access_expires_at?: string;
  refresh_token?: string | null;
  refresh_expires_at?: string | null;
  mode?: string;
}

export async function setSessionCookies(tokens: TokenPayload): Promise<void> {
  const jar = await cookies();
  const accessMaxAge = tokens.access_expires_at
    ? Math.max(
        60,
        Math.floor((new Date(tokens.access_expires_at).getTime() - Date.now()) / 1000),
      )
    : 60 * 15;

  jar.set(ACCESS_COOKIE, tokens.access_token, accessCookieOptions(accessMaxAge));

  if (tokens.refresh_token) {
    const refreshMaxAge = tokens.refresh_expires_at
      ? Math.max(
          60,
          Math.floor((new Date(tokens.refresh_expires_at).getTime() - Date.now()) / 1000),
        )
      : 60 * 60 * 24 * 14;
    jar.set(REFRESH_COOKIE, tokens.refresh_token, refreshCookieOptions(refreshMaxAge));
  }

  if (tokens.mode) {
    jar.set(MODE_COOKIE, tokens.mode, modeCookieOptions());
  }
}

export async function clearSessionCookies(): Promise<void> {
  const jar = await cookies();
  jar.delete(ACCESS_COOKIE);
  jar.delete(REFRESH_COOKIE);
  jar.delete(MODE_COOKIE);
}

export async function backendFetch(
  path: string,
  init: RequestInit = {},
  options: { auth?: boolean; token?: string } = {},
): Promise<Response> {
  const url = `${backendBaseUrl()}${path.startsWith("/") ? path : `/${path}`}`;
  const headers = new Headers(init.headers);

  if (options.auth !== false) {
    const token = options.token ?? (await getAccessToken());
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }

  return fetch(url, {
    ...init,
    headers,
    cache: "no-store",
  });
}
