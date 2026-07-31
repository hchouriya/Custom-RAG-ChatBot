"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { apiFetch } from "@/shared/api/client";
import { ApiError, isApiError } from "@/shared/api/errors";
import type { Mode, Principal, ProblemDetails } from "@/shared/api/types";

interface AuthContextValue {
  user: Principal | null;
  loading: boolean;
  error: string | null;
  login: (input: {
    email: string;
    password: string;
    mode?: Mode;
    totp_code?: string;
    challenge_token?: string;
  }) => Promise<{ mfaRequired?: boolean; challengeToken?: string; enrolmentRequired?: boolean }>;
  continueAsGuest: () => Promise<void>;
  logout: () => Promise<void>;
  refreshMe: () => Promise<void>;
  setModeLocally: (mode: Mode) => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function normalizePrincipal(raw: Record<string, unknown>): Principal {
  const user = (raw.user as Record<string, unknown> | undefined) ?? raw;
  const role = String(user.role ?? raw.role ?? "guest") as Principal["role"];
  const mode = String(raw.mode ?? user.mode ?? "customer") as Mode;
  const allowed =
    (raw.allowed_modes as Mode[] | undefined) ??
    (role === "customer" || role === "guest"
      ? (["customer"] as Mode[])
      : (["internal", "customer"] as Mode[]));

  return {
    id: (user.id as string | null | undefined) ?? null,
    email: (user.email as string | null | undefined) ?? null,
    full_name: (user.full_name as string | null | undefined) ?? null,
    role,
    mode,
    allowed_modes: allowed,
    permissions: (raw.permissions as string[] | undefined) ?? [],
    is_guest: Boolean(raw.is_guest ?? role === "guest"),
    must_change_password: Boolean(user.must_change_password ?? false),
    has_mfa: Boolean(user.has_mfa ?? false),
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Principal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshMe = useCallback(async () => {
    try {
      const me = await apiFetch<Record<string, unknown>>("/auth/me", {
        skipRefresh: false,
      });
      setUser(normalizePrincipal(me));
      setError(null);
    } catch (err) {
      if (isApiError(err) && err.status === 401) {
        setUser(null);
      } else {
        setError(err instanceof Error ? err.message : "Failed to load session");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshMe();
  }, [refreshMe]);

  const login = useCallback<AuthContextValue["login"]>(async (input) => {
    try {
      const result = await apiFetch<Record<string, unknown>>("/auth/login", {
        method: "POST",
        body: input,
        skipRefresh: true,
      });
      setUser(normalizePrincipal(result));
      return {};
    } catch (err) {
      if (err instanceof ApiError && err.code === "MFA_REQUIRED") {
        const problem = err.problem as ProblemDetails;
        return {
          mfaRequired: true,
          challengeToken: problem.challenge_token,
          enrolmentRequired: problem.enrolment_required,
        };
      }
      throw err;
    }
  }, []);

  const continueAsGuest = useCallback(async () => {
    const result = await apiFetch<Record<string, unknown>>("/auth/guest", {
      method: "POST",
      body: {},
      skipRefresh: true,
    });
    setUser(normalizePrincipal({ ...result, is_guest: true }));
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiFetch("/auth/logout", { method: "POST", skipRefresh: true });
    } finally {
      setUser(null);
    }
  }, []);

  const setModeLocally = useCallback((mode: Mode) => {
    setUser((prev) => (prev ? { ...prev, mode } : prev));
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      error,
      login,
      continueAsGuest,
      logout,
      refreshMe,
      setModeLocally,
    }),
    [user, loading, error, login, continueAsGuest, logout, refreshMe, setModeLocally],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
