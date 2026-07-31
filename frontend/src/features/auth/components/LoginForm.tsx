"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { useAuth } from "@/features/auth/hooks/useAuth";
import { isApiError } from "@/shared/api/errors";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";

export function LoginForm() {
  const router = useRouter();
  const { login, continueAsGuest } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [totp, setTotp] = useState("");
  const [challengeToken, setChallengeToken] = useState<string | undefined>();
  const [mfaStep, setMfaStep] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [guestLoading, setGuestLoading] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await login({
        email,
        password,
        totp_code: mfaStep ? totp : undefined,
        challenge_token: challengeToken,
      });
      if (result.mfaRequired) {
        setMfaStep(true);
        setChallengeToken(result.challengeToken);
        return;
      }
      router.replace("/chat");
    } catch (err) {
      if (isApiError(err) && err.code === "ACCOUNT_LOCKED") {
        setError(err.message);
      } else if (isApiError(err)) {
        setError(err.message || "Email or password is incorrect");
      } else {
        setError("Unable to sign in. Is the API running?");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function onGuest() {
    setError(null);
    setGuestLoading(true);
    try {
      await continueAsGuest();
      router.replace("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Guest session failed");
    } finally {
      setGuestLoading(false);
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      className="w-full max-w-md animate-fade-up rounded-2xl border border-ink-700/80 bg-ink-900/55 p-7 shadow-[0_30px_80px_-40px_rgba(0,0,0,0.9)] backdrop-blur-md"
    >
      {!mfaStep ? (
        <>
          <Input
            label="Work email"
            name="email"
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@company.com"
            disabled={submitting}
          />
          <div className="mt-4">
            <Input
              label="Password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              disabled={submitting}
            />
          </div>
        </>
      ) : (
        <div className="space-y-3">
          <p className="text-sm text-ink-300">
            Enter the 6-digit code from your authenticator app.
          </p>
          <Input
            label="Authentication code"
            name="totp"
            inputMode="numeric"
            autoComplete="one-time-code"
            autoFocus
            required
            maxLength={6}
            pattern="[0-9]{6}"
            value={totp}
            onChange={(e) => setTotp(e.target.value.replace(/\D/g, "").slice(0, 6))}
            placeholder="000000"
            disabled={submitting}
          />
        </div>
      )}

      {error ? (
        <p className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </p>
      ) : null}

      <Button
        type="submit"
        variant="amber"
        className="mt-6 w-full"
        loading={submitting}
      >
        {mfaStep ? "Verify and continue" : "Sign in"}
      </Button>

      {!mfaStep ? (
        <>
          <div className="my-5 flex items-center gap-3 text-xs uppercase tracking-[0.18em] text-ink-500">
            <span className="h-px flex-1 bg-ink-700" />
            or
            <span className="h-px flex-1 bg-ink-700" />
          </div>
          <button
            type="button"
            onClick={() => void onGuest()}
            disabled={guestLoading || submitting}
            className="w-full text-center text-sm text-amber-200/90 underline-offset-4 transition hover:text-amber-100 hover:underline disabled:opacity-50"
          >
            {guestLoading ? "Starting guest session…" : "Customer? Continue as guest →"}
          </button>
        </>
      ) : (
        <button
          type="button"
          className="mt-4 w-full text-center text-sm text-ink-400 hover:text-ink-200"
          onClick={() => {
            setMfaStep(false);
            setTotp("");
            setChallengeToken(undefined);
          }}
        >
          ← Back to password
        </button>
      )}
    </form>
  );
}
