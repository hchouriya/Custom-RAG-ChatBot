import { LoginForm } from "@/features/auth/components/LoginForm";

export default function LoginPage() {
  return (
    <main className="login-atmosphere relative flex min-h-dvh items-center justify-center overflow-hidden px-4 py-10">
      <div className="login-grid pointer-events-none absolute inset-0 animate-glow" aria-hidden />
      <div className="relative z-10 flex w-full max-w-lg flex-col items-center text-center">
        <p className="font-display text-6xl tracking-tight text-amber-300 sm:text-7xl animate-fade-up">
          Aegis
        </p>
        <p
          className="mt-3 max-w-sm text-base text-ink-300 animate-fade-up"
          style={{ animationDelay: "80ms" }}
        >
          Grounded answers from your documents
        </p>
        <div className="mt-10 w-full" style={{ animationDelay: "140ms" }}>
          <LoginForm />
        </div>
      </div>
    </main>
  );
}
