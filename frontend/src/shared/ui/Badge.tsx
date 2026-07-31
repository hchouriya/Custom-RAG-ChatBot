import { cn } from "@/shared/lib/cn";

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "amber" | "success" | "danger" | "info";
  className?: string;
}) {
  const tones = {
    neutral: "bg-ink-800 text-ink-200 border-ink-600",
    amber: "bg-amber-400/15 text-amber-200 border-amber-400/30",
    success: "bg-emerald-400/15 text-emerald-200 border-emerald-400/30",
    danger: "bg-rose-400/15 text-rose-200 border-rose-400/30",
    info: "bg-sky-400/15 text-sky-200 border-sky-400/30",
  } as const;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-medium tracking-wide",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
