import { cn } from "@/shared/lib/cn";

export function EmptyState({
  title,
  description,
  action,
  className,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 px-6 py-16 text-center animate-fade-up",
        className,
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full border border-amber-400/30 bg-amber-400/10 text-amber-300">
        <span className="font-display text-xl" aria-hidden>
          ◆
        </span>
      </div>
      <h2 className="font-display text-2xl text-ink-50">{title}</h2>
      {description ? <p className="max-w-md text-sm text-ink-400">{description}</p> : null}
      {action}
    </div>
  );
}
