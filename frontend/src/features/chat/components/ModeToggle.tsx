"use client";

import { cn } from "@/shared/lib/cn";
import type { Mode } from "@/shared/api/types";

export function ModeToggle({
  mode,
  allowed,
  onChange,
}: {
  mode: Mode;
  allowed: Mode[];
  onChange: (mode: Mode) => void;
}) {
  if (allowed.length < 2) {
    return (
      <div className="px-3">
        <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-500">
          Mode
        </p>
        <div className="rounded-lg border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-200 capitalize">
          {mode}
        </div>
      </div>
    );
  }

  return (
    <div className="px-3">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-500">
        Mode
      </p>
      <div
        role="radiogroup"
        aria-label="Assistant mode"
        className="grid grid-cols-2 gap-1 rounded-lg border border-ink-700 bg-ink-950 p-1"
      >
        {(["internal", "customer"] as Mode[]).map((value) => {
          const enabled = allowed.includes(value);
          return (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={mode === value}
              disabled={!enabled}
              onClick={() => onChange(value)}
              className={cn(
                "rounded-md px-2 py-1.5 text-xs font-medium capitalize transition",
                mode === value
                  ? value === "internal"
                    ? "bg-amber-400/20 text-amber-100"
                    : "bg-teal-400/20 text-teal-100"
                  : "text-ink-400 hover:text-ink-200",
                !enabled && "cursor-not-allowed opacity-40",
              )}
            >
              {value}
            </button>
          );
        })}
      </div>
    </div>
  );
}
