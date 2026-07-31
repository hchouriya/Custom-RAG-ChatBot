import { forwardRef, type InputHTMLAttributes } from "react";

import { cn } from "@/shared/lib/cn";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, label, hint, error, id, ...props },
  ref,
) {
  const inputId = id ?? props.name;
  return (
    <label className="flex w-full flex-col gap-1.5 text-sm text-ink-200">
      {label ? <span className="font-medium text-ink-100">{label}</span> : null}
      <input
        ref={ref}
        id={inputId}
        className={cn(
          "h-11 w-full rounded-lg border border-ink-600 bg-ink-900/80 px-3 text-ink-50",
          "placeholder:text-ink-500 transition duration-150",
          "focus:border-amber-400/70 focus:outline-none focus:ring-2 focus:ring-amber-400/25",
          error && "border-rose-400/60 focus:border-rose-400 focus:ring-rose-400/20",
          className,
        )}
        {...props}
      />
      {error ? <span className="text-xs text-rose-300">{error}</span> : null}
      {!error && hint ? <span className="text-xs text-ink-400">{hint}</span> : null}
    </label>
  );
});
