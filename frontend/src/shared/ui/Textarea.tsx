import { forwardRef, type TextareaHTMLAttributes } from "react";

import { cn } from "@/shared/lib/cn";

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { className, label, ...props },
  ref,
) {
  return (
    <label className="flex w-full flex-col gap-1.5 text-sm text-ink-200">
      {label ? <span className="font-medium text-ink-100">{label}</span> : null}
      <textarea
        ref={ref}
        className={cn(
          "min-h-[2.75rem] w-full resize-none rounded-lg border border-ink-600 bg-ink-900/80 px-3 py-2.5 text-ink-50",
          "placeholder:text-ink-500 transition duration-150",
          "focus:border-amber-400/70 focus:outline-none focus:ring-2 focus:ring-amber-400/25",
          className,
        )}
        {...props}
      />
    </label>
  );
});
