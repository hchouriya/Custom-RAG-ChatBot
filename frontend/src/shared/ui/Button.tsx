import { forwardRef, type ButtonHTMLAttributes } from "react";

import { cn } from "@/shared/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "amber";
type Size = "sm" | "md" | "lg" | "icon";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

const variants: Record<Variant, string> = {
  primary:
    "bg-amber-400 text-ink-950 hover:bg-amber-300 focus-visible:ring-amber-300/60 disabled:bg-amber-400/40",
  secondary:
    "bg-ink-800 text-ink-50 border border-ink-600 hover:bg-ink-700 focus-visible:ring-ink-400/40",
  ghost: "bg-transparent text-ink-200 hover:bg-ink-800/80 focus-visible:ring-ink-400/30",
  danger:
    "bg-rose-500/15 text-rose-200 border border-rose-500/30 hover:bg-rose-500/25 focus-visible:ring-rose-400/40",
  amber:
    "bg-gradient-to-b from-amber-300 to-amber-500 text-ink-950 shadow-[0_8px_24px_-12px_rgba(245,158,11,0.7)] hover:from-amber-200 hover:to-amber-400",
};

const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-xs gap-1.5",
  md: "h-10 px-4 text-sm gap-2",
  lg: "h-12 px-5 text-base gap-2",
  icon: "h-10 w-10",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    className,
    variant = "primary",
    size = "md",
    loading = false,
    disabled,
    children,
    ...props
  },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center rounded-lg font-medium transition duration-150 ease-out",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-ink-950",
        "disabled:cursor-not-allowed disabled:opacity-50",
        variants[variant],
        sizes[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span
          className="h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent"
          aria-hidden
        />
      ) : null}
      {children}
    </button>
  );
});
