"use client";

import type { Citation, UsageEvent } from "@/shared/api/types";
import { scoreLabel } from "@/shared/lib/format";
import { cn } from "@/shared/lib/cn";
import { Badge } from "@/shared/ui/Badge";

export function SourcesPanel({
  citations,
  selectedMarker,
  usage,
  open,
  onClose,
  onSelect,
}: {
  citations: Citation[];
  selectedMarker: number | null;
  usage: UsageEvent | null;
  open: boolean;
  onClose: () => void;
  onSelect: (marker: number) => void;
}) {
  const selected = citations.find((c) => c.marker === selectedMarker) ?? null;

  return (
    <aside
      className={cn(
        "flex h-full w-full flex-col border-l border-ink-800 bg-ink-950/90 transition duration-200 ease-out",
        open ? "translate-x-0 opacity-100" : "pointer-events-none translate-x-4 opacity-0 lg:pointer-events-auto lg:translate-x-0 lg:opacity-100",
      )}
      aria-label="Sources"
    >
      <div className="flex items-center justify-between border-b border-ink-800 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-ink-100">Sources</h2>
          <p className="text-xs text-ink-500">{citations.length} citations</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md px-2 py-1 text-ink-400 hover:bg-ink-800 hover:text-ink-100 lg:hidden"
          aria-label="Close sources"
        >
          ×
        </button>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto p-3">
        {citations.length === 0 ? (
          <p className="px-1 py-8 text-center text-sm text-ink-500">
            Sources appear here once retrieval finishes.
          </p>
        ) : (
          citations.map((citation) => {
            const score =
              citation.score_rerank ??
              citation.score_fused ??
              citation.score_dense ??
              null;
            const active = selectedMarker === citation.marker;
            return (
              <button
                key={`${citation.marker}-${citation.document_id}`}
                type="button"
                onClick={() => onSelect(citation.marker)}
                className={cn(
                  "w-full rounded-xl border px-3 py-3 text-left transition",
                  active
                    ? "border-amber-400/50 bg-amber-400/10"
                    : "border-ink-700 bg-ink-900/50 hover:border-ink-500",
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-ink-800 text-[11px] font-semibold text-amber-200">
                    {citation.marker}
                  </span>
                  {citation.was_used === false ? (
                    <Badge tone="neutral">unused</Badge>
                  ) : (
                    <Badge tone="amber">used</Badge>
                  )}
                </div>
                <p className="mt-2 line-clamp-2 text-sm font-medium text-ink-100">
                  {citation.document_title || "Untitled document"}
                </p>
                <p className="mt-1 text-xs text-ink-400">
                  {citation.page != null ? `p.${citation.page}` : "—"}
                  {citation.section ? ` · §${citation.section}` : ""}
                </p>
                <div className="mt-2 flex items-center gap-2">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
                    <div
                      className="h-full rounded-full bg-amber-400/80"
                      style={{ width: `${Math.min(100, (score ?? 0) * 100)}%` }}
                    />
                  </div>
                  <span className="text-[11px] tabular-nums text-ink-400">
                    {scoreLabel(score)}
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>

      {selected ? (
        <div className="animate-fade-up border-t border-ink-800 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-500">
            Quote
          </p>
          <p className="mt-2 text-sm leading-relaxed text-ink-200">
            {selected.quote || "No quote available for this citation."}
          </p>
        </div>
      ) : null}

      {usage ? (
        <div className="border-t border-ink-800 px-4 py-3 text-xs text-ink-500">
          {(usage.total_ms / 1000).toFixed(1)}s · {usage.prompt_tokens + usage.completion_tokens}{" "}
          tok · conf {usage.confidence.toFixed(2)}
          {usage.grounded ? " · grounded" : ""}
        </div>
      ) : null}
    </aside>
  );
}
