"use client";

import {
  parseCitationMarkers,
  renderInlineMarkdown,
} from "@/features/chat/lib/markers";
import type { Citation, Message, RefusalEvent } from "@/shared/api/types";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/lib/cn";

function CitationChip({
  marker,
  onClick,
}: {
  marker: number;
  onClick: (marker: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onClick(marker)}
      className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-amber-400/20 px-1 align-super text-[10px] font-semibold text-amber-200 transition hover:bg-amber-400/35"
      aria-label={`Open citation ${marker}`}
    >
      {marker}
    </button>
  );
}

function MessageBody({
  content,
  onCitationClick,
}: {
  content: string;
  onCitationClick: (marker: number) => void;
}) {
  const segments = parseCitationMarkers(content);
  return (
    <div className="text-[15px] leading-relaxed text-ink-100">
      {segments.map((seg, i) =>
        seg.type === "text" ? (
          <span
            key={i}
            dangerouslySetInnerHTML={{ __html: renderInlineMarkdown(seg.value) }}
          />
        ) : (
          <CitationChip key={i} marker={seg.marker} onClick={onCitationClick} />
        ),
      )}
    </div>
  );
}

export function MessageList({
  messages,
  streamingText,
  phase,
  refusal,
  error,
  onCitationClick,
  onFeedback,
  onRetry,
}: {
  messages: Message[];
  streamingText: string;
  phase: string;
  refusal: RefusalEvent | null;
  error: string | null;
  onCitationClick: (marker: number) => void;
  onFeedback: (messageId: string, rating: "up" | "down") => void;
  onRetry?: () => void;
}) {
  return (
    <div
      className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6"
      aria-live="polite"
    >
      {messages.map((message) => (
        <article
          key={message.id}
          className={cn(
            "animate-fade-up",
            message.role === "user" ? "self-end" : "self-stretch",
          )}
        >
          {message.role === "user" ? (
            <div className="max-w-[85%] rounded-2xl rounded-br-md bg-ink-800 px-4 py-3 text-[15px] text-ink-50">
              {message.content}
            </div>
          ) : (
            <div className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-4 py-4">
              <div className="mb-2 flex items-center gap-2 text-xs text-amber-200/80">
                <span aria-hidden>◆</span>
                <span>Aegis</span>
              </div>
              <MessageBody content={message.content} onCitationClick={onCitationClick} />
              <div className="mt-4 flex flex-wrap gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onFeedback(message.id, "up")}
                  aria-label="Helpful"
                >
                  👍
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onFeedback(message.id, "down")}
                  aria-label="Not helpful"
                >
                  👎
                </Button>
              </div>
            </div>
          )}
        </article>
      ))}

      {streamingText ? (
        <article className="rounded-2xl border border-ink-700/70 bg-ink-900/40 px-4 py-4">
          <div className="mb-2 flex items-center gap-2 text-xs text-amber-200/80">
            <span className="inline-block h-2 w-2 animate-pulse-soft rounded-full bg-amber-300" />
            Generating…
          </div>
          <MessageBody content={streamingText} onCitationClick={onCitationClick} />
          <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-amber-300 align-middle" />
        </article>
      ) : null}

      {phase === "searching" || phase === "citing" ? (
        <div className="flex items-center gap-2 text-sm text-ink-400">
          <span className="inline-block h-2 w-2 animate-pulse-soft rounded-full bg-amber-300" />
          {phase === "searching" ? "Searching collections…" : "Gathering sources…"}
        </div>
      ) : null}

      {refusal ? (
        <div className="rounded-2xl border border-amber-400/25 bg-amber-400/5 px-4 py-4 text-sm text-ink-200">
          <p className="font-medium text-amber-100">{refusal.message}</p>
          {refusal.detail ? <p className="mt-2 text-ink-400">{refusal.detail}</p> : null}
          {refusal.nearest_documents?.length ? (
            <ul className="mt-3 space-y-1 text-ink-400">
              {refusal.nearest_documents.map((doc) => (
                <li key={doc.title}>
                  Closest: {doc.title} ({doc.score.toFixed(2)})
                </li>
              ))}
            </ul>
          ) : null}
          {refusal.suggestions?.length ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {refusal.suggestions.map((s) => (
                <span
                  key={s}
                  className="rounded-md border border-ink-600 bg-ink-900 px-2 py-1 text-xs text-ink-300"
                >
                  Try: {s}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-4 text-sm text-rose-100">
          <p>{error}</p>
          {onRetry ? (
            <Button size="sm" variant="secondary" className="mt-3" onClick={onRetry}>
              Retry
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function SuggestedQuestions({
  suggestions,
  onPick,
}: {
  suggestions: string[];
  onPick: (q: string) => void;
}) {
  if (!suggestions.length) return null;
  return (
    <div className="mx-auto grid w-full max-w-3xl gap-3 px-4 sm:grid-cols-2">
      {suggestions.slice(0, 4).map((q) => (
        <button
          key={q}
          type="button"
          onClick={() => onPick(q)}
          className="rounded-xl border border-ink-700 bg-ink-900/50 px-4 py-3 text-left text-sm text-ink-200 transition hover:border-amber-400/40 hover:bg-ink-850 hover:text-ink-50"
        >
          {q}
        </button>
      ))}
    </div>
  );
}

// keep Citation type import used for potential extension
export type { Citation };
