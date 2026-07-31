"use client";

import { conversationGroupLabel } from "@/shared/lib/format";
import { cn } from "@/shared/lib/cn";
import type { Conversation } from "@/shared/api/types";
import { Skeleton } from "@/shared/ui/Skeleton";

export function ConversationSidebar({
  conversations,
  activeId,
  loading,
  onSelect,
  onNew,
}: {
  conversations: Conversation[];
  activeId: string | null;
  loading: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const grouped = conversations.reduce<Record<string, Conversation[]>>((acc, c) => {
    const label = conversationGroupLabel(c.last_message_at ?? c.created_at);
    (acc[label] ??= []).push(c);
    return acc;
  }, {});

  const order = ["Today", "Yesterday", "Last 7 days", "Earlier"];

  return (
    <div className="flex h-full flex-col">
      <button
        type="button"
        onClick={onNew}
        className="mx-3 mt-3 inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-amber-400/35 bg-amber-400/10 text-sm font-medium text-amber-100 transition hover:bg-amber-400/20"
      >
        <span aria-hidden>+</span> New chat
      </button>

      <div className="mt-4 flex-1 space-y-5 overflow-y-auto px-2 pb-4">
        {loading
          ? Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="mx-1 h-9" />
            ))
          : order.map((label) => {
              const items = grouped[label];
              if (!items?.length) return null;
              return (
                <div key={label}>
                  <p className="mb-1.5 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-500">
                    {label}
                  </p>
                  <ul className="space-y-0.5">
                    {items.map((c) => (
                      <li key={c.id}>
                        <button
                          type="button"
                          onClick={() => onSelect(c.id)}
                          className={cn(
                            "w-full truncate rounded-lg px-2.5 py-2 text-left text-sm transition",
                            activeId === c.id
                              ? "bg-ink-800 text-ink-50"
                              : "text-ink-300 hover:bg-ink-850 hover:text-ink-100",
                          )}
                          title={c.title ?? "Untitled"}
                        >
                          {c.title || "Untitled conversation"}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
      </div>
    </div>
  );
}
