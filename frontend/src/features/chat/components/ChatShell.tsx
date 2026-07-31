"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Composer } from "@/features/chat/components/Composer";
import { ConversationSidebar } from "@/features/chat/components/ConversationSidebar";
import {
  MessageList,
  SuggestedQuestions,
} from "@/features/chat/components/MessageList";
import { ModeToggle } from "@/features/chat/components/ModeToggle";
import { SourcesPanel } from "@/features/chat/components/SourcesPanel";
import { useChat } from "@/features/chat/hooks/useChat";
import { useAuth } from "@/features/auth/hooks/useAuth";
import type { Mode } from "@/shared/api/types";
import { EmptyState } from "@/shared/ui/EmptyState";
import { cn } from "@/shared/lib/cn";

export function ChatShell() {
  const { user, logout, setModeLocally } = useAuth();
  const chat = useChat();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1280px)");
    const apply = () => setSourcesOpen(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const streaming =
    chat.phase === "searching" ||
    chat.phase === "citing" ||
    chat.phase === "generating";

  const title =
    chat.conversations.find((c) => c.id === chat.activeId)?.title ?? "New conversation";

  const canDocs =
    user &&
    (user.role === "admin" ||
      user.role === "manager" ||
      user.permissions.includes("document:write") ||
      user.permissions.includes("document:read"));

  function onModeChange(mode: Mode) {
    setModeLocally(mode);
    // Mode is also sent via X-Assistant-Mode on subsequent BFF calls when we set a cookie.
    document.cookie = `aegis_mode=${mode}; path=/; SameSite=Lax`;
    void chat.startNewChat();
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-ink-950 text-ink-50">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen ? (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
        />
      ) : null}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-ink-800 bg-ink-950 transition duration-200 ease-out lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center gap-2 border-b border-ink-800 px-4 py-4">
          <span className="font-display text-2xl tracking-tight text-amber-300">Aegis</span>
        </div>

        <div className="py-3">
          <ModeToggle
            mode={user?.mode ?? "customer"}
            allowed={user?.allowed_modes ?? ["customer"]}
            onChange={onModeChange}
          />
        </div>

        <div className="min-h-0 flex-1">
          <ConversationSidebar
            conversations={chat.conversations}
            activeId={chat.activeId}
            loading={chat.loadingList}
            onSelect={(id) => {
              void chat.selectConversation(id);
              setSidebarOpen(false);
            }}
            onNew={() => {
              void chat.startNewChat();
              setSidebarOpen(false);
            }}
          />
        </div>

        <div className="space-y-1 border-t border-ink-800 p-3 text-sm">
          {canDocs ? (
            <Link
              href="/documents"
              className="block rounded-lg px-2.5 py-2 text-ink-300 transition hover:bg-ink-850 hover:text-ink-50"
            >
              Documents
            </Link>
          ) : null}
          <div className="flex items-center justify-between gap-2 rounded-lg px-2.5 py-2">
            <div className="min-w-0">
              <p className="truncate text-ink-100">
                {user?.full_name || user?.email || "Guest"}
              </p>
              <p className="truncate text-xs capitalize text-ink-500">{user?.role}</p>
            </div>
            <button
              type="button"
              onClick={() => void logout().then(() => (window.location.href = "/login"))}
              className="shrink-0 text-xs text-ink-400 hover:text-amber-200"
            >
              Sign out
            </button>
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between border-b border-ink-800 px-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              className="rounded-md px-2 py-1 text-ink-300 hover:bg-ink-850 lg:hidden"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open conversations"
            >
              ☰
            </button>
            <h1 className="truncate text-sm font-medium text-ink-100 sm:text-base">
              {title}
            </h1>
          </div>
          <button
            type="button"
            onClick={() => setSourcesOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-lg border border-ink-700 px-2.5 py-1.5 text-xs text-ink-300 hover:border-amber-400/40 hover:text-amber-100 xl:hidden"
          >
            Sources
            {chat.citations.length ? (
              <span className="rounded bg-amber-400/20 px-1.5 py-0.5 text-amber-200">
                {chat.citations.length}
              </span>
            ) : null}
          </button>
        </header>

        <div className="relative min-h-0 flex-1 overflow-y-auto">
          {!chat.activeId && chat.messages.length === 0 && !chat.streamingText ? (
            <div className="flex min-h-full flex-col justify-center gap-8 py-10">
              <EmptyState
                title="What can I help you find?"
                description="Answers come only from documents you have access to."
              />
              <SuggestedQuestions
                suggestions={chat.suggestions}
                onPick={(q) => void chat.ask(q)}
              />
            </div>
          ) : (
            <MessageList
              messages={chat.messages}
              streamingText={chat.streamingText}
              phase={chat.phase}
              refusal={chat.refusal}
              error={chat.error}
              onCitationClick={(marker) => {
                chat.setSelectedMarker(marker);
                setSourcesOpen(true);
              }}
              onFeedback={(id, rating) => void chat.submitFeedback(id, rating)}
            />
          )}
        </div>

        <Composer
          streaming={streaming}
          onSend={(q) => void chat.ask(q)}
          onStop={() => void chat.stopStreaming()}
        />
      </main>

      {/* Sources: desktop column + mobile overlay */}
      <div
        className={cn(
          "fixed inset-y-0 right-0 z-40 w-[320px] xl:static xl:z-0 xl:block xl:w-[320px]",
          sourcesOpen ? "block" : "hidden xl:block",
        )}
      >
        {sourcesOpen && (
          <button
            type="button"
            className="fixed inset-0 z-30 bg-black/40 xl:hidden"
            aria-label="Close sources overlay"
            onClick={() => setSourcesOpen(false)}
          />
        )}
        <div className="relative z-40 h-full">
          <SourcesPanel
            citations={chat.citations}
            selectedMarker={chat.selectedMarker}
            usage={chat.usage}
            open={sourcesOpen}
            onClose={() => setSourcesOpen(false)}
            onSelect={(marker) => chat.setSelectedMarker(marker)}
          />
        </div>
      </div>
    </div>
  );
}
