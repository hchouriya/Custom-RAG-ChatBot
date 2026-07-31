"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  cancelStream,
  createConversation,
  getConversation,
  getSuggestions,
  listConversations,
  sendFeedback,
  streamMessage,
} from "@/features/chat/lib/api";
import type {
  Citation,
  Conversation,
  FeedbackRating,
  Message,
  RefusalEvent,
  UsageEvent,
} from "@/shared/api/types";

export type StreamPhase =
  | "idle"
  | "searching"
  | "citing"
  | "generating"
  | "done"
  | "refused"
  | "error";

export interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  messages: Message[];
  citations: Citation[];
  suggestions: string[];
  phase: StreamPhase;
  streamingText: string;
  usage: UsageEvent | null;
  refusal: RefusalEvent | null;
  error: string | null;
  loadingList: boolean;
  loadingThread: boolean;
  selectedMarker: number | null;
}

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [phase, setPhase] = useState<StreamPhase>("idle");
  const [streamingText, setStreamingText] = useState("");
  const [usage, setUsage] = useState<UsageEvent | null>(null);
  const [refusal, setRefusal] = useState<RefusalEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingThread, setLoadingThread] = useState(false);
  const [selectedMarker, setSelectedMarker] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const activeIdRef = useRef<string | null>(null);
  activeIdRef.current = activeId;

  const refreshConversations = useCallback(async () => {
    setLoadingList(true);
    try {
      const items = await listConversations({ limit: 40 });
      setConversations(items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load conversations");
    } finally {
      setLoadingList(false);
    }
  }, []);

  const loadSuggestions = useCallback(async () => {
    try {
      setSuggestions(await getSuggestions());
    } catch {
      setSuggestions([]);
    }
  }, []);

  useEffect(() => {
    void refreshConversations();
    void loadSuggestions();
  }, [refreshConversations, loadSuggestions]);

  const selectConversation = useCallback(async (id: string) => {
    setActiveId(id);
    setLoadingThread(true);
    setError(null);
    setRefusal(null);
    setUsage(null);
    setStreamingText("");
    setPhase("idle");
    setSelectedMarker(null);
    try {
      const detail = await getConversation(id);
      setMessages(detail.messages);
      const lastAssistant = [...detail.messages]
        .reverse()
        .find((m) => m.role === "assistant");
      setCitations(lastAssistant?.citations ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load conversation");
      setMessages([]);
      setCitations([]);
    } finally {
      setLoadingThread(false);
    }
  }, []);

  const startNewChat = useCallback(async () => {
    abortRef.current?.abort();
    setActiveId(null);
    setMessages([]);
    setCitations([]);
    setStreamingText("");
    setPhase("idle");
    setRefusal(null);
    setUsage(null);
    setError(null);
    setSelectedMarker(null);
  }, []);

  const stopStreaming = useCallback(async () => {
    abortRef.current?.abort();
    const id = activeIdRef.current;
    if (id) {
      try {
        await cancelStream(id);
      } catch {
        // best-effort
      }
    }
    setPhase((p) => (p === "generating" || p === "searching" || p === "citing" ? "done" : p));
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed) return;

      setError(null);
      setRefusal(null);
      setUsage(null);
      setStreamingText("");
      setCitations([]);
      setSelectedMarker(null);
      setPhase("searching");

      let conversationId = activeIdRef.current;
      try {
        if (!conversationId) {
          const created = await createConversation({
            title: trimmed.slice(0, 80),
          });
          conversationId = created.id;
          setActiveId(created.id);
          setConversations((prev) => [created, ...prev]);
        }

        const userMessage: Message = {
          id: `local-user-${Date.now()}`,
          conversation_id: conversationId,
          role: "user",
          content: trimmed,
          created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMessage]);

        const controller = new AbortController();
        abortRef.current = controller;

        let assistantText = "";
        let assistantId: string | null = null;
        let streamCitations: Citation[] = [];

        for await (const event of streamMessage(
          conversationId,
          trimmed,
          controller.signal,
        )) {
          switch (event.event) {
            case "meta":
              setPhase("citing");
              if (event.data.message_id) assistantId = event.data.message_id;
              break;
            case "citations":
              streamCitations = event.data.citations ?? [];
              setCitations(streamCitations);
              setPhase("generating");
              break;
            case "token":
              setPhase("generating");
              assistantText += event.data.delta;
              setStreamingText(assistantText);
              break;
            case "usage":
              setUsage(event.data);
              break;
            case "refusal":
              setRefusal(event.data);
              setPhase("refused");
              break;
            case "clarify":
              setRefusal({
                reason: "clarify",
                message: event.data.question,
                suggestions: event.data.options,
              });
              setPhase("refused");
              break;
            case "error":
              setError(event.data.message);
              setPhase("error");
              break;
            case "done": {
              const finalId = event.data.message_id ?? assistantId ?? `local-asst-${Date.now()}`;
              const used = new Set(event.data.citations_used ?? []);
              const finalCitations =
                used.size > 0
                  ? streamCitations.map((c) => ({
                      ...c,
                      was_used: used.has(c.marker),
                    }))
                  : streamCitations;
              setCitations(finalCitations);
              setMessages((prev) => [
                ...prev,
                {
                  id: finalId,
                  conversation_id: conversationId!,
                  role: "assistant",
                  content: assistantText,
                  status: (event.data.status as Message["status"]) ?? "ok",
                  citations: finalCitations,
                  created_at: new Date().toISOString(),
                },
              ]);
              setStreamingText("");
              setPhase(event.data.status === "refused" ? "refused" : "done");
              void refreshConversations();
              break;
            }
            default:
              break;
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          setPhase("done");
          return;
        }
        setError(err instanceof Error ? err.message : "Stream failed");
        setPhase("error");
      }
    },
    [refreshConversations],
  );

  const submitFeedback = useCallback(
    async (messageId: string, rating: FeedbackRating) => {
      await sendFeedback(messageId, rating);
    },
    [],
  );

  const state: ChatState = useMemo(
    () => ({
      conversations,
      activeId,
      messages,
      citations,
      suggestions,
      phase,
      streamingText,
      usage,
      refusal,
      error,
      loadingList,
      loadingThread,
      selectedMarker,
    }),
    [
      conversations,
      activeId,
      messages,
      citations,
      suggestions,
      phase,
      streamingText,
      usage,
      refusal,
      error,
      loadingList,
      loadingThread,
      selectedMarker,
    ],
  );

  return {
    ...state,
    selectConversation,
    startNewChat,
    ask,
    stopStreaming,
    submitFeedback,
    setSelectedMarker,
    refreshConversations,
  };
}
