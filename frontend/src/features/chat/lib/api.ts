import { apiFetch, apiStream, parseSseStream } from "@/shared/api/client";
import type {
  Conversation,
  ConversationDetail,
  FeedbackRating,
  Message,
  Paginated,
  StreamEvent,
} from "@/shared/api/types";
import { idempotencyKey } from "@/shared/lib/format";

const bff = (path: string) => `/bff${path.startsWith("/") ? path : `/${path}`}`;

export async function listConversations(params?: {
  q?: string;
  archived?: boolean;
  pinned?: boolean;
  limit?: number;
}): Promise<Conversation[]> {
  const data = await apiFetch<Paginated<Conversation> | Conversation[]>(
    bff("/chat/conversations"),
    { query: params },
  );
  return Array.isArray(data) ? data : data.items;
}

export async function createConversation(input?: {
  title?: string;
  collection_ids?: string[];
}): Promise<Conversation> {
  return apiFetch<Conversation>(bff("/chat/conversations"), {
    method: "POST",
    body: input ?? {},
    headers: { "Idempotency-Key": idempotencyKey() },
  });
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const data = await apiFetch<
    ConversationDetail | (Conversation & { messages?: Message[] })
  >(bff(`/chat/conversations/${id}`));

  if ("conversation" in data && data.conversation) {
    return data as ConversationDetail;
  }

  const conversation = data as Conversation;
  return {
    conversation,
    messages: (data as { messages?: Message[] }).messages ?? [],
  };
}

export async function deleteConversation(id: string): Promise<void> {
  await apiFetch(bff(`/chat/conversations/${id}`), { method: "DELETE" });
}

export async function getSuggestions(): Promise<string[]> {
  const data = await apiFetch<{ suggestions?: string[] } | string[]>(
    bff("/chat/suggestions"),
  );
  return Array.isArray(data) ? data : (data.suggestions ?? []);
}

export async function cancelStream(conversationId: string): Promise<void> {
  await apiFetch(bff(`/chat/conversations/${conversationId}/stream`), {
    method: "DELETE",
  });
}

export async function sendFeedback(
  messageId: string,
  rating: FeedbackRating,
  reason?: string,
): Promise<void> {
  await apiFetch(bff(`/chat/messages/${messageId}/feedback`), {
    method: "POST",
    body: { rating, reason },
  });
}

export async function* streamMessage(
  conversationId: string,
  content: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await apiStream(bff(`/chat/conversations/${conversationId}/messages`), {
    method: "POST",
    signal,
    headers: { "Idempotency-Key": idempotencyKey() },
    body: {
      content,
      stream: true,
      options: { max_citations: 8 },
    },
  });

  yield* parseSseStream(response);
}
