import { ApiError } from "@/shared/api/errors";
import type { ProblemDetails, StreamEvent, StreamEventName } from "@/shared/api/types";

type Json = Record<string, unknown> | unknown[] | string | number | boolean | null;

export interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  body?: Json | FormData | BodyInit | null;
  query?: Record<string, string | number | boolean | null | undefined>;
  /** When true, do not attempt a silent refresh + retry. */
  skipRefresh?: boolean;
}

function buildQuery(query?: ApiRequestOptions["query"]): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") continue;
    params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

async function parseProblem(response: Response): Promise<ProblemDetails> {
  try {
    const data = (await response.json()) as ProblemDetails;
    return {
      status: data.status ?? response.status,
      title: data.title,
      detail: data.detail,
      code: data.code,
      type: data.type,
      instance: data.instance,
      request_id: data.request_id,
      errors: data.errors,
      challenge_token: data.challenge_token,
      enrolment_required: data.enrolment_required,
    };
  } catch {
    return {
      status: response.status,
      title: response.statusText || "Request failed",
      detail: await response.text().catch(() => "Request failed"),
      code: "HTTP_ERROR",
    };
  }
}

/**
 * Browser client talks only to the Next.js BFF under `/api/...`.
 * Tokens never leave httpOnly cookies.
 */
export async function apiFetch<T>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<T> {
  const { body, query, skipRefresh, headers, ...rest } = options;
  const url = `/api${path.startsWith("/") ? path : `/${path}`}${buildQuery(query)}`;

  const init: RequestInit = {
    ...rest,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(headers ?? {}),
    },
  };

  if (body !== undefined && body !== null) {
    if (body instanceof FormData || typeof body === "string" || body instanceof Blob) {
      init.body = body as BodyInit;
    } else {
      (init.headers as Record<string, string>)["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
  }

  let response = await fetch(url, init);

  if (response.status === 401 && !skipRefresh && !path.startsWith("/auth/")) {
    const refreshed = await fetch("/api/auth/refresh", {
      method: "POST",
      credentials: "include",
    });
    if (refreshed.ok) {
      response = await fetch(url, init);
    }
  }

  if (!response.ok) {
    throw new ApiError(await parseProblem(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return (await response.text()) as T;
}

export async function* parseSseStream(
  response: Response,
): AsyncGenerator<StreamEvent> {
  if (!response.body) {
    throw new ApiError({
      status: 502,
      title: "Empty stream",
      detail: "The server returned no response body.",
      code: "EMPTY_STREAM",
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName: StreamEventName | null = null;
  let dataLines: string[] = [];

  const flush = (): StreamEvent | null => {
    if (!eventName && dataLines.length === 0) return null;
    const raw = dataLines.join("\n");
    dataLines = [];
    const name = (eventName ?? "token") as StreamEventName;
    eventName = null;
    if (!raw || name === "heartbeat") {
      return { event: "heartbeat", data: {} };
    }
    try {
      const data = JSON.parse(raw) as StreamEvent["data"];
      return { event: name, data } as StreamEvent;
    } catch {
      return {
        event: "error",
        data: { code: "PARSE_ERROR", message: "Malformed SSE payload", retryable: false },
      };
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      let line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);

      if (line === "") {
        const event = flush();
        if (event) yield event;
      } else if (line.startsWith(":")) {
        // heartbeat comment
      } else if (line.startsWith("event:")) {
        eventName = line.slice(6).trim() as StreamEventName;
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }

      newlineIndex = buffer.indexOf("\n");
    }
  }

  const trailing = flush();
  if (trailing) yield trailing;
}

export async function apiStream(
  path: string,
  options: ApiRequestOptions = {},
): Promise<Response> {
  const { body, query, headers, ...rest } = options;
  const url = `/api${path.startsWith("/") ? path : `/${path}`}${buildQuery(query)}`;

  const response = await fetch(url, {
    ...rest,
    method: options.method ?? "POST",
    credentials: "include",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      ...(headers ?? {}),
    },
    body: body === undefined || body === null ? undefined : JSON.stringify(body),
  });

  if (!response.ok) {
    throw new ApiError(await parseProblem(response));
  }

  return response;
}
