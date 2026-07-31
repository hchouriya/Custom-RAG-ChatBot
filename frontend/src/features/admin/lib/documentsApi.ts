import { apiFetch } from "@/shared/api/client";
import type {
  Collection,
  Document,
  Paginated,
  RegisteredDocument,
  UploadTicket,
  Visibility,
} from "@/shared/api/types";
import { idempotencyKey } from "@/shared/lib/format";

const bff = (path: string) => `/bff${path.startsWith("/") ? path : `/${path}`}`;

export async function listDocuments(params?: {
  q?: string;
  collection_id?: string;
  limit?: number;
}): Promise<Paginated<Document> | Document[]> {
  return apiFetch(bff("/documents"), { query: params });
}

export async function getDocument(id: string): Promise<Document> {
  return apiFetch(bff(`/documents/${id}`));
}

export async function listCollections(): Promise<Collection[]> {
  const data = await apiFetch<Paginated<Collection> | Collection[]>(bff("/collections"));
  return Array.isArray(data) ? data : data.items;
}

export async function createUploadTicket(input: {
  filename: string;
  size_bytes: number;
  declared_mime: string;
  collection_id: string;
}): Promise<UploadTicket> {
  return apiFetch(bff("/documents/uploads"), {
    method: "POST",
    body: input,
    headers: { "Idempotency-Key": idempotencyKey() },
  });
}

export async function putToStorage(
  ticket: UploadTicket,
  file: File,
): Promise<void> {
  if (Object.keys(ticket.fields ?? {}).length > 0) {
    const form = new FormData();
    for (const [key, value] of Object.entries(ticket.fields)) {
      form.append(key, value);
    }
    form.append("file", file);
    const response = await fetch(ticket.url, { method: "POST", body: form });
    if (!response.ok) {
      throw new Error(`Upload to storage failed (${response.status})`);
    }
    return;
  }

  const response = await fetch(ticket.url, {
    method: "PUT",
    headers: {
      "Content-Type": ticket.declared_mime || file.type || "application/octet-stream",
    },
    body: file,
  });
  if (!response.ok) {
    throw new Error(`Upload to storage failed (${response.status})`);
  }
}

export async function registerDocument(input: {
  upload_id: string;
  title: string;
  visibility: Visibility;
  description?: string;
  tags?: string[];
}): Promise<RegisteredDocument | Document> {
  return apiFetch(bff("/documents"), {
    method: "POST",
    body: input,
    headers: { "Idempotency-Key": idempotencyKey() },
  });
}
