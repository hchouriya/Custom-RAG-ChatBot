export type Mode = "internal" | "customer";
export type Role =
  | "admin"
  | "manager"
  | "internal_employee"
  | "customer"
  | "guest";
export type Visibility =
  | "public"
  | "customer"
  | "internal"
  | "confidential"
  | "restricted";
export type IngestStatus =
  | "pending"
  | "scanning"
  | "parsing"
  | "chunking"
  | "embedding"
  | "indexing"
  | "indexed"
  | "failed"
  | "quarantined"
  | "superseded";
export type AnswerStatus =
  | "ok"
  | "no_answer"
  | "refused"
  | "clarify"
  | "escalated"
  | "error";
export type MessageRole = "user" | "assistant" | "system";
export type FeedbackRating = "up" | "down";

export interface ProblemDetails {
  type?: string;
  title?: string;
  status: number;
  detail?: string;
  instance?: string;
  request_id?: string;
  code?: string;
  errors?: Array<{ field?: string; message: string }>;
  challenge_token?: string;
  enrolment_required?: boolean;
}

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: Role;
  department_id?: string | null;
  is_active: boolean;
  must_change_password: boolean;
  has_mfa: boolean;
}

export interface AuthTokens {
  access_token: string;
  access_expires_at: string;
  refresh_token?: string | null;
  refresh_expires_at?: string | null;
  mode: Mode;
  is_guest?: boolean;
  user?: User | null;
}

export interface Principal {
  id: string | null;
  email: string | null;
  full_name: string | null;
  role: Role;
  mode: Mode;
  allowed_modes: Mode[];
  permissions: string[];
  is_guest: boolean;
  must_change_password: boolean;
  has_mfa: boolean;
}

export interface Conversation {
  id: string;
  mode: Mode;
  title: string | null;
  summary?: string | null;
  message_count: number;
  is_pinned: boolean;
  is_archived: boolean;
  created_at: string | null;
  last_message_at: string | null;
  collection_ids?: string[];
}

export interface Citation {
  marker: number;
  chunk_id?: string;
  document_id: string;
  document_title: string;
  version_no?: number;
  page?: number | null;
  section?: string | null;
  heading_path?: string[];
  quote?: string | null;
  score_rerank?: number | null;
  score_fused?: number | null;
  score_dense?: number | null;
  score_sparse?: number | null;
  rank?: number;
  was_used?: boolean;
  url?: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  parent_id?: string | null;
  status?: AnswerStatus | null;
  refusal_reason?: string | null;
  model?: string | null;
  confidence?: number | null;
  is_grounded?: boolean | null;
  created_at?: string | null;
  citations?: Citation[];
}

export interface ConversationDetail {
  conversation: Conversation;
  messages: Message[];
}

export interface Paginated<T> {
  items: T[];
  next_cursor?: string | null;
  has_more: boolean;
  total_estimate?: number | null;
}

export interface Collection {
  id: string;
  name: string;
  slug: string;
  description?: string | null;
  mode: Mode;
  default_visibility: Visibility;
  is_active: boolean;
}

export interface DocumentVersion {
  id: string;
  document_id: string;
  version_no: number;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  page_count?: number | null;
  chunk_count: number;
  status: IngestStatus;
  error_message?: string | null;
  created_at?: string | null;
  indexed_at?: string | null;
}

export interface Document {
  id: string;
  collection_id: string;
  title: string;
  description?: string | null;
  visibility: Visibility;
  department_id?: string | null;
  language?: string | null;
  owner_id?: string | null;
  active_version_id?: string | null;
  tags: string[];
  is_archived: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  active_version?: DocumentVersion | null;
}

export interface UploadTicket {
  upload_id: string;
  collection_id: string;
  storage_key: string;
  url: string;
  fields: Record<string, string>;
  max_bytes: number;
  expires_at: string;
  declared_mime: string;
  original_filename: string;
}

export interface RegisteredDocument {
  document_id: string;
  version_id: string;
  status: IngestStatus | string;
  job_id?: string | null;
  poll?: string;
  duplicate_of?: string | null;
}

export type StreamEventName =
  | "meta"
  | "citations"
  | "token"
  | "usage"
  | "refusal"
  | "clarify"
  | "done"
  | "error"
  | "heartbeat";

export interface MetaEvent {
  message_id?: string | null;
  trace_id?: string | null;
  model: string;
  mode: string;
  intent: string;
  rewritten?: string[];
  retrieved?: number;
  reranked?: number;
}

export interface CitationsEvent {
  citations: Citation[];
}

export interface TokenEvent {
  delta: string;
}

export interface UsageEvent {
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  ttft_ms?: number | null;
  total_ms: number;
  confidence: number;
  grounded: boolean;
}

export interface RefusalEvent {
  reason: string;
  message: string;
  detail?: string;
  nearest_documents?: Array<{ title: string; score: number }>;
  suggestions?: string[];
  escalation_available?: boolean;
}

export interface ClarifyEvent {
  question: string;
  options?: string[];
}

export interface DoneEvent {
  status: string;
  message_id?: string | null;
  citations_used?: number[];
}

export interface ErrorEvent {
  code: string;
  message: string;
  retryable?: boolean;
}

export type StreamEvent =
  | { event: "meta"; data: MetaEvent }
  | { event: "citations"; data: CitationsEvent }
  | { event: "token"; data: TokenEvent }
  | { event: "usage"; data: UsageEvent }
  | { event: "refusal"; data: RefusalEvent }
  | { event: "clarify"; data: ClarifyEvent }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: ErrorEvent }
  | { event: "heartbeat"; data: unknown };
