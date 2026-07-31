"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";

import {
  createUploadTicket,
  listCollections,
  listDocuments,
  putToStorage,
  registerDocument,
} from "@/features/admin/lib/documentsApi";
import { useAuth } from "@/features/auth/hooks/useAuth";
import type { Collection, Document, Visibility } from "@/shared/api/types";
import { formatBytes, formatRelativeTime } from "@/shared/lib/format";
import { Badge } from "@/shared/ui/Badge";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { Skeleton } from "@/shared/ui/Skeleton";

function statusTone(status?: string) {
  switch (status) {
    case "indexed":
      return "success" as const;
    case "failed":
    case "quarantined":
      return "danger" as const;
    case "pending":
    case "scanning":
    case "parsing":
    case "chunking":
    case "embedding":
    case "indexing":
      return "amber" as const;
    default:
      return "neutral" as const;
  }
}

export function DocumentsAdmin() {
  const { user, logout } = useAuth();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [showUpload, setShowUpload] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [collectionId, setCollectionId] = useState("");
  const [visibility, setVisibility] = useState<Visibility>("internal");
  const [uploading, setUploading] = useState(false);
  const [uploadStep, setUploadStep] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docs, cols] = await Promise.all([
        listDocuments({ q: query || undefined, limit: 50 }),
        listCollections(),
      ]);
      setDocuments(Array.isArray(docs) ? docs : docs.items);
      setCollections(cols);
      if (!collectionId && cols[0]) setCollectionId(cols[0].id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, [query, collectionId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const filtered = useMemo(() => documents, [documents]);

  async function onUpload(event: FormEvent) {
    event.preventDefault();
    if (!file || !collectionId) return;
    setUploading(true);
    setError(null);
    try {
      setUploadStep("Requesting upload ticket…");
      const ticket = await createUploadTicket({
        filename: file.name,
        size_bytes: file.size,
        declared_mime: file.type || "application/octet-stream",
        collection_id: collectionId,
      });

      setUploadStep("Uploading bytes to storage…");
      await putToStorage(ticket, file);

      setUploadStep("Registering document…");
      await registerDocument({
        upload_id: ticket.upload_id,
        title: title.trim() || file.name,
        visibility,
      });

      setShowUpload(false);
      setFile(null);
      setTitle("");
      setUploadStep(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      setUploadStep(null);
    }
  }

  return (
    <div className="min-h-dvh bg-ink-950 text-ink-50">
      <header className="flex items-center justify-between border-b border-ink-800 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-4">
          <Link href="/chat" className="font-display text-xl text-amber-300">
            Aegis
          </Link>
          <span className="text-sm text-ink-400">Documents</span>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="hidden text-ink-400 sm:inline">
            {user?.full_name || user?.email}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void logout().then(() => (window.location.href = "/login"))}
          >
            Sign out
          </Button>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="font-display text-3xl text-ink-50">Documents</h1>
            <p className="mt-1 text-sm text-ink-400">
              Upload, track ingestion, and keep the corpus searchable.
            </p>
          </div>
          <Button variant="amber" onClick={() => setShowUpload(true)}>
            + Upload
          </Button>
        </div>

        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
          <Input
            label="Search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Title or keyword"
            className="sm:max-w-sm"
          />
          <div className="flex items-end">
            <Button variant="secondary" onClick={() => void refresh()}>
              Refresh
            </Button>
          </div>
        </div>

        {error ? (
          <p className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {error}
          </p>
        ) : null}

        <div className="mt-6 overflow-x-auto rounded-xl border border-ink-800">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-ink-800 bg-ink-900/60 text-xs uppercase tracking-[0.12em] text-ink-500">
              <tr>
                <th className="px-4 py-3 font-medium">Title</th>
                <th className="px-4 py-3 font-medium">Visibility</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Chunks</th>
                <th className="px-4 py-3 font-medium">Updated</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className="border-b border-ink-900">
                      <td className="px-4 py-3" colSpan={5}>
                        <Skeleton className="h-6 w-full" />
                      </td>
                    </tr>
                  ))
                : filtered.map((doc) => {
                    const status = doc.active_version?.status ?? "pending";
                    return (
                      <tr
                        key={doc.id}
                        className="border-b border-ink-900/80 transition hover:bg-ink-900/40"
                      >
                        <td className="px-4 py-3">
                          <p className="font-medium text-ink-100">{doc.title}</p>
                          <p className="text-xs text-ink-500">
                            {doc.tags?.length ? doc.tags.join(", ") : "—"}
                          </p>
                        </td>
                        <td className="px-4 py-3 capitalize text-ink-300">
                          {doc.visibility}
                        </td>
                        <td className="px-4 py-3">
                          <Badge tone={statusTone(status)}>{status}</Badge>
                        </td>
                        <td className="px-4 py-3 tabular-nums text-ink-300">
                          {doc.active_version?.chunk_count ?? "—"}
                        </td>
                        <td className="px-4 py-3 text-ink-400">
                          {formatRelativeTime(doc.updated_at ?? doc.created_at)}
                        </td>
                      </tr>
                    );
                  })}
              {!loading && filtered.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-ink-500">
                    No documents yet. Upload one to get started.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>

      {showUpload ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <form
            onSubmit={onUpload}
            className="w-full max-w-lg animate-fade-up rounded-2xl border border-ink-700 bg-ink-950 p-6 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="font-display text-2xl text-ink-50">Upload document</h2>
                <p className="mt-1 text-sm text-ink-400">
                  Ticket → storage PUT → register.
                </p>
              </div>
              <button
                type="button"
                className="text-ink-400 hover:text-ink-100"
                onClick={() => setShowUpload(false)}
                aria-label="Close"
              >
                ×
              </button>
            </div>

            <div className="mt-5 space-y-4">
              <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-ink-600 bg-ink-900/40 px-4 py-8 text-center transition hover:border-amber-400/40">
                <span className="text-sm text-ink-200">
                  {file ? file.name : "Drop a file or click to choose"}
                </span>
                {file ? (
                  <span className="mt-1 text-xs text-ink-500">{formatBytes(file.size)}</span>
                ) : null}
                <input
                  type="file"
                  className="sr-only"
                  onChange={(e) => {
                    const next = e.target.files?.[0] ?? null;
                    setFile(next);
                    if (next && !title) setTitle(next.name.replace(/\.[^.]+$/, ""));
                  }}
                />
              </label>

              <Input
                label="Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                required
              />

              <label className="flex flex-col gap-1.5 text-sm text-ink-200">
                <span className="font-medium text-ink-100">Collection</span>
                <select
                  className="h-11 rounded-lg border border-ink-600 bg-ink-900 px-3 text-ink-50 focus:border-amber-400/70 focus:outline-none focus:ring-2 focus:ring-amber-400/25"
                  value={collectionId}
                  onChange={(e) => setCollectionId(e.target.value)}
                  required
                >
                  {collections.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-col gap-1.5 text-sm text-ink-200">
                <span className="font-medium text-ink-100">Visibility</span>
                <select
                  className="h-11 rounded-lg border border-ink-600 bg-ink-900 px-3 text-ink-50 focus:border-amber-400/70 focus:outline-none focus:ring-2 focus:ring-amber-400/25"
                  value={visibility}
                  onChange={(e) => setVisibility(e.target.value as Visibility)}
                >
                  {(
                    [
                      "public",
                      "customer",
                      "internal",
                      "confidential",
                      "restricted",
                    ] as Visibility[]
                  ).map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {uploadStep ? (
              <p className="mt-4 text-sm text-amber-200/90">{uploadStep}</p>
            ) : null}

            <div className="mt-6 flex justify-end gap-2">
              <Button
                type="button"
                variant="ghost"
                onClick={() => setShowUpload(false)}
                disabled={uploading}
              >
                Cancel
              </Button>
              <Button type="submit" variant="amber" loading={uploading} disabled={!file}>
                Upload
              </Button>
            </div>
          </form>
        </div>
      ) : null}
    </div>
  );
}
