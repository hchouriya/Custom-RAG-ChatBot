"use client";

import { DocumentsAdmin } from "@/features/admin/components/DocumentsAdmin";
import { RequireAuth } from "@/features/auth/components/RequireAuth";

export default function DocumentsPage() {
  return (
    <RequireAuth>
      <DocumentsAdmin />
    </RequireAuth>
  );
}
