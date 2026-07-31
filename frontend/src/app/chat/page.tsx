"use client";

import { ChatShell } from "@/features/chat/components/ChatShell";
import { RequireAuth } from "@/features/auth/components/RequireAuth";

export default function ChatPage() {
  return (
    <RequireAuth>
      <ChatShell />
    </RequireAuth>
  );
}
