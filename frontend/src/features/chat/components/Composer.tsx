"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { Button } from "@/shared/ui/Button";
import { Textarea } from "@/shared/ui/Textarea";

export function Composer({
  disabled,
  streaming,
  onSend,
  onStop,
}: {
  disabled?: boolean;
  streaming?: boolean;
  onSend: (value: string) => void;
  onStop: () => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (streaming) {
      onStop();
      return;
    }
    submit();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      className="mx-auto flex w-full max-w-3xl items-end gap-2 border-t border-ink-800 bg-ink-950/80 px-4 py-3 backdrop-blur"
    >
      <Textarea
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask about your documents…"
        disabled={disabled || streaming}
        rows={1}
        className="min-h-11 border-ink-700"
        aria-label="Message"
      />
      <Button
        type="submit"
        variant={streaming ? "danger" : "amber"}
        size="icon"
        aria-label={streaming ? "Stop generating" : "Send message"}
        disabled={!streaming && (!value.trim() || disabled)}
      >
        {streaming ? "■" : "↑"}
      </Button>
    </form>
  );
}
