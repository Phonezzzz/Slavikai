import { useEffect, useRef } from "react";

type ChatInputProps = {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  sending: boolean;
};

export default function ChatInput({ value, onChange, onSend, sending }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) {
      return;
    }
    element.style.height = "";
    element.style.height = `${element.scrollHeight}px`;
  }, [value]);

  return (
    <div className="rounded-3xl border border-slate-800/80 bg-slate-950/60 p-3">
      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
          rows={1}
          placeholder="Send a message"
          className="min-h-[44px] flex-1 resize-none rounded-2xl border border-slate-800/80 bg-slate-900/60 px-4 py-3 text-sm text-slate-100 outline-none focus:ring-2 focus:ring-indigo-500/60"
        />
        <button
          type="button"
          disabled={sending || value.trim().length === 0}
          onClick={onSend}
          className="flex h-11 items-center justify-center rounded-2xl bg-indigo-600 px-6 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition disabled:cursor-not-allowed disabled:opacity-50"
        >
          {sending ? "Sending..." : "Send"}
        </button>
      </div>
      <div className="mt-2 text-xs text-slate-500">
        Enter — отправить, Shift+Enter — новая строка.
      </div>
    </div>
  );
}
