import { useEffect, useRef } from "react";

import type { Message } from "../types";
import MessageBubble from "./MessageBubble";

type MessageListProps = {
  messages: Message[];
};

export default function MessageList({ messages }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
  }, [messages.length]);

  return (
    <div
      ref={containerRef}
      className="flex h-[55vh] flex-col gap-3 overflow-y-auto rounded-3xl border border-neutral-800/80 bg-neutral-950/40 px-4 py-4"
    >
      {messages.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-neutral-800/80 bg-neutral-900/60 px-4 py-6 text-sm text-neutral-400">
          Нет сообщений. Отправьте первое сообщение, чтобы начать диалог.
        </div>
      ) : (
        messages.map((message, index) => (
          <MessageBubble key={`${message.role}-${index}`} message={message} />
        ))
      )}
    </div>
  );
}
