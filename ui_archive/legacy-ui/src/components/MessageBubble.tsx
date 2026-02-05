import type { Message } from "../types";

type MessageBubbleProps = {
  message: Message;
  onEditMessage: (content: string) => void;
  onRefreshMessage: (content: string) => void;
};

const roleLabel = (role: Message["role"]): string => {
  if (role === "user") {
    return "You";
  }
  if (role === "assistant") {
    return "Assistant";
  }
  return "System";
};

export default function MessageBubble({
  message,
  onEditMessage,
  onRefreshMessage,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const isSystem = message.role === "system";

  const wrapperClass = isSystem
    ? "justify-center"
    : isUser
      ? "justify-end"
      : "justify-start";

  const bubbleClass = isSystem
    ? "max-w-[80%] rounded-2xl border border-neutral-700/60 bg-neutral-800/60 px-4 py-3 text-xs text-neutral-200"
    : isUser
      ? "max-w-[78%] rounded-3xl bg-neutral-200 px-4 py-3 text-sm text-neutral-950 shadow-lg shadow-black/20"
      : "max-w-[78%] rounded-3xl border border-neutral-700/70 bg-neutral-900/70 px-4 py-3 text-sm text-neutral-100 shadow-lg shadow-black/10";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
    } catch {
      return;
    }
  };

  const handleListen = () => {
    if (!("speechSynthesis" in window) || !message.content.trim()) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message.content);
    window.speechSynthesis.speak(utterance);
  };

  const handleStopListen = () => {
    if (!("speechSynthesis" in window)) {
      return;
    }
    window.speechSynthesis.cancel();
  };

  return (
    <div className={`flex ${wrapperClass}`}>
      <div className={bubbleClass}>
        <div className="flex items-start justify-between gap-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-400">
            {roleLabel(message.role)}
          </div>
          {!isSystem ? (
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => {
                  void handleCopy();
                }}
                className="rounded-md border border-neutral-700/70 px-1.5 py-0.5 text-[10px] text-neutral-400 hover:text-neutral-100"
                title="Copy"
              >
                ⧉
              </button>
              <button
                type="button"
                onClick={() => onEditMessage(message.content)}
                className="rounded-md border border-neutral-700/70 px-1.5 py-0.5 text-[10px] text-neutral-400 hover:text-neutral-100"
                title="Edit"
              >
                ✎
              </button>
              {isAssistant ? (
                <>
                  <button
                    type="button"
                    onClick={() => onRefreshMessage(message.content)}
                    className="rounded-md border border-neutral-700/70 px-1.5 py-0.5 text-[10px] text-neutral-400 hover:text-neutral-100"
                    title="Refresh"
                  >
                    ↻
                  </button>
                  <button
                    type="button"
                    onClick={handleListen}
                    className="rounded-md border border-neutral-700/70 px-1.5 py-0.5 text-[10px] text-neutral-400 hover:text-neutral-100"
                    title="Listen"
                  >
                    🔊
                  </button>
                  <button
                    type="button"
                    onClick={handleStopListen}
                    className="rounded-md border border-neutral-700/70 px-1.5 py-0.5 text-[10px] text-neutral-400 hover:text-neutral-100"
                    title="Stop"
                  >
                    ■
                  </button>
                </>
              ) : null}
            </div>
          ) : null}
        </div>
        <div className="mt-2 whitespace-pre-wrap leading-relaxed">{message.content}</div>
      </div>
    </div>
  );
}
