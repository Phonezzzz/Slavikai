import type { Message } from "../types";

type MessageBubbleProps = {
  message: Message;
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

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  const wrapperClass = isSystem
    ? "justify-center"
    : isUser
    ? "justify-end"
    : "justify-start";

  const bubbleClass = isSystem
    ? "max-w-[80%] rounded-2xl border border-slate-700/60 bg-slate-800/60 px-4 py-3 text-xs text-slate-200"
    : isUser
    ? "max-w-[78%] rounded-3xl bg-gradient-to-br from-indigo-500 to-indigo-700 px-4 py-3 text-sm text-white shadow-lg shadow-indigo-500/20"
    : "max-w-[78%] rounded-3xl border border-slate-700/70 bg-slate-900/70 px-4 py-3 text-sm text-slate-100 shadow-lg shadow-black/10";

  const labelClass = isSystem
    ? "text-[10px] uppercase tracking-[0.2em] text-slate-400"
    : "text-[10px] uppercase tracking-[0.2em] text-slate-400";

  return (
    <div className={`flex ${wrapperClass}`}>
      <div className={bubbleClass}>
        <div className={labelClass}>{roleLabel(message.role)}</div>
        <div className="mt-2 whitespace-pre-wrap leading-relaxed">
          {message.content}
        </div>
      </div>
    </div>
  );
}
