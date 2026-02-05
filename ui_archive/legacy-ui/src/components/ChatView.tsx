import type { Message } from "../types";
import ChatInput from "./ChatInput";
import MessageList from "./MessageList";

type ChatViewProps = {
  messages: Message[];
  activity: string[];
  input: string;
  sending: boolean;
  onInputChange: (value: string) => void;
  onSend: (contentOverride?: string) => void;
};

export default function ChatView({
  messages,
  activity,
  input,
  sending,
  onInputChange,
  onSend,
}: ChatViewProps) {
  return (
    <section className="flex flex-col gap-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/60 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-neutral-500">Chat</p>
          <h2 className="text-lg font-semibold text-neutral-100">Conversation</h2>
        </div>
        <span className="text-xs text-neutral-400">{messages.length} messages</span>
      </div>
      <MessageList messages={messages} />
      <div className="rounded-2xl border border-neutral-800/80 bg-neutral-950/50 px-3 py-2">
        <div className="text-[11px] uppercase tracking-[0.2em] text-neutral-500">Agent activity</div>
        {activity.length === 0 ? (
          <div className="mt-1 text-xs text-neutral-500">Awaiting SSE activity...</div>
        ) : (
          <div className="mt-1 max-h-20 space-y-1 overflow-y-auto font-mono text-[11px] text-neutral-400">
            {activity.slice(-8).map((line, index) => (
              <div key={`${line}-${index}`}>{line}</div>
            ))}
          </div>
        )}
      </div>
      <ChatInput value={input} onChange={onInputChange} onSend={onSend} sending={sending} />
    </section>
  );
}
