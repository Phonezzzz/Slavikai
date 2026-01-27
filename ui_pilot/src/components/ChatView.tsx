import type { Message } from "../types";
import ChatInput from "./ChatInput";
import MessageList from "./MessageList";

type ChatViewProps = {
  messages: Message[];
  input: string;
  sending: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
};

export default function ChatView({
  messages,
  input,
  sending,
  onInputChange,
  onSend,
}: ChatViewProps) {
  return (
    <section className="flex flex-col gap-4 rounded-3xl border border-slate-800/80 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Chat</p>
          <h2 className="text-lg font-semibold text-slate-100">Conversation</h2>
        </div>
        <span className="text-xs text-slate-400">{messages.length} messages</span>
      </div>
      <MessageList messages={messages} />
      <ChatInput value={input} onChange={onInputChange} onSend={onSend} sending={sending} />
    </section>
  );
}
