import { useEffect, useRef, useState } from 'react';
import { Copy, Send, Sparkles, User } from 'lucide-react';
import { motion } from 'motion/react';

import type { ChatMessage } from '../types';

type ChatAreaProps = {
  conversationId: string | null;
  messages: ChatMessage[];
  sending: boolean;
  statusMessage: string | null;
  onSend: (content: string) => Promise<boolean>;
};

const formatRole = (role: ChatMessage['role']): string => {
  if (role === 'assistant') {
    return 'Assistant';
  }
  if (role === 'user') {
    return 'You';
  }
  return 'System';
};

export function ChatArea({ conversationId, messages, sending, statusMessage, onSend }: ChatAreaProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || sending || !conversationId) {
      return;
    }
    const ok = await onSend(trimmed);
    if (ok) {
      setInput('');
    }
  };

  const handleCopy = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      return;
    }
  };

  return (
    <div className="relative flex flex-1 flex-col">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-white/[0.02] via-transparent to-white/[0.02]" />

      <div className="relative z-10 flex-1 overflow-y-auto px-6 py-8">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 ? (
            <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-6 text-center text-sm text-white/50">
              No messages in this chat yet.
            </div>
          ) : (
            messages.map((message, index) => {
              const isAssistant = message.role === 'assistant';
              const isUser = message.role === 'user';
              return (
                <motion.div
                  key={`${message.role}-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: Math.min(index * 0.03, 0.3) }}
                  className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  {isAssistant ? (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/20 bg-white/10">
                      <Sparkles className="h-4 w-4 text-white" />
                    </div>
                  ) : null}

                  <div className={`flex max-w-[74%] flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
                    <div
                      className={`rounded-2xl px-4 py-3 ${
                        isUser ? 'bg-white text-black' : 'bg-white/5 text-white/90'
                      }`}
                    >
                      <div className="mb-1 text-[10px] uppercase tracking-[0.2em] opacity-60">
                        {formatRole(message.role)}
                      </div>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        void handleCopy(message.content);
                      }}
                      className="rounded-md p-1 text-white/40 transition-colors hover:bg-white/5 hover:text-white/80"
                      title="Copy"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  {isUser ? (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/20 bg-white/10">
                      <User className="h-4 w-4 text-white/60" />
                    </div>
                  ) : null}
                </motion.div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="relative z-10 border-t border-white/5 bg-zinc-900/50 p-6 backdrop-blur-xl">
        <div className="mx-auto max-w-3xl">
          <div className="relative">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void handleSend();
                }
              }}
              placeholder={conversationId ? 'Type your message... (Shift+Enter for new line)' : 'Create a chat first'}
              className="w-full resize-none rounded-xl border border-white/10 bg-white/5 px-4 py-3 pr-12 text-white placeholder-white/30 transition-all duration-200 focus:border-white/20 focus:outline-none focus:ring-2 focus:ring-white/20"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '200px' }}
              disabled={!conversationId || sending}
            />
            <button
              type="button"
              onClick={() => {
                void handleSend();
              }}
              disabled={!conversationId || sending || !input.trim()}
              className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg bg-white transition-all duration-200 hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <Send className="h-4 w-4 text-black" />
            </button>
          </div>
          <div className="mt-2 min-h-5 text-xs text-white/45">
            {statusMessage || (sending ? 'Sending...' : '')}
          </div>
        </div>
      </div>
    </div>
  );
}
