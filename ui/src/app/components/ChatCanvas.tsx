import { ChevronLeft, ChevronRight, FileText } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import type { ChatMessage } from '../types';

interface ChatCanvasProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  messages: ChatMessage[];
}

const findLatestOutput = (messages: ChatMessage[]): string => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === 'assistant' && message.content.trim()) {
      return message.content;
    }
  }
  return '';
};

const markdownComponents: Components = {
  h1: ({ children }) => <h1 className="text-lg font-semibold text-white">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-semibold text-white">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold text-white">{children}</h3>,
  p: ({ children }) => <p className="leading-relaxed text-white/80">{children}</p>,
  ul: ({ children }) => <ul className="list-disc space-y-1 pl-5 text-white/80">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5 text-white/80">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-white/20 pl-3 text-white/70">{children}</blockquote>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      className="text-sky-300 underline decoration-sky-400/60 underline-offset-4"
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  ),
  code: ({ inline, className, children }) => {
    if (inline) {
      return (
        <code className="rounded bg-white/10 px-1 py-0.5 text-[12px] text-white/90">
          {children}
        </code>
      );
    }
    return (
      <code className={`block text-xs text-white/90 ${className ?? ''}`}>{children}</code>
    );
  },
  pre: ({ children }) => (
    <pre className="overflow-x-auto rounded-lg border border-white/10 bg-black/60 p-3 text-xs text-white/80">
      {children}
    </pre>
  ),
};

export function ChatCanvas({ collapsed, onToggleCollapse, messages }: ChatCanvasProps) {
  const output = useMemo(() => findLatestOutput(messages), [messages]);
  const hasOutput = output.trim().length > 0;

  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? '0px' : '480px' }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative bg-zinc-950"
    >
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex h-full w-[480px] flex-col"
          >
            <div className="p-4">
              <div className="flex items-center gap-2">
                <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white/70">
                  <FileText className="h-3.5 w-3.5" />
                  Preview / Output
                </span>
              </div>
              <div className="mt-2 text-xs text-white/40">
                Последний результат агента из текущей сессии.
              </div>
            </div>

            <div className="flex-1 overflow-y-auto px-4 pb-6">
              {!hasOutput ? (
                <div className="rounded-lg border border-white/10 bg-black/40 p-4 text-sm text-white/60">
                  Пока нет результата. После ответа ассистента он появится здесь.
                </div>
              ) : (
                <div className="space-y-4">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {output}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        onClick={onToggleCollapse}
        className="absolute -left-3 top-1/2 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full border border-white/10 bg-white/10 backdrop-blur-xl transition-all duration-200 hover:bg-white/20"
      >
        {collapsed ? (
          <ChevronLeft className="h-3.5 w-3.5 text-white/60" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-white/60" />
        )}
      </button>
    </motion.div>
  );
}
