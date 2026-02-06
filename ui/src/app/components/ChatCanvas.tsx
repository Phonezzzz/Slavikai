import { ChevronLeft, ChevronRight, FileText, FolderOpen } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import JSZip from 'jszip';
import { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import type { ChatMessage } from '../types';

type CanvasTab = 'output' | 'files';

type ExportFormat = 'txt' | 'md' | 'json' | 'zip';

type ArtifactItem = {
  id: string;
  title: string;
  content: string;
};

interface ChatCanvasProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  messages: ChatMessage[];
}

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

const findLatestOutput = (messages: ChatMessage[]): string => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === 'assistant' && message.content.trim()) {
      return message.content;
    }
  }
  return '';
};

const buildArtifacts = (messages: ChatMessage[]): ArtifactItem[] => {
  let counter = 0;
  return messages
    .filter((message) => message.role === 'assistant' && message.content.trim())
    .map((message) => {
      counter += 1;
      const firstLine = message.content.split('\n').find((line) => line.trim());
      const title = firstLine ? firstLine.trim().slice(0, 60) : `Output ${counter}`;
      return {
        id: `artifact-${counter}`,
        title,
        content: message.content,
      };
    });
};

const formatFileName = (title: string): string =>
  title
    .toLowerCase()
    .replace(/[^a-z0-9-_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '') || 'artifact';

const buildContentForFormat = (artifact: ArtifactItem, format: Exclude<ExportFormat, 'zip'>): string => {
  if (format === 'json') {
    return JSON.stringify({ title: artifact.title, content: artifact.content }, null, 2);
  }
  if (format === 'txt') {
    return artifact.content.replace(/\r\n/g, '\n');
  }
  return artifact.content;
};

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

export function ChatCanvas({ collapsed, onToggleCollapse, messages }: ChatCanvasProps) {
  const [activeTab, setActiveTab] = useState<CanvasTab>('output');
  const [format, setFormat] = useState<ExportFormat>('md');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const output = useMemo(() => findLatestOutput(messages), [messages]);
  const hasOutput = output.trim().length > 0;
  const artifacts = useMemo(() => buildArtifacts(messages), [messages]);

  const handleDownloadOne = async (artifact: ArtifactItem) => {
    setStatus(null);
    try {
      if (format === 'zip') {
        const zip = new JSZip();
        const fileName = `${formatFileName(artifact.title)}.md`;
        zip.file(fileName, artifact.content);
        const blob = await zip.generateAsync({ type: 'blob' });
        downloadBlob(blob, `${formatFileName(artifact.title)}.zip`);
        return;
      }
      const content = buildContentForFormat(artifact, format);
      const extension = format;
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      downloadBlob(blob, `${formatFileName(artifact.title)}.${extension}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Не удалось скачать файл.';
      setStatus(message);
    }
  };

  const handleDownloadAll = async () => {
    setStatus(null);
    if (artifacts.length === 0 || busy) {
      return;
    }
    setBusy(true);
    try {
      const zip = new JSZip();
      const exportFormat: Exclude<ExportFormat, 'zip'> = format === 'zip' ? 'md' : format;
      artifacts.forEach((artifact, index) => {
        const content = buildContentForFormat(artifact, exportFormat);
        const filename = `${formatFileName(artifact.title) || `artifact-${index + 1}`}.${
          exportFormat
        }`;
        zip.file(filename, content);
      });
      const blob = await zip.generateAsync({ type: 'blob' });
      downloadBlob(blob, 'chat-artifacts.zip');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Не удалось сформировать архив.';
      setStatus(message);
    } finally {
      setBusy(false);
    }
  };

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
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setActiveTab('output')}
                  className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide transition-colors ${
                    activeTab === 'output'
                      ? 'border-white/30 bg-white/15 text-white'
                      : 'border-white/10 bg-white/5 text-white/60 hover:text-white'
                  }`}
                >
                  <FileText className="h-3.5 w-3.5" />
                  Preview / Output
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('files')}
                  className={`flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide transition-colors ${
                    activeTab === 'files'
                      ? 'border-white/30 bg-white/15 text-white'
                      : 'border-white/10 bg-white/5 text-white/60 hover:text-white'
                  }`}
                >
                  <FolderOpen className="h-3.5 w-3.5" />
                  Files
                </button>
              </div>
              <div className="mt-2 text-xs text-white/40">
                {activeTab === 'output'
                  ? 'Последний результат агента из текущей сессии.'
                  : 'Экспорт артефактов, которые сгенерировал агент.'}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto px-4 pb-6">
              {activeTab === 'output' && (
                <div className="space-y-4">
                  {!hasOutput ? (
                    <div className="rounded-lg border border-white/10 bg-black/40 p-4 text-sm text-white/60">
                      Пока нет результата. После ответа ассистента он появится здесь.
                    </div>
                  ) : (
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                      {output}
                    </ReactMarkdown>
                  )}
                </div>
              )}

              {activeTab === 'files' && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-white/10 bg-black/40 p-3 text-xs text-white/60">
                    <div className="flex items-center gap-2">
                      <span className="text-white/70">Формат:</span>
                      <select
                        value={format}
                        onChange={(event) => setFormat(event.target.value as ExportFormat)}
                        className="rounded-md border border-white/10 bg-black/50 px-2 py-1 text-xs text-white/80"
                      >
                        <option value="txt">txt</option>
                        <option value="md">md</option>
                        <option value="json">json</option>
                        <option value="zip">zip</option>
                      </select>
                    </div>
                    <button
                      type="button"
                      onClick={() => void handleDownloadAll()}
                      disabled={busy || artifacts.length === 0}
                      className="rounded-md border border-white/10 bg-white/10 px-3 py-1 text-xs text-white/80 transition-colors hover:bg-white/20 disabled:opacity-40"
                    >
                      {busy ? 'Готовлю архив...' : 'Скачать всё'}
                    </button>
                  </div>

                  {status && (
                    <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-2 text-xs text-rose-200">
                      {status}
                    </div>
                  )}

                  {artifacts.length === 0 ? (
                    <div className="rounded-lg border border-white/10 bg-black/40 p-4 text-sm text-white/60">
                      Артефактов пока нет.
                    </div>
                  ) : (
                    artifacts.map((artifact) => (
                      <div
                        key={artifact.id}
                        className="rounded-lg border border-white/10 bg-black/40 p-3"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="truncate text-sm text-white">{artifact.title}</div>
                            <div className="text-xs text-white/40">
                              {artifact.content.length.toLocaleString()} символов
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => void handleDownloadOne(artifact)}
                            className="rounded-md border border-white/10 bg-white/10 px-3 py-1 text-xs text-white/80 transition-colors hover:bg-white/20"
                          >
                            Скачать
                          </button>
                        </div>
                      </div>
                    ))
                  )}
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
