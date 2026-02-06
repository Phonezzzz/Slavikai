import { ChevronLeft, ChevronRight, Copy, Download, FileText } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';

import type { CanvasOutput } from '../types';

type CanvasFormat = 'txt' | 'md' | 'py' | 'json' | 'yaml';

interface ChatCanvasProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  canvas: CanvasOutput | null;
}

const FORMAT_OPTIONS: Array<{ value: CanvasFormat; label: string }> = [
  { value: 'txt', label: 'txt' },
  { value: 'md', label: 'md' },
  { value: 'py', label: 'py' },
  { value: 'json', label: 'json' },
  { value: 'yaml', label: 'yaml' },
];

const normalizeContent = (content: string): string => content.replace(/\r\n/g, '\n');

const inferFormatFromFilename = (filename: string | null): CanvasFormat | null => {
  if (!filename) {
    return null;
  }
  const parts = filename.toLowerCase().split('.');
  if (parts.length < 2) {
    return null;
  }
  const ext = parts[parts.length - 1];
  if (ext === 'md') return 'md';
  if (ext === 'txt') return 'txt';
  if (ext === 'py') return 'py';
  if (ext === 'json') return 'json';
  if (ext === 'yaml' || ext === 'yml') return 'yaml';
  return null;
};

const inferFormatFromMime = (format: string | null): CanvasFormat | null => {
  if (!format) {
    return null;
  }
  const normalized = format.toLowerCase();
  if (normalized.includes('markdown')) return 'md';
  if (normalized.includes('json')) return 'json';
  if (normalized.includes('yaml')) return 'yaml';
  if (normalized.includes('python')) return 'py';
  if (normalized.includes('plain')) return 'txt';
  return null;
};

const inferLanguage = (format: string | null): string | null => {
  if (!format) {
    return null;
  }
  const normalized = format.toLowerCase();
  if (normalized.includes('python')) return 'python';
  if (normalized.includes('json')) return 'json';
  if (normalized.includes('yaml')) return 'yaml';
  if (normalized.includes('markdown')) return 'markdown';
  return null;
};

const formatToLanguage = (format: CanvasFormat): string | null => {
  if (format === 'py') return 'python';
  if (format === 'json') return 'json';
  if (format === 'yaml') return 'yaml';
  return null;
};

const resolveDefaultFormat = (canvas: CanvasOutput | null): CanvasFormat => {
  if (!canvas) {
    return 'txt';
  }
  const byName = inferFormatFromFilename(canvas.suggestedFilename);
  if (byName) {
    return byName;
  }
  const byMime = inferFormatFromMime(canvas.format);
  if (byMime) {
    return byMime;
  }
  return 'txt';
};

const resolveFilename = (canvas: CanvasOutput | null, format: CanvasFormat): string => {
  const extension = format;
  const raw = canvas?.suggestedFilename?.trim();
  if (!raw) {
    return `output.${extension}`;
  }
  const sanitized = raw.split('/').pop()?.split('\\').pop() || raw;
  const lower = sanitized.toLowerCase();
  if (lower.endsWith(`.${extension}`)) {
    return sanitized;
  }
  const base = sanitized.replace(/\.[^.]+$/, '');
  return `${base || 'output'}.${extension}`;
};

const shouldRenderMarkdown = (canvas: CanvasOutput | null, content: string): boolean => {
  if (!content.trim()) {
    return false;
  }
  if (content.includes('```')) {
    return true;
  }
  if (!canvas?.format) {
    return false;
  }
  return canvas.format.toLowerCase().includes('markdown');
};

const contentForDownload = (
  content: string,
  format: CanvasFormat,
  canvas: CanvasOutput | null,
): string => {
  const normalized = normalizeContent(content);
  if (format !== 'md') {
    return normalized;
  }
  if (normalized.includes('```')) {
    return normalized;
  }
  const lang = inferLanguage(canvas?.format);
  if (!lang || lang === 'markdown') {
    return normalized;
  }
  return `\`\`\`${lang}\n${normalized}\n\`\`\``;
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

const MIME_BY_FORMAT: Record<CanvasFormat, string> = {
  txt: 'text/plain;charset=utf-8',
  md: 'text/markdown;charset=utf-8',
  py: 'text/x-python;charset=utf-8',
  json: 'application/json;charset=utf-8',
  yaml: 'text/yaml;charset=utf-8',
};

export function ChatCanvas({ collapsed, onToggleCollapse, canvas }: ChatCanvasProps) {
  const [format, setFormat] = useState<CanvasFormat>(() => resolveDefaultFormat(canvas));
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    setFormat(resolveDefaultFormat(canvas));
    setStatus(null);
  }, [canvas?.updatedAt, canvas?.format, canvas?.suggestedFilename]);

  const content = canvas?.content ?? '';
  const hasOutput = content.trim().length > 0;
  const updatedAt = canvas?.updatedAt ?? null;
  const normalizedContent = normalizeContent(content);
  const renderMarkdown = shouldRenderMarkdown(canvas, normalizedContent);
  const selectedLanguage = formatToLanguage(format);
  const shouldWrapCode =
    !renderMarkdown &&
    selectedLanguage !== null &&
    selectedLanguage !== 'markdown' &&
    !normalizedContent.includes('```');
  const markdownSource = shouldWrapCode
    ? `\`\`\`${selectedLanguage}\n${normalizedContent}\n\`\`\``
    : normalizedContent;
  const renderMarkdownOutput = renderMarkdown || shouldWrapCode;

  const infoLine = useMemo(() => {
    if (!updatedAt) {
      return null;
    }
    const parsed = Date.parse(updatedAt);
    if (Number.isNaN(parsed)) {
      return updatedAt;
    }
    return new Date(parsed).toLocaleString();
  }, [updatedAt]);

  const handleCopy = async () => {
    if (!hasOutput) {
      return;
    }
    setStatus(null);
    try {
      await navigator.clipboard.writeText(content);
      setStatus('Скопировано');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Не удалось скопировать.';
      setStatus(message);
    }
  };

  const handleDownload = () => {
    if (!hasOutput) {
      return;
    }
    setStatus(null);
    try {
      const filename = resolveFilename(canvas, format);
      const output = contentForDownload(content, format, canvas);
      const blob = new Blob([output], { type: MIME_BY_FORMAT[format] });
      downloadBlob(blob, filename);
      setStatus(`Скачано: ${filename}`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Не удалось скачать.';
      setStatus(message);
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
            <div className="border-b border-white/10 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-white/70">
                  <FileText className="h-3.5 w-3.5 text-white/60" />
                  Canvas
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void handleCopy()}
                    disabled={!hasOutput}
                    className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-white/70 transition-colors hover:bg-white/10 disabled:opacity-40"
                  >
                    <Copy className="h-3.5 w-3.5" />
                    Copy
                  </button>
                  <button
                    type="button"
                    onClick={handleDownload}
                    disabled={!hasOutput}
                    className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-white/70 transition-colors hover:bg-white/10 disabled:opacity-40"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download
                  </button>
                  <select
                    value={format}
                    onChange={(event) => setFormat(event.target.value as CanvasFormat)}
                    className="rounded-md border border-white/10 bg-black/40 px-2 py-1 text-xs text-white/80"
                  >
                    {FORMAT_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value} className="bg-zinc-950 text-white">
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] text-white/40">
                <span>
                  {hasOutput
                    ? 'Последний длинный результат этой сессии.'
                    : 'Пока нет результата.'}
                </span>
                {infoLine ? <span>{infoLine}</span> : null}
              </div>
              {status ? <div className="mt-2 text-[11px] text-white/60">{status}</div> : null}
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-5">
              {!hasOutput ? (
                <div className="rounded-lg border border-white/10 bg-black/40 p-4 text-sm text-white/60">
                  Canvas обновится после того, как агент пришлёт длинный результат.
                </div>
              ) : (
                <div className="rounded-lg border border-white/10 bg-black/40 p-4">
                  {renderMarkdownOutput ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        h1: ({ children }) => (
                          <h1 className="mb-3 text-base font-semibold text-white/90">{children}</h1>
                        ),
                        h2: ({ children }) => (
                          <h2 className="mb-2 text-sm font-semibold text-white/85">{children}</h2>
                        ),
                        h3: ({ children }) => (
                          <h3 className="mb-2 text-sm font-semibold text-white/80">{children}</h3>
                        ),
                        p: ({ children }) => (
                          <p className="mb-3 text-sm leading-relaxed text-white/75">{children}</p>
                        ),
                        ul: ({ children }) => (
                          <ul className="mb-3 list-disc space-y-1 pl-5 text-sm text-white/75">
                            {children}
                          </ul>
                        ),
                        ol: ({ children }) => (
                          <ol className="mb-3 list-decimal space-y-1 pl-5 text-sm text-white/75">
                            {children}
                          </ol>
                        ),
                        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                        a: ({ href, children }) => (
                          <a
                            href={href}
                            className="text-emerald-300 underline underline-offset-4 transition-colors hover:text-emerald-200"
                            target="_blank"
                            rel="noreferrer"
                          >
                            {children}
                          </a>
                        ),
                        code: ({ inline, className, children }) => {
                          if (!inline) {
                            return (
                              <pre className="mb-3 overflow-x-auto rounded-lg border border-white/10 bg-black/60 p-4 text-xs leading-relaxed">
                                <code className={className}>{children}</code>
                              </pre>
                            );
                          }
                          const text = String(children ?? '').replace(/\n$/, '');
                          if (inline) {
                            return (
                              <code className="rounded bg-white/10 px-1.5 py-0.5 text-[0.85em] text-emerald-200">
                                {text}
                              </code>
                            );
                          }
                        },
                        blockquote: ({ children }) => (
                          <blockquote className="mb-3 border-l-2 border-white/10 pl-3 text-sm text-white/60">
                            {children}
                          </blockquote>
                        ),
                      }}
                    >
                      {markdownSource}
                    </ReactMarkdown>
                  ) : (
                    <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-emerald-200">
                      {normalizedContent}
                    </pre>
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
