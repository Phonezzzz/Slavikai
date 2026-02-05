import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot,
  Copy,
  Edit2,
  Globe,
  Mic,
  Paperclip,
  RefreshCcw,
  Send,
  Sparkles,
  Square,
  User,
  Volume2,
} from 'lucide-react';
import { motion } from 'motion/react';

import type { ChatMessage, ProviderModels, SelectedModel } from '../types';

type ChatAreaProps = {
  conversationId: string | null;
  messages: ChatMessage[];
  sending: boolean;
  statusMessage: string | null;
  selectedModel: SelectedModel | null;
  providerModels: ProviderModels[];
  modelsLoading: boolean;
  savingModel: boolean;
  onSend: (content: string) => Promise<boolean>;
  onSetModel: (provider: string, model: string) => Promise<boolean>;
};

type InputAttachment = {
  name: string;
  size: number;
  type: string;
  preview: string;
};

type SpeechRecognitionAlternativeLike = {
  transcript: string;
};

type SpeechRecognitionResultLike = {
  length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
};

type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
};

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

type SpeechWindow = Window & {
  SpeechRecognition?: SpeechRecognitionCtor;
  webkitSpeechRecognition?: SpeechRecognitionCtor;
};

const MAX_ATTACHMENTS = 6;
const MAX_PREVIEW_CHARS = 3000;

const canReadAsText = (file: File): boolean => {
  if (file.type.startsWith('text/')) {
    return true;
  }
  return /\.(txt|md|json|yaml|yml|toml|csv|py|ts|tsx|js|jsx|css|html|xml)$/i.test(file.name);
};

const formatSize = (bytes: number): string => {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const serializeAttachments = (attachments: InputAttachment[]): string => {
  const lines: string[] = ['[Attached files]'];
  attachments.forEach((item, index) => {
    lines.push(`#${index + 1} ${item.name} (${item.type || 'file'}, ${formatSize(item.size)})`);
    lines.push(item.preview.trim() ? item.preview.trim() : '(empty preview)');
    lines.push('---');
  });
  return lines.join('\n');
};

const readAttachment = async (file: File): Promise<InputAttachment> => {
  if (!canReadAsText(file)) {
    return {
      name: file.name,
      size: file.size,
      type: file.type || 'binary',
      preview: '[binary file attached]',
    };
  }
  try {
    const text = await file.text();
    return {
      name: file.name,
      size: file.size,
      type: file.type || 'text/plain',
      preview: text.slice(0, MAX_PREVIEW_CHARS),
    };
  } catch {
    return {
      name: file.name,
      size: file.size,
      type: file.type || 'text/plain',
      preview: '[failed to read file preview]',
    };
  }
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

const findPreviousUserMessage = (messages: ChatMessage[], index: number): string | null => {
  for (let current = index - 1; current >= 0; current -= 1) {
    const message = messages[current];
    if (message.role === 'user' && message.content.trim()) {
      return message.content;
    }
  }
  return null;
};

export function ChatArea({
  conversationId,
  messages,
  sending,
  statusMessage,
  selectedModel,
  providerModels,
  modelsLoading,
  savingModel,
  onSend,
  onSetModel,
}: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<InputAttachment[]>([]);
  const [webSearchMode, setWebSearchMode] = useState(false);
  const [recording, setRecording] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [modelProvider, setModelProvider] = useState('');
  const [modelId, setModelId] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const latestInputRef = useRef('');

  useEffect(() => {
    latestInputRef.current = input;
  }, [input]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      recognitionRef.current = null;
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  useEffect(() => {
    if (selectedModel) {
      setModelProvider(selectedModel.provider);
      setModelId(selectedModel.model);
      return;
    }
    if (providerModels.length > 0) {
      const fallbackProvider = providerModels[0].provider;
      const fallbackModel = providerModels[0].models[0] ?? '';
      setModelProvider(fallbackProvider);
      setModelId(fallbackModel);
      return;
    }
    setModelProvider('');
    setModelId('');
  }, [providerModels, selectedModel]);

  const speechCtor = useMemo(() => {
    const browserWindow = window as SpeechWindow;
    return browserWindow.SpeechRecognition ?? browserWindow.webkitSpeechRecognition ?? null;
  }, []);

  const modelsForProvider = useMemo(() => {
    const found = providerModels.find((item) => item.provider === modelProvider);
    return found?.models ?? [];
  }, [providerModels, modelProvider]);

  const sendDisabled =
    !conversationId || sending || (input.trim().length === 0 && attachments.length === 0);

  const handleAttachFiles = async (fileList: FileList | null) => {
    if (!fileList) {
      return;
    }
    const incoming = Array.from(fileList).slice(0, MAX_ATTACHMENTS);
    const parsed = await Promise.all(incoming.map((file) => readAttachment(file)));
    setAttachments((prev) => [...prev, ...parsed].slice(0, MAX_ATTACHMENTS));
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleToggleRecording = () => {
    if (!speechCtor) {
      return;
    }
    if (recording) {
      recognitionRef.current?.stop();
      setRecording(false);
      return;
    }

    const recognition = new speechCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'ru-RU';
    recognition.onresult = (event) => {
      const pieces: string[] = [];
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const alternative = result?.[0];
        if (alternative && typeof alternative.transcript === 'string') {
          pieces.push(alternative.transcript);
        }
      }
      const transcript = pieces.join(' ').trim();
      if (!transcript) {
        return;
      }
      const current = latestInputRef.current;
      const spacer = current.trim().length > 0 ? ' ' : '';
      setInput(`${current}${spacer}${transcript}`.trimStart());
    };
    recognition.onerror = () => {
      setRecording(false);
    };
    recognition.onend = () => {
      setRecording(false);
      recognitionRef.current = null;
    };
    recognitionRef.current = recognition;
    recognition.start();
    setRecording(true);
  };

  const handleSend = async (overrideContent?: string) => {
    const baseContent = typeof overrideContent === 'string' ? overrideContent : input.trim();
    const attachmentBlock = attachments.length > 0 ? serializeAttachments(attachments) : '';
    const composed = baseContent
      ? attachmentBlock
        ? `${baseContent}\n\n${attachmentBlock}`
        : baseContent
      : attachmentBlock;

    if (!composed.trim()) {
      return;
    }

    const payload =
      webSearchMode && !composed.trimStart().startsWith('/') ? `/web ${composed}` : composed;
    const ok = await onSend(payload);
    if (ok && typeof overrideContent !== 'string') {
      setInput('');
      setAttachments([]);
    }
  };

  const handleApplyModel = async () => {
    if (!conversationId || !modelProvider || !modelId || savingModel) {
      return;
    }
    const ok = await onSetModel(modelProvider, modelId);
    if (ok) {
      setModelOpen(false);
    }
  };

  const handleCopy = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      return;
    }
  };

  const handleListen = (content: string) => {
    if (!('speechSynthesis' in window) || !content.trim()) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(content);
    window.speechSynthesis.speak(utterance);
  };

  const handleStopListen = () => {
    if (!('speechSynthesis' in window)) {
      return;
    }
    window.speechSynthesis.cancel();
  };

  return (
    <div className="relative flex flex-1 flex-col">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-zinc-200/[0.03] via-transparent to-zinc-200/[0.03]" />

      <div className="relative z-10 flex-1 overflow-y-auto px-6 py-8">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 ? (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-6 text-center text-sm text-zinc-400">
              No messages in this chat yet.
            </div>
          ) : (
            messages.map((message, index) => {
              const isAssistant = message.role === 'assistant';
              const isUser = message.role === 'user';
              const previousUserMessage = isAssistant
                ? findPreviousUserMessage(messages, index)
                : null;
              return (
                <motion.div
                  key={`${message.role}-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: Math.min(index * 0.03, 0.3) }}
                  className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  {isAssistant ? (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-900">
                      <Sparkles className="h-4 w-4 text-zinc-300" />
                    </div>
                  ) : null}

                  <div
                    className={`flex max-w-[78%] flex-col gap-2 ${
                      isUser ? 'items-end' : 'items-start'
                    }`}
                  >
                    <div
                      className={`rounded-2xl px-4 py-3 ${
                        isUser
                          ? 'border border-zinc-600 bg-zinc-800 text-zinc-100'
                          : 'border border-zinc-800 bg-zinc-900/70 text-zinc-200'
                      }`}
                    >
                      <div className="mb-1 text-[10px] uppercase tracking-[0.2em] opacity-60">
                        {formatRole(message.role)}
                      </div>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => {
                          void handleCopy(message.content);
                        }}
                        className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                        title="Copy"
                      >
                        <Copy className="h-3.5 w-3.5" />
                      </button>

                      {isUser ? (
                        <button
                          type="button"
                          onClick={() => setInput(message.content)}
                          className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                          title="Edit"
                        >
                          <Edit2 className="h-3.5 w-3.5" />
                        </button>
                      ) : null}

                      {isAssistant ? (
                        <>
                          <button
                            type="button"
                            onClick={() => {
                              if (previousUserMessage) {
                                void handleSend(previousUserMessage);
                              }
                            }}
                            disabled={!previousUserMessage || sending}
                            className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30"
                            title="Refresh"
                          >
                            <RefreshCcw className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleListen(message.content)}
                            className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                            title="Listen"
                          >
                            <Volume2 className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={handleStopListen}
                            className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                            title="Stop"
                          >
                            <Square className="h-3.5 w-3.5" />
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>

                  {isUser ? (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-900">
                      <User className="h-4 w-4 text-zinc-300" />
                    </div>
                  ) : null}
                </motion.div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="relative z-10 border-t border-zinc-800 bg-zinc-950/85 p-6 backdrop-blur-xl">
        <div className="mx-auto max-w-3xl">
          <div className="mb-2 flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => setModelOpen((prev) => !prev)}
              className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
              title="Model selector"
            >
              <Bot className="h-3.5 w-3.5" />
              {selectedModel ? `${selectedModel.provider}/${selectedModel.model}` : 'Select model'}
            </button>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded-md border border-zinc-700 bg-zinc-900 p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                title="Attach files"
              >
                <Paperclip className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={handleToggleRecording}
                disabled={!speechCtor}
                className={`rounded-md border p-1.5 transition-colors ${
                  !speechCtor
                    ? 'border-zinc-700 bg-zinc-900 text-zinc-600'
                    : recording
                      ? 'border-rose-400/60 bg-rose-500/20 text-rose-200'
                      : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                }`}
                title="Rec / STT"
              >
                <Mic className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setWebSearchMode((prev) => !prev)}
                className={`rounded-md border p-1.5 transition-colors ${
                  webSearchMode
                    ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200'
                    : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                }`}
                title="Web search mode"
              >
                <Globe className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {modelOpen ? (
            <div className="mb-2 rounded-xl border border-zinc-700 bg-zinc-900/80 p-3">
              <div className="grid gap-2 sm:grid-cols-[1fr,1fr,auto]">
                <select
                  value={modelProvider}
                  onChange={(event) => {
                    const provider = event.target.value;
                    setModelProvider(provider);
                    const nextModels =
                      providerModels.find((item) => item.provider === provider)?.models ?? [];
                    setModelId(nextModels[0] ?? '');
                  }}
                  className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100"
                  disabled={modelsLoading || savingModel || providerModels.length === 0}
                >
                  {providerModels.map((item) => (
                    <option key={item.provider} value={item.provider} className="bg-zinc-950 text-zinc-100">
                      {item.provider}
                    </option>
                  ))}
                </select>

                <select
                  value={modelId}
                  onChange={(event) => setModelId(event.target.value)}
                  className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100"
                  disabled={modelsLoading || savingModel || modelsForProvider.length === 0}
                >
                  {modelsForProvider.map((item) => (
                    <option key={item} value={item} className="bg-zinc-950 text-zinc-100">
                      {item}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  onClick={() => {
                    void handleApplyModel();
                  }}
                  disabled={
                    !conversationId ||
                    !modelProvider ||
                    !modelId ||
                    modelsLoading ||
                    savingModel
                  }
                  className="rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-100 hover:bg-zinc-700 disabled:opacity-40"
                >
                  {savingModel ? '...' : 'Set'}
                </button>
              </div>
            </div>
          ) : null}

          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => {
              void handleAttachFiles(event.target.files);
            }}
          />

          {attachments.length > 0 ? (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((item, index) => (
                <button
                  key={`${item.name}-${index}`}
                  type="button"
                  onClick={() => {
                    setAttachments((prev) => prev.filter((_, current) => current !== index));
                  }}
                  className="rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  {item.name} · {formatSize(item.size)} · remove
                </button>
              ))}
            </div>
          ) : null}

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
              placeholder={
                conversationId
                  ? webSearchMode
                    ? 'Web search mode is ON. Type query...'
                    : 'Type your message... (Shift+Enter for new line)'
                  : 'Create a chat first'
              }
              className="w-full resize-none rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 pr-12 text-zinc-100 placeholder-zinc-500 transition-all duration-200 focus:border-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-600/50"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '220px' }}
              disabled={!conversationId || sending}
            />
            <button
              type="button"
              onClick={() => {
                void handleSend();
              }}
              disabled={sendDisabled}
              className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-800 transition-all duration-200 hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <Send className="h-4 w-4 text-zinc-100" />
            </button>
          </div>

          <div className="mt-2 min-h-5 text-xs text-zinc-500">
            {statusMessage ||
              (savingModel
                ? 'Saving model...'
                : modelsLoading
                  ? 'Loading models...'
                  : webSearchMode
                    ? 'Web search mode enabled'
                    : recording
                      ? 'Recording...'
                      : sending
                        ? 'Sending...'
                        : '')}
          </div>
        </div>
      </div>
    </div>
  );
}
