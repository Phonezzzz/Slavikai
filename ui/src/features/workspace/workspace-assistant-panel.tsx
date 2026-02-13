import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import {
  Bot,
  Check,
  ChevronDown,
  Copy,
  Edit2,
  LoaderCircle,
  Mic,
  RefreshCcw,
  Send,
  Square,
  ThumbsDown,
  ThumbsUp,
  Volume2,
} from 'lucide-react';

import type { PlanEnvelope, SessionMode, TaskExecutionState, UiDecision } from '../../app/types';
import type { CanvasMessage, CanvasSendPayload } from '../../app/components/canvas';
import { DecisionPanel } from '../../app/components/decision-panel';
import { PlanPanel } from '../../app/components/plan-panel';

export type WorkspaceContextChip = {
  key: string;
  label: string;
  enabled: boolean;
  onToggle: () => void;
};

export type WorkspaceModelOption = {
  value: string;
  label: string;
  provider: string;
  model: string;
};

type WorkspaceAssistantPanelProps = {
  contextChips: WorkspaceContextChip[];
  mode: SessionMode;
  modelOptions: WorkspaceModelOption[];
  selectedModelValue: string | null;
  modelsLoading: boolean;
  savingModel: boolean;
  onSelectModel: (provider: string, model: string) => void;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  modeBusy: boolean;
  modeError: string | null;
  onChangeMode: (mode: SessionMode) => Promise<void>;
  onPlanDraft: (goal: string) => Promise<void>;
  onPlanApprove: () => Promise<void>;
  onPlanExecute: () => Promise<void>;
  onPlanCancel: () => Promise<void>;
  decision: UiDecision | null | undefined;
  decisionBusy: boolean;
  decisionError: string | null;
  onDecisionRespond?: (
    choice: 'approve' | 'reject' | 'edit',
    editedAction?: Record<string, unknown> | null,
  ) => Promise<void> | void;
  messages: CanvasMessage[];
  terminalPendingText: string | null;
  agentInput: string;
  sending: boolean;
  isDecisionBlocking: boolean;
  canSend: boolean;
  onSendFeedback?: (interactionId: string, rating: 'good' | 'bad') => Promise<boolean>;
  onAgentInputChange: (value: string) => void;
  onAgentSend: () => void;
  onSendPayload: (payload: CanvasSendPayload) => Promise<boolean> | boolean;
};

export function WorkspaceAssistantPanel({
  contextChips,
  mode,
  modelOptions,
  selectedModelValue,
  modelsLoading,
  savingModel,
  onSelectModel,
  activePlan,
  activeTask,
  modeBusy,
  modeError,
  onChangeMode,
  onPlanDraft,
  onPlanApprove,
  onPlanExecute,
  onPlanCancel,
  decision,
  decisionBusy,
  decisionError,
  onDecisionRespond,
  messages,
  terminalPendingText,
  agentInput,
  sending,
  isDecisionBlocking,
  canSend,
  onSendFeedback,
  onAgentInputChange,
  onAgentSend,
  onSendPayload,
}: WorkspaceAssistantPanelProps) {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [feedbackByMessageId, setFeedbackByMessageId] = useState<Record<string, 'good' | 'bad'>>({});
  const [feedbackBusyMessageId, setFeedbackBusyMessageId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const visibleMessages = useMemo(() => messages.slice(-24), [messages]);

  const canUseMediaRecorder = useMemo(() => {
    if (typeof window === 'undefined') {
      return false;
    }
    return (
      typeof window.MediaRecorder !== 'undefined'
      && !!window.navigator?.mediaDevices
      && typeof window.navigator.mediaDevices.getUserMedia === 'function'
    );
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const lineHeightPx = 22;
    const maxHeight = lineHeightPx * 5;
    textarea.style.height = 'auto';
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, lineHeightPx), maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
  }, [agentInput]);

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const extractErrorMessage = (payload: unknown, fallback: string): string => {
    if (!payload || typeof payload !== 'object') {
      return fallback;
    }
    const body = payload as { error?: { message?: unknown } };
    if (body.error && typeof body.error.message === 'string' && body.error.message.trim()) {
      return body.error.message;
    }
    return fallback;
  };

  const buildMessageTextForCopy = (message: CanvasMessage): string => {
    const attachments = message.attachments ?? [];
    if (attachments.length === 0) {
      return message.content;
    }
    const lines: string[] = [];
    if (message.content.trim()) {
      lines.push(message.content.trim());
      lines.push('');
    }
    lines.push('[attachments]');
    attachments.forEach((attachment, index) => {
      lines.push(`#${index + 1} ${attachment.name} (${attachment.mime})`);
      lines.push(attachment.content);
      lines.push('---');
    });
    return lines.join('\n');
  };

  const insertTextIntoComposer = (rawText: string) => {
    const text = rawText.trim();
    if (!text) {
      return;
    }
    const textarea = textareaRef.current;
    const isActive =
      typeof document !== 'undefined'
      && textarea !== null
      && document.activeElement === textarea;
    const currentValue = agentInput;
    const startBase = isActive ? (textarea?.selectionStart ?? currentValue.length) : currentValue.length;
    const endBase = isActive ? (textarea?.selectionEnd ?? currentValue.length) : currentValue.length;
    const start = Math.max(0, Math.min(startBase, currentValue.length));
    const end = Math.max(start, Math.min(endBase, currentValue.length));
    const prefix = currentValue.slice(0, start);
    const suffix = currentValue.slice(end);
    const spacerBefore = prefix.length > 0 && !/\s$/.test(prefix) ? ' ' : '';
    const spacerAfter = suffix.length > 0 && !/^\s/.test(suffix) ? ' ' : '';
    const insertion = `${spacerBefore}${text}${spacerAfter}`;
    const nextValue = `${prefix}${insertion}${suffix}`;
    const nextCaret = prefix.length + insertion.length;

    onAgentInputChange(nextValue);
    if (isActive && textarea) {
      window.requestAnimationFrame(() => {
        textarea.focus();
        textarea.setSelectionRange(nextCaret, nextCaret);
      });
    }
  };

  const handleCopyMessage = async (message: CanvasMessage) => {
    try {
      await navigator.clipboard.writeText(buildMessageTextForCopy(message));
      setCopiedMessageId(message.messageId);
      window.setTimeout(() => {
        setCopiedMessageId((prev) => (prev === message.messageId ? null : prev));
      }, 1200);
    } catch {
      setCopiedMessageId(null);
    }
  };

  const handleEditMessage = (message: CanvasMessage) => {
    onAgentInputChange(message.content);
    textareaRef.current?.focus();
  };

  const handleRefreshMessage = (message: CanvasMessage) => {
    if (!message.parentUserMessageId || sending || isDecisionBlocking) {
      return;
    }
    const source = messages.find(
      (entry) =>
        entry.role === 'user'
        && entry.messageId === message.parentUserMessageId
        && !entry.transient
        && (entry.content.trim().length > 0 || (entry.attachments?.length ?? 0) > 0),
    );
    if (!source) {
      return;
    }
    void onSendPayload({
      content: source.content.trim(),
      attachments: source.attachments ?? [],
    });
  };

  const handleListenToggle = (message: CanvasMessage) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window) || !message.content.trim()) {
      return;
    }
    if (speakingMessageId === message.messageId) {
      window.speechSynthesis.cancel();
      setSpeakingMessageId(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message.content);
    utterance.onend = () => setSpeakingMessageId((prev) => (prev === message.messageId ? null : prev));
    utterance.onerror = () => setSpeakingMessageId((prev) => (prev === message.messageId ? null : prev));
    setSpeakingMessageId(message.messageId);
    window.speechSynthesis.speak(utterance);
  };

  const handleFeedback = async (message: CanvasMessage, rating: 'good' | 'bad') => {
    const interactionId = typeof message.traceId === 'string' ? message.traceId.trim() : '';
    if (!interactionId || !onSendFeedback) {
      return;
    }
    const previous = feedbackByMessageId[message.messageId] ?? null;
    if (previous === rating) {
      return;
    }
    setFeedbackBusyMessageId(message.messageId);
    setFeedbackByMessageId((prev) => ({ ...prev, [message.messageId]: rating }));
    const ok = await onSendFeedback(interactionId, rating);
    if (!ok) {
      setFeedbackByMessageId((prev) => {
        const next = { ...prev };
        if (previous === null) {
          delete next[message.messageId];
        } else {
          next[message.messageId] = previous;
        }
        return next;
      });
    }
    setFeedbackBusyMessageId((prev) => (prev === message.messageId ? null : prev));
  };

  const transcribeAudio = async (blob: Blob) => {
    setIsTranscribing(true);
    setSttError(null);
    try {
      const extension = blob.type.includes('ogg') ? 'ogg' : 'webm';
      const file = new File([blob], `recording.${extension}`, { type: blob.type || 'audio/webm' });
      const body = new FormData();
      body.append('audio', file);
      body.append('language', 'ru');
      const response = await fetch('/ui/api/stt/transcribe', {
        method: 'POST',
        body,
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'STT request failed.'));
      }
      const text = (payload as { text?: unknown }).text;
      if (typeof text !== 'string' || !text.trim()) {
        throw new Error('STT returned empty text.');
      }
      insertTextIntoComposer(text);
    } catch (error) {
      setSttError(error instanceof Error ? error.message : 'STT failed.');
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleToggleRecording = async () => {
    if (!canUseMediaRecorder || sending || isTranscribing || isDecisionBlocking) {
      return;
    }
    if (isRecording) {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
      }
      setIsRecording(false);
      return;
    }

    try {
      const stream = await window.navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
      ];
      const selectedMime = mimeCandidates.find((candidate) => {
        if (typeof window.MediaRecorder.isTypeSupported !== 'function') {
          return false;
        }
        return window.MediaRecorder.isTypeSupported(candidate);
      });
      const recorder = selectedMime
        ? new window.MediaRecorder(stream, { mimeType: selectedMime })
        : new window.MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        setSttError('Не удалось записать аудио.');
        setIsRecording(false);
      };
      recorder.onstop = () => {
        setIsRecording(false);
        const chunks = audioChunksRef.current;
        audioChunksRef.current = [];
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
        }
        mediaRecorderRef.current = null;
        if (chunks.length === 0) {
          return;
        }
        const audioBlob = new Blob(chunks, {
          type: recorder.mimeType || 'audio/webm',
        });
        void transcribeAudio(audioBlob);
      };
      recorder.start();
      setSttError(null);
      setIsRecording(true);
    } catch {
      setSttError('Микрофон недоступен.');
      setIsRecording(false);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
    }
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onAgentSend();
    }
  };

  const controlButtonClass =
    'rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40';

  return (
    <section className="min-h-0 border-r border-[#1f1f24] bg-[#0d0d11] flex flex-col overflow-hidden">
      <div className="h-9 border-b border-[#1f1f24] px-3 flex items-center gap-2 text-[12px] text-[#9a9aa3]">
        <Bot className="h-3.5 w-3.5 text-[#f59e0b]" />
        AI Assistant
      </div>

      <div className="border-b border-[#1f1f24] px-3 py-2 flex flex-wrap gap-2">
        {contextChips.length === 0 ? (
          <span className="text-[11px] text-[#666]">No context available.</span>
        ) : (
          contextChips.map((chip) => (
            <button
              key={chip.key}
              onClick={chip.onToggle}
              className={`rounded-full border px-2 py-0.5 text-[11px] ${
                chip.enabled
                  ? 'border-[#2f5dff] bg-[#1a2348] text-[#c8d7ff]'
                  : 'border-[#2a2a31] bg-[#111117] text-[#888893]'
              }`}
              title={chip.label}
            >
              {chip.label}
            </button>
          ))
        )}
      </div>

      <PlanPanel
        mode={mode}
        plan={activePlan}
        task={activeTask}
        busy={modeBusy}
        error={modeError}
        onChangeMode={onChangeMode}
        onDraft={onPlanDraft}
        onApprove={onPlanApprove}
        onExecute={onPlanExecute}
        onCancel={onPlanCancel}
        showModeControls={false}
      />

      {decision && decision.status === 'pending' ? (
        <div className="border-b border-[#1f1f24]">
          <DecisionPanel
            decision={decision}
            busy={decisionBusy}
            error={decisionError}
            onRespond={(choice, editedAction) => {
              if (!onDecisionRespond) {
                return;
              }
              void onDecisionRespond(choice, editedAction);
            }}
          />
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-auto px-3 py-3 space-y-2" data-scrollbar="always">
        {visibleMessages.length === 0 ? (
          <div className="text-[12px] text-[#777]">No messages yet.</div>
        ) : (
          visibleMessages.map((message) => {
            const isUser = message.role === 'user';
            const canFeedback =
              !!onSendFeedback
              && !isUser
              && typeof message.traceId === 'string'
              && message.traceId.trim().length > 0;
            const feedbackRating = feedbackByMessageId[message.messageId] ?? null;
            const canRefresh = !isUser && !!message.parentUserMessageId;
            const isSavedMessage = !message.transient;

            return (
              <div key={message.id}>
                <div
                  className={`rounded-lg border px-3 py-2 text-[12px] whitespace-pre-wrap break-words ${
                    isUser
                      ? 'bg-[#141418] border-[#252530] text-[#d0d0d8] ml-6'
                      : 'bg-[#101015] border-[#1f1f28] text-[#bfc2cb] mr-6'
                  }`}
                >
                  {message.content}
                </div>
                {isSavedMessage ? (
                  <div className={`mt-1 flex items-center gap-1 ${isUser ? 'justify-end mr-6' : 'ml-0'}`}>
                    <button
                      type="button"
                      onClick={() => {
                        void handleCopyMessage(message);
                      }}
                      className={controlButtonClass}
                      title="Copy"
                      aria-label="Copy"
                    >
                      {copiedMessageId === message.messageId ? (
                        <Check className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </button>

                    {isUser ? (
                      <button
                        type="button"
                        onClick={() => handleEditMessage(message)}
                        className={controlButtonClass}
                        title="Edit"
                        aria-label="Edit"
                      >
                        <Edit2 className="h-3.5 w-3.5" />
                      </button>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => handleRefreshMessage(message)}
                          disabled={!canRefresh || sending || isDecisionBlocking}
                          className={controlButtonClass}
                          title={canRefresh ? 'Refresh' : 'Refresh unavailable'}
                          aria-label="Refresh"
                        >
                          <RefreshCcw className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleListenToggle(message)}
                          className={controlButtonClass}
                          title={speakingMessageId === message.messageId ? 'Stop listen' : 'Listen'}
                          aria-label={speakingMessageId === message.messageId ? 'Stop listen' : 'Listen'}
                        >
                          {speakingMessageId === message.messageId ? (
                            <Square className="h-3.5 w-3.5" />
                          ) : (
                            <Volume2 className="h-3.5 w-3.5" />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            void handleFeedback(message, 'good');
                          }}
                          disabled={!canFeedback || feedbackBusyMessageId === message.messageId}
                          className={`${controlButtonClass} ${feedbackRating === 'good' ? 'text-emerald-300' : ''}`}
                          title={canFeedback ? 'Like' : 'Like unavailable'}
                          aria-label="Like"
                        >
                          <ThumbsUp className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            void handleFeedback(message, 'bad');
                          }}
                          disabled={!canFeedback || feedbackBusyMessageId === message.messageId}
                          className={`${controlButtonClass} ${feedbackRating === 'bad' ? 'text-rose-300' : ''}`}
                          title={canFeedback ? 'Dislike' : 'Dislike unavailable'}
                          aria-label="Dislike"
                        >
                          <ThumbsDown className="h-3.5 w-3.5" />
                        </button>
                      </>
                    )}
                  </div>
                ) : null}
              </div>
            );
          })
        )}
      </div>

      <div className="border-t border-[#1f1f24] p-3 space-y-2">
        {terminalPendingText ? (
          <div className="text-[11px] text-amber-300">{terminalPendingText}</div>
        ) : null}
        {sttError ? (
          <div className="rounded-md border border-rose-700/40 bg-rose-900/20 px-2.5 py-2 text-[11px] text-rose-200">
            {sttError}
          </div>
        ) : null}
        <div className="grid grid-cols-2 gap-2">
          <div className="relative">
            <select
              value={mode}
              onChange={(event) => {
                const nextMode = event.target.value;
                if (nextMode === 'ask' || nextMode === 'plan' || nextMode === 'act') {
                  void onChangeMode(nextMode);
                }
              }}
              disabled={modeBusy || isDecisionBlocking}
              className="w-full appearance-none rounded-md border border-[#252530] bg-[#111116] px-2.5 py-1.5 pr-8 text-[11px] uppercase tracking-wide text-[#d4d4db] outline-none"
            >
              <option value="ask">ask</option>
              <option value="plan">plan</option>
              <option value="act">act</option>
            </select>
            <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#666]" />
          </div>
          <div className="relative">
            <select
              value={selectedModelValue ?? ''}
              onChange={(event) => {
                const next = modelOptions.find((option) => option.value === event.target.value);
                if (next) {
                  onSelectModel(next.provider, next.model);
                }
              }}
              disabled={modelsLoading || savingModel || isDecisionBlocking || modelOptions.length === 0}
              className="w-full appearance-none rounded-md border border-[#252530] bg-[#111116] px-2.5 py-1.5 pr-8 text-[11px] text-[#d4d4db] outline-none"
            >
              <option value="" disabled>
                Select model
              </option>
              {modelOptions.map((option) => (
                <option key={option.value} value={option.value} className="bg-[#0b0b0d] text-[#ddd]">
                  {option.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#666]" />
          </div>
        </div>

        <div className="flex items-end gap-2 rounded-md border border-[#252530] bg-[#111116] px-2.5 py-2">
          <textarea
            ref={textareaRef}
            value={agentInput}
            onChange={(event) => onAgentInputChange(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Ask agent..."
            className="workspace-composer-textarea min-h-[24px] max-h-[110px] flex-1 resize-none bg-transparent text-[12px] text-[#d4d4db] outline-none"
            rows={1}
            disabled={sending || isTranscribing || isDecisionBlocking}
            data-scrollbar="always"
          />
          <button
            type="button"
            onClick={() => {
              void handleToggleRecording();
            }}
            disabled={!canUseMediaRecorder || sending || isTranscribing || isDecisionBlocking}
            className={`rounded p-1 transition-colors ${
              !canUseMediaRecorder
                ? 'text-[#444]'
                : isRecording
                  ? 'text-rose-300'
                  : isTranscribing
                    ? 'text-amber-300'
                    : 'text-[#6f6f78] hover:text-[#c1c1ca]'
            }`}
            title={
              !canUseMediaRecorder
                ? 'Microphone unavailable'
                : isRecording
                  ? 'Stop recording'
                  : isTranscribing
                    ? 'Transcribing...'
                    : 'Start recording'
            }
            aria-label="Toggle speech-to-text"
          >
            {isTranscribing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Mic className="h-4 w-4" />}
          </button>
          <button
            type="button"
            onClick={onAgentSend}
            disabled={sending || isTranscribing || isDecisionBlocking || !canSend}
            className={`rounded p-1.5 transition-colors ${
              !sending && !isTranscribing && !isDecisionBlocking && canSend
                ? 'bg-[#6366f1] text-white hover:bg-[#5558e6]'
                : 'bg-[#1b1b20] text-[#555]'
            }`}
            title="Send"
            aria-label="Send"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </section>
  );
}
