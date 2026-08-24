import { useMemo, useState } from 'react';
import {
  Check,
  Copy,
  Monitor,
  Pause,
  ThumbsDown,
  ThumbsUp,
  Volume2,
} from 'lucide-react';

import type {
  AutoState,
  ComputerActivityEvent,
  DecisionRespondChoice,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
  UiDecision,
} from '../../app/types';
import { SESSION_MODE_LABELS } from '../../app/types';
import type { CanvasMessage } from '../../app/components/canvas';
import { StatusBadge } from '../../app/components/ui/status-badge';
import { MessageRenderer } from '../messages';
import type { RenderableMessage } from '../messages';
import { TtsAudioPlayer, useTtsAudioPlayer } from '../audio';
import { deriveComputerStatus } from './computer-status';

type WorkspaceAssistantPanelProps = {
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  decision: UiDecision | null | undefined;
  decisionBusy: boolean;
  decisionError: string | null;
  onDecisionRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
  messages: CanvasMessage[];
  computerEvents?: ComputerActivityEvent[];
  terminalPendingText: string | null;
  onSendFeedback?: (interactionId: string, rating: 'good' | 'bad') => Promise<boolean>;
};

export function WorkspaceAssistantPanel({
  mode,
  activePlan,
  activeTask,
  autoState,
  decision,
  decisionBusy,
  decisionError,
  onDecisionRespond,
  messages,
  computerEvents = [],
  terminalPendingText,
  onSendFeedback,
}: WorkspaceAssistantPanelProps) {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [feedbackByMessageId, setFeedbackByMessageId] = useState<Record<string, 'good' | 'bad'>>({});
  const [feedbackBusyMessageId, setFeedbackBusyMessageId] = useState<string | null>(null);
  const ttsPlayer = useTtsAudioPlayer();

  const visibleMessages = useMemo(() => messages.slice(-24), [messages]);
  const renderItems = useMemo<RenderableMessage[]>(() => {
    const items: RenderableMessage[] = visibleMessages.map((message) => ({
      kind: 'message',
      message,
      meta: message.runtimeMeta ?? null,
    }));
    if (decision && decision.status === 'pending') {
      items.push({
        kind: 'decision',
        id: `decision-${decision.id}`,
        decision,
      });
    }
    return items;
  }, [decision, visibleMessages]);

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

  const controlButtonClass =
    'rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40';

  const status = deriveComputerStatus({ mode, activePlan, activeTask, autoState, decision });
  const modeLabel = SESSION_MODE_LABELS[mode];

  return (
    <section
      className="min-h-0 bg-zinc-900 flex flex-col overflow-hidden h-full"
      data-computer-surface="true"
    >
      <header className="shrink-0 border-b border-zinc-800 bg-zinc-900">
        <div className="flex items-center gap-2 px-3 py-2">
          <Monitor className="h-3.5 w-3.5 shrink-0 text-zinc-400" />
          <span className="text-[12px] font-medium text-zinc-300">Computer</span>
          <span className="rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
            {modeLabel}
          </span>
          <StatusBadge tone={status.tone}>{status.label}</StatusBadge>
        </div>
        {status.goal || status.stepLabel || status.detail ? (
          <div className="space-y-0.5 px-3 pb-2">
            {status.goal ? (
              <div className="truncate text-[12px] text-zinc-300" title={status.goal}>
                {status.goal}
              </div>
            ) : null}
            {status.stepLabel || status.detail ? (
              <div className="flex items-center gap-2 text-[11px]">
                {status.stepLabel ? (
                  <span className="text-zinc-400">{status.stepLabel}</span>
                ) : null}
                {status.stepLabel && status.detail ? (
                  <span className="text-zinc-500">·</span>
                ) : null}
                {status.detail ? (
                  <span
                    className={
                      status.tone === 'error'
                        ? 'text-rose-300'
                        : status.tone === 'waiting'
                          ? 'text-amber-300'
                          : 'text-zinc-400'
                    }
                  >
                    {status.detail}
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </header>

      {computerEvents.length > 0 ? (
        <div className="border-b border-zinc-800 bg-zinc-950 px-3 py-2 max-h-40 overflow-auto space-y-0.5">
          {computerEvents.slice(-20).map((ev, idx) => (
            <div
              key={`${ev.ts}-${idx}`}
              className={`flex items-center gap-2 text-[10px] font-mono ${ev.ok === false ? 'text-red-400' : 'text-zinc-500'}`}
            >
              <span className={`shrink-0 w-16 ${ev.ok === false ? 'text-red-400' : 'text-[#5a8a5a]'}`}>
                {ev.kind}
              </span>
              <span className="truncate text-zinc-400">
                {ev.path ?? ev.command ?? ev.tool}
              </span>
              {ev.duration_ms !== undefined ? (
                <span className="shrink-0 text-zinc-500">{ev.duration_ms}ms</span>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-auto px-3 py-3 space-y-2" data-scrollbar="always">
        {renderItems.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
            <Monitor className="h-8 w-8 text-zinc-600" />
            <p className="text-[13px] text-zinc-300">
              {status.isActive
                ? `${status.label}. The agent is working; progress will appear here.`
                : 'Nothing running yet.'}
            </p>
            <p className="max-w-sm text-[12px] leading-5 text-zinc-500">
              Start a task in Chat, or open Files / Changes / Terminal to inspect this session.
            </p>
          </div>
        ) : (
          renderItems.map((item) => {
            if (item.kind === 'decision') {
              return (
                <div key={item.id}>
                  <MessageRenderer
                    context="workspace"
                    message={item}
                    decisionBusy={decisionBusy}
                    decisionError={decisionError}
                    onDecisionRespond={onDecisionRespond}
                  />
                </div>
              );
            }

            const message = item.message;
            const isUser = message.role === 'user';
            const canFeedback =
              !!onSendFeedback
              && !isUser
              && typeof message.traceId === 'string'
              && message.traceId.trim().length > 0;
            const feedbackRating = feedbackByMessageId[message.messageId] ?? null;
            const isSavedMessage = !message.transient;

            return (
              <div key={message.id}>
                <MessageRenderer
                  context="workspace"
                  message={item}
                  decisionBusy={decisionBusy}
                  decisionError={decisionError}
                  onDecisionRespond={onDecisionRespond}
                />
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

                    {!isUser ? (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            void ttsPlayer.toggle(message.messageId, message.content);
                          }}
                          className={controlButtonClass}
                          title={
                            ttsPlayer.state.activeMessageId === message.messageId
                            && ttsPlayer.state.status === 'playing'
                              ? 'Pause listen'
                              : 'Listen'
                          }
                          aria-label={
                            ttsPlayer.state.activeMessageId === message.messageId
                            && ttsPlayer.state.status === 'playing'
                              ? 'Pause listen'
                              : 'Listen'
                          }
                        >
                          {ttsPlayer.state.activeMessageId === message.messageId
                          && ttsPlayer.state.status === 'playing' ? (
                            <Pause className="h-3.5 w-3.5" />
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
                    ) : null}
                  </div>
                ) : null}
                {message.role === 'assistant' && ttsPlayer.state.activeMessageId === message.messageId ? (
                  <TtsAudioPlayer
                    playback={ttsPlayer.state}
                    onPlayPause={() => {
                      void ttsPlayer.toggle(message.messageId, message.content);
                    }}
                    onSeek={ttsPlayer.seek}
                    onStop={ttsPlayer.stop}
                  />
                ) : null}
              </div>
            );
          })
        )}
      </div>

      {terminalPendingText ? (
        <div className="border-t border-zinc-800 px-3 py-2 text-[11px] text-amber-300">
          {terminalPendingText}
        </div>
      ) : null}
    </section>
  );
}
