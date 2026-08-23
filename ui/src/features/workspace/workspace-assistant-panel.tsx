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
import type { CanvasMessage } from '../../app/components/canvas';
import { MessageRenderer } from '../messages';
import type { RenderableMessage } from '../messages';
import { TtsAudioPlayer, useTtsAudioPlayer } from '../audio';

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

  return (
    <section
      className="min-h-0 bg-[#0d0d11] flex flex-col overflow-hidden h-full"
      data-computer-surface="true"
    >
      <div className="h-9 border-b border-[#1f1f24] px-3 flex items-center gap-2 text-[12px]">
        <Monitor className="h-3.5 w-3.5 text-[#7c7cff]" />
        <span className="font-medium text-[#d2d2d9]">Computer</span>
        <span className="text-[#444]">·</span>
        <span className="text-[#9a9aa3]">Agent Execution Surface</span>
      </div>

      <div className="border-b border-[#1f1f24] bg-[#0f0f14] px-3 py-2 text-[11px] text-[#8a8a94]">
        <div className="flex items-center justify-between gap-2">
          <span>
            mode:{' '}
            <span className="text-[#b0b0bd]">{mode}</span>
          </span>
          <span>
            plan:{' '}
            <span
              className={
                activePlan?.status === 'running'
                  ? 'text-[#7c7cff]'
                  : activePlan?.status === 'completed'
                    ? 'text-emerald-400'
                    : activePlan?.status === 'failed'
                      ? 'text-red-400'
                      : 'text-[#8a8a94]'
              }
            >
              {activePlan?.status ?? 'none'}
            </span>
          </span>
        </div>
        <div className="mt-1 flex items-center justify-between gap-2">
          <span>
            task:{' '}
            <span
              className={
                activeTask?.status === 'running'
                  ? 'text-[#7c7cff]'
                  : activeTask?.status === 'completed'
                    ? 'text-emerald-400'
                    : activeTask?.status === 'failed'
                      ? 'text-red-400'
                      : 'text-[#8a8a94]'
              }
            >
              {activeTask?.status ?? 'none'}
            </span>
          </span>
          <span>
            auto:{' '}
            <span
              className={
                autoState?.status === 'planning' || autoState?.status === 'coding'
                  ? 'text-[#7c7cff]'
                  : autoState?.status === 'completed'
                    ? 'text-emerald-400'
                    : autoState?.status?.startsWith('failed')
                      ? 'text-red-400'
                      : autoState?.status === 'waiting_approval'
                        ? 'text-amber-300'
                        : 'text-[#8a8a94]'
              }
            >
              {autoState?.status ?? 'idle'}
            </span>
          </span>
        </div>
        {autoState?.goal ? (
          <div className="mt-1 truncate text-[#666]" title={autoState.goal}>
            goal: {autoState.goal}
          </div>
        ) : null}
        {autoState?.skill ? (
          <div className="mt-1 truncate text-[#666]">
            skill: {autoState.skill.skill_id && autoState.skill.version
              ? `${autoState.skill.skill_id}@${autoState.skill.version}`
              : 'none'} · {autoState.skill.status}
          </div>
        ) : null}
      </div>

      {computerEvents.length > 0 ? (
        <div className="border-b border-[#1f1f24] bg-[#0a0a0f] px-3 py-2 max-h-40 overflow-auto space-y-0.5">
          {computerEvents.slice(-20).map((ev, idx) => (
            <div
              key={`${ev.ts}-${idx}`}
              className={`flex items-center gap-2 text-[10px] font-mono ${ev.ok === false ? 'text-red-400' : 'text-[#6a6a75]'}`}
            >
              <span className={`shrink-0 w-16 ${ev.ok === false ? 'text-red-400' : 'text-[#5a8a5a]'}`}>
                {ev.kind}
              </span>
              <span className="truncate text-[#8a8a94]">
                {ev.path ?? ev.command ?? ev.tool}
              </span>
              {ev.duration_ms !== undefined ? (
                <span className="shrink-0 text-[#555]">{ev.duration_ms}ms</span>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-auto px-3 py-3 space-y-2" data-scrollbar="always">
        {renderItems.length === 0 ? (
          <div className="text-[12px] text-[#777]">No messages yet.</div>
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
        <div className="border-t border-[#1f1f24] px-3 py-2 text-[11px] text-amber-300">
          {terminalPendingText}
        </div>
      ) : null}
    </section>
  );
}
