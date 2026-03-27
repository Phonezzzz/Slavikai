import { useEffect, useMemo, useState } from 'react';

import type { Artifact } from './components/artifacts-sidebar';
import type {
  CanvasComposerAttachment,
  CanvasMessage,
  CanvasSendPayload,
} from './components/canvas';
import type { SessionTransportBridge } from './session-bridges';
import {
  buildArtifactsFromSources,
  buildCanvasMessages,
  extractErrorMessage,
  extractSessionIdFromPayload,
  parseDisplayDecision,
  parseMwvReport,
  parseMessages,
  parseSessionArtifacts,
  parseSessionFiles,
  parseTraceId,
  toMessageRuntimeMeta,
  type ChatStreamState,
  type PendingUserMessage,
  type SessionArtifactRecord,
} from './session-payload';
import { saveLastSessionId } from './session-storage';
import type {
  ChatAttachment,
  ChatMessage,
  MessageLane,
  MessageRuntimeMeta,
  SessionSummary,
} from './types';

type UseSessionTransportOptions = {
  sessionHeader: string;
  selectedConversation: string | null;
  forceCanvasNext: boolean;
  consumeForceCanvasNext: () => void;
  onSessionIdChange: (sessionId: string) => void;
  onStatusMessage: (message: string | null) => void;
  onRuntimePayload: (payload: unknown) => void;
  onOpenStreamedArtifact: (artifactId: string) => void;
  setArtifactViewerArtifactId: (artifactId: string | null) => void;
  loadSessions: () => Promise<SessionSummary[]>;
};

export type SessionTransportResult = {
  chatMessages: ChatMessage[];
  workspaceMessagesState: ChatMessage[];
  sessionFiles: string[];
  sessionArtifacts: SessionArtifactRecord[];
  streamingContentByArtifactId: Record<string, string>;
  sending: boolean;
  pendingUserMessage: PendingUserMessage | null;
  pendingSessionId: string | null;
  chatStreamingState: ChatStreamState | null;
  workspaceStreamingState: ChatStreamState | null;
  awaitingFirstAssistantChunk: boolean;
  canvasMessages: CanvasMessage[];
  pendingCanvasMessage: CanvasMessage | null;
  streamingAssistantCanvasMessage: CanvasMessage | null;
  workspaceMessages: CanvasMessage[];
  artifacts: Artifact[];
  showAssistantLoading: boolean;
  handleSend: (payload: CanvasSendPayload, lane?: MessageLane) => Promise<boolean>;
  bridge: SessionTransportBridge;
};

const toOutputArtifactUiId = (artifactId: string): string => `output-${artifactId}`;

const normalizeComposerAttachments = (
  attachments: CanvasComposerAttachment[],
): ChatAttachment[] => {
  return attachments
    .map((item) => ({
      name: item.name.trim(),
      mime: item.mime.trim(),
      content: item.content,
    }))
    .filter(
      (item) =>
        item.name.length > 0
        && item.mime.length > 0
        && typeof item.content === 'string',
    );
};

const createPendingCanvasMessage = (
  pending: PendingUserMessage | null,
  lane: MessageLane,
): CanvasMessage | null => {
  if (!pending || pending.lane !== lane) {
    return null;
  }
  const prefix = lane === 'workspace' ? 'pending-ws' : 'pending';
  const pendingId = `${prefix}-${Date.now()}-${pending.content.length}-${pending.attachments.length}`;
  return {
    id: pendingId,
    messageId: pendingId,
    role: 'user',
    content: pending.content,
    createdAt: new Date().toISOString(),
    traceId: null,
    parentUserMessageId: null,
    attachments: pending.attachments,
    transient: true,
    runtimeMeta: {
      messageId: pendingId,
      lane,
      traceId: null,
      isFinal: false,
      mwvReport: null,
    },
  };
};

const createStreamingAssistantMessage = (
  streamingState: ChatStreamState | null,
  lane: MessageLane,
): CanvasMessage | null => {
  if (!streamingState || !streamingState.content.trim()) {
    return null;
  }
  const prefix = lane === 'workspace' ? 'stream-ws' : 'stream';
  const streamId = `${prefix}-${streamingState.streamId}`;
  return {
    id: streamId,
    messageId: streamId,
    role: 'assistant',
    content: streamingState.content,
    createdAt: new Date().toISOString(),
    traceId: null,
    parentUserMessageId: null,
    attachments: [],
    transient: true,
    runtimeMeta: {
      messageId: streamId,
      lane,
      traceId: null,
      isFinal: false,
      mwvReport: null,
    },
  };
};

export function useSessionTransport({
  sessionHeader,
  selectedConversation,
  forceCanvasNext,
  consumeForceCanvasNext,
  onSessionIdChange,
  onStatusMessage,
  onRuntimePayload,
  onOpenStreamedArtifact,
  setArtifactViewerArtifactId,
  loadSessions,
}: UseSessionTransportOptions): SessionTransportResult {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [workspaceMessagesState, setWorkspaceMessagesState] = useState<ChatMessage[]>([]);
  const [messageRuntimeMetaById, setMessageRuntimeMetaById] = useState<
    Record<string, MessageRuntimeMeta>
  >({});
  const [sessionFiles, setSessionFiles] = useState<string[]>([]);
  const [sessionArtifacts, setSessionArtifacts] = useState<SessionArtifactRecord[]>([]);
  const [streamingContentByArtifactId, setStreamingContentByArtifactId] = useState<Record<string, string>>({});
  const [sending, setSending] = useState(false);
  const [pendingUserMessage, setPendingUserMessage] = useState<PendingUserMessage | null>(null);
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [chatStreamingState, setChatStreamingState] = useState<ChatStreamState | null>(null);
  const [workspaceStreamingState, setWorkspaceStreamingState] = useState<ChatStreamState | null>(null);
  const [awaitingFirstAssistantChunk, setAwaitingFirstAssistantChunk] = useState(false);

  const clearConversationState = () => {
    setChatMessages([]);
    setWorkspaceMessagesState([]);
    setMessageRuntimeMetaById({});
    setSessionFiles([]);
    setSessionArtifacts([]);
    setStreamingContentByArtifactId({});
    setPendingUserMessage(null);
    setPendingSessionId(null);
    setChatStreamingState(null);
    setWorkspaceStreamingState(null);
    setAwaitingFirstAssistantChunk(false);
  };

  const applyLoadedConversation = (snapshot: {
    chatMessages: ChatMessage[];
    workspaceMessages: ChatMessage[];
    outputContent: string | null;
    files: string[];
    artifacts: SessionArtifactRecord[];
  }) => {
    setChatMessages(snapshot.chatMessages);
    setWorkspaceMessagesState(snapshot.workspaceMessages);
    setSessionFiles(snapshot.files);
    setSessionArtifacts(snapshot.artifacts);
    setStreamingContentByArtifactId({});
    setPendingUserMessage(null);
    setPendingSessionId(null);
    setChatStreamingState(null);
    setWorkspaceStreamingState(null);
    setAwaitingFirstAssistantChunk(false);
    setMessageRuntimeMetaById(() => {
      const next: Record<string, MessageRuntimeMeta> = {};
      snapshot.chatMessages.forEach((message) => {
        if (message.role === 'assistant' || message.role === 'user') {
          next[message.message_id] = toMessageRuntimeMeta(message, 'chat', null);
        }
      });
      snapshot.workspaceMessages.forEach((message) => {
        if (message.role === 'assistant' || message.role === 'user') {
          next[message.message_id] = toMessageRuntimeMeta(message, 'workspace', null);
        }
      });
      return next;
    });
  };

  const applySessionPayload = (
    payload: unknown,
    options: { applyDisplay: boolean },
  ): { lane: MessageLane } => {
    const body = payload as {
      messages?: unknown;
      workspace_messages?: unknown;
      lane?: unknown;
      output?: unknown;
      files?: unknown;
      artifacts?: unknown;
      display?: unknown;
      trace_id?: unknown;
      mwv_report?: unknown;
    };

    const lane: MessageLane = body.lane === 'workspace' ? 'workspace' : 'chat';
    const traceIdFromPayload = parseTraceId(body.trace_id);
    const mwvReportFromPayload = parseMwvReport(body.mwv_report);
    const parsedChatMessagesState =
      body.messages !== undefined ? parseMessages(body.messages) : null;
    const parsedWorkspaceMessagesState =
      body.workspace_messages !== undefined ? parseMessages(body.workspace_messages) : null;

    if (parsedChatMessagesState !== null) {
      setChatMessages(parsedChatMessagesState);
    }
    if (parsedWorkspaceMessagesState !== null) {
      setWorkspaceMessagesState(parsedWorkspaceMessagesState);
    }
    if (body.files !== undefined) {
      setSessionFiles(parseSessionFiles(body.files));
    }

    if (body.artifacts !== undefined) {
      setSessionArtifacts(parseSessionArtifacts(body.artifacts));
    }

    setMessageRuntimeMetaById((previous) => {
      const next = { ...previous };
      const upsertForLane = (
        list: ChatMessage[] | null,
        listLane: MessageLane,
      ) => {
        if (!list) {
          return;
        }
        list.forEach((message) => {
          if (message.role !== 'assistant' && message.role !== 'user') {
            return;
          }
          next[message.message_id] = toMessageRuntimeMeta(
            message,
            listLane,
            previous[message.message_id] ?? null,
          );
        });
      };

      upsertForLane(parsedChatMessagesState, 'chat');
      upsertForLane(parsedWorkspaceMessagesState, 'workspace');

      const laneMessages = lane === 'workspace'
        ? parsedWorkspaceMessagesState ?? workspaceMessagesState
        : parsedChatMessagesState ?? chatMessages;
      const lastAssistant = [...laneMessages]
        .reverse()
        .find((message) => message.role === 'assistant');
      if (lastAssistant) {
        const previousMeta = next[lastAssistant.message_id] ?? null;
        next[lastAssistant.message_id] = {
          messageId: lastAssistant.message_id,
          lane,
          traceId: traceIdFromPayload ?? previousMeta?.traceId ?? lastAssistant.trace_id ?? null,
          isFinal: true,
          mwvReport: mwvReportFromPayload ?? previousMeta?.mwvReport ?? null,
        };
      }

      return next;
    });

    if (options.applyDisplay) {
      const displayDecision = parseDisplayDecision(body.display);
      if (displayDecision?.target === 'canvas') {
        const artifactId = displayDecision.artifactId ? toOutputArtifactUiId(displayDecision.artifactId) : null;
        if (artifactId) {
          onOpenStreamedArtifact(artifactId);
        }
      } else {
        setArtifactViewerArtifactId(null);
      }
    }

    onRuntimePayload(payload);
    return { lane };
  };

  useEffect(() => {
    if (!sending) {
      setAwaitingFirstAssistantChunk(false);
    }
  }, [sending]);

  useEffect(() => {
    if (typeof window === 'undefined' || !selectedConversation) {
      return;
    }
    const activeSessionId = selectedConversation;
    const refreshWorkflowSnapshot = async () => {
      const stateResponse = await fetch('/ui/api/state', {
        headers: {
          [sessionHeader]: activeSessionId,
        },
      });
      const statePayload: unknown = await stateResponse.json();
      if (!stateResponse.ok) {
        return;
      }
      onRuntimePayload({
        mode: (statePayload as { mode?: unknown }).mode,
        active_plan: (statePayload as { active_plan?: unknown }).active_plan,
        active_task: (statePayload as { active_task?: unknown }).active_task,
        auto_state: (statePayload as { auto_state?: unknown }).auto_state,
        decision: (statePayload as { pending_decision?: unknown }).pending_decision,
      });
    };
    setChatStreamingState(null);
    setWorkspaceStreamingState(null);
    const streamUrl = `/ui/api/events/stream?session_id=${encodeURIComponent(selectedConversation)}`;
    const eventSource = new EventSource(streamUrl);
    eventSource.onmessage = (event) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data) as unknown;
      } catch {
        return;
      }
      if (!parsed || typeof parsed !== 'object') {
        return;
      }
      const envelope = parsed as { type?: unknown; payload?: unknown };
      if (typeof envelope.type !== 'string' || !envelope.payload || typeof envelope.payload !== 'object') {
        return;
      }
      const payload = envelope.payload as {
        artifact_id?: unknown;
        stream_id?: unknown;
        delta?: unknown;
        mode?: unknown;
        lane?: unknown;
        decision?: unknown;
        workflow?: unknown;
        auto_state?: unknown;
      };
      const lane: MessageLane = payload.lane === 'workspace' ? 'workspace' : 'chat';
      if (envelope.type === 'chat.stream.start') {
        const streamId =
          typeof payload.stream_id === 'string' ? payload.stream_id.trim() : '';
        if (!streamId) {
          return;
        }
        setAwaitingFirstAssistantChunk(false);
        if (lane === 'workspace') {
          setWorkspaceStreamingState({ streamId, content: '' });
        } else {
          setChatStreamingState({ streamId, content: '' });
        }
        return;
      }
      if (envelope.type === 'chat.stream.delta') {
        const streamId =
          typeof payload.stream_id === 'string' ? payload.stream_id.trim() : '';
        const delta = typeof payload.delta === 'string' ? payload.delta : '';
        const mode = payload.mode === 'replace' ? 'replace' : 'append';
        if (!streamId || !delta) {
          return;
        }
        setAwaitingFirstAssistantChunk(false);
        if (lane === 'workspace') {
          setWorkspaceStreamingState((prev) => {
            if (!prev || prev.streamId !== streamId) {
              return { streamId, content: delta };
            }
            return { streamId, content: mode === 'replace' ? delta : `${prev.content}${delta}` };
          });
        } else {
          setChatStreamingState((prev) => {
            if (!prev || prev.streamId !== streamId) {
              return { streamId, content: delta };
            }
            return { streamId, content: mode === 'replace' ? delta : `${prev.content}${delta}` };
          });
        }
        return;
      }
      if (envelope.type === 'chat.stream.done') {
        if (lane === 'workspace') {
          setWorkspaceStreamingState(null);
        } else {
          setChatStreamingState(null);
        }
        return;
      }
      if (envelope.type === 'decision.packet') {
        onRuntimePayload({
          decision: payload.decision,
          mode: payload.workflow && typeof payload.workflow === 'object' ? (payload.workflow as { mode?: unknown }).mode : undefined,
          active_plan:
            payload.workflow && typeof payload.workflow === 'object'
              ? (payload.workflow as { active_plan?: unknown }).active_plan
              : undefined,
          active_task:
            payload.workflow && typeof payload.workflow === 'object'
              ? (payload.workflow as { active_task?: unknown }).active_task
              : undefined,
          auto_state:
            payload.workflow && typeof payload.workflow === 'object'
              ? (payload.workflow as { auto_state?: unknown }).auto_state
              : undefined,
        });
        return;
      }
      if (envelope.type === 'session.workflow') {
        onRuntimePayload(payload);
        return;
      }
      if (envelope.type === 'auto.progress') {
        onRuntimePayload({ auto_state: payload.auto_state });
        return;
      }
      if (envelope.type === 'session.resync_required') {
        void refreshWorkflowSnapshot();
        return;
      }
      const artifactId = typeof payload.artifact_id === 'string' ? payload.artifact_id.trim() : '';
      if (!artifactId) {
        return;
      }
      const uiArtifactId = toOutputArtifactUiId(artifactId);
      if (envelope.type === 'canvas.stream.start') {
        setAwaitingFirstAssistantChunk(false);
        onOpenStreamedArtifact(uiArtifactId);
        setStreamingContentByArtifactId((prev) => ({ ...prev, [uiArtifactId]: '' }));
        return;
      }
      if (envelope.type === 'canvas.stream.delta') {
        const delta = typeof payload.delta === 'string' ? payload.delta : '';
        if (!delta) {
          return;
        }
        setAwaitingFirstAssistantChunk(false);
        setStreamingContentByArtifactId((prev) => ({
          ...prev,
          [uiArtifactId]: `${prev[uiArtifactId] ?? ''}${delta}`,
        }));
        return;
      }
      if (envelope.type === 'canvas.stream.done') {
        setStreamingContentByArtifactId((prev) => {
          const next = { ...prev };
          delete next[uiArtifactId];
          return next;
        });
      }
    };
    eventSource.onerror = () => {};
    return () => {
      eventSource.close();
      setChatStreamingState(null);
      setWorkspaceStreamingState(null);
    };
  }, [selectedConversation]);

  const handleSend = async (
    payload: CanvasSendPayload,
    lane: MessageLane = 'chat',
  ): Promise<boolean> => {
    if (!selectedConversation || sending) {
      return false;
    }
    const trimmed = payload.content.trim();
    const normalizedAttachments = normalizeComposerAttachments(payload.attachments ?? []);
    if (!trimmed && normalizedAttachments.length === 0) {
      return false;
    }
    setPendingUserMessage({ content: trimmed, attachments: normalizedAttachments, lane });
    setPendingSessionId(selectedConversation);
    if (lane === 'workspace') {
      setWorkspaceStreamingState(null);
    } else {
      setChatStreamingState(null);
    }
    const forceCanvasForRequest = lane === 'chat' ? forceCanvasNext : false;
    setAwaitingFirstAssistantChunk(true);
    setSending(true);
    try {
      const response = await fetch('/ui/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [sessionHeader]: selectedConversation,
        },
        body: JSON.stringify({
          content: trimmed,
          lane,
          force_canvas: forceCanvasForRequest,
          attachments: normalizedAttachments.length > 0 ? normalizedAttachments : undefined,
        }),
      });
      const responsePayload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(responsePayload, 'Failed to send message.'));
      }

      const headerSession = response.headers.get(sessionHeader);
      const payloadSession = extractSessionIdFromPayload(responsePayload);
      const nextSession =
        (headerSession && headerSession.trim()) || payloadSession || selectedConversation;
      if (nextSession !== selectedConversation) {
        onSessionIdChange(nextSession);
      }
      saveLastSessionId(nextSession);

      setPendingUserMessage(null);
      setPendingSessionId(null);
      if (lane === 'workspace') {
        setWorkspaceStreamingState(null);
      } else {
        setChatStreamingState(null);
      }
      if (lane === 'chat' && forceCanvasForRequest) {
        consumeForceCanvasNext();
      }
      applySessionPayload(responsePayload, { applyDisplay: true });
      await loadSessions();
      onStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send message.';
      onStatusMessage(message);
      setPendingUserMessage(null);
      setPendingSessionId(null);
      if (lane === 'workspace') {
        setWorkspaceStreamingState(null);
      } else {
        setChatStreamingState(null);
      }
      return false;
    } finally {
      setSending(false);
    }
  };

  const pendingForChat =
    pendingSessionId === selectedConversation && pendingUserMessage?.lane === 'chat'
      ? pendingUserMessage
      : null;
  const pendingForWorkspace =
    pendingSessionId === selectedConversation && pendingUserMessage?.lane === 'workspace'
      ? pendingUserMessage
      : null;

  const canvasMessages = useMemo(
    () => buildCanvasMessages(chatMessages, 'chat', messageRuntimeMetaById),
    [chatMessages, messageRuntimeMetaById],
  );
  const pendingCanvasMessage = useMemo(
    () => createPendingCanvasMessage(pendingForChat, 'chat'),
    [pendingForChat],
  );
  const streamingAssistantCanvasMessage = useMemo(
    () => createStreamingAssistantMessage(chatStreamingState, 'chat'),
    [chatStreamingState],
  );
  const workspaceCanvasMessages = useMemo(
    () => buildCanvasMessages(workspaceMessagesState, 'workspace', messageRuntimeMetaById),
    [messageRuntimeMetaById, workspaceMessagesState],
  );
  const pendingWorkspaceCanvasMessage = useMemo(
    () => createPendingCanvasMessage(pendingForWorkspace, 'workspace'),
    [pendingForWorkspace],
  );
  const streamingWorkspaceAssistantMessage = useMemo(
    () => createStreamingAssistantMessage(workspaceStreamingState, 'workspace'),
    [workspaceStreamingState],
  );
  const workspaceMessages = useMemo(() => {
    const next = [...workspaceCanvasMessages];
    if (pendingWorkspaceCanvasMessage) {
      next.push(pendingWorkspaceCanvasMessage);
    }
    if (streamingWorkspaceAssistantMessage) {
      next.push(streamingWorkspaceAssistantMessage);
    }
    return next;
  }, [workspaceCanvasMessages, pendingWorkspaceCanvasMessage, streamingWorkspaceAssistantMessage]);
  const showAssistantLoading = useMemo(
    () =>
      sending &&
      awaitingFirstAssistantChunk &&
      (!chatStreamingState || !chatStreamingState.content.trim()),
    [awaitingFirstAssistantChunk, chatStreamingState, sending],
  );
  const artifacts = useMemo(
    () => buildArtifactsFromSources(sessionArtifacts, sessionFiles, streamingContentByArtifactId),
    [sessionArtifacts, sessionFiles, streamingContentByArtifactId],
  );

  return {
    chatMessages,
    workspaceMessagesState,
    sessionFiles,
    sessionArtifacts,
    streamingContentByArtifactId,
    sending,
    pendingUserMessage,
    pendingSessionId,
    chatStreamingState,
    workspaceStreamingState,
    awaitingFirstAssistantChunk,
    canvasMessages,
    pendingCanvasMessage,
    streamingAssistantCanvasMessage,
    workspaceMessages,
    artifacts,
    showAssistantLoading,
    handleSend,
    bridge: {
      applyLoadedConversation,
      applySessionPayload,
      clearConversationState,
    },
  };
}
