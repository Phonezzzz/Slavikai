import { useEffect, useMemo, useState } from 'react';
import { PanelRight } from 'lucide-react';

import { ArtifactPanel } from '../../app/components/artifact-panel';
import {
  Canvas,
  type CanvasComposerAttachment,
  type CanvasMessage,
  type CanvasSendPayload,
} from '../../app/components/canvas';
import type { Artifact } from '../../app/components/artifacts-sidebar';
import { HistorySidebar } from '../../app/components/history-sidebar';
import { SearchModal } from '../../app/components/search-modal';
import { Settings } from '../../app/components/Settings';
import {
  WorkspaceSettingsModal,
  type WorkspaceGithubImportResult,
} from '../../app/components/workspace-settings-modal';
import { WorkspaceIde } from '../../app/components/workspace-ide';
import type {
  ChatAttachment,
  ChatMessage,
  FolderSummary,
  PlanEnvelope,
  ProviderModels,
  SessionMode,
  SelectedModel,
  SessionSummary,
  TaskExecutionState,
  UiDecision,
} from '../../app/types';
import { isRecord } from '../../codecs/guards';
import { decodeStoredModel } from '../../codecs/settings.codec';
import { decodeSessionWorkflowPayload } from '../../codecs/session_workflow.codec';
import { openSseConnection } from '../../services/sse/sse_client';
import { SseRouter } from '../../services/sse/sse_router';
import {
  type AppView,
  type ComposerUiSettings,
  type DisplayDecision,
  type SessionArtifactRecord,
  DEFAULT_COMPOSER_SETTINGS,
  SCROLLBAR_REVEAL_DISTANCE_PX,
  SESSION_HEADER,
  buildArtifactsFromSources,
  buildCanvasMessages,
  extractErrorMessage,
  extractFilenameFromDisposition,
  extractSessionIdFromPayload,
  groupSessionByDate,
  loadLastSessionId,
  parseComposerSettings,
  parseDisplayDecision,
  parseFolders,
  parseMessages,
  parsePlanEnvelope,
  parseProviderModels,
  parseSelectedModel,
  parseSessionArtifacts,
  parseSessionFiles,
  parseSessionMode,
  parseSessionOutput,
  parseSessions,
  parseTaskExecution,
  parseUiDecision,
  pathForView,
  saveLastSessionId,
  sortSessionsByUpdated,
  toOutputArtifactUiId,
  triggerBrowserDownload,
  viewFromPathname,
} from './app-shell-utils';

type ChatStreamState = {
  streamId: string;
  content: string;
};

type PendingUserMessage = {
  content: string;
  attachments: ChatAttachment[];
};

type SessionPayloadApplyOptions = {
  applyDisplay: boolean;
};
export default function App() {
  const [activeView, setActiveView] = useState<AppView>(() => {
    if (typeof window === 'undefined') {
      return 'chat';
    }
    return viewFromPathname(window.location.pathname);
  });
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<SelectedModel | null>(null);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [, setSessionOutput] = useState<string | null>(null);
  const [sessionFiles, setSessionFiles] = useState<string[]>([]);
  const [sessionArtifacts, setSessionArtifacts] = useState<SessionArtifactRecord[]>([]);
  const [streamingContentByArtifactId, setStreamingContentByArtifactId] = useState<Record<string, string>>({});
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);

  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [savingModel, setSavingModel] = useState(false);
  const [sending, setSending] = useState(false);
  const [pendingUserMessage, setPendingUserMessage] = useState<PendingUserMessage | null>(null);
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const [artifactPanelOpen, setArtifactPanelOpen] = useState(false);
  const [artifactViewerArtifactId, setArtifactViewerArtifactId] = useState<string | null>(null);
  const [forceCanvasNext, setForceCanvasNext] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [workspaceSettingsOpen, setWorkspaceSettingsOpen] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [lastModelApplied, setLastModelApplied] = useState(false);
  const [chatStreamingState, setChatStreamingState] = useState<ChatStreamState | null>(null);
  const [awaitingFirstAssistantChunk, setAwaitingFirstAssistantChunk] = useState(false);
  const [composerSettings, setComposerSettings] = useState<ComposerUiSettings>(
    DEFAULT_COMPOSER_SETTINGS,
  );
  const [pendingDecision, setPendingDecision] = useState<UiDecision | null>(null);
  const [decisionBusy, setDecisionBusy] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [sessionMode, setSessionMode] = useState<SessionMode>('ask');
  const [activePlan, setActivePlan] = useState<PlanEnvelope | null>(null);
  const [activeTask, setActiveTask] = useState<TaskExecutionState | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);
  const [workspaceRefreshToken, setWorkspaceRefreshToken] = useState(0);

  const pendingForCanvas =
    pendingSessionId === selectedConversation ? pendingUserMessage : null;
  const canvasMessages = useMemo(
    () => buildCanvasMessages(messages),
    [messages],
  );
  const pendingCanvasMessage = useMemo(() => {
    if (!pendingForCanvas) {
      return null;
    }
    const pendingId = `pending-${Date.now()}-${pendingForCanvas.content.length}-${pendingForCanvas.attachments.length}`;
    return {
      id: pendingId,
      messageId: pendingId,
      role: 'user' as const,
      content: pendingForCanvas.content,
      createdAt: new Date().toISOString(),
      traceId: null,
      parentUserMessageId: null,
      attachments: pendingForCanvas.attachments,
      transient: true,
    };
  }, [pendingForCanvas]);
  const streamingAssistantCanvasMessage = useMemo(() => {
    if (!chatStreamingState || !chatStreamingState.content.trim()) {
      return null;
    }
    return {
      id: `stream-${chatStreamingState.streamId}`,
      messageId: `stream-${chatStreamingState.streamId}`,
      role: 'assistant' as const,
      content: chatStreamingState.content,
      createdAt: new Date().toISOString(),
      traceId: null,
      parentUserMessageId: null,
      attachments: [],
      transient: true,
    };
  }, [chatStreamingState]);
  const workspaceMessages = useMemo(() => {
    const next = [...canvasMessages];
    if (pendingCanvasMessage) {
      next.push(pendingCanvasMessage);
    }
    if (streamingAssistantCanvasMessage) {
      next.push(streamingAssistantCanvasMessage);
    }
    return next;
  }, [canvasMessages, pendingCanvasMessage, streamingAssistantCanvasMessage]);
  const showAssistantLoading = useMemo(
    () =>
      sending &&
      awaitingFirstAssistantChunk &&
      (!chatStreamingState || !chatStreamingState.content.trim()),
    [awaitingFirstAssistantChunk, chatStreamingState, sending],
  );
  const historyChats = useMemo(
    () =>
      sessions.map((session) => ({
        id: session.session_id,
        title: session.title,
        messageCount: session.message_count,
        date: session.updated_at,
        group: groupSessionByDate(session.updated_at),
      })),
    [sessions],
  );
  const searchChats = useMemo(
    () =>
      sessions.map((session) => ({
        id: session.session_id,
        title: session.title,
        date: session.updated_at,
        messageCount: session.message_count,
        preview: '',
      })),
    [sessions],
  );
  const artifacts = useMemo(
    () =>
      buildArtifactsFromSources(
        sessionArtifacts,
        sessionFiles,
        streamingContentByArtifactId,
      ),
    [sessionArtifacts, sessionFiles, streamingContentByArtifactId],
  );
  const modelLabel = selectedModel
    ? `${selectedModel.provider}/${selectedModel.model}`
    : 'Model not selected';
  const modelOptions = useMemo(
    () =>
      providerModels.flatMap((provider) =>
        provider.models.map((model) => ({
          value: `${provider.provider}::${model}`,
          label: `${provider.provider}/${model}`,
          provider: provider.provider,
          model,
        })),
      ),
    [providerModels],
  );
  const selectedModelValue = selectedModel
    ? `${selectedModel.provider}::${selectedModel.model}`
    : null;


  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    let raf = 0;
    const clearReveal = () => {
      const elements = document.querySelectorAll<HTMLElement>('[data-scrollbar="auto"]');
      elements.forEach((element) => element.classList.remove('scrollbar-reveal'));
    };
    const handleMove = (event: MouseEvent) => {
      const x = event.clientX;
      const y = event.clientY;
      if (raf) {
        cancelAnimationFrame(raf);
      }
      raf = window.requestAnimationFrame(() => {
        const elements = document.querySelectorAll<HTMLElement>('[data-scrollbar="auto"]');
        elements.forEach((element) => {
          const rect = element.getBoundingClientRect();
          const hasVertical = element.scrollHeight > element.clientHeight;
          const hasHorizontal = element.scrollWidth > element.clientWidth;
          let reveal = false;

          if (hasVertical) {
            const withinY = y >= rect.top && y <= rect.bottom;
            const nearRight = x >= rect.right - SCROLLBAR_REVEAL_DISTANCE_PX && x <= rect.right;
            reveal = withinY && nearRight;
          }

          if (!reveal && hasHorizontal) {
            const withinX = x >= rect.left && x <= rect.right;
            const nearBottom = y >= rect.bottom - SCROLLBAR_REVEAL_DISTANCE_PX && y <= rect.bottom;
            reveal = withinX && nearBottom;
          }

          element.classList.toggle('scrollbar-reveal', reveal);
        });
      });
    };

    window.addEventListener('mousemove', handleMove, { passive: true });
    window.addEventListener('mouseleave', clearReveal);
    window.addEventListener('blur', clearReveal);

    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseleave', clearReveal);
      window.removeEventListener('blur', clearReveal);
      if (raf) {
        cancelAnimationFrame(raf);
      }
      clearReveal();
    };
  }, []);

  const loadLastModel = (): SelectedModel | null => {
    if (typeof window === 'undefined') {
      return null;
    }
    const raw = window.localStorage.getItem('slavik.last.model');
    const decoded = decodeStoredModel(raw);
    if (decoded.ok) {
      return decoded.value;
    }
    window.localStorage.removeItem('slavik.last.model');
    return null;
  };

  const saveLastModel = (model: SelectedModel | null) => {
    if (typeof window === 'undefined') {
      return;
    }
    if (!model) {
      return;
    }
    window.localStorage.setItem('slavik.last.model', JSON.stringify(model));
  };

  const syncViewFromLocation = () => {
    if (typeof window === 'undefined') {
      return;
    }
    setActiveView(viewFromPathname(window.location.pathname));
  };

  const setView = (view: AppView) => {
    setActiveView(view);
    if (typeof window === 'undefined') {
      return;
    }
    const nextPath = pathForView(view);
    if (window.location.pathname === nextPath) {
      return;
    }
    window.history.pushState({ view }, '', nextPath);
  };

  const isModelAvailable = (model: SelectedModel, providers: ProviderModels[]) =>
    providers.some((provider) => provider.provider === model.provider && provider.models.includes(model.model));

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }
    syncViewFromLocation();
    const handlePopState = () => {
      syncViewFromLocation();
    };
    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  useEffect(() => {
    if (activeView !== 'workspace' && workspaceSettingsOpen) {
      setWorkspaceSettingsOpen(false);
    }
  }, [activeView, workspaceSettingsOpen]);

  const extractErrorFromResponse = async (
    response: Response,
    fallback: string,
  ): Promise<string> => {
    try {
      const payload: unknown = await response.json();
      return extractErrorMessage(payload, fallback);
    } catch {
      return fallback;
    }
  };

  const loadSessions = async (): Promise<SessionSummary[]> => {
    const response = await fetch('/ui/api/sessions');
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to load chat list.'));
    }
    const parsed = sortSessionsByUpdated(
      parseSessions((payload as { sessions?: unknown }).sessions),
    );
    setSessions(parsed);
    return parsed;
  };

  const loadFolders = async (): Promise<FolderSummary[]> => {
    const response = await fetch('/ui/api/folders');
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to load folders.'));
    }
    const parsed = parseFolders((payload as { folders?: unknown }).folders);
    setFolders(parsed);
    return parsed;
  };

  const createFolder = async (name: string): Promise<FolderSummary> => {
    const response = await fetch('/ui/api/folders', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name }),
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to create folder.'));
    }
    const folder = (payload as { folder?: unknown }).folder;
    const parsed = parseFolders(folder ? [folder] : []);
    if (parsed.length === 0) {
      throw new Error('Failed to create folder.');
    }
    return parsed[0];
  };

  const renameSession = async (sessionId: string, title: string): Promise<void> => {
    const response = await fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/title`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ title }),
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to rename chat.'));
    }
  };

  const moveSessionToFolder = async (
    sessionId: string,
    folderId: string | null,
  ): Promise<void> => {
    const response = await fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/folder`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ folder_id: folderId }),
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to move chat to folder.'));
    }
  };

  const loadModels = async (): Promise<ProviderModels[]> => {
    setModelsLoading(true);
    try {
      const response = await fetch('/ui/api/models');
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to load models.'));
      }
      const parsed = parseProviderModels((payload as { providers?: unknown }).providers);
      setProviderModels(parsed);
      return parsed;
    } finally {
      setModelsLoading(false);
    }
  };

  const loadComposerSettings = async (): Promise<ComposerUiSettings> => {
    const response = await fetch('/ui/api/settings');
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to load composer settings.'));
    }
    const parsed = parseComposerSettings(payload);
    setComposerSettings(parsed);
    return parsed;
  };

  const setSessionModel = async (
    sessionId: string,
    provider: string,
    model: string,
  ): Promise<SelectedModel | null> => {
    const response = await fetch('/ui/api/session-model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        [SESSION_HEADER]: sessionId,
      },
      body: JSON.stringify({ provider, model }),
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to set model.'));
    }
    const parsed = parseSelectedModel((payload as { selected_model?: unknown }).selected_model);
    return parsed || { provider, model };
  };

  const loadConversation = async (sessionId: string): Promise<SelectedModel | null> => {
    const [sessionResponse, historyResponse, outputResponse, filesResponse] = await Promise.all([
      fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}`, {
        headers: {
          [SESSION_HEADER]: sessionId,
        },
      }),
      fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/history`, {
        headers: {
          [SESSION_HEADER]: sessionId,
        },
      }),
      fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/output`, {
        headers: {
          [SESSION_HEADER]: sessionId,
        },
      }),
      fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}/files`, {
        headers: {
          [SESSION_HEADER]: sessionId,
        },
      }),
    ]);

    const [sessionPayload, historyPayload, outputPayload, filesPayload]: unknown[] =
      await Promise.all([
        sessionResponse.json(),
        historyResponse.json(),
        outputResponse.json(),
        filesResponse.json(),
      ]);

    if (!sessionResponse.ok) {
      throw new Error(extractErrorMessage(sessionPayload, 'Failed to load chat session.'));
    }
    if (!historyResponse.ok) {
      throw new Error(extractErrorMessage(historyPayload, 'Failed to load chat history.'));
    }
    if (!outputResponse.ok) {
      throw new Error(extractErrorMessage(outputPayload, 'Failed to load canvas output.'));
    }
    if (!filesResponse.ok) {
      throw new Error(extractErrorMessage(filesPayload, 'Failed to load session files.'));
    }

    const session = (sessionPayload as { session?: { selected_model?: unknown } }).session;
    setMessages(parseMessages((historyPayload as { messages?: unknown }).messages));
    setChatStreamingState(null);
    setPendingDecision(parseUiDecision((session as { decision?: unknown } | undefined)?.decision));
    setDecisionError(null);
    const parsedOutput = parseSessionOutput((outputPayload as { output?: unknown }).output);
    setSessionOutput(parsedOutput.content);
    setSessionFiles(parseSessionFiles((filesPayload as { files?: unknown }).files));
    setSessionArtifacts(
      parseSessionArtifacts((session as { artifacts?: unknown } | undefined)?.artifacts),
    );
    setSessionMode(parseSessionMode((session as { mode?: unknown } | undefined)?.mode));
    setActivePlan(
      parsePlanEnvelope((session as { active_plan?: unknown } | undefined)?.active_plan),
    );
    setActiveTask(
      parseTaskExecution((session as { active_task?: unknown } | undefined)?.active_task),
    );
    const parsedSelectedModel = parseSelectedModel(session?.selected_model);
    setSelectedModel(parsedSelectedModel);
    return parsedSelectedModel;
  };

  const createConversation = async (): Promise<{
    sessionId: string | null;
    selectedModel: SelectedModel | null;
  }> => {
    const response = await fetch('/ui/api/sessions', {
      method: 'POST',
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to create chat.'));
    }
    const headerSession = response.headers.get(SESSION_HEADER);
    const payloadSession = extractSessionIdFromPayload(payload);
    const nextSession = (headerSession && headerSession.trim()) || payloadSession || null;
    const session = (payload as { session?: { messages?: unknown; selected_model?: unknown } }).session;
    const sessionModel = parseSelectedModel(session?.selected_model);
    setMessages(parseMessages(session?.messages));
    setChatStreamingState(null);
    setPendingDecision(parseUiDecision((session as { decision?: unknown } | undefined)?.decision));
    setDecisionError(null);
    const parsedOutput = parseSessionOutput(
      (session as { output?: unknown } | undefined)?.output,
    );
    setSessionOutput(parsedOutput.content);
    setSessionFiles(
      parseSessionFiles((session as { files?: unknown } | undefined)?.files),
    );
    setSessionArtifacts(
      parseSessionArtifacts((session as { artifacts?: unknown } | undefined)?.artifacts),
    );
    setSessionMode(parseSessionMode((session as { mode?: unknown } | undefined)?.mode));
    setActivePlan(
      parsePlanEnvelope((session as { active_plan?: unknown } | undefined)?.active_plan),
    );
    setActiveTask(
      parseTaskExecution((session as { active_task?: unknown } | undefined)?.active_task),
    );
    setSelectedModel(sessionModel);
    return { sessionId: nextSession, selectedModel: sessionModel };
  };

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      setSessionsLoading(true);
      try {
        const modelsPromise = loadModels();
        const foldersPromise = loadFolders();
        const composerPromise = loadComposerSettings().catch(() => DEFAULT_COMPOSER_SETTINGS);
        const listedSessions = await loadSessions();

        const storedSession = loadLastSessionId();
        const storedExists =
          storedSession && listedSessions.some((session) => session.session_id === storedSession);
        let nextSession = storedExists ? storedSession : null;
        if (!nextSession && listedSessions.length > 0) {
          nextSession = listedSessions[0].session_id;
        }
        if (!nextSession) {
          const created = await createConversation();
          nextSession = created.sessionId;
        }

        if (!cancelled) {
          let selectedFromSession: SelectedModel | null = null;
          if (nextSession) {
            setSelectedConversation(nextSession);
            selectedFromSession = await loadConversation(nextSession);
            saveLastSessionId(nextSession);
          }
          const models = await modelsPromise;
          await foldersPromise;
          await composerPromise;
          if (nextSession && !selectedFromSession && !lastModelApplied) {
            const lastModel = loadLastModel();
            if (lastModel && isModelAvailable(lastModel, models)) {
              try {
                const applied = await setSessionModel(
                  nextSession,
                  lastModel.provider,
                  lastModel.model,
                );
                setSelectedModel(applied);
                saveLastModel(applied);
                setLastModelApplied(true);
              } catch (error) {
                const message =
                  error instanceof Error ? error.message : 'Failed to restore last model.';
                setStatusMessage(message);
              }
            }
          }
          setStatusMessage(null);
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Failed to initialize chat.';
          setStatusMessage(message);
        }
      } finally {
        if (!cancelled) {
          setSessionsLoading(false);
        }
      }
    };

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (selectedModel) {
      saveLastModel(selectedModel);
    }
  }, [selectedModel]);

  useEffect(() => {
    if (!sending) {
      setAwaitingFirstAssistantChunk(false);
    }
  }, [sending]);

  useEffect(() => {
    if (typeof window === 'undefined' || !selectedConversation) {
      return;
    }
    setChatStreamingState(null);
    const streamUrl = `/ui/api/events/stream?session_id=${encodeURIComponent(selectedConversation)}`;
    const router = new SseRouter({
      onChatStreamStart: (streamId) => {
        setAwaitingFirstAssistantChunk(false);
        setChatStreamingState({ streamId, content: '' });
      },
      onChatStreamDelta: (streamId, delta) => {
        setAwaitingFirstAssistantChunk(false);
        setChatStreamingState((prev) => {
          if (!prev || prev.streamId !== streamId) {
            return { streamId, content: delta };
          }
          return { streamId, content: `${prev.content}${delta}` };
        });
      },
      onChatStreamDone: () => {},
      onDecisionPacket: (decisionPayload) => {
        setPendingDecision(parseUiDecision(decisionPayload));
        setDecisionError(null);
      },
      onSessionWorkflow: (workflow) => {
        setSessionMode(workflow.mode);
        setActivePlan(workflow.activePlan);
        setActiveTask(workflow.activeTask);
        setPlanError(null);
      },
      onCanvasStreamStart: (artifactId) => {
        const uiArtifactId = toOutputArtifactUiId(artifactId);
        setAwaitingFirstAssistantChunk(false);
        setArtifactPanelOpen(true);
        setArtifactViewerArtifactId(uiArtifactId);
        setStreamingContentByArtifactId((prev) => ({ ...prev, [uiArtifactId]: '' }));
      },
      onCanvasStreamDelta: (artifactId, delta) => {
        const uiArtifactId = toOutputArtifactUiId(artifactId);
        setAwaitingFirstAssistantChunk(false);
        setStreamingContentByArtifactId((prev) => {
          const nextChunk = `${prev[uiArtifactId] ?? ''}${delta}`;
          return { ...prev, [uiArtifactId]: nextChunk };
        });
      },
      onCanvasStreamDone: (artifactId) => {
        const uiArtifactId = toOutputArtifactUiId(artifactId);
        setStreamingContentByArtifactId((prev) => {
          const next = { ...prev };
          delete next[uiArtifactId];
          return next;
        });
      },
      onDropped: (reason, detail, counters) => {
        console.warn('[sse:dropped]', reason, detail, counters);
      },
    });

    const connection = openSseConnection(streamUrl, {
      onMessage: (rawData) => {
        router.routeRawMessage(rawData);
      },
      onError: () => {},
    });

    return () => {
      connection.close();
      setChatStreamingState(null);
    };
  }, [selectedConversation]);

  const handleSelectConversation = async (sessionId: string) => {
    if (!sessionId || sessionId === selectedConversation) {
      return;
    }
    setSelectedConversation(sessionId);
    setArtifactViewerArtifactId(null);
    setStreamingContentByArtifactId({});
    setChatStreamingState(null);
    setPendingDecision(null);
    setDecisionError(null);
    try {
      await loadConversation(sessionId);
      saveLastSessionId(sessionId);
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load selected chat.';
      setStatusMessage(message);
    }
  };

  const handleCreateFolder = async () => {
    if (typeof window === 'undefined') {
      return;
    }
    const name = window.prompt('Folder name');
    if (!name || !name.trim()) {
      return;
    }
    try {
      await createFolder(name.trim());
      await loadFolders();
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create folder.';
      setStatusMessage(message);
    }
  };

  const handleRenameChat = async (sessionId: string) => {
    if (typeof window === 'undefined') {
      return;
    }
    const session = sessions.find((item) => item.session_id === sessionId);
    const currentTitle = session?.title ?? '';
    const nextTitle = window.prompt('Rename chat', currentTitle);
    if (!nextTitle || !nextTitle.trim()) {
      return;
    }
    try {
      await renameSession(sessionId, nextTitle.trim());
      await loadSessions();
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to rename chat.';
      setStatusMessage(message);
    }
  };

  const handleMoveChatToFolder = async (sessionId: string, folderId: string | null) => {
    try {
      await moveSessionToFolder(sessionId, folderId);
      await loadSessions();
      setStatusMessage(null);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to move chat to folder.';
      setStatusMessage(message);
    }
  };

  const handleCreateConversation = async () => {
    try {
      const created = await createConversation();
      const nextSession = created.sessionId;
      if (!nextSession) {
        setStatusMessage('Failed to create chat.');
        return;
      }
      setSelectedConversation(nextSession);
    setArtifactViewerArtifactId(null);
    setStreamingContentByArtifactId({});
    setChatStreamingState(null);
    setPendingDecision(null);
    setDecisionError(null);
      saveLastSessionId(nextSession);
      await loadSessions();
      if (!created.selectedModel && providerModels.length > 0) {
        const lastModel = loadLastModel();
        if (lastModel && isModelAvailable(lastModel, providerModels)) {
          try {
            const applied = await setSessionModel(
              nextSession,
              lastModel.provider,
              lastModel.model,
            );
            setSelectedModel(applied);
            saveLastModel(applied);
            setLastModelApplied(true);
          } catch (error) {
            const message =
              error instanceof Error ? error.message : 'Failed to restore last model.';
            setStatusMessage(message);
          }
        }
      }
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create chat.';
      setStatusMessage(message);
    }
  };

  const handleDeleteConversation = async (sessionId: string) => {
    if (!sessionId || deletingSessionId === sessionId) {
      return;
    }
    setDeletingSessionId(sessionId);
    try {
      const response = await fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'DELETE',
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to delete chat.'));
      }

      const updatedSessions = await loadSessions();

      if (selectedConversation === sessionId) {
        if (updatedSessions.length > 0) {
          const nextSession = updatedSessions[0].session_id;
          setSelectedConversation(nextSession);
          await loadConversation(nextSession);
          saveLastSessionId(nextSession);
        } else {
          const created = await createConversation();
          if (created.sessionId) {
            setSelectedConversation(created.sessionId);
            await loadSessions();
            saveLastSessionId(created.sessionId);
          } else {
            setSelectedConversation(null);
            setMessages([]);
            setSessionOutput(null);
            setSessionFiles([]);
            setSessionArtifacts([]);
            setStreamingContentByArtifactId({});
            setChatStreamingState(null);
            setSelectedModel(null);
            saveLastSessionId(null);
          }
        }
      }

      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete chat.';
      setStatusMessage(message);
    } finally {
      setDeletingSessionId(null);
    }
  };

  const handleSetModel = async (provider: string, model: string): Promise<boolean> => {
    if (!selectedConversation || savingModel) {
      return false;
    }
    setSavingModel(true);
    try {
      const nextModel = await setSessionModel(selectedConversation, provider, model);
      setSelectedModel(nextModel);
      saveLastModel(nextModel);
      setStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to set model.';
      setStatusMessage(message);
      return false;
    } finally {
      setSavingModel(false);
    }
  };

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

  const applySessionPayload = (
    payload: unknown,
    options: SessionPayloadApplyOptions,
  ): { decision: UiDecision | null } => {
    if (!isRecord(payload)) {
      return { decision: null };
    }

    if (Object.prototype.hasOwnProperty.call(payload, 'messages')) {
      setMessages(parseMessages(payload.messages));
    }

    const parsedDecision = parseUiDecision(payload.decision);
    if (Object.prototype.hasOwnProperty.call(payload, 'decision')) {
      setPendingDecision(parsedDecision);
      setDecisionError(null);
    }

    if (Object.prototype.hasOwnProperty.call(payload, 'output')) {
      const parsedOutput = parseSessionOutput(payload.output);
      setSessionOutput(parsedOutput.content);
    }

    if (Object.prototype.hasOwnProperty.call(payload, 'files')) {
      setSessionFiles(parseSessionFiles(payload.files));
    }

    if (Object.prototype.hasOwnProperty.call(payload, 'artifacts')) {
      setSessionArtifacts(parseSessionArtifacts(payload.artifacts));
    }

    const parsedModel = parseSelectedModel(payload.selected_model);
    if (parsedModel) {
      setSelectedModel(parsedModel);
      saveLastModel(parsedModel);
    }

    if (Object.prototype.hasOwnProperty.call(payload, 'mode')) {
      setSessionMode(parseSessionMode(payload.mode));
    }
    if (Object.prototype.hasOwnProperty.call(payload, 'active_plan')) {
      setActivePlan(parsePlanEnvelope(payload.active_plan));
    }
    if (Object.prototype.hasOwnProperty.call(payload, 'active_task')) {
      setActiveTask(parseTaskExecution(payload.active_task));
    }

    if (options.applyDisplay) {
      const displayDecision = parseDisplayDecision(payload.display);
      if (displayDecision?.target === 'canvas') {
        setArtifactPanelOpen(true);
        if (displayDecision.artifactId) {
          setArtifactViewerArtifactId(`output-${displayDecision.artifactId}`);
        }
      } else {
        setArtifactViewerArtifactId(null);
      }
    }

    return { decision: parsedDecision };
  };

  const applyWorkflowPayload = (payload: unknown) => {
    const decoded = decodeSessionWorkflowPayload(payload);
    if (!decoded.ok) {
      return;
    }
    setSessionMode(decoded.value.mode);
    setActivePlan(decoded.value.activePlan);
    setActiveTask(decoded.value.activeTask);
  };

  const handleChangeMode = async (mode: SessionMode) => {
    if (!selectedConversation || planBusy) {
      return;
    }
    setPlanBusy(true);
    setPlanError(null);
    try {
      const response = await fetch('/ui/api/mode', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({
          mode,
          confirm: mode === 'act',
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to switch mode.'));
      }
      applyWorkflowPayload(payload);
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to switch mode.';
      setPlanError(message);
    } finally {
      setPlanBusy(false);
    }
  };

  const handlePlanDraft = async (goal: string) => {
    if (!selectedConversation || planBusy) {
      return;
    }
    setPlanBusy(true);
    setPlanError(null);
    try {
      const response = await fetch('/ui/api/plan/draft', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({ goal }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to build draft.'));
      }
      applyWorkflowPayload(payload);
      setStatusMessage('Plan draft updated.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to build draft.';
      setPlanError(message);
    } finally {
      setPlanBusy(false);
    }
  };

  const handlePlanApprove = async () => {
    if (!selectedConversation || planBusy) {
      return;
    }
    setPlanBusy(true);
    setPlanError(null);
    try {
      const response = await fetch('/ui/api/plan/approve', {
        method: 'POST',
        headers: {
          [SESSION_HEADER]: selectedConversation,
        },
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to approve plan.'));
      }
      applyWorkflowPayload(payload);
      setStatusMessage('Plan approved.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to approve plan.';
      setPlanError(message);
    } finally {
      setPlanBusy(false);
    }
  };

  const handlePlanExecute = async () => {
    if (!selectedConversation || !activePlan || planBusy) {
      return;
    }
    setPlanBusy(true);
    setPlanError(null);
    try {
      const response = await fetch('/ui/api/plan/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({ plan_hash: activePlan.plan_hash }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to execute plan.'));
      }
      applyWorkflowPayload(payload);
      setStatusMessage('Plan execution started.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to execute plan.';
      setPlanError(message);
    } finally {
      setPlanBusy(false);
    }
  };

  const handlePlanCancel = async () => {
    if (!selectedConversation || planBusy) {
      return;
    }
    setPlanBusy(true);
    setPlanError(null);
    try {
      const response = await fetch('/ui/api/plan/cancel', {
        method: 'POST',
        headers: {
          [SESSION_HEADER]: selectedConversation,
        },
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to cancel plan.'));
      }
      applyWorkflowPayload(payload);
      setStatusMessage('Plan execution cancelled.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to cancel plan.';
      setPlanError(message);
    } finally {
      setPlanBusy(false);
    }
  };

  const handleSend = async (payload: CanvasSendPayload): Promise<boolean> => {
    if (!selectedConversation || sending) {
      return false;
    }
    const trimmed = payload.content.trim();
    const normalizedAttachments = normalizeComposerAttachments(payload.attachments ?? []);
    if (!trimmed && normalizedAttachments.length === 0) {
      return false;
    }
    setPendingUserMessage({ content: trimmed, attachments: normalizedAttachments });
    setPendingSessionId(selectedConversation);
    setChatStreamingState(null);
    setAwaitingFirstAssistantChunk(true);
    const forceCanvasForRequest = forceCanvasNext;
    if (forceCanvasForRequest) {
      setForceCanvasNext(false);
    }
    setSending(true);
    try {
      const response = await fetch('/ui/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({
          content: trimmed,
          force_canvas: forceCanvasForRequest,
          attachments: normalizedAttachments.length > 0 ? normalizedAttachments : undefined,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to send message.'));
      }

      const headerSession = response.headers.get(SESSION_HEADER);
      const payloadSession = extractSessionIdFromPayload(payload);
      const nextSession =
        (headerSession && headerSession.trim()) || payloadSession || selectedConversation;
      if (nextSession !== selectedConversation) {
        setSelectedConversation(nextSession);
      }
      saveLastSessionId(nextSession);

      setPendingUserMessage(null);
      setPendingSessionId(null);
      setChatStreamingState(null);
      applySessionPayload(payload, { applyDisplay: true });
      await loadSessions();
      setStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send message.';
      setStatusMessage(message);
      setPendingUserMessage(null);
      setPendingSessionId(null);
      setChatStreamingState(null);
      if (forceCanvasForRequest) {
        setForceCanvasNext(true);
      }
      return false;
    } finally {
      setSending(false);
    }
  };

  const handleDownloadArtifact = async (artifact: Artifact) => {
    if (!selectedConversation) {
      return;
    }
    try {
      if (artifact.sourceArtifactId) {
        const response = await fetch(
          `/ui/api/sessions/${encodeURIComponent(selectedConversation)}/artifacts/${encodeURIComponent(
            artifact.sourceArtifactId,
          )}/download`,
          {
            headers: {
              [SESSION_HEADER]: selectedConversation,
            },
          },
        );
        if (!response.ok) {
          const errorMessage = await extractErrorFromResponse(
            response,
            'Failed to download artifact.',
          );
          throw new Error(errorMessage);
        }
        const blob = await response.blob();
        const fallbackName = artifact.fileName?.trim() || `${artifact.name}.${artifact.type.toLowerCase()}`;
        const fileName = extractFilenameFromDisposition(
          response.headers.get('Content-Disposition'),
          fallbackName,
        );
        triggerBrowserDownload(blob, fileName);
        setStatusMessage(null);
        return;
      }
      if (artifact.sessionFilePath) {
        const response = await fetch(
          `/ui/api/sessions/${encodeURIComponent(selectedConversation)}/files/download?path=${encodeURIComponent(
            artifact.sessionFilePath,
          )}`,
          {
            headers: {
              [SESSION_HEADER]: selectedConversation,
            },
          },
        );
        if (!response.ok) {
          const errorMessage = await extractErrorFromResponse(
            response,
            'Failed to download file.',
          );
          throw new Error(errorMessage);
        }
        const blob = await response.blob();
        const fallbackName = artifact.sessionFilePath.split('/').pop() || artifact.name;
        const fileName = extractFilenameFromDisposition(
          response.headers.get('Content-Disposition'),
          fallbackName,
        );
        triggerBrowserDownload(blob, fileName);
        setStatusMessage(null);
        return;
      }
      const fallbackName = artifact.fileName?.trim() || `${artifact.name}.${artifact.type.toLowerCase()}`;
      triggerBrowserDownload(
        new Blob([artifact.content ?? ''], { type: 'text/plain' }),
        fallbackName,
      );
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to download artifact.';
      setStatusMessage(message);
    }
  };

  const handleDownloadAllArtifacts = async () => {
    if (!selectedConversation) {
      return;
    }
    try {
      const response = await fetch(
        `/ui/api/sessions/${encodeURIComponent(selectedConversation)}/artifacts/download-all`,
        {
          headers: {
            [SESSION_HEADER]: selectedConversation,
          },
        },
      );
      if (!response.ok) {
        const message = await extractErrorFromResponse(
          response,
          'No downloadable file artifacts.',
        );
        throw new Error(message);
      }
      const blob = await response.blob();
      const fileName = extractFilenameFromDisposition(
        response.headers.get('Content-Disposition'),
        'artifacts.zip',
      );
      triggerBrowserDownload(blob, fileName);
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to download artifacts.';
      setStatusMessage(message);
    }
  };

  const handleSendFeedback = async (
    interactionId: string,
    rating: 'good' | 'bad',
  ): Promise<boolean> => {
    try {
      const response = await fetch('/slavik/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interaction_id: interactionId,
          rating,
          labels: [],
          free_text: null,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to save feedback.'));
      }
      setStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save feedback.';
      setStatusMessage(message);
      return false;
    }
  };

  const handleSettingsSaved = async () => {
    try {
      await Promise.all([loadModels(), loadComposerSettings()]);
      setStatusMessage('Settings saved.');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Settings saved, but models refresh failed.';
      setStatusMessage(message);
    }
  };

  const handleWorkspaceGithubImport = async (
    repoUrl: string,
    branch?: string,
  ): Promise<WorkspaceGithubImportResult> => {
    if (!selectedConversation) {
      throw new Error('No active session. Create chat first.');
    }
    const normalizedRepoUrl = repoUrl.trim();
    if (!normalizedRepoUrl) {
      throw new Error('Repository URL is required.');
    }
    const normalizedBranch = (branch ?? '').trim();
    const args = normalizedBranch
      ? `${normalizedRepoUrl} --branch ${normalizedBranch}`
      : normalizedRepoUrl;

    const response = await fetch('/ui/api/project/command', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        [SESSION_HEADER]: selectedConversation,
      },
      body: JSON.stringify({
        command: 'github_import',
        args,
      }),
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to run GitHub import.'));
    }

    const headerSession = response.headers.get(SESSION_HEADER);
    const payloadSession = extractSessionIdFromPayload(payload);
    const nextSession =
      (headerSession && headerSession.trim()) || payloadSession || selectedConversation;
    if (nextSession !== selectedConversation) {
      setSelectedConversation(nextSession);
    }
    saveLastSessionId(nextSession);

    const { decision } = applySessionPayload(payload, { applyDisplay: false });
    await loadSessions();
    const pending = decision?.status === 'pending' && decision.blocking === true;
    if (pending) {
      return {
        status: 'pending',
        message: 'Действие ожидает подтверждения в DecisionPanel (AI Assistant).',
      };
    }
    setWorkspaceRefreshToken((value) => value + 1);
    return {
      status: 'done',
      message: 'GitHub import completed.',
    };
  };

  const handleDecisionRespond = async (
    choice: 'approve' | 'reject' | 'edit',
    editedAction?: Record<string, unknown> | null,
  ) => {
    if (!selectedConversation || !pendingDecision) {
      return;
    }
    setDecisionBusy(true);
    setDecisionError(null);
    try {
      const response = await fetch('/ui/api/decision/respond', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({
          session_id: selectedConversation,
          decision_id: pendingDecision.id,
          choice,
          edited_action: choice === 'edit' ? (editedAction ?? {}) : null,
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to resolve decision.'));
      }
      applySessionPayload(payload, { applyDisplay: false });
      const resumeRaw = (payload as { resume?: unknown }).resume;
      if (resumeRaw && typeof resumeRaw === 'object') {
        const resume = resumeRaw as {
          ok?: unknown;
          data?: unknown;
          source_endpoint?: unknown;
          tool_name?: unknown;
          error?: unknown;
        };
        if (resume.source_endpoint === 'workspace.tool') {
          const toolName = typeof resume.tool_name === 'string' ? resume.tool_name : 'workspace tool';
          if (resume.ok === true) {
            setStatusMessage(`Workspace: ${toolName} completed.`);
          } else {
            const errorText =
              typeof resume.error === 'string' && resume.error.trim()
                ? resume.error
                : `Workspace: ${toolName} failed.`;
            setStatusMessage(errorText);
          }
        } else if (resume.source_endpoint === 'project.command') {
          const toolName = typeof resume.tool_name === 'string' ? resume.tool_name : 'project';
          const data = resume.data && typeof resume.data === 'object'
            ? (resume.data as { command?: unknown; output?: unknown })
            : null;
          const command = data && typeof data.command === 'string' ? data.command : null;
          if (resume.ok === true) {
            if (command === 'github_import') {
              setWorkspaceRefreshToken((value) => value + 1);
            }
            const outputPreview =
              data && typeof data.output === 'string' && data.output.trim()
                ? data.output.trim()
                : null;
            setStatusMessage(
              outputPreview
                ? outputPreview
                : `Project command (${toolName}) completed.`,
            );
          } else {
            const errorText =
              typeof resume.error === 'string' && resume.error.trim()
                ? resume.error
                : `Project command (${toolName}) failed.`;
            setStatusMessage(errorText);
          }
        } else {
          setStatusMessage(null);
        }
      } else {
        setStatusMessage(null);
      }
      await loadSessions();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to resolve decision.';
      setDecisionError(message);
    } finally {
      setDecisionBusy(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-foreground">
      <HistorySidebar
        chats={historyChats}
        folders={folders.map((folder) => ({ id: folder.folder_id, name: folder.name }))}
        activeChatId={selectedConversation}
        onSelectChat={(sessionId) => {
          setView('chat');
          void handleSelectConversation(sessionId);
        }}
        onNewChat={() => {
          setView('chat');
          void handleCreateConversation();
        }}
        onDeleteChat={(sessionId) => {
          void handleDeleteConversation(sessionId);
        }}
        onRenameChat={(sessionId) => {
          void handleRenameChat(sessionId);
        }}
        onMoveChatToFolder={(sessionId, folderId) => {
          void handleMoveChatToFolder(sessionId, folderId);
        }}
        onOpenSearch={() => setSearchOpen(true)}
        onOpenWorkspace={() => setView('workspace')}
        onOpenSettings={() => setSettingsOpen(true)}
        onCreateFolder={() => {
          void handleCreateFolder();
        }}
        compact={activeView === 'workspace'}
      />

      <div className="flex-1 min-w-0 relative">
        {activeView === 'workspace' ? (
          <WorkspaceIde
            sessionId={selectedConversation}
            sessionHeader={SESSION_HEADER}
            modelLabel={modelLabel}
            modelOptions={modelOptions}
            selectedModelValue={selectedModelValue}
            modelsLoading={modelsLoading}
            savingModel={savingModel}
            onSelectModel={(provider, model) => {
              void handleSetModel(provider, model);
            }}
            messages={workspaceMessages}
            sending={sending}
            statusMessage={statusMessage}
            onBackToChat={() => {
              setWorkspaceSettingsOpen(false);
              setView('chat');
            }}
            onOpenWorkspaceSettings={() => setWorkspaceSettingsOpen(true)}
            onSendAgentMessage={(payload) => handleSend(payload)}
            onSendFeedback={(interactionId, rating) => handleSendFeedback(interactionId, rating)}
            mode={sessionMode}
            activePlan={activePlan}
            activeTask={activeTask}
            modeBusy={planBusy}
            modeError={planError}
            onChangeMode={handleChangeMode}
            onPlanDraft={handlePlanDraft}
            onPlanApprove={handlePlanApprove}
            onPlanExecute={handlePlanExecute}
            onPlanCancel={handlePlanCancel}
            decision={pendingDecision}
            decisionBusy={decisionBusy}
            decisionError={decisionError}
            onDecisionRespond={(choice, editedAction) => {
              void handleDecisionRespond(choice, editedAction);
            }}
            refreshToken={workspaceRefreshToken}
          />
        ) : (
          <>
            <Canvas
              className="h-full"
              messages={canvasMessages}
              pendingMessage={pendingCanvasMessage}
              streamingAssistantMessage={streamingAssistantCanvasMessage}
              showAssistantLoading={showAssistantLoading}
              sending={sending}
              onSendMessage={(payload) => handleSend(payload)}
              onSendFeedback={(interactionId, rating) => handleSendFeedback(interactionId, rating)}
              modelName={modelLabel}
              onOpenSettings={() => setSettingsOpen(true)}
              statusMessage={statusMessage}
              longPasteToFileEnabled={composerSettings.longPasteToFileEnabled}
              longPasteThresholdChars={composerSettings.longPasteThresholdChars}
              modelOptions={modelOptions}
              selectedModelValue={selectedModelValue}
              onSelectModel={(provider, model) => {
                void handleSetModel(provider, model);
              }}
              modelsLoading={modelsLoading}
              savingModel={savingModel}
              forceCanvasNext={forceCanvasNext}
              onToggleForceCanvasNext={() => {
                setForceCanvasNext((prev) => !prev);
              }}
              decision={pendingDecision}
              decisionBusy={decisionBusy}
              decisionError={decisionError}
              onDecisionRespond={(choice, editedAction) => {
                void handleDecisionRespond(choice, editedAction);
              }}
            />

            {!artifactPanelOpen ? (
              <button
                onClick={() => {
                  setArtifactViewerArtifactId(null);
                  setArtifactPanelOpen(true);
                }}
                className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-[#141418] border border-[#1f1f24] hover:border-[#2a2a30] hover:bg-[#1b1b20] flex items-center justify-center transition-all cursor-pointer shadow-lg shadow-black/30"
                title="Open Artifacts"
              >
                <PanelRight className="w-4.5 h-4.5 text-[#888]" />
              </button>
            ) : null}
          </>
        )}
      </div>

      {activeView === 'chat' ? (
        <ArtifactPanel
          isOpen={artifactPanelOpen}
          onClose={() => {
            setArtifactPanelOpen(false);
            setArtifactViewerArtifactId(null);
          }}
          artifacts={artifacts}
          autoOpenArtifactId={artifactViewerArtifactId}
          onDownloadArtifact={(artifact) => {
            void handleDownloadArtifact(artifact);
          }}
          onDownloadAll={() => {
            void handleDownloadAllArtifacts();
          }}
        />
      ) : null}

      <SearchModal
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        chats={searchChats}
        onSelectChat={(sessionId) => {
          setView('chat');
          void handleSelectConversation(sessionId);
          setSearchOpen(false);
        }}
        onNewChat={() => {
          setView('chat');
          void handleCreateConversation();
        }}
      />

      <WorkspaceSettingsModal
        isOpen={workspaceSettingsOpen}
        onClose={() => setWorkspaceSettingsOpen(false)}
        pendingDecision={pendingDecision}
        onRunGithubImport={(repoUrl, branch) => handleWorkspaceGithubImport(repoUrl, branch)}
      />

      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={() => {
          void handleSettingsSaved();
        }}
        sessionId={selectedConversation}
        sessionHeader={SESSION_HEADER}
      />
    </div>
  );
}
