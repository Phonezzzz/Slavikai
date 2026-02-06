import { useEffect, useState } from 'react';

import { ChatArea } from './components/ChatArea';
import { Settings } from './components/Settings';
import { Sidebar } from './components/Sidebar';
import { ChatCanvas } from './components/ChatCanvas';
import type {
  ChatMessage,
  ProviderModels,
  SelectedModel,
  SessionSummary,
  UploadHistoryItem,
} from './types';

const SESSION_HEADER = 'X-Slavik-Session';
const SCROLLBAR_REVEAL_DISTANCE_PX = 38;

const isChatMessage = (value: unknown): value is ChatMessage => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as { role?: unknown; content?: unknown };
  if (candidate.role !== 'user' && candidate.role !== 'assistant' && candidate.role !== 'system') {
    return false;
  }
  return typeof candidate.content === 'string';
};

const parseMessages = (value: unknown): ChatMessage[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(isChatMessage);
};

const parseSelectedModel = (value: unknown): SelectedModel | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as { provider?: unknown; model?: unknown };
  if (typeof candidate.provider !== 'string' || typeof candidate.model !== 'string') {
    return null;
  }
  return { provider: candidate.provider, model: candidate.model };
};

const parseProviderModels = (value: unknown): ProviderModels[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const providers: ProviderModels[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as { provider?: unknown; models?: unknown; error?: unknown };
    if (typeof candidate.provider !== 'string' || !Array.isArray(candidate.models)) {
      continue;
    }
    providers.push({
      provider: candidate.provider,
      models: candidate.models.filter((entry): entry is string => typeof entry === 'string'),
      error:
        typeof candidate.error === 'string' || candidate.error === null ? candidate.error : null,
    });
  }
  return providers;
};

const uploadStorageKey = (sessionId: string) => `slavik.uploads.${sessionId}`;

const parseUploads = (value: unknown): UploadHistoryItem[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const uploads: UploadHistoryItem[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      id?: unknown;
      name?: unknown;
      size?: unknown;
      type?: unknown;
      preview?: unknown;
      previewType?: unknown;
      previewUrl?: unknown;
      createdAt?: unknown;
    };
    if (
      typeof candidate.id !== 'string' ||
      typeof candidate.name !== 'string' ||
      typeof candidate.size !== 'number' ||
      typeof candidate.type !== 'string' ||
      typeof candidate.preview !== 'string' ||
      typeof candidate.createdAt !== 'string'
    ) {
      continue;
    }
    if (
      candidate.previewType !== 'text' &&
      candidate.previewType !== 'image' &&
      candidate.previewType !== 'binary'
    ) {
      continue;
    }
    uploads.push({
      id: candidate.id,
      name: candidate.name,
      size: candidate.size,
      type: candidate.type,
      preview: candidate.preview,
      previewType: candidate.previewType,
      previewUrl: typeof candidate.previewUrl === 'string' ? candidate.previewUrl : null,
      createdAt: candidate.createdAt,
    });
  }
  return uploads;
};

const parseSessions = (value: unknown): SessionSummary[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const sessions: SessionSummary[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      session_id?: unknown;
      title?: unknown;
      created_at?: unknown;
      updated_at?: unknown;
      message_count?: unknown;
    };
    if (
      typeof candidate.session_id !== 'string' ||
      typeof candidate.created_at !== 'string' ||
      typeof candidate.updated_at !== 'string' ||
      typeof candidate.message_count !== 'number'
    ) {
      continue;
    }
    const title =
      typeof candidate.title === 'string' && candidate.title.trim()
        ? candidate.title.trim()
        : candidate.session_id.slice(0, 8);
    sessions.push({
      session_id: candidate.session_id,
      title,
      created_at: candidate.created_at,
      updated_at: candidate.updated_at,
      message_count: candidate.message_count,
    });
  }
  return sessions;
};

const extractSessionIdFromPayload = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }
  const maybeSession = payload as {
    session_id?: unknown;
    session?: { session_id?: unknown };
  };
  if (typeof maybeSession.session_id === 'string' && maybeSession.session_id.trim()) {
    return maybeSession.session_id.trim();
  }
  const nested = maybeSession.session;
  if (nested && typeof nested === 'object' && typeof nested.session_id === 'string') {
    return nested.session_id.trim() || null;
  }
  return null;
};

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

export default function App() {
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<SelectedModel | null>(null);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [uploadsBySession, setUploadsBySession] = useState<Record<string, UploadHistoryItem[]>>(
    {},
  );

  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [savingModel, setSavingModel] = useState(false);
  const [sending, setSending] = useState(false);
  const [pendingUserMessage, setPendingUserMessage] = useState<ChatMessage | null>(null);
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [chatCanvasCollapsed, setChatCanvasCollapsed] = useState(() => {
    if (typeof window === 'undefined') {
      return true;
    }
    const stored = window.localStorage.getItem('slavik.chatcanvas.collapsed');
    return stored ? stored === 'true' : true;
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [lastModelApplied, setLastModelApplied] = useState(false);

  useEffect(() => {
    if (!selectedConversation || typeof window === 'undefined') {
      return;
    }
    setUploadsBySession((prev) => {
      if (prev[selectedConversation]) {
        return prev;
      }
      const raw = window.localStorage.getItem(uploadStorageKey(selectedConversation));
      if (!raw) {
        return { ...prev, [selectedConversation]: [] };
      }
      try {
        const parsed = parseUploads(JSON.parse(raw));
        return { ...prev, [selectedConversation]: parsed };
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to parse upload history.';
        console.warn(message);
        return { ...prev, [selectedConversation]: [] };
      }
    });
  }, [selectedConversation]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    let raf = 0;
    const clearReveal = () => {
      const elements = document.querySelectorAll<HTMLElement>('[data-scrollbar]');
      elements.forEach((element) => element.classList.remove('scrollbar-reveal'));
    };
    const handleMove = (event: MouseEvent) => {
      const x = event.clientX;
      const y = event.clientY;
      if (raf) {
        cancelAnimationFrame(raf);
      }
      raf = window.requestAnimationFrame(() => {
        const elements = document.querySelectorAll<HTMLElement>('[data-scrollbar]');
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
    if (!raw) {
      return null;
    }
    try {
      const parsed = JSON.parse(raw) as { provider?: unknown; model?: unknown };
      if (typeof parsed.provider === 'string' && typeof parsed.model === 'string') {
        return { provider: parsed.provider, model: parsed.model };
      }
    } catch {
      return null;
    }
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

  const isModelAvailable = (model: SelectedModel, providers: ProviderModels[]) =>
    providers.some((provider) => provider.provider === model.provider && provider.models.includes(model.model));

  const loadSessions = async (): Promise<SessionSummary[]> => {
    const response = await fetch('/ui/api/sessions');
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to load chat list.'));
    }
    const parsed = parseSessions((payload as { sessions?: unknown }).sessions);
    setSessions(parsed);
    return parsed;
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

  const handleRecordUploads = (sessionId: string, uploads: UploadHistoryItem[]) => {
    setUploadsBySession((prev) => {
      const current = prev[sessionId] ?? [];
      const next = [...current, ...uploads];
      if (typeof window !== 'undefined') {
        try {
          window.localStorage.setItem(uploadStorageKey(sessionId), JSON.stringify(next));
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Failed to save upload history.';
          console.warn(message);
        }
      }
      return { ...prev, [sessionId]: next };
    });
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

  const loadConversation = async (sessionId: string): Promise<void> => {
    const response = await fetch(`/ui/api/sessions/${encodeURIComponent(sessionId)}`, {
      headers: {
        [SESSION_HEADER]: sessionId,
      },
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, 'Failed to load chat history.'));
    }
    const session = (payload as { session?: { messages?: unknown; selected_model?: unknown } }).session;
    setMessages(parseMessages(session?.messages));
    setSelectedModel(parseSelectedModel(session?.selected_model));
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
    setSelectedModel(sessionModel);
    return { sessionId: nextSession, selectedModel: sessionModel };
  };

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      setSessionsLoading(true);
      try {
        const statusResp = await fetch('/ui/api/status');
        const statusPayload: unknown = await statusResp.json();
        if (!statusResp.ok) {
          throw new Error(extractErrorMessage(statusPayload, 'Failed to initialize UI session.'));
        }
        const fromHeader = statusResp.headers.get(SESSION_HEADER);
        const fromPayload = extractSessionIdFromPayload(statusPayload);
        const statusSession = (fromHeader && fromHeader.trim()) || fromPayload || null;
        const statusSelected = parseSelectedModel(
          (statusPayload as { selected_model?: unknown }).selected_model,
        );
        setSelectedModel(statusSelected);

        const modelsPromise = loadModels();
        const listedSessions = await loadSessions();

        let nextSession = statusSession;
        if (!nextSession && listedSessions.length > 0) {
          nextSession = listedSessions[0].session_id;
        }
        if (!nextSession) {
          const created = await createConversation();
          nextSession = created.sessionId;
        }

        if (!cancelled) {
          if (nextSession) {
            setSelectedConversation(nextSession);
            await loadConversation(nextSession);
          }
          const models = await modelsPromise;
          if (nextSession && !statusSelected && !lastModelApplied) {
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

  const handleSelectConversation = async (sessionId: string) => {
    if (!sessionId || sessionId === selectedConversation) {
      return;
    }
    setSelectedConversation(sessionId);
    try {
      await loadConversation(sessionId);
      setStatusMessage(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load selected chat.';
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
        } else {
          const created = await createConversation();
          if (created.sessionId) {
            setSelectedConversation(created.sessionId);
            await loadSessions();
          } else {
            setSelectedConversation(null);
            setMessages([]);
            setSelectedModel(null);
          }
        }
      }

      if (typeof window !== 'undefined') {
        window.localStorage.removeItem(uploadStorageKey(sessionId));
      }
      setUploadsBySession((prev) => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });

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

  const handleSend = async (content: string): Promise<boolean> => {
    if (!selectedConversation || sending) {
      return false;
    }
    const trimmed = content.trim();
    if (!trimmed) {
      return false;
    }
    setPendingUserMessage({ role: 'user', content: trimmed });
    setPendingSessionId(selectedConversation);
    setSending(true);
    try {
      const response = await fetch('/ui/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({ content }),
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

      setPendingUserMessage(null);
      setPendingSessionId(null);
      setMessages(parseMessages((payload as { messages?: unknown }).messages));
      const parsedModel = parseSelectedModel((payload as { selected_model?: unknown }).selected_model);
      if (parsedModel) {
        setSelectedModel(parsedModel);
        saveLastModel(parsedModel);
      }
      await loadSessions();
      setStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send message.';
      setStatusMessage(message);
      setPendingUserMessage(null);
      setPendingSessionId(null);
      return false;
    } finally {
      setSending(false);
    }
  };

  const handleSettingsSaved = async () => {
    try {
      await loadModels();
      setStatusMessage('Settings saved.');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Settings saved, but models refresh failed.';
      setStatusMessage(message);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-foreground">
      <Sidebar
        selectedConversation={selectedConversation}
        conversations={sessions}
        loading={sessionsLoading}
        deletingSessionId={deletingSessionId}
        onSelectConversation={(sessionId) => {
          void handleSelectConversation(sessionId);
        }}
        onCreateConversation={() => {
          void handleCreateConversation();
        }}
        onDeleteConversation={(sessionId) => {
          void handleDeleteConversation(sessionId);
        }}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => {
          const next = !sidebarCollapsed;
          setSidebarCollapsed(next);
          if (typeof window !== 'undefined') {
            window.localStorage.setItem('slavik.sidebar.collapsed', String(next));
          }
        }}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <ChatArea
        conversationId={selectedConversation}
        messages={messages}
        sending={sending}
        pendingUserMessage={
          pendingSessionId === selectedConversation ? pendingUserMessage : null
        }
        statusMessage={statusMessage}
        selectedModel={selectedModel}
        providerModels={providerModels}
        modelsLoading={modelsLoading}
        savingModel={savingModel}
        onSend={handleSend}
        onSetModel={handleSetModel}
        onRecordUploads={handleRecordUploads}
      />

      <ChatCanvas
        messages={messages}
        uploads={selectedConversation ? uploadsBySession[selectedConversation] ?? [] : []}
        collapsed={chatCanvasCollapsed}
        onToggleCollapse={() => {
          const next = !chatCanvasCollapsed;
          setChatCanvasCollapsed(next);
          if (typeof window !== 'undefined') {
            window.localStorage.setItem('slavik.chatcanvas.collapsed', String(next));
          }
        }}
      />

      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={() => {
          void handleSettingsSaved();
        }}
      />
    </div>
  );
}
