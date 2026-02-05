import { useEffect, useState } from 'react';

import { ChatArea } from './components/ChatArea';
import { Settings } from './components/Settings';
import { Sidebar } from './components/Sidebar';
import { Workspace } from './components/Workspace';
import type { ChatMessage, ProviderModels, SelectedModel, SessionSummary } from './types';

const SESSION_HEADER = 'X-Slavik-Session';

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

  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [savingModel, setSavingModel] = useState(false);
  const [sending, setSending] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [workspaceCollapsed, setWorkspaceCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

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

  const createConversation = async (): Promise<string | null> => {
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
    setMessages(parseMessages(session?.messages));
    setSelectedModel(parseSelectedModel(session?.selected_model));
    return nextSession;
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
        setSelectedModel(parseSelectedModel((statusPayload as { selected_model?: unknown }).selected_model));

        const modelsPromise = loadModels();
        const listedSessions = await loadSessions();

        let nextSession = statusSession;
        if (!nextSession && listedSessions.length > 0) {
          nextSession = listedSessions[0].session_id;
        }
        if (!nextSession) {
          nextSession = await createConversation();
        }

        if (!cancelled) {
          if (nextSession) {
            setSelectedConversation(nextSession);
            await loadConversation(nextSession);
          }
          await modelsPromise;
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
      const nextSession = await createConversation();
      if (!nextSession) {
        setStatusMessage('Failed to create chat.');
        return;
      }
      setSelectedConversation(nextSession);
      await loadSessions();
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
          if (created) {
            setSelectedConversation(created);
            await loadSessions();
          } else {
            setSelectedConversation(null);
            setMessages([]);
            setSelectedModel(null);
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
      const response = await fetch('/ui/api/session-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({ provider, model }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to set model.'));
      }
      const parsed = parseSelectedModel((payload as { selected_model?: unknown }).selected_model);
      setSelectedModel(parsed || { provider, model });
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

      setMessages(parseMessages((payload as { messages?: unknown }).messages));
      const parsedModel = parseSelectedModel((payload as { selected_model?: unknown }).selected_model);
      if (parsedModel) {
        setSelectedModel(parsedModel);
      }
      await loadSessions();
      setStatusMessage(null);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send message.';
      setStatusMessage(message);
      return false;
    } finally {
      setSending(false);
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
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <ChatArea
        conversationId={selectedConversation}
        messages={messages}
        sending={sending}
        statusMessage={statusMessage}
        selectedModel={selectedModel}
        providerModels={providerModels}
        modelsLoading={modelsLoading}
        savingModel={savingModel}
        onSend={handleSend}
        onSetModel={handleSetModel}
      />

      <Workspace
        collapsed={workspaceCollapsed}
        onToggleCollapse={() => setWorkspaceCollapsed(!workspaceCollapsed)}
      />

      <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
