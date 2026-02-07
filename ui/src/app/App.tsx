import { useEffect, useMemo, useState } from 'react';
import { PanelRight } from 'lucide-react';

import { ArtifactPanel } from './components/artifact-panel';
import { Canvas, type CanvasMessage } from './components/canvas';
import type { Artifact } from './components/artifacts-sidebar';
import { HistorySidebar } from './components/history-sidebar';
import { SearchModal } from './components/search-modal';
import { Settings } from './components/Settings';
import type {
  ChatMessage,
  FolderSummary,
  ProviderModels,
  SelectedModel,
  SessionSummary,
} from './types';

const SESSION_HEADER = 'X-Slavik-Session';
const SCROLLBAR_REVEAL_DISTANCE_PX = 38;
const LAST_SESSION_KEY = 'slavik.last.session';

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

const loadLastSessionId = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  const raw = window.localStorage.getItem(LAST_SESSION_KEY);
  if (!raw || !raw.trim()) {
    return null;
  }
  return raw.trim();
};

const saveLastSessionId = (sessionId: string | null) => {
  if (typeof window === 'undefined') {
    return;
  }
  if (!sessionId) {
    window.localStorage.removeItem(LAST_SESSION_KEY);
    return;
  }
  window.localStorage.setItem(LAST_SESSION_KEY, sessionId);
};

const sortSessionsByUpdated = (value: SessionSummary[]): SessionSummary[] => {
  return [...value].sort((a, b) => {
    const aTime = Date.parse(a.updated_at);
    const bTime = Date.parse(b.updated_at);
    const aValue = Number.isNaN(aTime) ? 0 : aTime;
    const bValue = Number.isNaN(bTime) ? 0 : bTime;
    return bValue - aValue;
  });
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
      title_override?: unknown;
      folder_id?: unknown;
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
      title_override: typeof candidate.title_override === 'string' ? candidate.title_override : null,
      folder_id: typeof candidate.folder_id === 'string' ? candidate.folder_id : null,
    });
  }
  return sessions;
};

const parseFolders = (value: unknown): FolderSummary[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const folders: FolderSummary[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      folder_id?: unknown;
      name?: unknown;
      created_at?: unknown;
      updated_at?: unknown;
    };
    if (
      typeof candidate.folder_id !== 'string' ||
      typeof candidate.name !== 'string' ||
      typeof candidate.created_at !== 'string' ||
      typeof candidate.updated_at !== 'string'
    ) {
      continue;
    }
    folders.push({
      folder_id: candidate.folder_id,
      name: candidate.name,
      created_at: candidate.created_at,
      updated_at: candidate.updated_at,
    });
  }
  return folders;
};

const parseSessionOutput = (value: unknown): { content: string | null; updatedAt: string | null } => {
  if (!value || typeof value !== 'object') {
    return { content: null, updatedAt: null };
  }
  const candidate = value as { content?: unknown; updated_at?: unknown };
  const content = typeof candidate.content === 'string' ? candidate.content : null;
  const updatedAt = typeof candidate.updated_at === 'string' ? candidate.updated_at : null;
  return { content, updatedAt };
};

const parseSessionFiles = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
};

const groupSessionByDate = (value: string): 'today' | 'yesterday' | 'older' => {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return 'older';
  }
  const date = new Date(parsed);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfYesterday = new Date(startOfToday);
  startOfYesterday.setDate(startOfYesterday.getDate() - 1);
  if (date >= startOfToday) {
    return 'today';
  }
  if (date >= startOfYesterday) {
    return 'yesterday';
  }
  return 'older';
};

const buildCanvasMessages = (messages: ChatMessage[]): CanvasMessage[] => {
  return messages
    .filter((message) => message.role === 'user' || message.role === 'assistant')
    .map((message, index) => ({
      id: `${message.role}-${index}`,
      role: message.role,
      content: message.content,
    }));
};

const inferArtifactType = (content: string): Artifact['type'] => {
  if (content.includes('```')) {
    const match = content.match(/```\\s*([a-zA-Z0-9_-]+)/);
    const lang = match ? match[1].toLowerCase() : '';
    if (lang === 'python' || lang === 'py') return 'PY';
    if (lang === 'javascript' || lang === 'js') return 'JS';
    if (lang === 'typescript' || lang === 'ts') return 'TS';
    if (lang === 'json') return 'JSON';
    if (lang === 'html') return 'HTML';
    if (lang === 'css') return 'CSS';
    return 'MD';
  }
  return 'TXT';
};

const inferArtifactTypeFromPath = (path: string): Artifact['type'] => {
  const normalized = path.trim().toLowerCase();
  if (normalized.endsWith('.py')) return 'PY';
  if (normalized.endsWith('.ts') || normalized.endsWith('.tsx')) return 'TS';
  if (normalized.endsWith('.js') || normalized.endsWith('.jsx')) return 'JS';
  if (normalized.endsWith('.json')) return 'JSON';
  if (normalized.endsWith('.html')) return 'HTML';
  if (normalized.endsWith('.css')) return 'CSS';
  if (normalized.endsWith('.md') || normalized.endsWith('.markdown')) return 'MD';
  return 'TXT';
};

const inferArtifactCategory = (type: Artifact['type']): Artifact['category'] => {
  if (type === 'TXT' || type === 'MD') return 'Document';
  if (type === 'JSON') return 'Config';
  return 'Code';
};

const buildArtifactsFromSources = (
  outputText: string | null,
  files: string[],
): Artifact[] => {
  const artifacts: Artifact[] = [];
  if (outputText && outputText.trim()) {
    const firstLine = outputText.split('\n').find((line) => line.trim());
    const title = firstLine ? firstLine.trim().slice(0, 60) : 'Latest output';
    const type = inferArtifactType(outputText);
    artifacts.push({
      id: 'output-latest',
      name: title,
      type,
      category: inferArtifactCategory(type),
      content: outputText,
    });
  }

  const seen = new Set<string>();
  for (const rawPath of files) {
    const path = rawPath.trim();
    if (!path || seen.has(path)) {
      continue;
    }
    seen.add(path);
    const type = inferArtifactTypeFromPath(path);
    artifacts.push({
      id: `file-${path}`,
      name: path,
      type,
      category: inferArtifactCategory(type),
      content: `File path: ${path}`,
    });
  }

  return artifacts;
};

const shouldAutoOpenCanvas = (outputText: string | null): boolean => {
  if (!outputText) {
    return false;
  }
  const normalized = outputText.trim();
  if (!normalized) {
    return false;
  }
  return normalized.includes('```') || normalized.length >= 280;
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
  const [sessionOutput, setSessionOutput] = useState<string | null>(null);
  const [sessionFiles, setSessionFiles] = useState<string[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);

  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [savingModel, setSavingModel] = useState(false);
  const [sending, setSending] = useState(false);
  const [pendingUserMessage, setPendingUserMessage] = useState<ChatMessage | null>(null);
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [streamingAssistantText, setStreamingAssistantText] = useState<string | null>(null);
  const [streamingSessionId, setStreamingSessionId] = useState<string | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const [artifactPanelOpen, setArtifactPanelOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [lastModelApplied, setLastModelApplied] = useState(false);

  const pendingForCanvas =
    pendingSessionId === selectedConversation ? pendingUserMessage : null;
  const streamingForCanvas =
    streamingSessionId === selectedConversation ? streamingAssistantText : null;
  const canvasMessages = useMemo(
    () => {
      const base = buildCanvasMessages(messages);
      if (streamingForCanvas && streamingForCanvas.trim()) {
        base.push({
          id: 'assistant-stream',
          role: 'assistant',
          content: streamingForCanvas,
        });
      }
      return base;
    },
    [messages, streamingForCanvas],
  );
  const pendingCanvasMessage = useMemo(() => {
    if (!pendingForCanvas || pendingForCanvas.role !== 'user') {
      return null;
    }
    return {
      id: `pending-${pendingForCanvas.content.length}`,
      role: 'user' as const,
      content: pendingForCanvas.content,
    };
  }, [pendingForCanvas]);
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
    () => buildArtifactsFromSources(sessionOutput, sessionFiles),
    [sessionOutput, sessionFiles],
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
    const parsedOutput = parseSessionOutput((outputPayload as { output?: unknown }).output);
    setSessionOutput(parsedOutput.content);
    setSessionFiles(parseSessionFiles((filesPayload as { files?: unknown }).files));
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
    const parsedOutput = parseSessionOutput(
      (session as { output?: unknown } | undefined)?.output,
    );
    setSessionOutput(parsedOutput.content);
    setSessionFiles(
      parseSessionFiles((session as { files?: unknown } | undefined)?.files),
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

  const handleSelectConversation = async (sessionId: string) => {
    if (!sessionId || sessionId === selectedConversation) {
      return;
    }
    setSelectedConversation(sessionId);
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
    setStreamingSessionId(selectedConversation);
    setStreamingAssistantText('');
    setSending(true);
    try {
      const response = await fetch('/ui/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [SESSION_HEADER]: selectedConversation,
        },
        body: JSON.stringify({ content }),
      });
      if (!response.ok) {
        const payload: unknown = await response.json();
        throw new Error(extractErrorMessage(payload, 'Failed to send message.'));
      }
      if (!response.body) {
        throw new Error('Streaming response body is empty.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let streamedAssistant = '';
      let donePayload: unknown = null;
      let streamSessionId = selectedConversation;

      const processEventChunk = (chunk: string) => {
        const lines = chunk.split('\n');
        const dataLines: string[] = [];
        for (const lineRaw of lines) {
          const line = lineRaw.trimEnd();
          if (!line || line.startsWith(':')) {
            continue;
          }
          if (line.startsWith('data:')) {
            dataLines.push(line.slice('data:'.length).trimStart());
          }
        }
        if (dataLines.length === 0) {
          return;
        }
        const dataRaw = dataLines.join('\n');
        let parsed: unknown;
        try {
          parsed = JSON.parse(dataRaw);
        } catch {
          return;
        }
        if (!parsed || typeof parsed !== 'object') {
          return;
        }
        const event = parsed as { type?: unknown; delta?: unknown; message?: unknown; session_id?: unknown };
        if (typeof event.session_id === 'string' && event.session_id.trim()) {
          streamSessionId = event.session_id.trim();
          setStreamingSessionId(streamSessionId);
        }
        if (event.type === 'delta') {
          if (typeof event.delta === 'string' && event.delta) {
            streamedAssistant += event.delta;
            setStreamingAssistantText(streamedAssistant);
          }
          return;
        }
        if (event.type === 'error') {
          const errorMessage =
            typeof event.message === 'string' && event.message.trim()
              ? event.message
              : 'Failed to send message.';
          throw new Error(errorMessage);
        }
        if (event.type === 'done') {
          donePayload = parsed;
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let separatorIndex = buffer.indexOf('\n\n');
        while (separatorIndex >= 0) {
          const eventChunk = buffer.slice(0, separatorIndex);
          buffer = buffer.slice(separatorIndex + 2);
          processEventChunk(eventChunk);
          separatorIndex = buffer.indexOf('\n\n');
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) {
        processEventChunk(buffer);
      }

      const payload = donePayload;
      if (!payload || typeof payload !== 'object') {
        throw new Error('Streaming finished without final payload.');
      }
      const payloadSession = extractSessionIdFromPayload(payload);
      const nextSession = payloadSession || streamSessionId || selectedConversation;
      if (nextSession !== selectedConversation) {
        setSelectedConversation(nextSession);
      }
      saveLastSessionId(nextSession);
      setMessages(parseMessages((payload as { messages?: unknown }).messages));
      const parsedOutput = parseSessionOutput((payload as { output?: unknown }).output);
      setSessionOutput(parsedOutput.content);
      const parsedFiles = parseSessionFiles((payload as { files?: unknown }).files);
      setSessionFiles(parsedFiles);
      if (shouldAutoOpenCanvas(parsedOutput.content)) {
        setArtifactPanelOpen(true);
      }
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
      return false;
    } finally {
      setPendingUserMessage(null);
      setPendingSessionId(null);
      setStreamingAssistantText(null);
      setStreamingSessionId(null);
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
      <HistorySidebar
        chats={historyChats}
        folders={folders.map((folder) => ({ id: folder.folder_id, name: folder.name }))}
        activeChatId={selectedConversation}
        onSelectChat={(sessionId) => {
          void handleSelectConversation(sessionId);
        }}
        onNewChat={() => {
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
        onOpenSettings={() => setSettingsOpen(true)}
        onCreateFolder={() => {
          void handleCreateFolder();
        }}
      />

      <div className="flex-1 min-w-0 relative">
        <Canvas
          className="h-full"
          messages={canvasMessages}
          pendingMessage={pendingCanvasMessage}
          sending={sending}
          onSendMessage={(content) => {
            void handleSend(content);
          }}
          modelName={modelLabel}
          onOpenSettings={() => setSettingsOpen(true)}
          statusMessage={statusMessage}
          modelOptions={modelOptions}
          selectedModelValue={selectedModelValue}
          onSelectModel={(provider, model) => {
            void handleSetModel(provider, model);
          }}
          modelsLoading={modelsLoading}
          savingModel={savingModel}
        />

        {!artifactPanelOpen ? (
          <button
            onClick={() => setArtifactPanelOpen(true)}
            className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-[#141418] border border-[#1f1f24] hover:border-[#2a2a30] hover:bg-[#1b1b20] flex items-center justify-center transition-all cursor-pointer shadow-lg shadow-black/30"
            title="Open Artifacts"
          >
            <PanelRight className="w-4.5 h-4.5 text-[#888]" />
          </button>
        ) : null}
      </div>

      <ArtifactPanel
        isOpen={artifactPanelOpen}
        onClose={() => setArtifactPanelOpen(false)}
        artifacts={artifacts}
      />

      <SearchModal
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        chats={searchChats}
        onSelectChat={(sessionId) => {
          void handleSelectConversation(sessionId);
          setSearchOpen(false);
        }}
        onNewChat={() => {
          void handleCreateConversation();
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
