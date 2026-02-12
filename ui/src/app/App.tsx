import { useEffect, useMemo, useState } from 'react';
import { PanelRight } from 'lucide-react';

import { ArtifactPanel } from './components/artifact-panel';
import {
  Canvas,
  type CanvasComposerAttachment,
  type CanvasMessage,
  type CanvasSendPayload,
} from './components/canvas';
import type { Artifact } from './components/artifacts-sidebar';
import { HistorySidebar } from './components/history-sidebar';
import { SearchModal } from './components/search-modal';
import { Settings } from './components/Settings';
import { WorkspaceIde } from './components/workspace-ide';
import type {
  ChatAttachment,
  ChatMessage,
  FolderSummary,
  ProviderModels,
  SelectedModel,
  SessionSummary,
  UiDecision,
} from './types';

const SESSION_HEADER = 'X-Slavik-Session';
const SCROLLBAR_REVEAL_DISTANCE_PX = 38;
const LAST_SESSION_KEY = 'slavik.last.session';
const WORKSPACE_PATHS = new Set(['/workspace', '/ui/workspace']);

type SessionArtifactRecord = {
  id: string;
  kind: 'output';
  title: string;
  content: string;
  createdAt: string | null;
  displayTarget: 'chat' | 'canvas';
  artifactKind: 'text' | 'file';
  fileName: string | null;
  fileExt: string | null;
  language: string | null;
  fileContent: string | null;
};

type DisplayDecision = {
  target: 'chat' | 'canvas';
  artifactId: string | null;
  forced: boolean;
};

type ChatStreamState = {
  streamId: string;
  content: string;
};

type PendingUserMessage = {
  content: string;
  attachments: ChatAttachment[];
};

type AppView = 'chat' | 'workspace';

const toOutputArtifactUiId = (artifactId: string): string => `output-${artifactId}`;
const DEFAULT_LONG_PASTE_THRESHOLD_CHARS = 12000;

type ComposerUiSettings = {
  longPasteToFileEnabled: boolean;
  longPasteThresholdChars: number;
};

const DEFAULT_COMPOSER_SETTINGS: ComposerUiSettings = {
  longPasteToFileEnabled: true,
  longPasteThresholdChars: DEFAULT_LONG_PASTE_THRESHOLD_CHARS,
};

const parseChatAttachments = (value: unknown): ChatAttachment[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: ChatAttachment[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as { name?: unknown; mime?: unknown; content?: unknown };
    if (
      typeof candidate.name !== 'string'
      || !candidate.name.trim()
      || typeof candidate.mime !== 'string'
      || !candidate.mime.trim()
      || typeof candidate.content !== 'string'
    ) {
      continue;
    }
    normalized.push({
      name: candidate.name.trim(),
      mime: candidate.mime.trim(),
      content: candidate.content,
    });
  }
  return normalized;
};

const isChatMessage = (value: unknown): value is ChatMessage => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as {
    message_id?: unknown;
    role?: unknown;
    content?: unknown;
    created_at?: unknown;
    trace_id?: unknown;
    parent_user_message_id?: unknown;
    attachments?: unknown;
  };
  if (typeof candidate.message_id !== 'string' || !candidate.message_id.trim()) {
    return false;
  }
  if (candidate.role !== 'user' && candidate.role !== 'assistant' && candidate.role !== 'system') {
    return false;
  }
  if (typeof candidate.content !== 'string') {
    return false;
  }
  if (typeof candidate.created_at !== 'string' || !candidate.created_at.trim()) {
    return false;
  }
  if (candidate.trace_id !== null && typeof candidate.trace_id !== 'string') {
    return false;
  }
  if (
    candidate.parent_user_message_id !== null
    && typeof candidate.parent_user_message_id !== 'string'
  ) {
    return false;
  }
  if (candidate.attachments !== undefined && !Array.isArray(candidate.attachments)) {
    return false;
  }
  return true;
};

const parseMessages = (value: unknown): ChatMessage[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(isChatMessage).map((message) => ({
    ...message,
    attachments: parseChatAttachments((message as { attachments?: unknown }).attachments),
  }));
};

const parseComposerSettings = (value: unknown): ComposerUiSettings => {
  if (!value || typeof value !== 'object') {
    return DEFAULT_COMPOSER_SETTINGS;
  }
  const settings = (value as { settings?: unknown }).settings;
  if (!settings || typeof settings !== 'object') {
    return DEFAULT_COMPOSER_SETTINGS;
  }
  const composer = (settings as { composer?: unknown }).composer;
  if (!composer || typeof composer !== 'object') {
    return DEFAULT_COMPOSER_SETTINGS;
  }
  const candidate = composer as {
    long_paste_to_file_enabled?: unknown;
    long_paste_threshold_chars?: unknown;
  };
  const enabled =
    typeof candidate.long_paste_to_file_enabled === 'boolean'
      ? candidate.long_paste_to_file_enabled
      : DEFAULT_COMPOSER_SETTINGS.longPasteToFileEnabled;
  const threshold =
    typeof candidate.long_paste_threshold_chars === 'number'
      && Number.isFinite(candidate.long_paste_threshold_chars)
      && candidate.long_paste_threshold_chars > 0
      ? Math.floor(candidate.long_paste_threshold_chars)
      : DEFAULT_COMPOSER_SETTINGS.longPasteThresholdChars;
  return {
    longPasteToFileEnabled: enabled,
    longPasteThresholdChars: threshold,
  };
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

const viewFromPathname = (pathname: string): AppView => {
  if (WORKSPACE_PATHS.has(pathname)) {
    return 'workspace';
  }
  return 'chat';
};

const pathForView = (view: AppView): string => {
  if (view === 'workspace') {
    return '/workspace';
  }
  return '/ui/';
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

const parseSessionArtifacts = (value: unknown): SessionArtifactRecord[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const artifacts: SessionArtifactRecord[] = [];
  for (const item of value) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const candidate = item as {
      id?: unknown;
      kind?: unknown;
      title?: unknown;
      content?: unknown;
      created_at?: unknown;
      display_target?: unknown;
      artifact_kind?: unknown;
      file_name?: unknown;
      file_ext?: unknown;
      language?: unknown;
      file_content?: unknown;
    };
    if (typeof candidate.id !== 'string' || !candidate.id.trim()) {
      continue;
    }
    if (candidate.kind !== 'output') {
      continue;
    }
    if (typeof candidate.content !== 'string' || !candidate.content.trim()) {
      continue;
    }
    const title =
      typeof candidate.title === 'string' && candidate.title.trim()
        ? candidate.title.trim()
        : candidate.content.split('\n').find((line) => line.trim())?.trim().slice(0, 80) || 'Result';
    artifacts.push({
      id: candidate.id.trim(),
      kind: 'output',
      title,
      content: candidate.content,
      createdAt: typeof candidate.created_at === 'string' ? candidate.created_at : null,
      displayTarget: candidate.display_target === 'canvas' ? 'canvas' : 'chat',
      artifactKind: candidate.artifact_kind === 'file' ? 'file' : 'text',
      fileName: typeof candidate.file_name === 'string' ? candidate.file_name : null,
      fileExt: typeof candidate.file_ext === 'string' ? candidate.file_ext : null,
      language: typeof candidate.language === 'string' ? candidate.language : null,
      fileContent: typeof candidate.file_content === 'string' ? candidate.file_content : null,
    });
  }
  return artifacts;
};

const parseDisplayDecision = (value: unknown): DisplayDecision | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as { target?: unknown; artifact_id?: unknown; forced?: unknown };
  if (candidate.target !== 'chat' && candidate.target !== 'canvas') {
    return null;
  }
  return {
    target: candidate.target,
    artifactId: typeof candidate.artifact_id === 'string' ? candidate.artifact_id : null,
    forced: candidate.forced === true,
  };
};

const parseUiDecision = (value: unknown): UiDecision | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const candidate = value as {
    id?: unknown;
    kind?: unknown;
    status?: unknown;
    blocking?: unknown;
    reason?: unknown;
    summary?: unknown;
    proposed_action?: unknown;
    options?: unknown;
    default_option_id?: unknown;
    context?: unknown;
    created_at?: unknown;
    updated_at?: unknown;
    resolved_at?: unknown;
  };
  if (typeof candidate.id !== 'string' || !candidate.id.trim()) {
    return null;
  }
  if (candidate.kind !== 'approval' && candidate.kind !== 'decision') {
    return null;
  }
  if (
    candidate.status !== 'pending'
    && candidate.status !== 'approved'
    && candidate.status !== 'rejected'
    && candidate.status !== 'executing'
    && candidate.status !== 'resolved'
  ) {
    return null;
  }
  if (typeof candidate.reason !== 'string' || typeof candidate.summary !== 'string') {
    return null;
  }
  if (typeof candidate.created_at !== 'string' || typeof candidate.updated_at !== 'string') {
    return null;
  }
  const optionsRaw = candidate.options;
  const options = Array.isArray(optionsRaw)
    ? optionsRaw
        .filter((item): item is { id: string; title: string; action: string; payload?: unknown; risk?: unknown } => {
          return (
            !!item
            && typeof item === 'object'
            && typeof (item as { id?: unknown }).id === 'string'
            && typeof (item as { title?: unknown }).title === 'string'
            && typeof (item as { action?: unknown }).action === 'string'
          );
        })
        .map((item) => ({
          id: item.id,
          title: item.title,
          action: item.action,
          payload: item.payload && typeof item.payload === 'object' ? (item.payload as Record<string, unknown>) : {},
          risk: typeof item.risk === 'string' ? item.risk : 'low',
        }))
    : [];
  const proposedAction =
    candidate.proposed_action && typeof candidate.proposed_action === 'object'
      ? (candidate.proposed_action as Record<string, unknown>)
      : {};
  const context =
    candidate.context && typeof candidate.context === 'object'
      ? (candidate.context as Record<string, unknown>)
      : {};
  return {
    id: candidate.id.trim(),
    kind: candidate.kind,
    status: candidate.status,
    blocking: candidate.blocking === true,
    reason: candidate.reason,
    summary: candidate.summary,
    proposed_action: proposedAction,
    options,
    default_option_id:
      typeof candidate.default_option_id === 'string' ? candidate.default_option_id : null,
    context,
    created_at: candidate.created_at,
    updated_at: candidate.updated_at,
    resolved_at: typeof candidate.resolved_at === 'string' ? candidate.resolved_at : null,
  };
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
    .map((message) => ({
      id: message.message_id,
      messageId: message.message_id,
      role: message.role,
      content: message.content,
      createdAt: message.created_at,
      traceId: message.trace_id,
      parentUserMessageId: message.parent_user_message_id,
      attachments: message.attachments,
      transient: false,
    }));
};

const inferArtifactType = (content: string): Artifact['type'] => {
  if (content.includes('```')) {
    const match = content.match(/```\s*([a-zA-Z0-9_-]+)/);
    const lang = match ? match[1].toLowerCase() : '';
    if (lang === 'python' || lang === 'py') return 'PY';
    if (lang === 'javascript' || lang === 'js') return 'JS';
    if (lang === 'typescript' || lang === 'ts') return 'TS';
    if (lang === 'json') return 'JSON';
    if (lang === 'html') return 'HTML';
    if (lang === 'css') return 'CSS';
    if (lang === 'sh' || lang === 'shell' || lang === 'bash') return 'SH';
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
  if (normalized.endsWith('.sh')) return 'SH';
  if (normalized.endsWith('.md') || normalized.endsWith('.markdown')) return 'MD';
  return 'TXT';
};

const inferArtifactTypeFromLanguage = (languageRaw: string): Artifact['type'] => {
  const language = languageRaw.trim().toLowerCase();
  if (!language) {
    return 'TXT';
  }
  if (language === 'python' || language === 'py') return 'PY';
  if (language === 'typescript' || language === 'ts' || language === 'tsx') return 'TS';
  if (language === 'javascript' || language === 'js' || language === 'jsx') return 'JS';
  if (language === 'json') return 'JSON';
  if (language === 'html') return 'HTML';
  if (language === 'css') return 'CSS';
  if (language === 'shell' || language === 'bash' || language === 'sh') return 'SH';
  if (language === 'md' || language === 'markdown') return 'MD';
  return 'TXT';
};

const inferArtifactCategory = (type: Artifact['type']): Artifact['category'] => {
  if (type === 'TXT' || type === 'MD') return 'Document';
  if (type === 'JSON') return 'Config';
  if (type === 'SH') return 'Script';
  return 'Code';
};

const buildArtifactsFromSources = (
  artifactsHistory: SessionArtifactRecord[],
  files: string[],
  streamingContentByArtifactId: Record<string, string>,
): Artifact[] => {
  const artifacts: Artifact[] = [];
  const seen = new Set<string>();
  const seenFileNames = new Set<string>();

  for (const item of artifactsHistory) {
    if (!item.content.trim()) {
      continue;
    }
    const id = `output-${item.id}`;
    const hasStreamOverride = Object.prototype.hasOwnProperty.call(streamingContentByArtifactId, id);
    const content = hasStreamOverride ? streamingContentByArtifactId[id] : item.content;
    let type: Artifact['type'];
    if (item.fileExt) {
      type = inferArtifactTypeFromPath(`file.${item.fileExt}`);
    } else if (item.fileName) {
      type = inferArtifactTypeFromPath(item.fileName);
    } else if (item.language) {
      type = inferArtifactTypeFromLanguage(item.language);
    } else {
      type = inferArtifactType(item.content);
    }
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    const normalizedFileName = item.fileName?.trim().toLowerCase() ?? '';
    if (normalizedFileName) {
      seenFileNames.add(normalizedFileName);
    }
    artifacts.push({
      id,
      name: item.fileName || item.title,
      type,
      category: inferArtifactCategory(type),
      content,
      artifactKind: item.artifactKind,
      sourceArtifactId: item.id,
      fileName: item.fileName,
      fileExt: item.fileExt,
      language: item.language,
      fileContent: item.fileContent,
    });
  }

  for (const rawPath of files) {
    const path = rawPath.trim();
    const fileId = `file-${path}`;
    if (!path || seen.has(fileId)) {
      continue;
    }
    const pathBaseName = path.split('/').pop()?.toLowerCase() ?? '';
    if (pathBaseName && seenFileNames.has(pathBaseName)) {
      continue;
    }
    seen.add(fileId);
    const type = inferArtifactTypeFromPath(path);
    artifacts.push({
      id: fileId,
      name: path,
      type,
      category: inferArtifactCategory(type),
      content: `File path: ${path}`,
      artifactKind: 'file',
      sessionFilePath: path,
      fileName: path.split('/').pop() ?? path,
    });
  }

  return artifacts;
};

const extractFilenameFromDisposition = (
  disposition: string | null,
  fallbackName: string,
): string => {
  if (!disposition) {
    return fallbackName;
  }
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match && utf8Match[1]) {
    try {
      const decoded = decodeURIComponent(utf8Match[1].trim());
      return decoded || fallbackName;
    } catch {
      return utf8Match[1].trim() || fallbackName;
    }
  }
  const basicMatch = disposition.match(/filename="?([^";]+)"?/i);
  if (basicMatch && basicMatch[1]) {
    return basicMatch[1].trim() || fallbackName;
  }
  return fallbackName;
};

const triggerBrowserDownload = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
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
        decision?: unknown;
      };
      if (envelope.type === 'chat.stream.start') {
        const streamId =
          typeof payload.stream_id === 'string' ? payload.stream_id.trim() : '';
        if (!streamId) {
          return;
        }
        setAwaitingFirstAssistantChunk(false);
        setChatStreamingState({ streamId, content: '' });
        return;
      }
      if (envelope.type === 'chat.stream.delta') {
        const streamId =
          typeof payload.stream_id === 'string' ? payload.stream_id.trim() : '';
        const delta = typeof payload.delta === 'string' ? payload.delta : '';
        if (!streamId || !delta) {
          return;
        }
        setAwaitingFirstAssistantChunk(false);
        setChatStreamingState((prev) => {
          if (!prev || prev.streamId !== streamId) {
            return { streamId, content: delta };
          }
          return { streamId, content: `${prev.content}${delta}` };
        });
        return;
      }
      if (envelope.type === 'chat.stream.done') {
        return;
      }
      if (envelope.type === 'decision.packet') {
        setPendingDecision(parseUiDecision(payload.decision));
        setDecisionError(null);
        return;
      }
      const artifactId = typeof payload.artifact_id === 'string' ? payload.artifact_id.trim() : '';
      if (!artifactId) {
        return;
      }
      const uiArtifactId = toOutputArtifactUiId(artifactId);
      if (envelope.type === 'canvas.stream.start') {
        setAwaitingFirstAssistantChunk(false);
        setArtifactPanelOpen(true);
        setArtifactViewerArtifactId(uiArtifactId);
        setStreamingContentByArtifactId((prev) => ({ ...prev, [uiArtifactId]: '' }));
        return;
      }
      if (envelope.type === 'canvas.stream.delta') {
        const delta = typeof payload.delta === 'string' ? payload.delta : '';
        if (!delta) {
          return;
        }
        setAwaitingFirstAssistantChunk(false);
        setStreamingContentByArtifactId((prev) => {
          const nextChunk = `${prev[uiArtifactId] ?? ''}${delta}`;
          return { ...prev, [uiArtifactId]: nextChunk };
        });
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
      setMessages(parseMessages((payload as { messages?: unknown }).messages));
      setPendingDecision(parseUiDecision((payload as { decision?: unknown }).decision));
      setDecisionError(null);
      const parsedOutput = parseSessionOutput((payload as { output?: unknown }).output);
      setSessionOutput(parsedOutput.content);
      const parsedFiles = parseSessionFiles((payload as { files?: unknown }).files);
      setSessionFiles(parsedFiles);
      setSessionArtifacts(parseSessionArtifacts((payload as { artifacts?: unknown }).artifacts));
      const displayDecision = parseDisplayDecision((payload as { display?: unknown }).display);
      if (displayDecision?.target === 'canvas') {
        setArtifactPanelOpen(true);
        if (displayDecision.artifactId) {
          setArtifactViewerArtifactId(`output-${displayDecision.artifactId}`);
        }
      } else {
        setArtifactViewerArtifactId(null);
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
      const parsedDecision = parseUiDecision((payload as { decision?: unknown }).decision);
      setPendingDecision(parsedDecision);
      const maybeMessages = (payload as { messages?: unknown }).messages;
      if (Array.isArray(maybeMessages)) {
        setMessages(parseMessages(maybeMessages));
      }
      const maybeOutput = (payload as { output?: unknown }).output;
      if (maybeOutput !== undefined) {
        const parsedOutput = parseSessionOutput(maybeOutput);
        setSessionOutput(parsedOutput.content);
      }
      const maybeFiles = (payload as { files?: unknown }).files;
      if (maybeFiles !== undefined) {
        setSessionFiles(parseSessionFiles(maybeFiles));
      }
      const maybeArtifacts = (payload as { artifacts?: unknown }).artifacts;
      if (maybeArtifacts !== undefined) {
        setSessionArtifacts(parseSessionArtifacts(maybeArtifacts));
      }
      const parsedModel = parseSelectedModel((payload as { selected_model?: unknown }).selected_model);
      if (parsedModel) {
        setSelectedModel(parsedModel);
        saveLastModel(parsedModel);
      }
      const resumeRaw = (payload as { resume?: unknown }).resume;
      if (resumeRaw && typeof resumeRaw === 'object') {
        const resume = resumeRaw as {
          ok?: unknown;
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
            messages={workspaceMessages}
            sending={sending}
            statusMessage={statusMessage}
            onBackToChat={() => setView('chat')}
            onOpenSettings={() => setSettingsOpen(true)}
            onSendAgentMessage={(payload) => handleSend(payload)}
            decision={pendingDecision}
            decisionBusy={decisionBusy}
            decisionError={decisionError}
            onDecisionRespond={(choice, editedAction) => {
              void handleDecisionRespond(choice, editedAction);
            }}
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
