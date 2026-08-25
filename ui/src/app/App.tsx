import { useEffect, useMemo, useRef, useState } from 'react';

import type { Artifact } from './components/artifacts-sidebar';
import { ChatSessionScreen } from './components/chat-session-screen';
import { GlobalSettingsShell } from './components/global-settings-shell';
import { HistorySidebar } from './components/history-sidebar';
import { RepositoryPanel } from './components/repository-panel';
import { SearchModal } from './components/search-modal';
import { SessionControlShell } from './components/session-control-shell';
import { WorkspaceSessionScreen } from './components/workspace-session-screen';
import type { SessionTransportBridge } from './session-bridges';
import {
  DEFAULT_COMPOSER_SETTINGS,
  extractErrorMessage,
  extractFilenameFromDisposition,
  groupSessionByDate,
  parseComposerSettings,
  parseFolders,
  parseProviderModels,
  parseSessions,
  parseUiTheme,
  sortSessionsByUpdated,
  triggerBrowserDownload,
  type ComposerUiSettings,
  type UiTheme,
} from './session-payload';
import {
  loadWorkspaceExplorerVisible,
  pathForView,
  saveWorkspaceExplorerVisible,
  viewFromPathname,
  type AppView,
} from './session-storage';
import type { FolderSummary, ProviderModels, SessionSummary } from './types';
import { useRepositoryActions } from './use-repository-actions';
import { useSessionOverlays } from './use-session-overlays';
import { useSessionRuntimeController } from './use-session-runtime-controller';
import { useSessionTransport } from './use-session-transport';

const SESSION_HEADER = 'X-Slavik-Session';
const SCROLLBAR_REVEAL_DISTANCE_PX = 38;

const upsertProviderModels = (
  current: ProviderModels[],
  nextProvider: ProviderModels,
): ProviderModels[] => {
  const updated = current.filter((item) => item.provider !== nextProvider.provider);
  updated.push(nextProvider);
  return updated;
};

export default function App() {
  const transportRef = useRef<SessionTransportBridge | null>(null);
  const [activeView, setActiveView] = useState<AppView>(() => {
    if (typeof window === 'undefined') {
      return 'chat';
    }
    return viewFromPathname(window.location.pathname);
  });
  const [workspaceExplorerVisible] = useState<boolean>(() =>
    loadWorkspaceExplorerVisible(),
  );
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [composerSettings, setComposerSettings] = useState<ComposerUiSettings>(
    DEFAULT_COMPOSER_SETTINGS,
  );
  const [uiTheme, setUiTheme] = useState<UiTheme>('default');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const overlays = useSessionOverlays({ activeView });

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

  const loadModels = async (): Promise<ProviderModels[]> => {
    setModelsLoading(true);
    try {
      const response = await fetch('/ui/api/models?summary=1');
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

  const loadProviderModels = async (provider: string): Promise<ProviderModels | null> => {
    const normalized = provider.trim().toLowerCase();
    if (!normalized) {
      return null;
    }
    setModelsLoading(true);
    try {
      const params = new URLSearchParams({ provider: normalized, strict: '1' });
      const response = await fetch(`/ui/api/models?${params.toString()}`);
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, `Failed to load ${normalized} models.`));
      }
      const parsed = parseProviderModels((payload as { providers?: unknown }).providers);
      const next = parsed[0] ?? null;
      if (next) {
        setProviderModels((current) => upsertProviderModels(current, next));
      }
      return next;
    } finally {
      setModelsLoading(false);
    }
  };

  const startLocalOllama = async (): Promise<ProviderModels | null> => {
    setModelsLoading(true);
    try {
      const response = await fetch('/ui/api/local/ollama/start', { method: 'POST' });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to start Local Ollama.'));
      }
      const body = payload as { provider?: unknown; models?: unknown };
      const provider = typeof body.provider === 'string' ? body.provider : 'local';
      const models = Array.isArray(body.models)
        ? body.models.filter((item): item is string => typeof item === 'string')
        : [];
      const next: ProviderModels = { provider, models, error: null };
      setProviderModels((current) => upsertProviderModels(current, next));
      return next;
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
    setUiTheme(parseUiTheme(payload));
    return parsed;
  };

  const runtime = useSessionRuntimeController({
    sessionHeader: SESSION_HEADER,
    providerModels,
    loadSessions,
    loadModels,
    loadComposerSettings,
    onStatusMessage: setStatusMessage,
    transportRef,
  });

  const transport = useSessionTransport({
    sessionHeader: SESSION_HEADER,
    selectedConversation: runtime.selectedConversation,
    forceCanvasNext: overlays.forceCanvasNext,
    consumeForceCanvasNext: overlays.consumeForceCanvasNext,
    onSessionIdChange: async (sessionId) => {
      overlays.resetSessionSurfaceState();
      await runtime.handleSelectConversation(sessionId);
    },
    onStatusMessage: setStatusMessage,
    onRuntimePayload: runtime.applyRuntimePayload,
    onOpenStreamedArtifact: overlays.openStreamedArtifact,
    setArtifactViewerArtifactId: overlays.setArtifactViewerArtifactId,
    loadSessions,
  });
  transportRef.current = transport.bridge;

  const repositoryActions = useRepositoryActions({
    sessionId: runtime.selectedConversation,
    sessionHeader: SESSION_HEADER,
    pendingDecision: runtime.pendingDecision,
    transportRef,
    applyRuntimePayload: runtime.applyRuntimePayload,
    loadSessions,
    onSessionIdChange: async (sessionId) => {
      overlays.resetSessionSurfaceState();
      await runtime.handleSelectConversation(sessionId);
    },
    onStatusMessage: setStatusMessage,
  });

  const modelLabel = runtime.selectedModel
    ? `${runtime.selectedModel.provider}/${runtime.selectedModel.model}`
    : 'Model not selected';
  const sessionTitle =
    sessions.find((session) => session.session_id === runtime.selectedConversation)?.title ??
    null;
  const selectedModelValue = runtime.selectedModel
    ? `${runtime.selectedModel.provider}::${runtime.selectedModel.model}`
    : null;
  const historyChats = useMemo(
    () =>
      sessions.map((session) => ({
        id: session.session_id,
        title: session.title,
        messageCount: session.message_count,
        chatMessageCount: session.chat_message_count,
        workspaceMessageCount: session.workspace_message_count,
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

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }
    const syncViewFromLocation = () => {
      setActiveView(viewFromPathname(window.location.pathname));
    };
    syncViewFromLocation();
    window.addEventListener('popstate', syncViewFromLocation);
    return () => {
      window.removeEventListener('popstate', syncViewFromLocation);
    };
  }, []);

  useEffect(() => {
    saveWorkspaceExplorerVisible(workspaceExplorerVisible);
  }, [workspaceExplorerVisible]);

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

  const handleWorkspaceSidebarAction = () => {
    setView(activeView === 'workspace' ? 'chat' : 'workspace');
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

  const handleSettingsSaved = async () => {
    try {
      await Promise.all([loadModels(), loadComposerSettings(), loadSessions()]);
      setStatusMessage('Settings saved.');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Settings saved, but models refresh failed.';
      setStatusMessage(message);
    }
  };

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

  const handleDownloadArtifact = async (artifact: Artifact) => {
    if (!runtime.selectedConversation) {
      return;
    }
    try {
      if (artifact.sourceArtifactId) {
        const response = await fetch(
          `/ui/api/sessions/${encodeURIComponent(runtime.selectedConversation)}/artifacts/${encodeURIComponent(
            artifact.sourceArtifactId,
          )}/download`,
          {
            headers: {
              [SESSION_HEADER]: runtime.selectedConversation,
            },
          },
        );
        if (!response.ok) {
          throw new Error(await extractErrorFromResponse(response, 'Failed to download artifact.'));
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
          `/ui/api/sessions/${encodeURIComponent(runtime.selectedConversation)}/files/download?path=${encodeURIComponent(
            artifact.sessionFilePath,
          )}`,
          {
            headers: {
              [SESSION_HEADER]: runtime.selectedConversation,
            },
          },
        );
        if (!response.ok) {
          throw new Error(await extractErrorFromResponse(response, 'Failed to download file.'));
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
    if (!runtime.selectedConversation) {
      return;
    }
    try {
      const response = await fetch(
        `/ui/api/sessions/${encodeURIComponent(runtime.selectedConversation)}/artifacts/download-all`,
        {
          headers: {
            [SESSION_HEADER]: runtime.selectedConversation,
          },
        },
      );
      if (!response.ok) {
        throw new Error(await extractErrorFromResponse(response, 'No downloadable file artifacts.'));
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
      const response = await fetch('/ui/api/feedback', {
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

  return (
    <div
      className={`flex h-screen overflow-hidden text-foreground ${
        uiTheme === 'oled' ? 'bg-black' : 'bg-zinc-950'
      }`}
      data-ui-theme={uiTheme}
    >
      <HistorySidebar
        chats={historyChats}
        folders={folders.map((folder) => ({ id: folder.folder_id, name: folder.name }))}
        activeChatId={runtime.selectedConversation}
        onSelectChat={(sessionId) => {
          setView('chat');
          overlays.resetSessionSurfaceState();
          void runtime.handleSelectConversation(sessionId);
        }}
        onNewChat={() => {
          setView('chat');
          overlays.resetSessionSurfaceState();
          void runtime.handleCreateConversation();
        }}
        onDeleteChat={(sessionId) => {
          void runtime.handleDeleteConversation(sessionId);
        }}
        onRenameChat={(sessionId) => {
          void handleRenameChat(sessionId);
        }}
        onMoveChatToFolder={(sessionId, folderId) => {
          void handleMoveChatToFolder(sessionId, folderId);
        }}
        onOpenSearch={() => overlays.setSearchOpen(true)}
        onOpenWorkspace={handleWorkspaceSidebarAction}
        onOpenSettings={() => overlays.setSettingsOpen(true)}
        onCreateFolder={() => {
          void handleCreateFolder();
        }}
        compact={false}
        workspaceActive={activeView === 'workspace'}
      />

      <div className="relative flex h-full min-h-0 min-w-0 flex-1 overflow-hidden">
        <div className="min-w-0 flex-1">
          <ChatSessionScreen
            sessionId={runtime.selectedConversation}
            messages={transport.canvasMessages}
            pendingMessage={transport.pendingCanvasMessage}
            streamingAssistantMessage={transport.streamingAssistantCanvasMessage}
            sending={transport.sending}
            cancelling={transport.cancelling}
            modelLabel={modelLabel}
            modelProvider={runtime.selectedModel?.provider ?? null}
            sessionTitle={sessionTitle}
            statusMessage={statusMessage}
            longPasteToFileEnabled={composerSettings.longPasteToFileEnabled}
            longPasteThresholdChars={composerSettings.longPasteThresholdChars}
            forceCanvasNext={overlays.forceCanvasNext}
            artifactPanelOpen={overlays.artifactPanelOpen}
            artifactViewerArtifactId={overlays.artifactViewerArtifactId}
            artifacts={transport.artifacts}
            decision={runtime.pendingDecision}
            decisionBusy={runtime.decisionBusy}
            decisionError={runtime.decisionError}
            runtimeMode={runtime.sessionMode}
            onSendMessage={transport.handleSendChat}
            onCancelSend={transport.handleCancelChat}
            onSendFeedback={handleSendFeedback}
            onOpenSessionDrawer={() => overlays.setSessionDrawerOpen(true)}
            onToggleForceCanvasNext={() => overlays.setForceCanvasNext((prev) => !prev)}
            onDecisionRespond={(choice, editedAction) => {
              void runtime.handleDecisionRespond(
                choice,
                editedAction,
                repositoryActions.handleDecisionResume,
                repositoryActions.handleDecisionRejected,
              );
            }}
            onOpenArtifactPanel={() => {
              overlays.setArtifactViewerArtifactId(null);
              overlays.setArtifactPanelOpen(true);
            }}
            onCloseArtifactPanel={() => {
              overlays.setArtifactPanelOpen(false);
              overlays.setArtifactViewerArtifactId(null);
            }}
            onDownloadArtifact={handleDownloadArtifact}
            onDownloadAll={handleDownloadAllArtifacts}
          />
        </div>
        {activeView === 'workspace' ? (
          <div className="h-full min-h-0 w-[48vw] min-w-[420px] max-w-[920px] border-l border-zinc-800 bg-zinc-950 shadow-[-12px_0_30px_rgba(0,0,0,0.28)] max-lg:absolute max-lg:inset-y-0 max-lg:right-0 max-lg:z-20 max-lg:w-[min(92vw,760px)] max-lg:min-w-0">
            <WorkspaceSessionScreen
              sessionId={runtime.selectedConversation}
              sessionHeader={SESSION_HEADER}
              modelLabel={modelLabel}
              workspaceRoot={runtime.workspaceRoot}
              sessionPolicyLabel={runtime.sessionSecuritySummary.policyLabel}
              sessionYoloActive={runtime.sessionSecuritySummary.yoloActive}
              sessionSafeMode={runtime.sessionSecuritySummary.safeMode}
              messages={[]}
              computerEvents={runtime.computerEvents}
              statusMessage={statusMessage}
              onBackToChat={() => {
                overlays.setRepositoryPanelOpen(false);
                setView('chat');
              }}
              onOpenSessionDrawer={() => overlays.setSessionDrawerOpen(true)}
              onOpenRepositoryPanel={() => overlays.setRepositoryPanelOpen(true)}
              onApplyWorkspaceRoot={runtime.applyWorkspaceRoot}
              mode={runtime.sessionMode}
              activePlan={runtime.activePlan}
              activeTask={runtime.activeTask}
              autoState={runtime.autoState}
              decision={runtime.pendingDecision}
              decisionBusy={runtime.decisionBusy}
              decisionError={runtime.decisionError}
              onDecisionRespond={(choice, editedAction) => {
                void runtime.handleDecisionRespond(
                  choice,
                  editedAction,
                  repositoryActions.handleDecisionResume,
                  repositoryActions.handleDecisionRejected,
                );
              }}
              refreshToken={repositoryActions.workspaceRefreshToken}
              gitDecisionOutcome={repositoryActions.gitDecisionOutcome}
              explorerVisible={workspaceExplorerVisible}
            />
          </div>
        ) : null}
      </div>

      <SearchModal
        isOpen={overlays.searchOpen}
        onClose={() => overlays.setSearchOpen(false)}
        chats={searchChats}
        onSelectChat={(sessionId) => {
          setView('chat');
          overlays.resetSessionSurfaceState();
          void runtime.handleSelectConversation(sessionId);
          overlays.setSearchOpen(false);
        }}
        onNewChat={() => {
          setView('chat');
          overlays.resetSessionSurfaceState();
          void runtime.handleCreateConversation();
        }}
      />

      <RepositoryPanel
        isOpen={overlays.repositoryPanelOpen}
        onClose={() => overlays.setRepositoryPanelOpen(false)}
        pendingDecision={runtime.pendingDecision}
        onRunGithubImport={(repoUrl, branch) => repositoryActions.handleWorkspaceGithubImport(repoUrl, branch)}
      />

      <SessionControlShell
        isOpen={overlays.sessionDrawerOpen}
        onClose={() => overlays.setSessionDrawerOpen(false)}
        onSaved={() => {
          setStatusMessage('Session controls updated.');
          void runtime.refreshSessionSecuritySummary().catch(() => {
            setStatusMessage('Session controls updated, but session summary refresh failed.');
          });
        }}
        sessionId={runtime.selectedConversation}
        sessionHeader={SESSION_HEADER}
        mode={runtime.sessionMode}
        modeTransitions={runtime.modeTransitions}
        modeBusy={runtime.modeBusy}
        onChangeMode={runtime.handleChangeMode}
        modelLabel={modelLabel}
        providerModels={providerModels}
        selectedModelValue={selectedModelValue}
        modelsLoading={modelsLoading}
        savingModel={runtime.savingModel}
        onLoadProviderModels={loadProviderModels}
        onStartLocalOllama={startLocalOllama}
        onSelectModel={(provider, model) => {
          void runtime.handleSetModel(provider, model);
        }}
      />

      <GlobalSettingsShell
        isOpen={overlays.settingsOpen}
        onClose={() => overlays.setSettingsOpen(false)}
        onSaved={() => {
          void handleSettingsSaved();
        }}
      />
    </div>
  );
}
