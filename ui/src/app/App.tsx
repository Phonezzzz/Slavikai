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
  compactProviderError,
  DEFAULT_COMPOSER_SETTINGS,
  extractErrorMessage,
  extractFilenameFromDisposition,
  groupSessionByDate,
  parseComposerSettings,
  parseFolders,
  parseProviderModels,
  parseSessions,
  sortSessionsByUpdated,
  triggerBrowserDownload,
  type ComposerUiSettings,
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

export default function App() {
  const transportRef = useRef<SessionTransportBridge | null>(null);
  const [activeView, setActiveView] = useState<AppView>(() => {
    if (typeof window === 'undefined') {
      return 'chat';
    }
    return viewFromPathname(window.location.pathname);
  });
  const [workspaceExplorerVisible, setWorkspaceExplorerVisible] = useState<boolean>(() =>
    loadWorkspaceExplorerVisible(),
  );
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [composerSettings, setComposerSettings] = useState<ComposerUiSettings>(
    DEFAULT_COMPOSER_SETTINGS,
  );
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
  const modelOptions = useMemo(
    () =>
      providerModels.flatMap((provider) => {
        if (provider.models.length > 0) {
          return provider.models.map((model) => ({
            value: `${provider.provider}::${model}`,
            label: `${provider.provider}/${model}`,
            provider: provider.provider,
            model,
            disabled: false,
          }));
        }
        const unavailableReason = provider.error
          ? `unavailable: ${compactProviderError(provider.error)}`
          : 'unavailable';
        return [
          {
            value: `${provider.provider}::__unavailable__`,
            label: `${provider.provider}/${unavailableReason}`,
            provider: provider.provider,
            model: '',
            disabled: true,
          },
        ];
      }),
    [providerModels],
  );
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
    if (activeView !== 'workspace') {
      setView('workspace');
      return;
    }
    setWorkspaceExplorerVisible((prev) => !prev);
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

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-foreground">
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
        compact={activeView === 'workspace'}
        workspaceActive={activeView === 'workspace'}
        workspaceExplorerVisible={workspaceExplorerVisible}
      />

      <div className="flex-1 min-w-0 relative">
        {activeView === 'workspace' ? (
          <WorkspaceSessionScreen
            sessionId={runtime.selectedConversation}
            sessionHeader={SESSION_HEADER}
            modelLabel={modelLabel}
            sessionPolicyLabel={runtime.sessionSecuritySummary.policyLabel}
            sessionYoloActive={runtime.sessionSecuritySummary.yoloActive}
            sessionSafeMode={runtime.sessionSecuritySummary.safeMode}
            messages={transport.workspaceMessages}
            sending={transport.sending}
            statusMessage={statusMessage}
            onBackToChat={() => {
              overlays.setRepositoryPanelOpen(false);
              setView('chat');
            }}
            onOpenSessionDrawer={() => overlays.setSessionDrawerOpen(true)}
            onOpenRepositoryPanel={() => overlays.setRepositoryPanelOpen(true)}
            onSendAgentMessage={(payload) => transport.handleSend(payload, 'workspace')}
            mode={runtime.sessionMode}
            activePlan={runtime.activePlan}
            activeTask={runtime.activeTask}
            autoState={runtime.autoState}
            modeBusy={runtime.modeBusy}
            modeError={runtime.modeError}
            onChangeMode={runtime.handleChangeMode}
            onPlanDraft={runtime.handlePlanDraft}
            onPlanApprove={runtime.handlePlanApprove}
            onPlanExecute={runtime.handlePlanExecute}
            onPlanCancel={runtime.handlePlanCancel}
            decision={runtime.pendingDecision}
            decisionBusy={runtime.decisionBusy}
            decisionError={runtime.decisionError}
            onDecisionRespond={(choice, editedAction) => {
              void runtime.handleDecisionRespond(
                choice,
                editedAction,
                repositoryActions.handleDecisionResume,
              );
            }}
            refreshToken={repositoryActions.workspaceRefreshToken}
            explorerVisible={workspaceExplorerVisible}
          />
        ) : (
          <ChatSessionScreen
            messages={transport.canvasMessages}
            pendingMessage={transport.pendingCanvasMessage}
            streamingAssistantMessage={transport.streamingAssistantCanvasMessage}
            showAssistantLoading={transport.showAssistantLoading}
            sending={transport.sending}
            modelLabel={modelLabel}
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
            onSendMessage={(payload) => transport.handleSend(payload, 'chat')}
            onSendFeedback={handleSendFeedback}
            onOpenSessionDrawer={() => overlays.setSessionDrawerOpen(true)}
            onToggleForceCanvasNext={() => overlays.setForceCanvasNext((prev) => !prev)}
            onDecisionRespond={(choice, editedAction) => {
              void runtime.handleDecisionRespond(
                choice,
                editedAction,
                repositoryActions.handleDecisionResume,
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
        )}
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
        modeBusy={runtime.modeBusy}
        onChangeMode={runtime.handleChangeMode}
        modelLabel={modelLabel}
        modelOptions={modelOptions}
        selectedModelValue={selectedModelValue}
        modelsLoading={modelsLoading}
        savingModel={runtime.savingModel}
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
