import { useEffect, useMemo, useRef, useState } from 'react';
import type { OnMount } from '@monaco-editor/react';
import type * as Monaco from 'monaco-editor';

import type {
  AutoState,
  ComputerActivityEvent,
  DecisionRespondChoice,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
  UiDecision,
} from '../types';
import { getDecisionDisplayState } from '../decision-display';
import type { CanvasMessage } from './canvas';
import {
  terminalTimestamp,
  type WorkspaceNode,
  type WorkspaceTreeMeta,
} from '../../features/workspace/workspace-helpers';
import {
  WorkspaceEditorPane,
  type WorkspaceOpenFileTab,
} from '../../features/workspace/workspace-editor-pane';
import { WorkspaceAssistantPanel } from '../../features/workspace/workspace-assistant-panel';
import { WorkspaceExplorer } from '../../features/workspace/workspace-explorer';
import {
  WorkspaceQuickOpen,
  type WorkspaceQuickOpenItem,
} from '../../features/workspace/workspace-quick-open';
import {
  collectQuickOpenItems,
  filterQuickOpenItems,
  nextRecentWorkspacePaths,
  type QuickOpenIndexCache,
} from '../../features/workspace/workspace-quick-open-index';
import { WorkspaceToolbar } from '../../features/workspace/workspace-toolbar';
import { ProjectPicker } from '../../features/workspace/project-picker';
import { GitPanel } from '../../features/workspace/git-panel';
import { useWorkspaceLayout } from '../../features/workspace/use-workspace-layout';
import {
  deleteWorkspaceFile,
  fetchWorkspaceFile,
  fetchWorkspaceTree,
  postWorkspaceFileCreate,
  postWorkspaceFileMove,
  postWorkspaceFileRename,
  postWorkspaceIndex,
  postWorkspaceRootSelect,
  postWorkspaceRun,
  postWorkspaceCommandRunnerRun,
  putWorkspaceFile,
} from '../../features/workspace/workspace-api';

type WorkspaceIdeProps = {
  sessionId: string | null;
  sessionHeader: string;
  modelLabel: string;
  workspaceRoot: string;
  sessionPolicyLabel: string;
  sessionYoloActive: boolean;
  sessionSafeMode: boolean;
  messages: CanvasMessage[];
  computerEvents?: ComputerActivityEvent[];
  statusMessage?: string | null;
  onBackToChat: () => void;
  onOpenSessionDrawer: () => void;
  onOpenRepositoryPanel: () => void;
  onApplyWorkspaceRoot: (workspaceRoot: string) => void;
  onSendFeedback?: (interactionId: string, rating: 'good' | 'bad') => Promise<boolean>;
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  decision?: UiDecision | null;
  decisionBusy?: boolean;
  decisionError?: string | null;
  onDecisionRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
  refreshToken?: number;
  gitDecisionOutcome?: {
    kind: 'success' | 'rejected' | 'failed';
    message: string;
  } | null;
  explorerVisible: boolean;
};

const ROOT_TREE_DEBOUNCE_MS = 150;
const CHILD_TREE_DEBOUNCE_MS = 80;

export function WorkspaceIde({
  sessionId,
  sessionHeader,
  modelLabel,
  workspaceRoot,
  sessionPolicyLabel,
  sessionYoloActive,
  sessionSafeMode,
  messages,
  computerEvents = [],
  statusMessage,
  onBackToChat,
  onOpenSessionDrawer,
  onOpenRepositoryPanel,
  onApplyWorkspaceRoot,
  onSendFeedback,
  mode,
  activePlan,
  activeTask,
  autoState,
  decision,
  decisionBusy = false,
  decisionError = null,
  onDecisionRespond,
  refreshToken = 0,
  gitDecisionOutcome = null,
  explorerVisible,
}: WorkspaceIdeProps) {
  const [tree, setTree] = useState<WorkspaceNode[]>([]);
  const [treeLoading, setTreeLoading] = useState(false);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [treeMeta, setTreeMeta] = useState<WorkspaceTreeMeta | null>(null);
  const [loadingTreePaths, setLoadingTreePaths] = useState<Set<string>>(new Set());
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [activeExplorerPath, setActiveExplorerPath] = useState<string | null>(null);

  const [openFiles, setOpenFiles] = useState<WorkspaceOpenFileTab[]>([]);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [editorSaving, setEditorSaving] = useState(false);

  const [terminalLines, setTerminalLines] = useState<string[]>([
    `[${terminalTimestamp()}] Command runner ready.`,
  ]);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalBusy, setTerminalBusy] = useState(false);

  const [projectPickerOpen, setProjectPickerOpen] = useState(false);
  const [projectRootBusy, setProjectRootBusy] = useState(false);
  const [gitRefreshToken, setGitRefreshToken] = useState(0);
  const [indexing, setIndexing] = useState(false);
  const [quickOpenOpen, setQuickOpenOpen] = useState(false);
  const [quickOpenQuery, setQuickOpenQuery] = useState('');
  const [quickOpenLoading, setQuickOpenLoading] = useState(false);
  const [quickOpenPartial, setQuickOpenPartial] = useState(false);
  const [quickOpenItems, setQuickOpenItems] = useState<WorkspaceQuickOpenItem[]>([]);
  const [recentPaths, setRecentPaths] = useState<string[]>([]);
  const [computerTab, setComputerTab] = useState<
    'overview' | 'files' | 'changes' | 'terminal'
  >('overview');

  const {
    terminalHeight,
    filesTabColumns,
    workspaceGridRef,
    startExplorerResize,
    startTerminalResize,
  } = useWorkspaceLayout({ explorerVisible });

  const terminalEndRef = useRef<HTMLDivElement>(null);
  const assistantSeenRef = useRef<Set<string>>(new Set());
  const assistantInitRef = useRef(false);
  const decisionSeenRef = useRef<Set<string>>(new Set());
  const statusSeenRef = useRef<string | null>(null);
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);
  const previousDecisionStatusRef = useRef<string | null>(null);
  const treeDebounceTimersRef = useRef<Map<string, number>>(new Map());
  const treeAbortControllersRef = useRef<Map<string, AbortController>>(new Map());
  const treeInFlightPathsRef = useRef<Set<string>>(new Set());
  const quickOpenFileIndexRef = useRef<QuickOpenIndexCache | null>(null);
  const quickOpenLoadedForRoot = useRef<string | null>(null);
  const quickOpenAbortControllerRef = useRef<AbortController | null>(null);
  const rootScopeGenerationRef = useRef(0);

  const activeTab = useMemo(
    () => openFiles.find((item) => item.id === activeFileId) ?? null,
    [activeFileId, openFiles],
  );
  const hasUnsavedChanges = Boolean(activeTab && activeTab.content !== activeTab.savedContent);
  const decisionState = getDecisionDisplayState(decision, decisionBusy, decisionError);
  const isDecisionBlocking = decisionState.isBlocking;

  const requestHeaders = useMemo(() => {
    if (!sessionId) {
      return {} as Record<string, string>;
    }
    return { [sessionHeader]: sessionId };
  }, [sessionHeader, sessionId]);

  const pushRecentPath = (path: string) => {
    const normalized = path.trim();
    if (!normalized) {
      return;
    }
    setRecentPaths((prev) => nextRecentWorkspacePaths(prev, normalized));
  };

  const quickOpenResults = useMemo(
    () => filterQuickOpenItems(quickOpenItems, quickOpenQuery, recentPaths),
    [quickOpenItems, quickOpenQuery, recentPaths],
  );

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLines]);

  const resetRootScopedState = () => {
    rootScopeGenerationRef.current += 1;
    treeDebounceTimersRef.current.forEach((timerId) => {
      window.clearTimeout(timerId);
    });
    treeDebounceTimersRef.current.clear();
    treeAbortControllersRef.current.forEach((controller) => {
      controller.abort();
    });
    treeAbortControllersRef.current.clear();
    treeInFlightPathsRef.current.clear();
    setTree([]);
    setTreeError(null);
    setTreeLoading(false);
    setLoadingTreePaths(new Set());
    setTreeMeta(null);
    setExpandedNodes(new Set());
    setActiveExplorerPath(null);
    setOpenFiles([]);
    setActiveFileId(null);
    setQuickOpenOpen(false);
    setQuickOpenQuery('');
    setQuickOpenLoading(false);
    setQuickOpenPartial(false);
    setQuickOpenItems([]);
    setRecentPaths([]);
    quickOpenAbortControllerRef.current?.abort();
    quickOpenAbortControllerRef.current = null;
    quickOpenFileIndexRef.current = null;
    quickOpenLoadedForRoot.current = null;
    setGitRefreshToken((prev) => prev + 1);
  };

  useEffect(() => {
    assistantInitRef.current = false;
    assistantSeenRef.current = new Set();
    decisionSeenRef.current = new Set();
    statusSeenRef.current = null;
  }, [sessionId]);

  const workspaceScopeKeyRef = useRef<string | null>(null);
  useEffect(() => {
    const scopeKey = sessionId ? `${sessionId}\u0000${workspaceRoot}` : null;
    if (workspaceScopeKeyRef.current === scopeKey) {
      return;
    }
    workspaceScopeKeyRef.current = scopeKey;
    resetRootScopedState();
    if (sessionId) {
      requestTreeLoad(undefined, 'root_change');
    }
  }, [sessionId, workspaceRoot]);

  useEffect(() => {
    const assistantMessages = messages.filter((item) => item.role === 'assistant');
    if (!assistantInitRef.current) {
      assistantMessages.forEach((message) => {
        assistantSeenRef.current.add(message.messageId);
      });
      assistantInitRef.current = true;
      return;
    }
    const fresh = assistantMessages.filter((message) => !assistantSeenRef.current.has(message.messageId));
    if (fresh.length === 0) {
      return;
    }
    setTerminalLines((prev) => {
      const next = [...prev];
      for (const message of fresh) {
        const content = message.content.trim();
        if (!content) {
          continue;
        }
        next.push(`[${terminalTimestamp()}] assistant:`);
        next.push(content);
      }
      return next;
    });
    fresh.forEach((message) => {
      assistantSeenRef.current.add(message.messageId);
    });
  }, [messages]);

  useEffect(() => {
    if (!decision || decision.status !== 'pending') {
      return;
    }
    if (decisionSeenRef.current.has(decision.id)) {
      return;
    }
    decisionSeenRef.current.add(decision.id);
    setTerminalLines((prev) => [
      ...prev,
      `[${terminalTimestamp()}] pending approval: ${decision.summary}`,
    ]);
  }, [decision]);

  useEffect(() => {
    const currentStatus = decision?.status ?? null;
    const previousStatus = previousDecisionStatusRef.current;
    previousDecisionStatusRef.current = currentStatus;
    if (previousStatus === 'pending' && currentStatus !== 'pending') {
      requestTreeLoad(undefined, 'decision_resume');
      void refreshOpenTabsFromDisk();
      setGitRefreshToken((prev) => prev + 1);
    }
  }, [decision?.status]);

  useEffect(() => {
    if (!statusMessage || statusSeenRef.current === statusMessage) {
      return;
    }
    statusSeenRef.current = statusMessage;
    setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] ${statusMessage}`]);
  }, [statusMessage]);

  useEffect(() => {
    return () => {
      treeDebounceTimersRef.current.forEach((timerId) => {
        window.clearTimeout(timerId);
      });
      treeDebounceTimersRef.current.clear();
      treeAbortControllersRef.current.forEach((controller) => {
        controller.abort();
      });
      treeAbortControllersRef.current.clear();
      treeInFlightPathsRef.current.clear();
      quickOpenAbortControllerRef.current?.abort();
      quickOpenAbortControllerRef.current = null;
      rootScopeGenerationRef.current += 1;
    };
  }, []);

  const replaceNodeChildren = (
    nodes: WorkspaceNode[],
    targetPath: string,
    children: WorkspaceNode[],
    childrenTruncated: boolean,
  ): WorkspaceNode[] =>
    nodes.map((node) => {
      if (node.type === 'dir' && node.path === targetPath) {
        return {
          ...node,
          children,
          hasChildren: children.length > 0,
          childrenTruncated,
        };
      }
      if (node.type === 'dir' && node.children && node.children.length > 0) {
        return {
          ...node,
          children: replaceNodeChildren(
            node.children,
            targetPath,
            children,
            childrenTruncated,
          ),
        };
      }
      return node;
    });

  const treePathKey = (path?: string): string => {
    const normalized = path?.trim() ?? '';
    return normalized || '__root__';
  };

  const setPathLoading = (path: string, loading: boolean) => {
    setLoadingTreePaths((prev) => {
      const next = new Set(prev);
      if (loading) {
        next.add(path);
      } else {
        next.delete(path);
      }
      return next;
    });
  };

  const loadTreeNow = async (path?: string, reason?: string): Promise<void> => {
    void reason;
    if (!sessionId) {
      setTree([]);
      setTreeError('No active session. Create chat first.');
      return;
    }
    const normalizedPath = path?.trim() ?? '';
    const requestKey = treePathKey(normalizedPath);
    const existingController = treeAbortControllersRef.current.get(requestKey);
    if (existingController) {
      existingController.abort();
    }
    const abortController = new AbortController();
    treeAbortControllersRef.current.set(requestKey, abortController);
    treeInFlightPathsRef.current.add(requestKey);
    const scopeGeneration = rootScopeGenerationRef.current;
    const isRootLoad = normalizedPath.length === 0;
    if (isRootLoad) {
      setTreeLoading(true);
      setTreeError(null);
    } else {
      setPathLoading(normalizedPath, true);
    }
    try {
      const { pendingApproval, tree: parsedTree, treeMeta: loadedTreeMeta } = await fetchWorkspaceTree(
        normalizedPath ? { path: normalizedPath, recursive: false } : { recursive: false },
        requestHeaders,
        abortController.signal,
      );
      if (
        abortController.signal.aborted
        || treeAbortControllersRef.current.get(requestKey) !== abortController
        || rootScopeGenerationRef.current !== scopeGeneration
      ) {
        return;
      }
      if (pendingApproval) {
        setTreeError('Ожидает подтверждения действия для доступа к Computer.');
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: computer tree request`,
        ]);
        return;
      }
      setTreeMeta(loadedTreeMeta);
      if (normalizedPath) {
        setTree((prev) =>
          replaceNodeChildren(prev, normalizedPath, parsedTree, loadedTreeMeta.truncated),
        );
      } else {
        setTree(parsedTree);
      }
    } catch (error) {
      if (
        abortController.signal.aborted
        || treeAbortControllersRef.current.get(requestKey) !== abortController
        || rootScopeGenerationRef.current !== scopeGeneration
      ) {
        return;
      }
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      const message = error instanceof Error ? error.message : 'Failed to load Computer tree.';
      if (isRootLoad) {
        setTreeError(message);
      } else {
        setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
      }
    } finally {
      if (
        treeAbortControllersRef.current.get(requestKey) === abortController
        && rootScopeGenerationRef.current === scopeGeneration
      ) {
        treeAbortControllersRef.current.delete(requestKey);
        treeInFlightPathsRef.current.delete(requestKey);
        if (isRootLoad) {
          setTreeLoading(false);
        } else {
          setPathLoading(normalizedPath, false);
        }
      }
    }
  };

  const requestTreeLoad = (path?: string, reason?: string): void => {
    const normalizedPath = path?.trim() ?? '';
    const requestKey = treePathKey(normalizedPath);
    const delay = normalizedPath ? CHILD_TREE_DEBOUNCE_MS : ROOT_TREE_DEBOUNCE_MS;
    const existingTimer = treeDebounceTimersRef.current.get(requestKey);
    if (typeof existingTimer === 'number') {
      window.clearTimeout(existingTimer);
    }
    const timerId = window.setTimeout(() => {
      treeDebounceTimersRef.current.delete(requestKey);
      void loadTreeNow(normalizedPath, reason);
    }, delay);
    treeDebounceTimersRef.current.set(requestKey, timerId);
  };

  const ensureQuickOpenIndex = async (): Promise<void> => {
    if (!sessionId) {
      setQuickOpenItems([]);
      setQuickOpenPartial(false);
      return;
    }
    const rootKey = workspaceRoot.trim();
    if (rootKey && quickOpenLoadedForRoot.current === rootKey && quickOpenFileIndexRef.current) {
      setQuickOpenItems(quickOpenFileIndexRef.current.items);
      setQuickOpenPartial(quickOpenFileIndexRef.current.partial);
      return;
    }
    quickOpenAbortControllerRef.current?.abort();
    const abortController = new AbortController();
    quickOpenAbortControllerRef.current = abortController;
    const scopeGeneration = rootScopeGenerationRef.current;
    setQuickOpenLoading(true);
    try {
      const { pendingApproval, tree: parsedTree, treeMeta: loadedTreeMeta } = await fetchWorkspaceTree(
        { recursive: true, maxDepth: 12 },
        requestHeaders,
        abortController.signal,
      );
      if (
        abortController.signal.aborted
        || quickOpenAbortControllerRef.current !== abortController
        || rootScopeGenerationRef.current !== scopeGeneration
      ) {
        return;
      }
      if (pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: quick open index`,
        ]);
        return;
      }
      const indexedItems = collectQuickOpenItems(parsedTree);
      setQuickOpenItems(indexedItems);
      setQuickOpenPartial(loadedTreeMeta.truncated);
      const cache: QuickOpenIndexCache = {
        rootKey,
        items: indexedItems,
        partial: loadedTreeMeta.truncated,
        loadedAt: Date.now(),
      };
      quickOpenFileIndexRef.current = cache;
      quickOpenLoadedForRoot.current = rootKey;
    } catch (error) {
      if (
        abortController.signal.aborted
        || quickOpenAbortControllerRef.current !== abortController
        || rootScopeGenerationRef.current !== scopeGeneration
      ) {
        return;
      }
      const message = error instanceof Error ? error.message : 'Failed to load quick open file list.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
      setQuickOpenItems([]);
      setQuickOpenPartial(false);
    } finally {
      if (
        quickOpenAbortControllerRef.current === abortController
        && rootScopeGenerationRef.current === scopeGeneration
      ) {
        quickOpenAbortControllerRef.current = null;
        setQuickOpenLoading(false);
      }
    }
  };

  const openQuickOpen = () => {
    setQuickOpenOpen(true);
    setQuickOpenQuery('');
    void ensureQuickOpenIndex();
  };

  const handleQuickOpenSelect = (path: string) => {
    void openFileInTab(path);
    setQuickOpenOpen(false);
  };


  useEffect(() => {
    requestTreeLoad(undefined, 'session_init');
  }, [refreshToken, sessionId]);

  useEffect(() => {
    const isEditableTarget = (target: EventTarget | null): boolean => {
      if (!(target instanceof HTMLElement)) {
        return false;
      }
      if (target.isContentEditable) {
        return true;
      }
      if (
        target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
      ) {
        return true;
      }
      if (target.closest('.monaco-editor')) {
        return true;
      }
      return false;
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) {
        return;
      }
      if (isEditableTarget(event.target)) {
        return;
      }

      if (event.key.toLowerCase() !== 'p') {
        return;
      }
      event.preventDefault();
      openQuickOpen();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [sessionId, workspaceRoot, requestHeaders]);

  const findNodeByPath = (nodes: WorkspaceNode[], path: string): WorkspaceNode | null => {
    for (const node of nodes) {
      if (node.path === path) {
        return node;
      }
      if (node.type === 'dir' && node.children && node.children.length > 0) {
        const nested = findNodeByPath(node.children, path);
        if (nested) {
          return nested;
        }
      }
    }
    return null;
  };

  const refreshOpenTabsFromDisk = async (): Promise<void> => {
    const tabsSnapshot = [...openFiles];
    for (const tab of tabsSnapshot) {
      if (tab.content !== tab.savedContent) {
        continue;
      }
      try {
        const fileData = await readFileContent(tab.path);
        if (!fileData) {
          continue;
        }
        setOpenFiles((prev) =>
          prev.map((item) =>
            item.id === tab.id
              ? {
                  ...item,
                  content: fileData.content,
                  savedContent: fileData.content,
                  version: fileData.version,
                }
              : item,
          ),
        );
      } catch {
        // best effort refresh; keep current tab state on errors
      }
    }
  };

  const readFileContent = async (
    path: string,
  ): Promise<{ content: string; version: string | null } | null> => {
    const result = await fetchWorkspaceFile(path, requestHeaders);
    if (result.pendingApproval) {
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] pending approval: read ${path}`,
      ]);
      return null;
    }
    return { content: result.content, version: result.version };
  };

  const openFileInTab = async (path: string): Promise<void> => {
    const normalizedPath = path.trim();
    if (!normalizedPath) {
      return;
    }
    pushRecentPath(normalizedPath);
    setActiveExplorerPath(normalizedPath);
    const existing = openFiles.find((item) => item.path === normalizedPath);
    if (existing) {
      setActiveFileId(existing.id);
      return;
    }
    const tab: WorkspaceOpenFileTab = {
      id: `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      path: normalizedPath,
      name: normalizedPath.split('/').pop() || normalizedPath,
      content: '',
      savedContent: '',
      version: null,
      loading: true,
    };
    setOpenFiles((prev) => [...prev, tab]);
    setActiveFileId(tab.id);
    try {
      const fileData = await readFileContent(normalizedPath);
      if (fileData === null) {
        setOpenFiles((prev) => prev.filter((item) => item.id !== tab.id));
        if (activeFileId === tab.id) {
          setActiveFileId(null);
        }
        return;
      }
      setOpenFiles((prev) =>
        prev.map((item) =>
          item.id === tab.id
            ? {
                ...item,
                content: fileData.content,
                savedContent: fileData.content,
                version: fileData.version,
                loading: false,
              }
            : item,
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to open file.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
      setOpenFiles((prev) => prev.filter((item) => item.id !== tab.id));
      if (activeFileId === tab.id) {
        setActiveFileId(null);
      }
    }
  };

  const closeTab = (tabId: string) => {
    const target = openFiles.find((item) => item.id === tabId);
    if (!target) {
      return;
    }
    if (target.content !== target.savedContent) {
      const allowClose = window.confirm(`File ${target.name} has unsaved changes. Close anyway?`);
      if (!allowClose) {
        return;
      }
    }
    setOpenFiles((prev) => prev.filter((item) => item.id !== tabId));
    if (activeFileId === tabId) {
      const rest = openFiles.filter((item) => item.id !== tabId);
      setActiveFileId(rest.length > 0 ? rest[rest.length - 1].id : null);
    }
  };

  const updateActiveContent = (value: string) => {
    if (!activeTab) {
      return;
    }
    setOpenFiles((prev) => prev.map((item) => (item.id === activeTab.id ? { ...item, content: value } : item)));
  };

  const handleSave = async () => {
    if (!activeTab || editorSaving || isDecisionBlocking) {
      return;
    }
    setEditorSaving(true);
    try {
      const result = await putWorkspaceFile(
        activeTab.path,
        activeTab.content,
        activeTab.version,
        requestHeaders,
      );
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: save ${activeTab.path}`,
        ]);
        return;
      }
      setOpenFiles((prev) =>
        prev.map((item) =>
          item.id === activeTab.id
            ? {
                ...item,
                savedContent: item.content,
                version: result.version ?? item.version,
              }
            : item,
        ),
      );
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] saved: ${activeTab.path}`]);
      requestTreeLoad(undefined, 'save');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save file.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setEditorSaving(false);
    }
  };

  const handleRunActiveFile = async () => {
    if (!activeTab || terminalBusy || isDecisionBlocking) {
      return;
    }
    setTerminalBusy(true);
    setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] run: ${activeTab.path}`]);
    try {
      const result = await postWorkspaceRun(activeTab.path, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: run ${activeTab.path}`,
        ]);
        return;
      }
      setTerminalLines((prev) => {
        const next = [...prev];
        if (result.stdout) {
          next.push(result.stdout);
        }
        if (result.stderr) {
          next.push(`stderr: ${result.stderr}`);
        }
        next.push(`[${terminalTimestamp()}] exit=${result.exitCode}`);
        return next;
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to run file.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setTerminalBusy(false);
    }
  };

  const handleTerminalSubmit = async () => {
    const command = terminalInput.trim();
    if (!command || terminalBusy || isDecisionBlocking) {
      return;
    }
    setTerminalInput('');
    setTerminalBusy(true);
    setTerminalLines((prev) => [...prev, `$ ${command}`]);
    try {
      const result = await postWorkspaceCommandRunnerRun(command, 'session_root', requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: command runner ${command}`,
        ]);
        return;
      }
      setTerminalLines((prev) => {
        const next = [...prev];
        if (result.stdout.trim()) {
          next.push(result.stdout.trim());
        }
        if (result.stderr.trim()) {
          next.push(`stderr: ${result.stderr.trim()}`);
        }
        next.push(
          `[${terminalTimestamp()}] exit=${result.exitCode} cwd=${result.cwd || workspaceRoot || '.'}`,
        );
        return next;
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to run command runner command.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setTerminalBusy(false);
    }
  };

  const handleApplyProjectRoot = async (nextRoot: string) => {
    const trimmed = nextRoot.trim();
    if (!trimmed || projectRootBusy || !sessionId) {
      return;
    }
    if (trimmed === workspaceRoot.trim()) {
      setProjectPickerOpen(false);
      return;
    }
    setProjectRootBusy(true);
    try {
      const result = await postWorkspaceRootSelect(trimmed, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: switch Computer root`,
        ]);
        setProjectPickerOpen(false);
        return;
      }
      const appliedRoot = result.rootPath.trim() || trimmed;
      onApplyWorkspaceRoot(appliedRoot);
      setProjectPickerOpen(false);
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] Computer root: ${appliedRoot}`]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to change Computer root.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setProjectRootBusy(false);
    }
  };

  const handleReindex = async () => {
    if (!sessionId || indexing) {
      return;
    }
    setIndexing(true);
    try {
      const { indexedCode, indexedDocs, skipped } = await postWorkspaceIndex(requestHeaders);
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] index complete: code=${indexedCode} docs=${indexedDocs} skipped=${skipped}`,
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to index Computer.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setIndexing(false);
    }
  };

  const handleCreateFile = async () => {
    const raw = window.prompt('Новый путь файла (относительно Computer root):');
    const nextPath = raw?.trim() ?? '';
    if (!nextPath) {
      return;
    }
    try {
      const result = await postWorkspaceFileCreate(nextPath, '', false, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: create ${nextPath}`,
        ]);
        return;
      }
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] created: ${nextPath}`]);
      setActiveExplorerPath(nextPath);
      requestTreeLoad(undefined, 'create');
      void openFileInTab(nextPath);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create file.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

  const handleRenamePath = async (sourcePath: string | null) => {
    const currentPath = sourcePath?.trim() ?? '';
    if (!currentPath) {
      return;
    }
    const raw = window.prompt('Новое имя/путь:', currentPath);
    const nextPath = raw?.trim() ?? '';
    if (!nextPath || nextPath === currentPath) {
      return;
    }
    try {
      const result = await postWorkspaceFileRename(currentPath, nextPath, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: rename ${currentPath}`,
        ]);
        return;
      }
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] renamed: ${currentPath} -> ${nextPath}`,
      ]);
      setActiveExplorerPath(nextPath);
      setOpenFiles((prev) =>
        prev.map((item) =>
          item.path === currentPath || item.path.startsWith(`${currentPath}/`)
            ? {
                ...item,
                path: item.path.replace(currentPath, nextPath),
                name: item.path
                  .replace(currentPath, nextPath)
                  .split('/')
                  .pop() || item.name,
              }
            : item,
        ),
      );
      requestTreeLoad(undefined, 'rename');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to rename path.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

  const handleMovePath = async (sourcePath: string | null) => {
    const fromPath = sourcePath?.trim() ?? '';
    if (!fromPath) {
      return;
    }
    const raw = window.prompt('Новый путь (перемещение):', fromPath);
    const toPath = raw?.trim() ?? '';
    if (!toPath || toPath === fromPath) {
      return;
    }
    try {
      const result = await postWorkspaceFileMove(fromPath, toPath, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: move ${fromPath}`,
        ]);
        return;
      }
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] moved: ${fromPath} -> ${toPath}`,
      ]);
      setActiveExplorerPath(toPath);
      setOpenFiles((prev) =>
        prev.map((item) =>
          item.path === fromPath || item.path.startsWith(`${fromPath}/`)
            ? {
                ...item,
                path: item.path.replace(fromPath, toPath),
                name: item.path
                  .replace(fromPath, toPath)
                  .split('/')
                  .pop() || item.name,
              }
            : item,
        ),
      );
      requestTreeLoad(undefined, 'move');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to move path.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

  const handleDeletePath = async (sourcePath: string | null) => {
    const targetPath = sourcePath?.trim() ?? '';
    if (!targetPath) {
      return;
    }
    const node = findNodeByPath(tree, targetPath);
    const recursive = node?.type === 'dir';
    const confirmed = window.confirm(
      recursive
        ? `Удалить директорию ${targetPath} рекурсивно?`
        : `Удалить файл ${targetPath}?`,
    );
    if (!confirmed) {
      return;
    }
    try {
      const result = await deleteWorkspaceFile(targetPath, recursive, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: delete ${targetPath}`,
        ]);
        return;
      }
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] deleted: ${targetPath}`]);
      setOpenFiles((prev) =>
        prev.filter((item) => item.path !== targetPath && !item.path.startsWith(`${targetPath}/`)),
      );
      if (activeTab?.path === targetPath) {
        setActiveFileId(null);
      }
      if (activeExplorerPath === targetPath) {
        setActiveExplorerPath(null);
      }
      requestTreeLoad(undefined, 'delete');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete path.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

  const handleEditorMount: OnMount = (editor) => {
    editorRef.current = editor;
  };

  const terminalPendingText = isDecisionBlocking
    ? 'Ожидает подтверждения решения. Отправка временно заблокирована.'
    : null;
  const COMPUTER_TABS = [
    { id: 'overview' as const, label: 'Overview' },
    { id: 'files' as const, label: 'Files' },
    { id: 'changes' as const, label: 'Changes' },
    { id: 'terminal' as const, label: 'Terminal' },
  ];

  const tabBtnClass = (id: typeof computerTab) =>
    `px-3 py-1 text-xs rounded-md transition-colors border ${
      computerTab === id
        ? 'border-zinc-700 bg-zinc-800 text-zinc-100'
        : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
    }`;

  return (
    <div className="h-full min-h-0 flex flex-col bg-zinc-950 text-zinc-300">
      <WorkspaceToolbar
        modelLabel={modelLabel}
        indexing={indexing}
        workspaceRoot={workspaceRoot}
        sessionPolicyLabel={sessionPolicyLabel}
        sessionYoloActive={sessionYoloActive}
        sessionSafeMode={sessionSafeMode}
        statusMessage={statusMessage}
        onBackToChat={onBackToChat}
        onOpenSessionDrawer={onOpenSessionDrawer}
        onOpenProjectPicker={() => setProjectPickerOpen(true)}
        onReindex={() => {
          void handleReindex();
        }}
        onSwitchToFiles={() => setComputerTab('files')}
        onOpenRepositoryPanel={onOpenRepositoryPanel}
      />

      <div className="h-9 border-b border-zinc-800/80 flex items-center px-2 gap-1 shrink-0 overflow-x-auto">
        {COMPUTER_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setComputerTab(tab.id)}
            className={tabBtnClass(tab.id)}
            data-computer-tab={tab.id}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0">
        {computerTab === 'overview' && (
          <WorkspaceAssistantPanel
            sessionId={sessionId}
            mode={mode}
            activePlan={activePlan}
            activeTask={activeTask}
            autoState={autoState}
            decision={decision}
            decisionBusy={decisionBusy}
            decisionError={decisionError}
            onDecisionRespond={onDecisionRespond}
            messages={messages}
            computerEvents={computerEvents}
            terminalPendingText={terminalPendingText}
            onSendFeedback={onSendFeedback}
          />
        )}

        {computerTab === 'terminal' && (
          <div className="h-full flex flex-col" data-computer-tab-content="terminal">
            <div className="flex-1 overflow-auto px-3 py-2 font-mono text-xs text-zinc-400">
              {terminalLines.map((line, idx) => (
                <div key={idx} className="whitespace-pre-wrap break-all leading-5">{line}</div>
              ))}
              <div ref={terminalEndRef} />
            </div>
            {terminalPendingText ? (
              <div className="border-t border-zinc-800/80 px-3 py-2 text-xs text-amber-300">
                {terminalPendingText}
              </div>
            ) : null}
            <div className="border-t border-zinc-800/80 flex items-center gap-2 px-3 py-2">
              <span className="text-zinc-500 font-mono text-xs select-none">$</span>
              <input
                value={terminalInput}
                onChange={(event) => setTerminalInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    void handleTerminalSubmit();
                  }
                }}
                disabled={!sessionId || terminalBusy || isDecisionBlocking}
                placeholder="command..."
                className="flex-1 bg-transparent outline-none text-xs font-mono text-zinc-300 placeholder-zinc-600 disabled:opacity-50"
              />
              {terminalBusy ? (
                <span className="text-zinc-500 text-xs shrink-0">running…</span>
              ) : null}
            </div>
          </div>
        )}

        {computerTab === 'changes' && (
          <div className="h-full overflow-auto" data-computer-tab-content="changes">
            <GitPanel
              sessionId={sessionId}
              sessionHeader={sessionHeader}
              requestHeaders={requestHeaders}
              workspaceRoot={workspaceRoot}
              refreshToken={gitRefreshToken}
              decisionOutcome={gitDecisionOutcome}
            />
          </div>
        )}

        {computerTab === 'files' && (
          <div
            ref={workspaceGridRef}
            className="h-full min-h-0 grid"
            style={{ gridTemplateColumns: filesTabColumns }}
            data-computer-tab-content="files"
          >
            {explorerVisible ? (
              <>
                <WorkspaceExplorer
                  tree={tree}
                  treeLoading={treeLoading}
                  treeError={treeError}
                  loadingTreePaths={loadingTreePaths}
                  treeMeta={treeMeta}
                  expandedNodes={expandedNodes}
                  activePath={activeTab?.path ?? null}
                  activeExplorerPath={activeExplorerPath}
                  readOnly={!sessionYoloActive}
                  onToggleNode={(node, key, expanded) => {
                    setExpandedNodes((prev) => {
                      const next = new Set(prev);
                      if (expanded) {
                        next.delete(key);
                      } else {
                        next.add(key);
                      }
                      return next;
                    });
                    if (!expanded && node.type === 'dir' && (node.children?.length ?? 0) === 0 && node.hasChildren) {
                      requestTreeLoad(node.path, 'expand_dir');
                    }
                  }}
                  onSelectPath={setActiveExplorerPath}
                  onOpenFile={(path) => {
                    void openFileInTab(path);
                  }}
                  onCreateFile={() => {
                    void handleCreateFile();
                  }}
                  onRenamePath={(path) => {
                    void handleRenamePath(path);
                  }}
                  onMovePath={(path) => {
                    void handleMovePath(path);
                  }}
                  onDeletePath={(path) => {
                    void handleDeletePath(path);
                  }}
                />
                <button
                  onMouseDown={startExplorerResize}
                  className="cursor-col-resize bg-zinc-900 hover:bg-zinc-800"
                  aria-label="Resize explorer"
                  title="Resize explorer"
                />
              </>
            ) : null}
            <WorkspaceEditorPane
              openFiles={openFiles}
              activeFileId={activeFileId}
              activeTab={activeTab}
              hasUnsavedChanges={hasUnsavedChanges}
              editorSaving={editorSaving}
              terminalBusy={terminalBusy}
              isDecisionBlocking={isDecisionBlocking}
              terminalHeight={terminalHeight}
              terminalLines={terminalLines}
              terminalInput={terminalInput}
              terminalInputDisabled={!sessionId || terminalBusy || isDecisionBlocking}
              terminalEndRef={terminalEndRef}
              readOnly={!sessionYoloActive}
              onSelectTab={setActiveFileId}
              onCloseTab={closeTab}
              onRunActiveFile={() => {
                void handleRunActiveFile();
              }}
              onSaveActiveFile={() => {
                void handleSave();
              }}
              onEditorMount={handleEditorMount}
              onEditorChange={updateActiveContent}
              onTerminalResizeStart={startTerminalResize}
              onTerminalInputChange={setTerminalInput}
              onTerminalSubmit={() => {
                void handleTerminalSubmit();
              }}
            />
          </div>
        )}

      </div>

      {projectPickerOpen ? (
        <ProjectPicker
          sessionId={sessionId}
          sessionHeader={sessionHeader}
          workspaceRoot={workspaceRoot}
          loading={projectRootBusy}
          onApplyRoot={(path) => {
            void handleApplyProjectRoot(path);
          }}
          onClose={() => setProjectPickerOpen(false)}
        />
      ) : null}

      <WorkspaceQuickOpen
        open={quickOpenOpen}
        query={quickOpenQuery}
        items={quickOpenResults}
        loading={quickOpenLoading}
        partial={quickOpenPartial}
        onQueryChange={setQuickOpenQuery}
        onSelect={handleQuickOpenSelect}
        onClose={() => setQuickOpenOpen(false)}
      />
    </div>
  );
}
