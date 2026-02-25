import { useEffect, useMemo, useRef, useState } from 'react';
import type { OnMount } from '@monaco-editor/react';
import type * as Monaco from 'monaco-editor';

import type {
  AutoState,
  DecisionRespondChoice,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
  UiDecision,
} from '../types';
import type { CanvasMessage, CanvasSendPayload } from './canvas';
import {
  findFirstFilePath,
  policyLabel,
  terminalTimestamp,
  type WorkspaceNode,
  type WorkspaceTreeMeta,
} from '../../features/workspace/workspace-helpers';
import {
  WorkspaceEditorPane,
  type WorkspaceOpenFileTab,
} from '../../features/workspace/workspace-editor-pane';
import {
  WorkspaceAssistantPanel,
  type WorkspaceModelOption,
  type WorkspaceContextChip,
} from '../../features/workspace/workspace-assistant-panel';
import { WorkspaceExplorer } from '../../features/workspace/workspace-explorer';
import {
  WorkspaceQuickOpen,
  type WorkspaceQuickOpenItem,
} from '../../features/workspace/workspace-quick-open';
import { WorkspaceToolbar } from '../../features/workspace/workspace-toolbar';
import {
  deleteWorkspaceFile,
  fetchWorkspaceFile,
  fetchWorkspaceGitDiff,
  fetchWorkspaceRoot,
  fetchWorkspaceTree,
  postWorkspaceFileCreate,
  postWorkspaceFileMove,
  postWorkspaceFileRename,
  postWorkspaceIndex,
  postWorkspaceRootSelect,
  postWorkspaceRun,
  postWorkspaceTerminalRun,
  putWorkspaceFile,
} from '../../features/workspace/workspace-api';
import {
  buildWorkspaceContextAttachments,
  buildWorkspaceContextChips,
} from '../../features/workspace/workspace-context';

type WorkspaceIdeProps = {
  sessionId: string | null;
  sessionHeader: string;
  modelLabel: string;
  modelOptions: WorkspaceModelOption[];
  selectedModelValue: string | null;
  modelsLoading: boolean;
  savingModel: boolean;
  onSelectModel: (provider: string, model: string) => void;
  messages: CanvasMessage[];
  sending: boolean;
  statusMessage?: string | null;
  onBackToChat: () => void;
  onOpenWorkspaceSettings: () => void;
  onSendAgentMessage: (payload: CanvasSendPayload) => Promise<boolean>;
  onSendFeedback?: (interactionId: string, rating: 'good' | 'bad') => Promise<boolean>;
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  modeBusy?: boolean;
  modeError?: string | null;
  onChangeMode: (mode: SessionMode) => Promise<void>;
  onPlanDraft: (goal: string) => Promise<void>;
  onPlanApprove: () => Promise<void>;
  onPlanExecute: () => Promise<void>;
  onPlanCancel: () => Promise<void>;
  decision?: UiDecision | null;
  decisionBusy?: boolean;
  decisionError?: string | null;
  onDecisionRespond?: (
    choice: DecisionRespondChoice,
    editedPayload?: Record<string, unknown> | null,
  ) => Promise<void> | void;
  refreshToken?: number;
  explorerVisible: boolean;
};

const MIN_EXPLORER_WIDTH = 240;
const MAX_EXPLORER_WIDTH = 420;
const MIN_ASSISTANT_WIDTH = 340;
const ASSISTANT_MAX_SCREEN_SHARE = 0.5;
const MIN_TERMINAL_HEIGHT = 140;
const MAX_TERMINAL_HEIGHT = 420;
const MIN_EDITOR_WIDTH = 420;
const ASSISTANT_RESIZER_WIDTH = 6;
const EXPLORER_RESIZER_WIDTH = 6;
const ROOT_TREE_DEBOUNCE_MS = 150;
const CHILD_TREE_DEBOUNCE_MS = 80;
const QUICK_OPEN_MAX_RESULTS = 120;
const QUICK_OPEN_MAX_RECENT = 24;

type QuickOpenIndexCache = {
  rootKey: string;
  items: WorkspaceQuickOpenItem[];
  partial: boolean;
  loadedAt: number;
};

export function WorkspaceIde({
  sessionId,
  sessionHeader,
  modelLabel,
  modelOptions,
  selectedModelValue,
  modelsLoading,
  savingModel,
  onSelectModel,
  messages,
  sending,
  statusMessage,
  onBackToChat,
  onOpenWorkspaceSettings,
  onSendAgentMessage,
  onSendFeedback,
  mode,
  activePlan,
  activeTask,
  autoState,
  modeBusy = false,
  modeError = null,
  onChangeMode,
  onPlanDraft,
  onPlanApprove,
  onPlanExecute,
  onPlanCancel,
  decision,
  decisionBusy = false,
  decisionError = null,
  onDecisionRespond,
  refreshToken = 0,
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
  const [selectionText, setSelectionText] = useState('');

  const [terminalLines, setTerminalLines] = useState<string[]>([
    `[${terminalTimestamp()}] Workspace terminal ready.`,
  ]);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalBusy, setTerminalBusy] = useState(false);

  const [agentInput, setAgentInput] = useState('');
  const [includeOpenTabs, setIncludeOpenTabs] = useState(true);
  const [includeSelection, setIncludeSelection] = useState(true);
  const [includeGitDiff, setIncludeGitDiff] = useState(true);
  const [includeTerminal, setIncludeTerminal] = useState(true);

  const [workspaceRoot, setWorkspaceRoot] = useState('');
  const [workspacePolicy, setWorkspacePolicy] = useState('Sandbox');
  const [workspaceYoloActive, setWorkspaceYoloActive] = useState(false);
  const [workspaceSafeMode, setWorkspaceSafeMode] = useState(true);
  const [rootPickerOpen, setRootPickerOpen] = useState(false);
  const [rootInput, setRootInput] = useState('');
  const [rootBusy, setRootBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [gitDiffLoading, setGitDiffLoading] = useState(false);
  const [gitDiff, setGitDiff] = useState('');
  const [quickOpenOpen, setQuickOpenOpen] = useState(false);
  const [quickOpenQuery, setQuickOpenQuery] = useState('');
  const [quickOpenLoading, setQuickOpenLoading] = useState(false);
  const [quickOpenPartial, setQuickOpenPartial] = useState(false);
  const [quickOpenItems, setQuickOpenItems] = useState<WorkspaceQuickOpenItem[]>([]);
  const [recentPaths, setRecentPaths] = useState<string[]>([]);

  const [explorerWidth, setExplorerWidth] = useState(280);
  const [assistantWidth, setAssistantWidth] = useState(390);
  const [terminalHeight, setTerminalHeight] = useState(220);
  const [draggingExplorer, setDraggingExplorer] = useState(false);
  const [draggingAssistant, setDraggingAssistant] = useState(false);
  const [draggingTerminal, setDraggingTerminal] = useState(false);

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
  const workspaceGridRef = useRef<HTMLDivElement>(null);

  const clampAssistantWidth = (nextWidth: number): number => {
    const fallbackWidth = typeof window === 'undefined' ? 1280 : window.innerWidth;
    const gridWidth = workspaceGridRef.current?.clientWidth ?? fallbackWidth;
    const fixedColumns = explorerVisible
      ? explorerWidth + EXPLORER_RESIZER_WIDTH + ASSISTANT_RESIZER_WIDTH
      : ASSISTANT_RESIZER_WIDTH;
    const maxByEditor = Math.max(
      MIN_ASSISTANT_WIDTH,
      Math.floor(gridWidth - fixedColumns - MIN_EDITOR_WIDTH),
    );
    const maxByHalfScreen = Math.max(
      MIN_ASSISTANT_WIDTH,
      Math.floor(gridWidth * ASSISTANT_MAX_SCREEN_SHARE),
    );
    const maxAllowed = Math.min(maxByEditor, maxByHalfScreen);
    return Math.min(maxAllowed, Math.max(MIN_ASSISTANT_WIDTH, nextWidth));
  };

  const activeTab = useMemo(
    () => openFiles.find((item) => item.id === activeFileId) ?? null,
    [activeFileId, openFiles],
  );
  const hasUnsavedChanges = Boolean(activeTab && activeTab.content !== activeTab.savedContent);
  const isDecisionBlocking = decision?.status === 'pending' && decision.blocking === true;

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
    setRecentPaths((prev) => {
      const next = [normalized, ...prev.filter((item) => item !== normalized)];
      return next.slice(0, QUICK_OPEN_MAX_RECENT);
    });
  };

  const collectQuickOpenItems = (nodes: WorkspaceNode[]): WorkspaceQuickOpenItem[] => {
    const output: WorkspaceQuickOpenItem[] = [];
    const walk = (items: WorkspaceNode[]) => {
      for (const node of items) {
        if (node.type === 'file') {
          const path = node.path?.trim() ?? '';
          if (!path) {
            continue;
          }
          const slash = path.lastIndexOf('/');
          const name = slash >= 0 ? path.slice(slash + 1) : path;
          const dir = slash >= 0 ? path.slice(0, slash) : '';
          output.push({ path, name, dir });
          continue;
        }
        if (node.children && node.children.length > 0) {
          walk(node.children);
        }
      }
    };
    walk(nodes);
    output.sort((a, b) => {
      const byName = a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
      if (byName !== 0) {
        return byName;
      }
      return a.path.localeCompare(b.path, undefined, { sensitivity: 'base' });
    });
    return output;
  };

  const quickOpenResults = useMemo(() => {
    const rawQuery = quickOpenQuery.trim().toLowerCase();
    if (!rawQuery) {
      return quickOpenItems.slice(0, QUICK_OPEN_MAX_RESULTS);
    }
    const recentBoost = new Set(recentPaths);
    const scored = quickOpenItems
      .map((item) => {
        const name = item.name.toLowerCase();
        const path = item.path.toLowerCase();
        let score = -1;
        if (name === rawQuery) {
          score = 400;
        } else if (name.startsWith(rawQuery)) {
          score = 300;
        } else if (name.includes(rawQuery)) {
          score = 200;
        } else if (path.includes(rawQuery)) {
          score = 120;
        }
        if (score < 0) {
          return null;
        }
        if (recentBoost.has(item.path)) {
          score += 35;
        }
        return { item, score };
      })
      .filter((entry): entry is { item: WorkspaceQuickOpenItem; score: number } => entry !== null)
      .sort((a, b) => {
        if (b.score !== a.score) {
          return b.score - a.score;
        }
        return a.item.path.localeCompare(b.item.path, undefined, { sensitivity: 'base' });
      });
    return scored.slice(0, QUICK_OPEN_MAX_RESULTS).map((entry) => entry.item);
  }, [quickOpenItems, quickOpenQuery, recentPaths]);

  const lastTerminalOutput = useMemo(() => {
    for (let index = terminalLines.length - 1; index >= 0; index -= 1) {
      const line = terminalLines[index].trim();
      if (!line) {
        continue;
      }
      if (line.startsWith('[') && line.includes(']')) {
        continue;
      }
      return line;
    }
    return '';
  }, [terminalLines]);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLines]);

  useEffect(() => {
    assistantInitRef.current = false;
    assistantSeenRef.current = new Set();
    decisionSeenRef.current = new Set();
    statusSeenRef.current = null;
    treeDebounceTimersRef.current.forEach((timerId) => {
      window.clearTimeout(timerId);
    });
    treeDebounceTimersRef.current.clear();
    treeAbortControllersRef.current.forEach((controller) => {
      controller.abort();
    });
    treeAbortControllersRef.current.clear();
    treeInFlightPathsRef.current.clear();
    setOpenFiles([]);
    setActiveFileId(null);
    setActiveExplorerPath(null);
    setLoadingTreePaths(new Set());
    setTreeMeta(null);
    setSelectionText('');
    setQuickOpenOpen(false);
    setQuickOpenQuery('');
    setQuickOpenPartial(false);
    setQuickOpenItems([]);
    setRecentPaths([]);
    quickOpenFileIndexRef.current = null;
    quickOpenLoadedForRoot.current = null;
  }, [sessionId]);

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
      void loadGitDiff();
      void refreshOpenTabsFromDisk();
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
    };
  }, []);

  useEffect(() => {
    if (!(draggingExplorer || draggingAssistant || draggingTerminal)) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      if (explorerVisible && draggingExplorer) {
        setExplorerWidth((prev) => Math.min(MAX_EXPLORER_WIDTH, Math.max(MIN_EXPLORER_WIDTH, prev + event.movementX)));
      }
      if (draggingAssistant) {
        setAssistantWidth((prev) => clampAssistantWidth(prev + event.movementX));
      }
      if (draggingTerminal) {
        setTerminalHeight((prev) => Math.min(MAX_TERMINAL_HEIGHT, Math.max(MIN_TERMINAL_HEIGHT, prev - event.movementY)));
      }
    };
    const handleUp = () => {
      setDraggingExplorer(false);
      setDraggingAssistant(false);
      setDraggingTerminal(false);
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [draggingAssistant, draggingExplorer, draggingTerminal, explorerVisible]);

  useEffect(() => {
    if (!explorerVisible && draggingExplorer) {
      setDraggingExplorer(false);
    }
  }, [draggingExplorer, explorerVisible]);

  useEffect(() => {
    const syncAssistantWidth = () => {
      setAssistantWidth((prev) => clampAssistantWidth(prev));
    };
    syncAssistantWidth();
    window.addEventListener('resize', syncAssistantWidth);
    return () => {
      window.removeEventListener('resize', syncAssistantWidth);
    };
  }, [explorerVisible, explorerWidth]);

  const loadWorkspaceRoot = async (): Promise<void> => {
    if (!sessionId) {
      setWorkspaceRoot('');
      setWorkspacePolicy('Sandbox');
      setWorkspaceYoloActive(false);
      setWorkspaceSafeMode(true);
      return;
    }
    try {
      const { rootPath, policy } = await fetchWorkspaceRoot(requestHeaders);
      setWorkspaceRoot(rootPath);
      setRootInput(rootPath);
      if (policy) {
        setWorkspacePolicy(policyLabel(policy.profile));
        setWorkspaceYoloActive(policy.yolo_armed === true);
        setWorkspaceSafeMode(policy.safe_mode_effective !== false);
      } else {
        setWorkspacePolicy('Sandbox');
        setWorkspaceYoloActive(false);
        setWorkspaceSafeMode(true);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load workspace root.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

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
      if (pendingApproval) {
        setTreeError('Ожидает подтверждения действия для доступа к workspace.');
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: workspace tree request`,
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
      if (isRootLoad && !activeFileId) {
        const first = findFirstFilePath(parsedTree);
        if (first) {
          void openFileInTab(first);
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      const message = error instanceof Error ? error.message : 'Failed to load workspace tree.';
      if (isRootLoad) {
        setTreeError(message);
      } else {
        setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
      }
    } finally {
      if (treeAbortControllersRef.current.get(requestKey) === abortController) {
        treeAbortControllersRef.current.delete(requestKey);
      }
      treeInFlightPathsRef.current.delete(requestKey);
      if (isRootLoad) {
        setTreeLoading(false);
      } else {
        setPathLoading(normalizedPath, false);
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
    setQuickOpenLoading(true);
    try {
      const { pendingApproval, tree: parsedTree, treeMeta: loadedTreeMeta } = await fetchWorkspaceTree(
        { recursive: true, maxDepth: 12 },
        requestHeaders,
      );
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
      const message = error instanceof Error ? error.message : 'Failed to load quick open file list.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
      setQuickOpenItems([]);
      setQuickOpenPartial(false);
    } finally {
      setQuickOpenLoading(false);
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

  const loadGitDiff = async (): Promise<void> => {
    if (!sessionId) {
      setGitDiff('');
      return;
    }
    setGitDiffLoading(true);
    try {
      setGitDiff(await fetchWorkspaceGitDiff(requestHeaders));
    } catch {
      setGitDiff('');
    } finally {
      setGitDiffLoading(false);
    }
  };

  useEffect(() => {
    void loadWorkspaceRoot();
    requestTreeLoad(undefined, 'session_init');
    void loadGitDiff();
  }, [refreshToken, sessionId]);

  useEffect(() => {
    const rootKey = workspaceRoot.trim();
    if (quickOpenLoadedForRoot.current === rootKey) {
      return;
    }
    quickOpenLoadedForRoot.current = null;
    quickOpenFileIndexRef.current = null;
    setQuickOpenItems([]);
    setQuickOpenPartial(false);
  }, [workspaceRoot]);

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

      const key = event.key.toLowerCase();
      const isPrimaryShortcut = event.code === 'Space' || key === ' ';
      const isSecondaryShortcut = key === 'd';
      if (!isPrimaryShortcut && !isSecondaryShortcut) {
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
      void loadGitDiff();
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
      void loadGitDiff();
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
      const result = await postWorkspaceTerminalRun(command, 'session_root', requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: terminal ${command}`,
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
      const message = error instanceof Error ? error.message : 'Failed to run terminal command.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setTerminalBusy(false);
    }
  };

  const handleSelectRoot = async () => {
    const nextRoot = rootInput.trim();
    if (!nextRoot || rootBusy || !sessionId) {
      return;
    }
    setRootBusy(true);
    try {
      const result = await postWorkspaceRootSelect(nextRoot, requestHeaders);
      if (result.pendingApproval) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: switch workspace root`,
        ]);
        setRootPickerOpen(false);
        return;
      }
      setWorkspaceRoot(nextRoot);
      setRootPickerOpen(false);
      setOpenFiles([]);
      setActiveFileId(null);
      void loadWorkspaceRoot();
      requestTreeLoad(undefined, 'root_change');
      void loadGitDiff();
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] workspace root: ${nextRoot}`]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to change workspace root.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setRootBusy(false);
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
      const message = error instanceof Error ? error.message : 'Failed to index workspace.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setIndexing(false);
    }
  };

  const handleCreateFile = async () => {
    const raw = window.prompt('Новый путь файла (относительно workspace):');
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
      void loadGitDiff();
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
      void loadGitDiff();
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
      void loadGitDiff();
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
      void loadGitDiff();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete path.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

  const buildContextAttachments = () =>
    buildWorkspaceContextAttachments({
      includeOpenTabs,
      includeSelection,
      includeGitDiff,
      includeTerminal,
      openFiles,
      activeFile: activeTab,
      selectionText,
      gitDiff,
      lastTerminalOutput,
    });

  const handleAgentSend = async () => {
    const content = agentInput.trim();
    const attachments = buildContextAttachments();
    if ((!content && attachments.length === 0) || sending || isDecisionBlocking) {
      return;
    }
    const previousInput = agentInput;
    setAgentInput('');
    const ok = await onSendAgentMessage({
      content,
      attachments: attachments.length > 0 ? attachments : undefined,
    });
    if (!ok) {
      setAgentInput((current) => (current.trim().length === 0 ? previousInput : current));
    }
  };

  const handleEditorMount: OnMount = (editor) => {
    editorRef.current = editor;
    editor.onDidChangeCursorSelection(() => {
      const model = editor.getModel();
      const selection = editor.getSelection();
      const selected =
        model && selection ? model.getValueInRange(selection) : '';
      setSelectionText(selected.trim());
    });
  };

  const aiContextChips: WorkspaceContextChip[] = buildWorkspaceContextChips({
    openFilesCount: openFiles.length,
    includeOpenTabs,
    onToggleOpenTabs: () => setIncludeOpenTabs((prev) => !prev),
    selectionText,
    includeSelection,
    onToggleSelection: () => setIncludeSelection((prev) => !prev),
    gitDiff,
    gitDiffLoading,
    includeGitDiff,
    onToggleGitDiff: () => setIncludeGitDiff((prev) => !prev),
    lastTerminalOutput,
    includeTerminal,
    onToggleTerminal: () => setIncludeTerminal((prev) => !prev),
  });

  const terminalPendingText = isDecisionBlocking
    ? 'Ожидает подтверждения решения. Отправка временно заблокирована.'
    : null;
  const canSendWithContext = Boolean(agentInput.trim() || buildContextAttachments().length > 0);
  const workspaceGridColumns = explorerVisible
    ? `${explorerWidth}px ${EXPLORER_RESIZER_WIDTH}px ${assistantWidth}px ${ASSISTANT_RESIZER_WIDTH}px minmax(${MIN_EDITOR_WIDTH}px,1fr)`
    : `${assistantWidth}px ${ASSISTANT_RESIZER_WIDTH}px minmax(${MIN_EDITOR_WIDTH}px,1fr)`;

  return (
    <div className="h-full min-h-0 flex flex-col bg-[#0a0a0d] text-[#d2d2d9]">
      <WorkspaceToolbar
        modelLabel={modelLabel}
        indexing={indexing}
        workspaceRoot={workspaceRoot}
        workspacePolicy={workspacePolicy}
        workspaceYoloActive={workspaceYoloActive}
        workspaceSafeMode={workspaceSafeMode}
        rootPickerOpen={rootPickerOpen}
        rootInput={rootInput}
        rootBusy={rootBusy}
        statusMessage={statusMessage}
        onBackToChat={onBackToChat}
        onToggleRootPicker={() => setRootPickerOpen((prev) => !prev)}
        onReindex={() => {
          void handleReindex();
        }}
        onRefreshGitDiff={() => {
          void loadGitDiff();
        }}
        onOpenWorkspaceSettings={onOpenWorkspaceSettings}
        onOpenQuickOpen={openQuickOpen}
        onRootInputChange={setRootInput}
        onApplyRoot={() => {
          void handleSelectRoot();
        }}
        onCancelRootPicker={() => setRootPickerOpen(false)}
      />

      <div
        ref={workspaceGridRef}
        className="flex-1 min-h-0 grid"
        style={{
          gridTemplateColumns: workspaceGridColumns,
        }}
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
              onMouseDown={() => setDraggingExplorer(true)}
              className="cursor-col-resize bg-[#121218] hover:bg-[#1b1b23]"
              aria-label="Resize explorer"
              title="Resize explorer"
            />
          </>
        ) : null}

        <WorkspaceAssistantPanel
          contextChips={aiContextChips}
        mode={mode}
        modelOptions={modelOptions}
        selectedModelValue={selectedModelValue}
        modelsLoading={modelsLoading}
        savingModel={savingModel}
        onSelectModel={onSelectModel}
        activePlan={activePlan}
        activeTask={activeTask}
        autoState={autoState}
        modeBusy={modeBusy}
          modeError={modeError}
          onChangeMode={onChangeMode}
          onPlanDraft={onPlanDraft}
          onPlanApprove={onPlanApprove}
          onPlanExecute={onPlanExecute}
          onPlanCancel={onPlanCancel}
          decision={decision}
          decisionBusy={decisionBusy}
          decisionError={decisionError}
          onDecisionRespond={onDecisionRespond}
          messages={messages}
          terminalPendingText={terminalPendingText}
          agentInput={agentInput}
          sending={sending}
          isDecisionBlocking={isDecisionBlocking}
          canSend={canSendWithContext}
          onSendFeedback={onSendFeedback}
          onAgentInputChange={setAgentInput}
          onAgentSend={() => {
            void handleAgentSend();
          }}
          onSendPayload={onSendAgentMessage}
        />

        <button
          onMouseDown={() => setDraggingAssistant(true)}
          className="cursor-col-resize bg-[#121218] hover:bg-[#1b1b23]"
          aria-label="Resize assistant"
          title="Resize assistant"
        />

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
          onTerminalResizeStart={() => setDraggingTerminal(true)}
          onTerminalInputChange={setTerminalInput}
          onTerminalSubmit={() => {
            void handleTerminalSubmit();
          }}
        />
      </div>

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
