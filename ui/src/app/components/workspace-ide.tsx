import { useEffect, useMemo, useRef, useState } from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import type * as Monaco from 'monaco-editor';
import {
  ArrowLeft,
  Bot,
  ChevronDown,
  ChevronRight,
  FileCode2,
  FileText,
  Folder,
  FolderOpen,
  Play,
  RefreshCcw,
  Save,
  Send,
  Settings,
  Terminal,
  X,
} from 'lucide-react';

import type { UiDecision } from '../types';
import type { CanvasMessage, CanvasSendPayload } from './canvas';
import { DecisionPanel } from './decision-panel';

type WorkspaceNode = {
  name: string;
  type: 'dir' | 'file';
  path?: string;
  children?: WorkspaceNode[];
};

type WorkspaceRootPayload = {
  session_id?: unknown;
  root_path?: unknown;
  policy?: unknown;
};

type OpenFileTab = {
  id: string;
  path: string;
  name: string;
  content: string;
  savedContent: string;
  loading: boolean;
};

type WorkspaceIdeProps = {
  sessionId: string | null;
  sessionHeader: string;
  modelLabel: string;
  messages: CanvasMessage[];
  sending: boolean;
  statusMessage?: string | null;
  onBackToChat: () => void;
  onOpenSettings: () => void;
  onSendAgentMessage: (payload: CanvasSendPayload) => Promise<boolean>;
  decision?: UiDecision | null;
  decisionBusy?: boolean;
  decisionError?: string | null;
  onDecisionRespond?: (
    choice: 'approve' | 'reject' | 'edit',
    editedAction?: Record<string, unknown> | null,
  ) => Promise<void> | void;
};

type ApiErrorPayload = {
  message: string;
  code: string | null;
  details: Record<string, unknown> | null;
};

const MIN_EXPLORER_WIDTH = 240;
const MAX_EXPLORER_WIDTH = 420;
const MIN_ASSISTANT_WIDTH = 340;
const MAX_ASSISTANT_WIDTH = 520;
const MIN_TERMINAL_HEIGHT = 140;
const MAX_TERMINAL_HEIGHT = 420;

const extractApiError = (payload: unknown, fallback: string): ApiErrorPayload => {
  if (!payload || typeof payload !== 'object') {
    return { message: fallback, code: null, details: null };
  }
  const maybeError = payload as {
    error?: {
      message?: unknown;
      code?: unknown;
      details?: unknown;
    };
  };
  if (maybeError.error && typeof maybeError.error.message === 'string' && maybeError.error.message.trim()) {
    return {
      message: maybeError.error.message,
      code: typeof maybeError.error.code === 'string' ? maybeError.error.code : null,
      details:
        maybeError.error.details && typeof maybeError.error.details === 'object'
          ? (maybeError.error.details as Record<string, unknown>)
          : null,
    };
  }
  return { message: fallback, code: null, details: null };
};

const explainWorkspaceFailure = (error: ApiErrorPayload): string => {
  if (error.code === 'approval_required') {
    return 'Ожидает подтверждения действия (approval required).';
  }
  const normalized = error.message.toLowerCase();
  if (normalized.includes('safe mode')) {
    return `Блокировано safe-mode: ${error.message}`;
  }
  if (normalized.includes('инструмент') && normalized.includes('отключ')) {
    return `Инструмент отключён в Settings: ${error.message}`;
  }
  return error.message;
};

const parseWorkspaceTree = (value: unknown): WorkspaceNode[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const parseNode = (raw: unknown): WorkspaceNode | null => {
    if (!raw || typeof raw !== 'object') {
      return null;
    }
    const candidate = raw as {
      name?: unknown;
      type?: unknown;
      path?: unknown;
      children?: unknown;
    };
    if (typeof candidate.name !== 'string' || !candidate.name.trim()) {
      return null;
    }
    if (candidate.type !== 'dir' && candidate.type !== 'file') {
      return null;
    }
    if (candidate.type === 'file') {
      const path =
        typeof candidate.path === 'string' && candidate.path.trim()
          ? candidate.path.trim()
          : candidate.name.trim();
      return {
        name: candidate.name.trim(),
        type: 'file',
        path,
      };
    }
    const childrenRaw = Array.isArray(candidate.children) ? candidate.children : [];
    const children = childrenRaw
      .map((item) => parseNode(item))
      .filter((item): item is WorkspaceNode => item !== null);
    return {
      name: candidate.name.trim(),
      type: 'dir',
      children,
    };
  };

  return value
    .map((item) => parseNode(item))
    .filter((item): item is WorkspaceNode => item !== null);
};

const findFirstFilePath = (nodes: WorkspaceNode[]): string | null => {
  for (const node of nodes) {
    if (node.type === 'file' && node.path) {
      return node.path;
    }
    if (node.type === 'dir' && node.children) {
      const nested = findFirstFilePath(node.children);
      if (nested) {
        return nested;
      }
    }
  }
  return null;
};

const nodeKey = (node: WorkspaceNode, parentKey: string): string => {
  const base = parentKey ? `${parentKey}/${node.name}` : node.name;
  if (node.type === 'file' && node.path) {
    return `file:${node.path}`;
  }
  return `dir:${base}`;
};

const fileIcon = (path: string) => {
  const normalized = path.toLowerCase();
  if (
    normalized.endsWith('.py')
    || normalized.endsWith('.ts')
    || normalized.endsWith('.tsx')
    || normalized.endsWith('.js')
    || normalized.endsWith('.jsx')
  ) {
    return <FileCode2 className="h-3.5 w-3.5 text-[#6f9cff]" />;
  }
  return <FileText className="h-3.5 w-3.5 text-[#8f8f97]" />;
};

const terminalTimestamp = (): string => new Date().toLocaleTimeString();

const extensionFromPath = (path: string): string => {
  const index = path.lastIndexOf('.');
  if (index < 0 || index === path.length - 1) {
    return '';
  }
  return path.slice(index + 1).toLowerCase();
};

const monacoLanguageFromPath = (path: string): string => {
  const ext = extensionFromPath(path);
  if (ext === 'py') return 'python';
  if (ext === 'ts') return 'typescript';
  if (ext === 'tsx') return 'typescript';
  if (ext === 'js') return 'javascript';
  if (ext === 'jsx') return 'javascript';
  if (ext === 'json') return 'json';
  if (ext === 'md') return 'markdown';
  if (ext === 'toml') return 'ini';
  if (ext === 'yaml' || ext === 'yml') return 'yaml';
  if (ext === 'html') return 'html';
  if (ext === 'css') return 'css';
  if (ext === 'sh') return 'shell';
  if (ext === 'sql') return 'sql';
  return 'plaintext';
};

const compactPath = (path: string, max = 60): string => {
  if (path.length <= max) {
    return path;
  }
  const keep = Math.max(16, max - 3);
  return `...${path.slice(path.length - keep)}`;
};

const policyLabel = (value: unknown): string => {
  if (value === 'index') return 'Index';
  if (value === 'yolo') return 'YOLO';
  return 'Sandbox';
};

export function WorkspaceIde({
  sessionId,
  sessionHeader,
  modelLabel,
  messages,
  sending,
  statusMessage,
  onBackToChat,
  onOpenSettings,
  onSendAgentMessage,
  decision,
  decisionBusy = false,
  decisionError = null,
  onDecisionRespond,
}: WorkspaceIdeProps) {
  const [tree, setTree] = useState<WorkspaceNode[]>([]);
  const [treeLoading, setTreeLoading] = useState(false);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  const [openFiles, setOpenFiles] = useState<OpenFileTab[]>([]);
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
  const [rootPickerOpen, setRootPickerOpen] = useState(false);
  const [rootInput, setRootInput] = useState('');
  const [rootBusy, setRootBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [gitDiffLoading, setGitDiffLoading] = useState(false);
  const [gitDiff, setGitDiff] = useState('');

  const [explorerWidth, setExplorerWidth] = useState(280);
  const [assistantWidth, setAssistantWidth] = useState(390);
  const [terminalHeight, setTerminalHeight] = useState(220);
  const [draggingExplorer, setDraggingExplorer] = useState(false);
  const [draggingAssistant, setDraggingAssistant] = useState(false);
  const [draggingTerminal, setDraggingTerminal] = useState(false);

  const terminalEndRef = useRef<HTMLDivElement | null>(null);
  const assistantSeenRef = useRef<Set<string>>(new Set());
  const assistantInitRef = useRef(false);
  const decisionSeenRef = useRef<Set<string>>(new Set());
  const statusSeenRef = useRef<string | null>(null);
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);

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

  const recentMessages = useMemo(() => messages.slice(-24), [messages]);

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
    setOpenFiles([]);
    setActiveFileId(null);
    setSelectionText('');
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
    if (!statusMessage || statusSeenRef.current === statusMessage) {
      return;
    }
    statusSeenRef.current = statusMessage;
    setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] ${statusMessage}`]);
  }, [statusMessage]);

  useEffect(() => {
    if (!(draggingExplorer || draggingAssistant || draggingTerminal)) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      if (draggingExplorer) {
        setExplorerWidth((prev) => Math.min(MAX_EXPLORER_WIDTH, Math.max(MIN_EXPLORER_WIDTH, prev + event.movementX)));
      }
      if (draggingAssistant) {
        setAssistantWidth((prev) => Math.min(MAX_ASSISTANT_WIDTH, Math.max(MIN_ASSISTANT_WIDTH, prev + event.movementX)));
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
  }, [draggingAssistant, draggingExplorer, draggingTerminal]);

  const loadWorkspaceRoot = async (): Promise<void> => {
    if (!sessionId) {
      setWorkspaceRoot('');
      setWorkspacePolicy('Sandbox');
      return;
    }
    try {
      const response = await fetch('/ui/api/workspace/root', { headers: requestHeaders });
      const payload: unknown = await response.json();
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to load workspace root.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      const parsed = payload as WorkspaceRootPayload;
      const rootPath = typeof parsed.root_path === 'string' ? parsed.root_path : '';
      setWorkspaceRoot(rootPath);
      setRootInput(rootPath);
      const policyRaw = parsed.policy;
      if (policyRaw && typeof policyRaw === 'object') {
        const policyObj = policyRaw as { profile?: unknown; yolo_armed?: unknown };
        setWorkspacePolicy(policyLabel(policyObj.profile));
        setWorkspaceYoloActive(policyObj.yolo_armed === true);
      } else {
        setWorkspacePolicy('Sandbox');
        setWorkspaceYoloActive(false);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load workspace root.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    }
  };

  const loadTree = async (): Promise<void> => {
    if (!sessionId) {
      setTree([]);
      setTreeError('No active session. Create chat first.');
      return;
    }
    setTreeLoading(true);
    setTreeError(null);
    try {
      const response = await fetch('/ui/api/workspace/tree', { headers: requestHeaders });
      const payload: unknown = await response.json();
      if (response.status === 202) {
        setTreeError('Ожидает подтверждения действия для доступа к workspace.');
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: workspace tree request`,
        ]);
        return;
      }
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to load workspace tree.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      const parsedTree = parseWorkspaceTree((payload as { tree?: unknown }).tree);
      setTree(parsedTree);
      const expanded = new Set<string>();
      const collectDirs = (nodes: WorkspaceNode[], parent: string) => {
        for (const node of nodes) {
          if (node.type !== 'dir') {
            continue;
          }
          const key = nodeKey(node, parent);
          expanded.add(key);
          collectDirs(node.children ?? [], key);
        }
      };
      collectDirs(parsedTree, '');
      setExpandedNodes(expanded);
      if (!activeFileId) {
        const first = findFirstFilePath(parsedTree);
        if (first) {
          void openFileInTab(first);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load workspace tree.';
      setTreeError(message);
    } finally {
      setTreeLoading(false);
    }
  };

  const loadGitDiff = async (): Promise<void> => {
    if (!sessionId) {
      setGitDiff('');
      return;
    }
    setGitDiffLoading(true);
    try {
      const response = await fetch('/ui/api/workspace/git-diff', { headers: requestHeaders });
      const payload: unknown = await response.json();
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to load git diff.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      const diffRaw = (payload as { diff?: unknown }).diff;
      setGitDiff(typeof diffRaw === 'string' ? diffRaw : '');
    } catch {
      setGitDiff('');
    } finally {
      setGitDiffLoading(false);
    }
  };

  useEffect(() => {
    void loadWorkspaceRoot();
    void loadTree();
    void loadGitDiff();
  }, [sessionId]);

  const readFileContent = async (path: string): Promise<string | null> => {
    const response = await fetch(`/ui/api/workspace/file?path=${encodeURIComponent(path)}`, {
      headers: requestHeaders,
    });
    const payload: unknown = await response.json();
    if (response.status === 202) {
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] pending approval: read ${path}`,
      ]);
      return null;
    }
    if (!response.ok) {
      const errorInfo = extractApiError(payload, 'Failed to load file.');
      throw new Error(explainWorkspaceFailure(errorInfo));
    }
    const contentRaw = (payload as { content?: unknown }).content;
    return typeof contentRaw === 'string' ? contentRaw : '';
  };

  const openFileInTab = async (path: string): Promise<void> => {
    const normalizedPath = path.trim();
    if (!normalizedPath) {
      return;
    }
    const existing = openFiles.find((item) => item.path === normalizedPath);
    if (existing) {
      setActiveFileId(existing.id);
      return;
    }
    const tab: OpenFileTab = {
      id: `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      path: normalizedPath,
      name: normalizedPath.split('/').pop() || normalizedPath,
      content: '',
      savedContent: '',
      loading: true,
    };
    setOpenFiles((prev) => [...prev, tab]);
    setActiveFileId(tab.id);
    try {
      const content = await readFileContent(normalizedPath);
      if (content === null) {
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
                content,
                savedContent: content,
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
      const response = await fetch('/ui/api/workspace/file', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({ path: activeTab.path, content: activeTab.content }),
      });
      const payload: unknown = await response.json();
      if (response.status === 202) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: save ${activeTab.path}`,
        ]);
        return;
      }
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to save file.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      setOpenFiles((prev) =>
        prev.map((item) =>
          item.id === activeTab.id
            ? {
                ...item,
                savedContent: item.content,
              }
            : item,
        ),
      );
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] saved: ${activeTab.path}`]);
      void loadTree();
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
      const response = await fetch('/ui/api/workspace/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({ path: activeTab.path }),
      });
      const payload: unknown = await response.json();
      if (response.status === 202) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: run ${activeTab.path}`,
        ]);
        return;
      }
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to run file.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      const stdoutRaw = (payload as { stdout?: unknown }).stdout;
      const stderrRaw = (payload as { stderr?: unknown }).stderr;
      const exitCodeRaw = (payload as { exit_code?: unknown }).exit_code;
      const stdout = typeof stdoutRaw === 'string' ? stdoutRaw.trim() : '';
      const stderr = typeof stderrRaw === 'string' ? stderrRaw.trim() : '';
      const exitCode = typeof exitCodeRaw === 'number' ? exitCodeRaw : 0;
      setTerminalLines((prev) => {
        const next = [...prev];
        if (stdout) {
          next.push(stdout);
        }
        if (stderr) {
          next.push(`stderr: ${stderr}`);
        }
        next.push(`[${terminalTimestamp()}] exit=${exitCode}`);
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
    const ok = await onSendAgentMessage({ content: `/sh ${command}` });
    if (!ok) {
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] command failed`]);
    }
    setTerminalBusy(false);
  };

  const handleSelectRoot = async () => {
    const nextRoot = rootInput.trim();
    if (!nextRoot || rootBusy || !sessionId) {
      return;
    }
    setRootBusy(true);
    try {
      const response = await fetch('/ui/api/workspace/root/select', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({ root_path: nextRoot }),
      });
      const payload: unknown = await response.json();
      if (response.status === 202) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: switch workspace root`,
        ]);
        setRootPickerOpen(false);
        return;
      }
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to change workspace root.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      setWorkspaceRoot(nextRoot);
      setRootPickerOpen(false);
      setOpenFiles([]);
      setActiveFileId(null);
      void loadWorkspaceRoot();
      void loadTree();
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
      const response = await fetch('/ui/api/workspace/index', {
        method: 'POST',
        headers: requestHeaders,
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to index workspace.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      const body = payload as {
        indexed_code?: unknown;
        indexed_docs?: unknown;
        skipped?: unknown;
      };
      const indexedCode = typeof body.indexed_code === 'number' ? body.indexed_code : 0;
      const indexedDocs = typeof body.indexed_docs === 'number' ? body.indexed_docs : 0;
      const skipped = typeof body.skipped === 'number' ? body.skipped : 0;
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

  const buildContextAttachments = (): Array<{ name: string; mime: string; content: string }> => {
    const attachments: Array<{ name: string; mime: string; content: string }> = [];
    if (includeOpenTabs && openFiles.length > 0) {
      attachments.push({
        name: 'open-tabs.json',
        mime: 'application/json',
        content: JSON.stringify(
          {
            active_file: activeTab?.path ?? null,
            open_files: openFiles.map((item) => ({
              path: item.path,
              dirty: item.content !== item.savedContent,
            })),
          },
          null,
          2,
        ),
      });
    }
    if (includeSelection && selectionText.trim()) {
      attachments.push({
        name: 'selection.txt',
        mime: 'text/plain',
        content: selectionText,
      });
    }
    if (includeGitDiff && gitDiff.trim()) {
      attachments.push({
        name: 'git-diff.patch',
        mime: 'text/x-diff',
        content: gitDiff,
      });
    }
    if (includeTerminal && lastTerminalOutput.trim()) {
      attachments.push({
        name: 'terminal-last.txt',
        mime: 'text/plain',
        content: lastTerminalOutput,
      });
    }
    return attachments;
  };

  const handleAgentSend = async () => {
    const content = agentInput.trim();
    const attachments = buildContextAttachments();
    if ((!content && attachments.length === 0) || sending || isDecisionBlocking) {
      return;
    }
    const ok = await onSendAgentMessage({
      content,
      attachments: attachments.length > 0 ? attachments : undefined,
    });
    if (ok) {
      setAgentInput('');
    }
  };

  const renderNode = (node: WorkspaceNode, parent: string, depth: number): JSX.Element => {
    const key = nodeKey(node, parent);
    if (node.type === 'dir') {
      const expanded = expandedNodes.has(key);
      return (
        <div key={key}>
          <button
            onClick={() => {
              setExpandedNodes((prev) => {
                const next = new Set(prev);
                if (next.has(key)) {
                  next.delete(key);
                } else {
                  next.add(key);
                }
                return next;
              });
            }}
            className="flex w-full items-center gap-1.5 px-2 py-1.5 text-left text-[12px] text-[#a4a4ad] hover:bg-[#15151a]"
            style={{ paddingLeft: `${8 + depth * 14}px` }}
          >
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            {expanded ? <FolderOpen className="h-3.5 w-3.5 text-[#f59e0b]" /> : <Folder className="h-3.5 w-3.5 text-[#f59e0b]" />}
            <span className="truncate">{node.name}</span>
          </button>
          {expanded && (node.children ?? []).map((child) => renderNode(child, key, depth + 1))}
        </div>
      );
    }
    const path = node.path ?? node.name;
    const isActive = activeTab?.path === path;
    return (
      <button
        key={key}
        onClick={() => {
          void openFileInTab(path);
        }}
        className={`flex w-full items-center gap-1.5 px-2 py-1.5 text-left text-[12px] transition-colors ${
          isActive ? 'bg-[#1b1b22] text-[#d6d6de]' : 'text-[#9a9aa3] hover:bg-[#15151a]'
        }`}
        style={{ paddingLeft: `${28 + depth * 14}px` }}
      >
        {fileIcon(path)}
        <span className="truncate">{node.name}</span>
      </button>
    );
  };

  const handleEditorMount: OnMount = (editor) => {
    editorRef.current = editor;
    editor.onDidChangeCursorSelection(() => {
      const selected = editor.getModel()?.getValueInRange(editor.getSelection() ?? undefined) ?? '';
      setSelectionText(selected.trim());
    });
  };

  const aiContextChips = [
    {
      key: 'tabs',
      label: `Open tabs (${openFiles.length})`,
      enabled: includeOpenTabs,
      onToggle: () => setIncludeOpenTabs((prev) => !prev),
      hidden: openFiles.length === 0,
    },
    {
      key: 'selection',
      label: selectionText.trim() ? `Selection (${Math.min(selectionText.length, 120)} chars)` : 'Selection',
      enabled: includeSelection,
      onToggle: () => setIncludeSelection((prev) => !prev),
      hidden: !selectionText.trim(),
    },
    {
      key: 'diff',
      label: gitDiff.trim() ? 'Git diff' : gitDiffLoading ? 'Git diff (loading...)' : 'Git diff (empty)',
      enabled: includeGitDiff,
      onToggle: () => setIncludeGitDiff((prev) => !prev),
      hidden: false,
    },
    {
      key: 'terminal',
      label: lastTerminalOutput.trim() ? 'Last terminal output' : 'Terminal output',
      enabled: includeTerminal,
      onToggle: () => setIncludeTerminal((prev) => !prev),
      hidden: !lastTerminalOutput.trim(),
    },
  ].filter((item) => !item.hidden);

  const terminalPendingText = isDecisionBlocking
    ? 'Ожидает подтверждения решения. Отправка временно заблокирована.'
    : null;

  return (
    <div className="h-full min-h-0 flex flex-col bg-[#0a0a0d] text-[#d2d2d9]">
      <div className="h-12 border-b border-[#1f1f24] px-3 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="flex items-center gap-2">
          <button
            onClick={onBackToChat}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Chat
          </button>
          <span className="text-[11px] text-[#666]">{modelLabel}</span>
        </div>

        <div className="flex items-center justify-center gap-2">
          <button
            onClick={() => setRootPickerOpen((prev) => !prev)}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2.5 py-1 text-[12px] text-[#c7c7d0] hover:bg-[#181820]"
          >
            Open Workspace Root
          </button>
          <button
            onClick={() => {
              void handleReindex();
            }}
            disabled={indexing}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2.5 py-1 text-[12px] text-[#c7c7d0] hover:bg-[#181820] disabled:opacity-50"
          >
            {indexing ? 'Indexing...' : 'Re-index'}
          </button>
          {workspaceRoot ? (
            <span className="max-w-[380px] truncate rounded-md border border-[#2a2a31] bg-[#111117] px-2 py-1 text-[11px] text-[#8f8f99]" title={workspaceRoot}>
              {compactPath(workspaceRoot, 68)}
            </span>
          ) : null}
        </div>

        <div className="flex items-center justify-end gap-2">
          <span className={`text-[11px] ${workspaceYoloActive ? 'text-red-300' : 'text-[#8f8f98]'}`}>
            Policy: {workspacePolicy}
          </span>
          <button
            onClick={() => {
              void loadGitDiff();
            }}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
            title="Refresh git diff"
          >
            <RefreshCcw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onOpenSettings}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
          >
            <Settings className="h-3.5 w-3.5" />
            Settings
          </button>
        </div>
      </div>

      {rootPickerOpen ? (
        <div className="border-b border-[#1f1f24] px-3 py-2 flex items-center gap-2 bg-[#0d0d12]">
          <input
            value={rootInput}
            onChange={(event) => setRootInput(event.target.value)}
            placeholder="/path/to/workspace/root"
            className="flex-1 rounded-md border border-[#2a2a31] bg-[#111117] px-3 py-1.5 text-[12px] text-[#d0d0d8] outline-none"
          />
          <button
            onClick={() => {
              void handleSelectRoot();
            }}
            disabled={rootBusy || !rootInput.trim()}
            className="rounded-md border border-[#2a2a31] bg-[#15151b] px-3 py-1.5 text-[12px] text-[#d0d0d8] disabled:opacity-50"
          >
            {rootBusy ? 'Applying...' : 'Apply'}
          </button>
          <button
            onClick={() => setRootPickerOpen(false)}
            className="rounded-md border border-[#2a2a31] bg-[#111117] px-3 py-1.5 text-[12px] text-[#b3b3bc]"
          >
            Cancel
          </button>
        </div>
      ) : null}

      {statusMessage ? (
        <div className="border-b border-[#1f1f24] px-3 py-2 text-[12px] text-[#8d8d96]">{statusMessage}</div>
      ) : null}

      <div
        className="flex-1 min-h-0 grid"
        style={{
          gridTemplateColumns: `${explorerWidth}px 6px ${assistantWidth}px 6px minmax(420px,1fr)`,
        }}
      >
        <aside className="min-h-0 border-r border-[#1f1f24] bg-[#0d0d11] flex flex-col overflow-hidden">
          <div className="h-9 flex items-center px-3 border-b border-[#1f1f24] text-[11px] uppercase tracking-wider text-[#686873]">
            Explorer
          </div>
          <div className="flex-1 min-h-0 overflow-auto" data-scrollbar="always">
            {treeLoading ? (
              <div className="px-3 py-2 text-[12px] text-[#777]">Loading tree...</div>
            ) : treeError ? (
              <div className="px-3 py-2 text-[12px] text-red-400">{treeError}</div>
            ) : tree.length === 0 ? (
              <div className="px-3 py-2 text-[12px] text-[#777]">Workspace is empty.</div>
            ) : (
              tree.map((node) => renderNode(node, '', 0))
            )}
          </div>
        </aside>

        <button
          onMouseDown={() => setDraggingExplorer(true)}
          className="cursor-col-resize bg-[#121218] hover:bg-[#1b1b23]"
          aria-label="Resize explorer"
          title="Resize explorer"
        />

        <section className="min-h-0 border-r border-[#1f1f24] bg-[#0d0d11] flex flex-col overflow-hidden">
          <div className="h-9 border-b border-[#1f1f24] px-3 flex items-center gap-2 text-[12px] text-[#9a9aa3]">
            <Bot className="h-3.5 w-3.5 text-[#f59e0b]" />
            AI Assistant
          </div>

          <div className="border-b border-[#1f1f24] px-3 py-2 flex flex-wrap gap-2">
            {aiContextChips.length === 0 ? (
              <span className="text-[11px] text-[#666]">No context available.</span>
            ) : (
              aiContextChips.map((chip) => (
                <button
                  key={chip.key}
                  onClick={chip.onToggle}
                  className={`rounded-full border px-2 py-0.5 text-[11px] ${
                    chip.enabled
                      ? 'border-[#2f5dff] bg-[#1a2348] text-[#c8d7ff]'
                      : 'border-[#2a2a31] bg-[#111117] text-[#888893]'
                  }`}
                  title={chip.label}
                >
                  {chip.label}
                </button>
              ))
            )}
          </div>

          {decision && decision.status === 'pending' ? (
            <div className="border-b border-[#1f1f24]">
              <DecisionPanel
                decision={decision}
                busy={decisionBusy}
                errorMessage={decisionError}
                onRespond={onDecisionRespond}
              />
            </div>
          ) : null}

          <div className="flex-1 min-h-0 overflow-auto px-3 py-3 space-y-2" data-scrollbar="always">
            {recentMessages.length === 0 ? (
              <div className="text-[12px] text-[#777]">No messages yet.</div>
            ) : (
              recentMessages.map((message) => (
                <div
                  key={message.id}
                  className={`rounded-lg border px-3 py-2 text-[12px] whitespace-pre-wrap break-words ${
                    message.role === 'user'
                      ? 'bg-[#141418] border-[#252530] text-[#d0d0d8] ml-6'
                      : 'bg-[#101015] border-[#1f1f28] text-[#bfc2cb] mr-6'
                  }`}
                >
                  {message.content}
                </div>
              ))
            )}
          </div>

          <div className="border-t border-[#1f1f24] p-3 space-y-2">
            {terminalPendingText ? (
              <div className="text-[11px] text-amber-300">{terminalPendingText}</div>
            ) : null}
            <textarea
              value={agentInput}
              onChange={(event) => setAgentInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void handleAgentSend();
                }
              }}
              placeholder="Ask agent..."
              className="min-h-[56px] max-h-40 w-full resize-y rounded-md border border-[#252530] bg-[#111116] px-3 py-2 text-[12px] text-[#d4d4db] outline-none"
              disabled={isDecisionBlocking}
            />
            <button
              onClick={() => {
                void handleAgentSend();
              }}
              disabled={sending || isDecisionBlocking || (!agentInput.trim() && buildContextAttachments().length === 0)}
              className="w-full rounded-md border border-[#2a2a31] bg-[#141418] px-3 py-2 text-[12px] text-[#bcbcc6] disabled:opacity-50"
            >
              Send with Context
            </button>
          </div>
        </section>

        <button
          onMouseDown={() => setDraggingAssistant(true)}
          className="cursor-col-resize bg-[#121218] hover:bg-[#1b1b23]"
          aria-label="Resize assistant"
          title="Resize assistant"
        />

        <section className="min-h-0 flex flex-col overflow-hidden">
          <div className="h-9 border-b border-[#1f1f24] px-3 flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2 overflow-auto" data-scrollbar="auto">
              {openFiles.length === 0 ? (
                <span className="text-[12px] text-[#666]">No file selected</span>
              ) : (
                openFiles.map((tab) => {
                  const isActive = tab.id === activeFileId;
                  const dirty = tab.content !== tab.savedContent;
                  return (
                    <div
                      key={tab.id}
                      className={`group inline-flex max-w-[220px] items-center gap-2 rounded-md border px-2 py-1 text-[12px] ${
                        isActive
                          ? 'border-[#3a3a46] bg-[#1a1a22] text-[#d4d4dd]'
                          : 'border-[#252530] bg-[#121218] text-[#9999a4]'
                      }`}
                    >
                      <button
                        onClick={() => setActiveFileId(tab.id)}
                        className="truncate"
                        title={tab.path}
                      >
                        {tab.name}
                        {dirty ? ' *' : ''}
                      </button>
                      <button
                        onClick={() => closeTab(tab.id)}
                        className="opacity-70 hover:opacity-100"
                        title="Close tab"
                        aria-label="Close tab"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRunActiveFile}
                disabled={!activeTab || terminalBusy || isDecisionBlocking}
                className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                Run
              </button>
              <button
                onClick={handleSave}
                disabled={!hasUnsavedChanges || editorSaving || isDecisionBlocking}
                className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
              >
                <Save className="h-3.5 w-3.5" />
                {editorSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>

          <div className="flex-1 min-h-0 bg-[#0b0b0f]">
            {activeTab ? (
              <Editor
                theme="vs-dark"
                language={monacoLanguageFromPath(activeTab.path)}
                value={activeTab.content}
                onChange={(value) => updateActiveContent(value ?? '')}
                onMount={handleEditorMount}
                options={{
                  minimap: { enabled: false },
                  fontSize: 13,
                  lineHeight: 22,
                  automaticLayout: true,
                  wordWrap: 'off',
                  renderLineHighlight: 'all',
                  scrollBeyondLastLine: false,
                }}
              />
            ) : (
              <div className="h-full w-full p-4 text-[12px] text-[#70707b]">Select a file from Explorer.</div>
            )}
          </div>

          <button
            onMouseDown={() => setDraggingTerminal(true)}
            className="h-1.5 cursor-row-resize bg-[#121218] hover:bg-[#1b1b23]"
            aria-label="Resize terminal"
            title="Resize terminal"
          />

          <div className="border-t border-[#1f1f24] bg-[#09090c] flex flex-col" style={{ height: `${terminalHeight}px` }}>
            <div className="h-8 border-b border-[#1f1f24] px-3 flex items-center gap-2 text-[12px] text-[#8f8f98]">
              <Terminal className="h-3.5 w-3.5" />
              Terminal
            </div>
            <div className="flex-1 min-h-0 overflow-auto px-3 py-2 font-mono text-[12px] text-[#c4c4cd]" data-scrollbar="always">
              {terminalLines.map((line, index) => (
                <div key={`${line}-${index}`} className="whitespace-pre-wrap break-words">
                  {line}
                </div>
              ))}
              <div ref={terminalEndRef} />
            </div>
            <div className="h-9 border-t border-[#1f1f24] px-3 flex items-center gap-2">
              <span className="font-mono text-[12px] text-[#777]">$</span>
              <input
                value={terminalInput}
                onChange={(event) => setTerminalInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    void handleTerminalSubmit();
                  }
                }}
                placeholder="Type shell command"
                className="flex-1 bg-transparent border-0 outline-none text-[12px] text-[#d0d0d8]"
                disabled={!sessionId || terminalBusy || isDecisionBlocking}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
