import { useEffect, useMemo, useRef, useState } from 'react';
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
  Save,
  Send,
  Settings,
  Terminal,
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
      const path = typeof candidate.path === 'string' && candidate.path.trim()
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
  if (normalized.endsWith('.py') || normalized.endsWith('.ts') || normalized.endsWith('.tsx') || normalized.endsWith('.js') || normalized.endsWith('.jsx')) {
    return <FileCode2 className="h-3.5 w-3.5 text-[#6f9cff]" />;
  }
  return <FileText className="h-3.5 w-3.5 text-[#8f8f97]" />;
};

const terminalTimestamp = (): string => new Date().toLocaleTimeString();

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

  const [activeFilePath, setActiveFilePath] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState('');
  const [savedContent, setSavedContent] = useState('');
  const [fileLoading, setFileLoading] = useState(false);
  const [fileSaving, setFileSaving] = useState(false);

  const [terminalLines, setTerminalLines] = useState<string[]>([
    `[${terminalTimestamp()}] Workspace terminal ready.`,
  ]);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalBusy, setTerminalBusy] = useState(false);

  const [agentInput, setAgentInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement | null>(null);
  const assistantSeenRef = useRef<Set<string>>(new Set());
  const assistantInitRef = useRef(false);
  const decisionSeenRef = useRef<Set<string>>(new Set());
  const statusSeenRef = useRef<string | null>(null);

  const hasUnsavedChanges = activeFilePath !== null && editorContent !== savedContent;
  const isDecisionBlocking = decision?.status === 'pending' && decision.blocking === true;

  const requestHeaders = useMemo(() => {
    if (!sessionId) {
      return {} as Record<string, string>;
    }
    return { [sessionHeader]: sessionId };
  }, [sessionHeader, sessionId]);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLines]);

  useEffect(() => {
    assistantInitRef.current = false;
    assistantSeenRef.current = new Set();
    decisionSeenRef.current = new Set();
    statusSeenRef.current = null;
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

  const loadTree = async (): Promise<void> => {
    if (!sessionId) {
      setTree([]);
      setActiveFilePath(null);
      setTreeError('No active session. Create chat first.');
      return;
    }
    setTreeLoading(true);
    setTreeError(null);
    try {
      const response = await fetch('/ui/api/workspace/tree', {
        headers: requestHeaders,
      });
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
      if (!activeFilePath) {
        const first = findFirstFilePath(parsedTree);
        if (first) {
          setActiveFilePath(first);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load workspace tree.';
      setTreeError(message);
    } finally {
      setTreeLoading(false);
    }
  };

  useEffect(() => {
    void loadTree();
  }, [sessionId]);

  useEffect(() => {
    if (!activeFilePath || !sessionId) {
      setEditorContent('');
      setSavedContent('');
      return;
    }
    const loadFile = async () => {
      setFileLoading(true);
      try {
        const response = await fetch(
          `/ui/api/workspace/file?path=${encodeURIComponent(activeFilePath)}`,
          { headers: requestHeaders },
        );
        const payload: unknown = await response.json();
        if (response.status === 202) {
          setTerminalLines((prev) => [
            ...prev,
            `[${terminalTimestamp()}] pending approval: read ${activeFilePath}`,
          ]);
          return;
        }
        if (!response.ok) {
          const errorInfo = extractApiError(payload, 'Failed to load file.');
          throw new Error(explainWorkspaceFailure(errorInfo));
        }
        const content = (payload as { content?: unknown }).content;
        const normalized = typeof content === 'string' ? content : '';
        setEditorContent(normalized);
        setSavedContent(normalized);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to load file.';
        setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
      } finally {
        setFileLoading(false);
      }
    };
    void loadFile();
  }, [activeFilePath, requestHeaders, sessionId]);

  const handleSave = async () => {
    if (!activeFilePath || !sessionId || fileSaving || isDecisionBlocking) {
      return;
    }
    setFileSaving(true);
    try {
      const response = await fetch('/ui/api/workspace/file', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({ path: activeFilePath, content: editorContent }),
      });
      const payload: unknown = await response.json();
      if (response.status === 202) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: save ${activeFilePath}`,
        ]);
        return;
      }
      if (!response.ok) {
        const errorInfo = extractApiError(payload, 'Failed to save file.');
        throw new Error(explainWorkspaceFailure(errorInfo));
      }
      setSavedContent(editorContent);
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] saved: ${activeFilePath}`]);
      void loadTree();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save file.';
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setFileSaving(false);
    }
  };

  const handleRunActiveFile = async () => {
    if (!activeFilePath || !sessionId || terminalBusy || isDecisionBlocking) {
      return;
    }
    setTerminalBusy(true);
    setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] run: ${activeFilePath}`]);
    try {
      const response = await fetch('/ui/api/workspace/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({ path: activeFilePath }),
      });
      const payload: unknown = await response.json();
      if (response.status === 202) {
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: run ${activeFilePath}`,
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

  const handleAgentSend = async () => {
    const content = agentInput.trim();
    if (!content || sending || isDecisionBlocking) {
      return;
    }
    const ok = await onSendAgentMessage({ content });
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
    const isActive = activeFilePath === node.path;
    const path = node.path ?? node.name;
    return (
      <button
        key={key}
        onClick={() => setActiveFilePath(path)}
        className={`flex w-full items-center gap-1.5 px-2 py-1.5 text-left text-[12px] transition-colors ${
          isActive
            ? 'bg-[#1b1b22] text-[#d6d6de]'
            : 'text-[#9a9aa3] hover:bg-[#15151a]'
        }`}
        style={{ paddingLeft: `${28 + depth * 14}px` }}
      >
        {fileIcon(path)}
        <span className="truncate">{node.name}</span>
      </button>
    );
  };

  const recentMessages = useMemo(() => messages.slice(-24), [messages]);

  return (
    <div className="h-full min-h-0 flex flex-col bg-[#0a0a0d] text-[#d2d2d9]">
      <div className="h-11 border-b border-[#1f1f24] px-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={onBackToChat}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Chat
          </button>
          <span className="text-[12px] text-[#888893]">Workspace</span>
          <span className="text-[11px] text-[#666]">{modelLabel}</span>
        </div>
        <button
          onClick={onOpenSettings}
          className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
        >
          <Settings className="h-3.5 w-3.5" />
          Settings
        </button>
      </div>

      {statusMessage ? (
        <div className="border-b border-[#1f1f24] px-3 py-2 text-[12px] text-[#8d8d96]">{statusMessage}</div>
      ) : null}

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

      <div className="flex-1 min-h-0 flex">
        <aside className="w-[260px] min-w-[220px] border-r border-[#1f1f24] flex flex-col bg-[#0d0d11]">
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

        <section className="flex-1 min-w-0 min-h-0 flex flex-col">
          <div className="h-9 border-b border-[#1f1f24] px-3 flex items-center justify-between">
            <span className="text-[12px] text-[#9a9aa3] truncate">{activeFilePath ?? 'No file selected'}</span>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRunActiveFile}
                disabled={!activeFilePath || terminalBusy || isDecisionBlocking}
                className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                Run
              </button>
              <button
                onClick={handleSave}
                disabled={!hasUnsavedChanges || fileSaving || isDecisionBlocking}
                className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
              >
                <Save className="h-3.5 w-3.5" />
                {fileSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>

          <div className="flex-1 min-h-0 bg-[#0b0b0f]">
            <textarea
              value={editorContent}
              onChange={(event) => setEditorContent(event.target.value)}
              spellCheck={false}
              placeholder={fileLoading ? 'Loading file...' : 'Select file from Explorer'}
              className="h-full w-full resize-none border-0 bg-transparent p-4 font-mono text-[13px] leading-6 text-[#d6d6de] outline-none"
              disabled={!activeFilePath || fileLoading || isDecisionBlocking}
            />
          </div>

          <div className="h-[220px] border-t border-[#1f1f24] bg-[#09090c] flex flex-col">
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

        <aside className="w-[380px] min-w-[320px] border-l border-[#1f1f24] bg-[#0d0d11] flex flex-col">
          <div className="h-9 border-b border-[#1f1f24] px-3 flex items-center gap-2 text-[12px] text-[#9a9aa3]">
            <Bot className="h-3.5 w-3.5 text-[#f59e0b]" />
            AI Assistant
          </div>
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
          <div className="border-t border-[#1f1f24] p-3 flex items-end gap-2">
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
              className="min-h-[42px] max-h-40 flex-1 resize-none rounded-md border border-[#252530] bg-[#111116] px-3 py-2 text-[12px] text-[#d4d4db] outline-none"
              disabled={isDecisionBlocking}
            />
            <button
              onClick={() => {
                void handleAgentSend();
              }}
              disabled={sending || !agentInput.trim() || isDecisionBlocking}
              className="h-10 w-10 rounded-md border border-[#2a2a31] bg-[#141418] text-[#bcbcc6] flex items-center justify-center disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}
