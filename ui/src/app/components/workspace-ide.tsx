import { useEffect, useMemo, useRef, useState } from 'react';
import type { OnMount } from '@monaco-editor/react';
import type * as Monaco from 'monaco-editor';

import type { UiDecision } from '../types';
import type { PlanEnvelope, SessionMode, TaskExecutionState } from '../types';
import type { CanvasMessage, CanvasSendPayload } from './canvas';
import {
  findFirstFilePath,
  nodeKey,
  policyLabel,
  terminalTimestamp,
  type WorkspaceNode,
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
import { WorkspaceToolbar } from '../../features/workspace/workspace-toolbar';
import {
  fetchWorkspaceFile,
  fetchWorkspaceGitDiff,
  fetchWorkspaceRoot,
  fetchWorkspaceTree,
  postWorkspaceIndex,
  postWorkspaceRootSelect,
  postWorkspaceRun,
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
    choice: 'approve' | 'reject' | 'edit',
    editedAction?: Record<string, unknown> | null,
  ) => Promise<void> | void;
  refreshToken?: number;
  workspaceIndexProgress?: {
    total: number;
    processed: number;
    indexedCode: number;
    indexedDocs: number;
    skipped: number;
    done: boolean;
    rootPath: string;
  } | null;
};

const MIN_EXPLORER_WIDTH = 240;
const MAX_EXPLORER_WIDTH = 420;
const MIN_ASSISTANT_WIDTH = 340;
const MAX_ASSISTANT_WIDTH = 520;
const MIN_TERMINAL_HEIGHT = 140;
const MAX_TERMINAL_HEIGHT = 420;

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
  workspaceIndexProgress = null,
}: WorkspaceIdeProps) {
  const [tree, setTree] = useState<WorkspaceNode[]>([]);
  const [treeLoading, setTreeLoading] = useState(false);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

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
  const [rootPickerOpen, setRootPickerOpen] = useState(false);
  const [rootInput, setRootInput] = useState('');
  const [rootBusy, setRootBusy] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [indexModalOpen, setIndexModalOpen] = useState(false);
  const [indexModalStatus, setIndexModalStatus] = useState<'confirm' | 'running' | 'done' | 'error'>('confirm');
  const [indexModalError, setIndexModalError] = useState<string | null>(null);
  const [indexModalProgress, setIndexModalProgress] = useState<{
    total: number;
    processed: number;
    indexedCode: number;
    indexedDocs: number;
    skipped: number;
    done: boolean;
    rootPath: string;
  } | null>(null);
  const [gitDiffLoading, setGitDiffLoading] = useState(false);
  const [gitDiff, setGitDiff] = useState('');

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
    setIndexModalOpen(false);
    setIndexModalStatus('confirm');
    setIndexModalError(null);
    setIndexModalProgress(null);
  }, [sessionId]);

  useEffect(() => {
    if (!workspaceIndexProgress) {
      return;
    }
    setIndexModalProgress(workspaceIndexProgress);
    if (indexModalStatus === 'running' && workspaceIndexProgress.done) {
      setIndexModalStatus('done');
    }
  }, [indexModalStatus, workspaceIndexProgress]);

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
      const { rootPath, policy } = await fetchWorkspaceRoot(requestHeaders);
      setWorkspaceRoot(rootPath);
      setRootInput(rootPath);
      if (policy) {
        setWorkspacePolicy(policyLabel(policy.profile));
        setWorkspaceYoloActive(policy.yolo_armed === true);
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
      const { pendingApproval, tree: parsedTree } = await fetchWorkspaceTree(requestHeaders);
      if (pendingApproval) {
        setTreeError('Ожидает подтверждения действия для доступа к workspace.');
        setTerminalLines((prev) => [
          ...prev,
          `[${terminalTimestamp()}] pending approval: workspace tree request`,
        ]);
        return;
      }
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
      setGitDiff(await fetchWorkspaceGitDiff(requestHeaders));
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
  }, [refreshToken, sessionId]);

  const readFileContent = async (path: string): Promise<string | null> => {
    const result = await fetchWorkspaceFile(path, requestHeaders);
    if (result.pendingApproval) {
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] pending approval: read ${path}`,
      ]);
      return null;
    }
    return result.content;
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
    const tab: WorkspaceOpenFileTab = {
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
      const result = await putWorkspaceFile(activeTab.path, activeTab.content, requestHeaders);
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
    if (!sessionId || indexing || isDecisionBlocking) {
      return;
    }
    setIndexModalOpen(true);
    setIndexModalStatus('confirm');
    setIndexModalError(null);
    setIndexModalProgress(null);
  };

  const handleStartReindex = async () => {
    if (!sessionId || indexing) {
      return;
    }
    setIndexModalStatus('running');
    setIndexModalError(null);
    if (
      !indexModalProgress
      || indexModalProgress.rootPath !== workspaceRoot
    ) {
      setIndexModalProgress({
        total: 0,
        processed: 0,
        indexedCode: 0,
        indexedDocs: 0,
        skipped: 0,
        done: false,
        rootPath: workspaceRoot,
      });
    }
    setIndexing(true);
    try {
      const { total, processed, indexedCode, indexedDocs, skipped } = await postWorkspaceIndex(
        requestHeaders,
      );
      setIndexModalProgress({
        total,
        processed,
        indexedCode,
        indexedDocs,
        skipped,
        done: true,
        rootPath: workspaceRoot,
      });
      setIndexModalStatus('done');
      setTerminalLines((prev) => [
        ...prev,
        `[${terminalTimestamp()}] index complete: ${processed}/${total} code=${indexedCode} docs=${indexedDocs} skipped=${skipped}`,
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to index workspace.';
      setIndexModalStatus('error');
      setIndexModalError(message);
      setTerminalLines((prev) => [...prev, `[${terminalTimestamp()}] error: ${message}`]);
    } finally {
      setIndexing(false);
    }
  };

  const buildContextAttachments = () =>
    buildWorkspaceContextAttachments({
      includeOpenTabs,
      includeSelection,
      includeGitDiff,
      includeTerminal,
      openFiles,
      activeFilePath: activeTab?.path ?? null,
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
  const indexPercent = useMemo(() => {
    if (!indexModalProgress || indexModalProgress.total <= 0) {
      return 0;
    }
    return Math.min(
      100,
      Math.round((indexModalProgress.processed / indexModalProgress.total) * 100),
    );
  }, [indexModalProgress]);

  return (
    <div className="h-full min-h-0 flex flex-col bg-[#0a0a0d] text-[#d2d2d9]">
      <WorkspaceToolbar
        modelLabel={modelLabel}
        indexing={indexing}
        workspaceRoot={workspaceRoot}
        workspacePolicy={workspacePolicy}
        workspaceYoloActive={workspaceYoloActive}
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
        onRootInputChange={setRootInput}
        onApplyRoot={() => {
          void handleSelectRoot();
        }}
        onCancelRootPicker={() => setRootPickerOpen(false)}
      />

      <div
        className="flex-1 min-h-0 grid"
        style={{
          gridTemplateColumns: `${explorerWidth}px 6px ${assistantWidth}px 6px minmax(420px,1fr)`,
        }}
      >
        <WorkspaceExplorer
          tree={tree}
          treeLoading={treeLoading}
          treeError={treeError}
          expandedNodes={expandedNodes}
          activePath={activeTab?.path ?? null}
          onToggleNode={(key) => {
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
          onOpenFile={(path) => {
            void openFileInTab(path);
          }}
        />

        <button
          onMouseDown={() => setDraggingExplorer(true)}
          className="cursor-col-resize bg-[#121218] hover:bg-[#1b1b23]"
          aria-label="Resize explorer"
          title="Resize explorer"
        />

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
      {indexModalOpen ? (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-xl rounded-xl border border-[#2a2a31] bg-[#111117] p-5 shadow-2xl">
            <h3 className="text-sm font-semibold text-[#ececf3]">Индексация может занять время</h3>
            <p className="mt-2 text-xs text-[#a8a8b4]">
              Текущий root: <span className="text-[#d2d2db]">{workspaceRoot || '—'}</span>
            </p>
            {indexModalProgress ? (
              <div className="mt-4 space-y-2 text-xs text-[#c8c8d2]">
                <div className="flex items-center justify-between">
                  <span>
                    {indexModalProgress.processed} / {indexModalProgress.total}
                  </span>
                  <span>{indexPercent}%</span>
                </div>
                <div className="h-2 w-full rounded bg-[#1b1b22]">
                  <div
                    className="h-2 rounded bg-[#3b82f6]"
                    style={{ width: `${indexPercent}%` }}
                  />
                </div>
                <div>
                  code={indexModalProgress.indexedCode} docs={indexModalProgress.indexedDocs} skipped=
                  {indexModalProgress.skipped}
                </div>
              </div>
            ) : null}
            {indexModalError ? (
              <div className="mt-3 rounded border border-red-900/70 bg-red-950/30 px-3 py-2 text-xs text-red-200">
                {indexModalError}
              </div>
            ) : null}
            {indexModalStatus === 'done' ? (
              <div className="mt-3 rounded border border-emerald-900/70 bg-emerald-950/30 px-3 py-2 text-xs text-emerald-200">
                Индексация завершена.
              </div>
            ) : null}
            <div className="mt-5 flex items-center justify-end gap-2">
              <button
                onClick={() => setIndexModalOpen(false)}
                disabled={indexModalStatus === 'running'}
                className="rounded-md border border-[#2a2a31] bg-[#16161d] px-3 py-1.5 text-xs text-[#c5c5cf] disabled:opacity-50"
              >
                {indexModalStatus === 'confirm' ? 'Cancel' : 'Close'}
              </button>
              {indexModalStatus === 'confirm' ? (
                <button
                  onClick={() => {
                    void handleStartReindex();
                  }}
                  className="rounded-md border border-[#2a2a31] bg-[#1f2536] px-3 py-1.5 text-xs text-[#d9e5ff]"
                >
                  Start
                </button>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
