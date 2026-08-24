import { useEffect, useRef, useState } from 'react';
import { FolderGit2, RefreshCw } from 'lucide-react';
import {
  fetchWorkspaceGitStatus,
  postWorkspaceGitStage,
  postWorkspaceGitUnstage,
  postWorkspaceGitCommit,
  type GitStatusResult,
  type GitFileEntry,
} from './workspace-api';

type GitPanelProps = {
  sessionId: string | null;
  sessionHeader: string;
  requestHeaders: Record<string, string>;
  workspaceRoot: string;
  refreshToken: number;
  decisionOutcome?: {
    kind: 'success' | 'rejected' | 'failed';
    message: string;
  } | null;
};

export function GitPanel({
  sessionId,
  sessionHeader: _sessionHeader,
  requestHeaders,
  workspaceRoot,
  refreshToken,
  decisionOutcome = null,
}: GitPanelProps) {
  const [status, setStatus] = useState<GitStatusResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [commitMessage, setCommitMessage] = useState('');
  const [actionBusy, setActionBusy] = useState(false);
  const [pendingApproval, setPendingApproval] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [outcomeKind, setOutcomeKind] = useState<'success' | 'rejected' | 'failed' | null>(null);

  const statusGenerationRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortControllerRef.current?.abort();
    };
  }, []);

  const loadStatus = async () => {
    if (!sessionId || !mountedRef.current) return;
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const generation = statusGenerationRef.current + 1;
    statusGenerationRef.current = generation;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchWorkspaceGitStatus(requestHeaders, controller.signal);
      if (!mountedRef.current || statusGenerationRef.current !== generation) {
        return;
      }
      if (!result.ok && result.error) {
        setError(result.error);
        setStatus(null);
      } else {
        setStatus(result);
      }
    } catch (e) {
      if (!mountedRef.current || statusGenerationRef.current !== generation) {
        return;
      }
      if (e instanceof DOMException && e.name === 'AbortError') {
        return;
      }
      const msg = e instanceof Error ? e.message : 'Failed to load git status.';
      setError(msg);
      setStatus(null);
    } finally {
      if (mountedRef.current && statusGenerationRef.current === generation) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    setPendingApproval(false);
    setActionMessage(null);
    setOutcomeKind(null);
    void loadStatus();
  }, [sessionId, workspaceRoot, refreshToken]);

  useEffect(() => {
    if (!decisionOutcome) {
      return;
    }
    setPendingApproval(false);
    setActionMessage(decisionOutcome.message);
    setOutcomeKind(decisionOutcome.kind);
    if (decisionOutcome.kind === 'success') {
      void loadStatus();
    }
  }, [decisionOutcome]);

  const handleStage = async (paths: string[] | null, all: boolean) => {
    setActionBusy(true);
    setActionMessage(null);
    setPendingApproval(false);
    setOutcomeKind(null);
    try {
      const result = await postWorkspaceGitStage(paths, all, requestHeaders);
      if (result.pendingApproval) {
        setPendingApproval(true);
        setActionMessage('Stage ожидает подтверждения в DecisionPanel.');
        return;
      }
      setActionMessage(result.ok ? result.message : `Stage failed: ${result.message}`);
      if (result.ok) void loadStatus();
    } catch (e) {
      setActionMessage(e instanceof Error ? e.message : 'Stage failed.');
    } finally {
      setActionBusy(false);
    }
  };

  const handleUnstage = async (paths: string[] | null, all: boolean) => {
    setActionBusy(true);
    setActionMessage(null);
    setPendingApproval(false);
    setOutcomeKind(null);
    try {
      const result = await postWorkspaceGitUnstage(paths, all, requestHeaders);
      if (result.pendingApproval) {
        setPendingApproval(true);
        setActionMessage('Unstage ожидает подтверждения в DecisionPanel.');
        return;
      }
      setActionMessage(result.ok ? result.message : `Unstage failed: ${result.message}`);
      if (result.ok) void loadStatus();
    } catch (e) {
      setActionMessage(e instanceof Error ? e.message : 'Unstage failed.');
    } finally {
      setActionBusy(false);
    }
  };

  const handleCommit = async () => {
    const msg = commitMessage.trim();
    if (!msg || actionBusy) return;
    setActionBusy(true);
    setActionMessage(null);
    setPendingApproval(false);
    setOutcomeKind(null);
    try {
      const result = await postWorkspaceGitCommit(msg, requestHeaders);
      if (result.pendingApproval) {
        setPendingApproval(true);
        setActionMessage('Commit ожидает подтверждения в DecisionPanel.');
        return;
      }
      if (result.ok) {
        setCommitMessage('');
        setActionMessage(result.message);
        void loadStatus();
      } else {
        setActionMessage(`Commit failed: ${result.message}`);
      }
    } catch (e) {
      setActionMessage(e instanceof Error ? e.message : 'Commit failed.');
    } finally {
      setActionBusy(false);
    }
  };

  const fileEntry = (
    entries: GitFileEntry[],
    label: string,
    color: string,
    renderActions?: (entry: GitFileEntry) => React.ReactNode,
  ) =>
    entries.length > 0 ? (
      <div className="mb-2">
        <div className={`text-[11px] font-medium ${color} mb-1`}>
          {label} ({entries.length})
        </div>
        {entries.map((entry) => (
          <div
            key={entry.path}
            className="flex items-center justify-between gap-2 text-[11px] text-zinc-400 py-0.5"
          >
            <span className="truncate font-mono" title={entry.path}>
              {entry.path}
            </span>
            <span className="flex items-center gap-1 shrink-0">
              <span className="text-zinc-500">{entry.status}</span>
              {renderActions ? renderActions(entry) : null}
            </span>
          </div>
        ))}
      </div>
    ) : null;

  const hasChanges =
    status &&
    (status.staged.length > 0 ||
      status.unstaged.length > 0 ||
      status.untracked.length > 0 ||
      status.conflicted.length > 0);

  const allUnstagedPaths = [
    ...(status?.unstaged ?? []),
    ...(status?.untracked ?? []),
  ];

  const fileActionButton = (
    label: string,
    onClick: () => void,
    disabled: boolean,
  ) => (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-50"
    >
      {label}
    </button>
  );

  return (
    <div className="h-full overflow-auto px-3 py-3" data-computer-tab-content="git-panel">
      {!sessionId ? (
        <div className="text-[12px] text-zinc-500">No active session.</div>
      ) : loading ? (
        <div className="text-[12px] text-zinc-500">Loading git status...</div>
      ) : error ? (
        <div className="text-[12px] text-red-400">{error}</div>
      ) : !status ? null : (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FolderGit2 className="h-4 w-4 text-zinc-400" />
              <span className="text-[12px] font-medium text-zinc-300">
                {status.branch || 'detached'}
              </span>
              {status.upstream ? (
                <span className="text-[11px] text-zinc-500">
                  {status.upstream}
                  {status.ahead > 0 ? ` ↑${status.ahead}` : ''}
                  {status.behind > 0 ? ` ↓${status.behind}` : ''}
                </span>
              ) : null}
            </div>
            <button
              onClick={() => void loadStatus()}
              disabled={loading}
              className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-900 disabled:opacity-50"
            >
              <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>

          {!status.ok ? (
            <div className="text-[12px] text-red-400">
              {status.error || 'Git error'}
            </div>
          ) : !hasChanges ? (
            <div className="text-[12px] text-zinc-500">Clean — no changes.</div>
          ) : (
            <>
              {fileEntry(
                status.staged,
                'Staged',
                'text-emerald-300',
                (entry) =>
                  fileActionButton(
                    'Unstage',
                    () => {
                      void handleUnstage([entry.path], false);
                    },
                    actionBusy,
                  ),
              )}
              {fileEntry(
                status.unstaged,
                'Unstaged',
                'text-amber-300',
                (entry) =>
                  fileActionButton(
                    'Stage',
                    () => {
                      void handleStage([entry.path], false);
                    },
                    actionBusy,
                  ),
              )}
              {fileEntry(
                status.untracked,
                'Untracked',
                'text-zinc-400',
                (entry) =>
                  fileActionButton(
                    'Stage',
                    () => {
                      void handleStage([entry.path], false);
                    },
                    actionBusy,
                  ),
              )}
              {fileEntry(status.conflicted, 'Conflicted', 'text-red-400')}

              <div className="flex flex-wrap items-center gap-1.5 border-t border-zinc-800 pt-2">
                {status.unstaged.length > 0 || status.untracked.length > 0 ? (
                  <button
                    onClick={() => {
                      void handleStage(allUnstagedPaths.map((e) => e.path), false);
                    }}
                    disabled={actionBusy}
                    className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-900 disabled:opacity-50"
                  >
                    Stage all
                  </button>
                ) : null}
                {status.staged.length > 0 ? (
                  <button
                    onClick={() => {
                      void handleUnstage(null, true);
                    }}
                    disabled={actionBusy}
                    className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-900 disabled:opacity-50"
                  >
                    Unstage all
                  </button>
                ) : null}
              </div>

              <div className="border-t border-zinc-800 pt-2 space-y-1.5">
                <input
                  value={commitMessage}
                  onChange={(event) => setCommitMessage(event.target.value)}
                  placeholder="Commit message..."
                  className="w-full rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-[12px] text-zinc-300 outline-none placeholder:text-zinc-500"
                  disabled={actionBusy}
                />
                <button
                  onClick={() => {
                    void handleCommit();
                  }}
                  disabled={actionBusy || !commitMessage.trim() || status.staged.length === 0}
                  className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1 text-[11px] text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                >
                  {actionBusy ? 'Committing...' : 'Commit'}
                </button>
              </div>
            </>
          )}

          {pendingApproval ? (
            <div className="rounded-md border border-amber-800/60 bg-amber-950/30 px-3 py-2 text-[11px] text-amber-200">
              {actionMessage}
            </div>
          ) : outcomeKind === 'failed' ? (
            <div className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-[11px] text-red-300">
              {actionMessage}
            </div>
          ) : outcomeKind === 'rejected' ? (
            <div className="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-[11px] text-zinc-400">
              {actionMessage}
            </div>
          ) : actionMessage ? (
            <div className="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-[11px] text-zinc-400">
              {actionMessage}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
