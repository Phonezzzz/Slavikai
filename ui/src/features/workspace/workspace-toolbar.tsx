import { ArrowLeft, FolderGit2, RefreshCcw, SlidersHorizontal } from 'lucide-react';

import { compactPath } from './workspace-helpers';

type WorkspaceToolbarProps = {
  modelLabel: string;
  indexing: boolean;
  workspaceRoot: string;
  sessionPolicyLabel: string;
  sessionYoloActive: boolean;
  sessionSafeMode: boolean;
  rootPickerOpen: boolean;
  rootInput: string;
  rootBusy: boolean;
  statusMessage: string | null | undefined;
  onBackToChat: () => void;
  onOpenSessionDrawer: () => void;
  onToggleRootPicker: () => void;
  onReindex: () => void;
  onRefreshGitDiff: () => void;
  onOpenRepositoryPanel: () => void;
  onOpenQuickOpen: () => void;
  onRootInputChange: (value: string) => void;
  onApplyRoot: () => void;
  onCancelRootPicker: () => void;
};

export function WorkspaceToolbar({
  modelLabel,
  indexing,
  workspaceRoot,
  sessionPolicyLabel,
  sessionYoloActive,
  sessionSafeMode,
  rootPickerOpen,
  rootInput,
  rootBusy,
  statusMessage,
  onBackToChat,
  onOpenSessionDrawer,
  onToggleRootPicker,
  onReindex,
  onRefreshGitDiff,
  onOpenRepositoryPanel,
  onOpenQuickOpen,
  onRootInputChange,
  onApplyRoot,
  onCancelRootPicker,
}: WorkspaceToolbarProps) {
  const policyBadge =
    sessionPolicyLabel === 'YOLO'
      ? `Session policy: YOLO (${sessionYoloActive ? 'armed' : 'disarmed'})`
      : `Session policy: ${sessionPolicyLabel}`;
  const policyClass =
    sessionPolicyLabel === 'YOLO'
      ? sessionYoloActive
        ? 'text-red-300'
        : 'text-amber-300'
      : 'text-[#8f8f98]';
  return (
    <>
      <div className="h-12 border-b border-[#1f1f24] px-3 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="flex items-center gap-2">
          <button
            onClick={onBackToChat}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Chat
          </button>
          <button
            onClick={onOpenSessionDrawer}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Session
          </button>
          <span className="text-[11px] text-[#666]">{modelLabel}</span>
        </div>

        <div className="flex items-center justify-center gap-2">
          <button
            onClick={onToggleRootPicker}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2.5 py-1 text-[12px] text-[#c7c7d0] hover:bg-[#181820]"
          >
            Open Workspace Root
          </button>
          <button
            onClick={onReindex}
            disabled={indexing}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2.5 py-1 text-[12px] text-[#c7c7d0] hover:bg-[#181820] disabled:opacity-50"
          >
            {indexing ? 'Indexing...' : 'Re-index'}
          </button>
          {workspaceRoot ? (
            <button
              onClick={onOpenQuickOpen}
              className="inline-flex max-w-[420px] items-center gap-1 rounded-md border border-[#2a2a31] bg-[#111117] px-2 py-1 text-left text-[11px] text-[#8f8f99] hover:bg-[#171720]"
              title="Quick Open (Ctrl+Space, Ctrl+D)"
              aria-label="Quick Open (Ctrl+Space, Ctrl+D)"
            >
              <span className="truncate">{compactPath(workspaceRoot, 60)}</span>
              <span className="rounded border border-[#32405d] bg-[#182137] px-1 py-0.5 text-[10px] text-[#9ec0ff]">
                Ctrl+Space
              </span>
              <span className="rounded border border-[#2d2d39] bg-[#171721] px-1 py-0.5 text-[10px] text-[#9a9aa8]">
                Ctrl+D
              </span>
            </button>
          ) : null}
        </div>

        <div className="flex items-center justify-end gap-2">
          <span className={`text-[11px] ${policyClass}`}>
            {policyBadge}
          </span>
          <span className={`text-[11px] ${sessionSafeMode ? 'text-amber-300' : 'text-emerald-300'}`}>
            Session safe mode: {sessionSafeMode ? 'ON' : 'OFF'}
          </span>
          <button
            onClick={onRefreshGitDiff}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
            title="Refresh git diff"
          >
            <RefreshCcw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onOpenRepositoryPanel}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
          >
            <FolderGit2 className="h-3.5 w-3.5" />
            Repository
          </button>
        </div>
      </div>

      {rootPickerOpen ? (
        <div className="border-b border-[#1f1f24] px-3 py-2 flex items-center gap-2 bg-[#0d0d12]">
          <input
            value={rootInput}
            onChange={(event) => onRootInputChange(event.target.value)}
            placeholder="/path/to/workspace/root"
            className="flex-1 rounded-md border border-[#2a2a31] bg-[#111117] px-3 py-1.5 text-[12px] text-[#d0d0d8] outline-none"
          />
          <button
            onClick={onApplyRoot}
            disabled={rootBusy || !rootInput.trim()}
            className="rounded-md border border-[#2a2a31] bg-[#15151b] px-3 py-1.5 text-[12px] text-[#d0d0d8] disabled:opacity-50"
          >
            {rootBusy ? 'Applying...' : 'Apply'}
          </button>
          <button
            onClick={onCancelRootPicker}
            className="rounded-md border border-[#2a2a31] bg-[#111117] px-3 py-1.5 text-[12px] text-[#b3b3bc]"
          >
            Cancel
          </button>
        </div>
      ) : null}

      {statusMessage ? (
        <div className="border-b border-[#1f1f24] px-3 py-2 text-[12px] text-[#8d8d96]">{statusMessage}</div>
      ) : null}
    </>
  );
}
