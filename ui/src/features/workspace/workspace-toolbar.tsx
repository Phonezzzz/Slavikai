import { ArrowLeft, RefreshCcw, Settings } from 'lucide-react';

import { compactPath } from './workspace-helpers';

type WorkspaceToolbarProps = {
  modelLabel: string;
  indexing: boolean;
  workspaceRoot: string;
  workspacePolicy: string;
  workspaceYoloActive: boolean;
  rootPickerOpen: boolean;
  rootInput: string;
  rootBusy: boolean;
  statusMessage: string | null | undefined;
  onBackToChat: () => void;
  onToggleRootPicker: () => void;
  onReindex: () => void;
  onRefreshGitDiff: () => void;
  onOpenWorkspaceSettings: () => void;
  onRootInputChange: (value: string) => void;
  onApplyRoot: () => void;
  onCancelRootPicker: () => void;
};

export function WorkspaceToolbar({
  modelLabel,
  indexing,
  workspaceRoot,
  workspacePolicy,
  workspaceYoloActive,
  rootPickerOpen,
  rootInput,
  rootBusy,
  statusMessage,
  onBackToChat,
  onToggleRootPicker,
  onReindex,
  onRefreshGitDiff,
  onOpenWorkspaceSettings,
  onRootInputChange,
  onApplyRoot,
  onCancelRootPicker,
}: WorkspaceToolbarProps) {
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
            <span
              className="max-w-[380px] truncate rounded-md border border-[#2a2a31] bg-[#111117] px-2 py-1 text-[11px] text-[#8f8f99]"
              title={workspaceRoot}
            >
              {compactPath(workspaceRoot, 68)}
            </span>
          ) : null}
        </div>

        <div className="flex items-center justify-end gap-2">
          <span className={`text-[11px] ${workspaceYoloActive ? 'text-red-300' : 'text-[#8f8f98]'}`}>
            Policy: {workspacePolicy}
          </span>
          <button
            onClick={onRefreshGitDiff}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] hover:bg-[#181820]"
            title="Refresh git diff"
          >
            <RefreshCcw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onOpenWorkspaceSettings}
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
