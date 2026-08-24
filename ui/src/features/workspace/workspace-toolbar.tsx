import { ArrowLeft, FolderGit2, Monitor, SlidersHorizontal } from 'lucide-react';

import { compactPath } from './workspace-helpers';

type WorkspaceToolbarProps = {
  modelLabel: string;
  indexing: boolean;
  workspaceRoot: string;
  sessionPolicyLabel: string;
  sessionYoloActive: boolean;
  sessionSafeMode: boolean;
  statusMessage: string | null | undefined;
  onBackToChat: () => void;
  onOpenSessionDrawer: () => void;
  onOpenProjectPicker: () => void;
  onReindex: () => void;
  onSwitchToFiles: () => void;
  onOpenRepositoryPanel: () => void;
};

export function WorkspaceToolbar({
  modelLabel,
  indexing,
  workspaceRoot,
  sessionPolicyLabel,
  sessionSafeMode,
  statusMessage,
  onBackToChat,
  onOpenSessionDrawer,
  onOpenProjectPicker,
  onReindex,
  onSwitchToFiles,
  onOpenRepositoryPanel,
}: WorkspaceToolbarProps) {
  const policyBadge = `Session policy: ${sessionPolicyLabel}`;
  const policyClass = sessionPolicyLabel === 'YOLO' ? 'text-red-300' : 'text-zinc-400';
  return (
    <>
      <div className="h-12 border-b border-zinc-800 px-3 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="flex items-center gap-2">
          <button
            onClick={onBackToChat}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-300 hover:bg-zinc-900"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Chat
          </button>
          <button
            onClick={onOpenSessionDrawer}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-300 hover:bg-zinc-900"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Session
          </button>
          <span className="text-[11px] text-zinc-500">{modelLabel}</span>
        </div>

        <div className="flex items-center justify-center gap-2">
          <div className="flex items-center gap-1.5 text-[12px]">
            <Monitor className="h-3.5 w-3.5 text-zinc-400" />
            <span className="font-medium text-zinc-300">Slavik Computer</span>
          </div>
          <span className="text-[#333]">·</span>
          <button
            onClick={onOpenProjectPicker}
            className="inline-flex max-w-[280px] items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-[12px] text-zinc-300 hover:bg-zinc-900"
            title={workspaceRoot || 'Open project'}
          >
            <span className="truncate">
              {workspaceRoot ? compactPath(workspaceRoot, 40) : 'Open Project'}
            </span>
          </button>
          <button
            onClick={onSwitchToFiles}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-[12px] text-zinc-300 hover:bg-zinc-900"
          >
            Files
          </button>
          <button
            onClick={onReindex}
            disabled={indexing}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-[12px] text-zinc-300 hover:bg-zinc-900 disabled:opacity-50"
          >
            {indexing ? 'Indexing...' : 'Re-index'}
          </button>
        </div>

        <div className="flex items-center justify-end gap-2">
          <span className={`text-[11px] ${policyClass}`}>
            {policyBadge}
          </span>
          <span className={`text-[11px] ${sessionSafeMode ? 'text-amber-300' : 'text-emerald-300'}`}>
            Session safe mode: {sessionSafeMode ? 'ON' : 'OFF'}
          </span>
          <button
            onClick={onOpenRepositoryPanel}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-300 hover:bg-zinc-900"
          >
            <FolderGit2 className="h-3.5 w-3.5" />
            Repository
          </button>
        </div>
      </div>

      {statusMessage ? (
        <div className="border-b border-zinc-800 px-3 py-2 text-[12px] text-zinc-400">{statusMessage}</div>
      ) : null}
    </>
  );
}
