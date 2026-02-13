import type { WorkspaceContextChip } from './workspace-assistant-panel';
import type { WorkspaceOpenFileTab } from './workspace-editor-pane';

export type WorkspaceContextAttachment = {
  name: string;
  mime: string;
  content: string;
};

type BuildContextAttachmentsParams = {
  includeOpenTabs: boolean;
  includeSelection: boolean;
  includeGitDiff: boolean;
  includeTerminal: boolean;
  openFiles: WorkspaceOpenFileTab[];
  activeFilePath: string | null;
  selectionText: string;
  gitDiff: string;
  lastTerminalOutput: string;
};

export const buildWorkspaceContextAttachments = ({
  includeOpenTabs,
  includeSelection,
  includeGitDiff,
  includeTerminal,
  openFiles,
  activeFilePath,
  selectionText,
  gitDiff,
  lastTerminalOutput,
}: BuildContextAttachmentsParams): WorkspaceContextAttachment[] => {
  const attachments: WorkspaceContextAttachment[] = [];
  if (includeOpenTabs && openFiles.length > 0) {
    attachments.push({
      name: 'open-tabs.json',
      mime: 'application/json',
      content: JSON.stringify(
        {
          active_file: activeFilePath,
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

type BuildContextChipsParams = {
  openFilesCount: number;
  includeOpenTabs: boolean;
  onToggleOpenTabs: () => void;
  selectionText: string;
  includeSelection: boolean;
  onToggleSelection: () => void;
  gitDiff: string;
  gitDiffLoading: boolean;
  includeGitDiff: boolean;
  onToggleGitDiff: () => void;
  lastTerminalOutput: string;
  includeTerminal: boolean;
  onToggleTerminal: () => void;
};

export const buildWorkspaceContextChips = ({
  openFilesCount,
  includeOpenTabs,
  onToggleOpenTabs,
  selectionText,
  includeSelection,
  onToggleSelection,
  gitDiff,
  gitDiffLoading,
  includeGitDiff,
  onToggleGitDiff,
  lastTerminalOutput,
  includeTerminal,
  onToggleTerminal,
}: BuildContextChipsParams): WorkspaceContextChip[] => {
  return [
    {
      key: 'tabs',
      label: `Open tabs (${openFilesCount})`,
      enabled: includeOpenTabs,
      onToggle: onToggleOpenTabs,
      hidden: openFilesCount === 0,
    },
    {
      key: 'selection',
      label: selectionText.trim() ? `Selection (${Math.min(selectionText.length, 120)} chars)` : 'Selection',
      enabled: includeSelection,
      onToggle: onToggleSelection,
      hidden: !selectionText.trim(),
    },
    {
      key: 'diff',
      label: gitDiff.trim() ? 'Git diff' : gitDiffLoading ? 'Git diff (loading...)' : 'Git diff (empty)',
      enabled: includeGitDiff,
      onToggle: onToggleGitDiff,
      hidden: false,
    },
    {
      key: 'terminal',
      label: lastTerminalOutput.trim() ? 'Last terminal output' : 'Terminal output',
      enabled: includeTerminal,
      onToggle: onToggleTerminal,
      hidden: !lastTerminalOutput.trim(),
    },
  ].filter((item) => !item.hidden);
};
