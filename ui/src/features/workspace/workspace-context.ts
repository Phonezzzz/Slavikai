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
  activeFile: WorkspaceOpenFileTab | null;
  selectionText: string;
  gitDiff: string;
  lastTerminalOutput: string;
};

export const MAX_ATTACHMENTS = 8;
export const MAX_ATTACHMENT_CHARS = 80_000;
export const MAX_TOTAL_ATTACHMENT_CHARS = 160_000;

const truncateAttachmentContent = (content: string, limit: number): string => {
  if (content.length <= limit) {
    return content;
  }
  const suffix = '\n...[truncated by client attachment budget]';
  const trimmedLimit = Math.max(0, limit - suffix.length);
  return `${content.slice(0, trimmedLimit)}${suffix}`;
};

const pushAttachmentWithinBudget = (
  attachments: WorkspaceContextAttachment[],
  state: { totalChars: number },
  attachment: WorkspaceContextAttachment,
): void => {
  if (attachments.length >= MAX_ATTACHMENTS) {
    return;
  }
  const remaining = MAX_TOTAL_ATTACHMENT_CHARS - state.totalChars;
  if (remaining <= 0) {
    return;
  }
  const perAttachmentLimit = Math.min(MAX_ATTACHMENT_CHARS, remaining);
  const normalized = truncateAttachmentContent(attachment.content, perAttachmentLimit);
  if (!normalized.trim()) {
    return;
  }
  attachments.push({ ...attachment, content: normalized });
  state.totalChars += normalized.length;
};

export const mergeWorkspaceAttachments = (
  primary: WorkspaceContextAttachment[],
  secondary: WorkspaceContextAttachment[],
): WorkspaceContextAttachment[] => {
  const merged: WorkspaceContextAttachment[] = [];
  const budgetState = { totalChars: 0 };
  primary.forEach((attachment) => pushAttachmentWithinBudget(merged, budgetState, attachment));
  secondary.forEach((attachment) => pushAttachmentWithinBudget(merged, budgetState, attachment));
  return merged;
};

export const buildWorkspaceContextAttachments = ({
  includeOpenTabs,
  includeSelection,
  includeGitDiff,
  includeTerminal,
  openFiles,
  activeFile,
  selectionText,
  gitDiff,
  lastTerminalOutput,
}: BuildContextAttachmentsParams): WorkspaceContextAttachment[] => {
  const attachments: WorkspaceContextAttachment[] = [];
  const budgetState = { totalChars: 0 };
  if (includeOpenTabs && openFiles.length > 0) {
    pushAttachmentWithinBudget(attachments, budgetState, {
      name: 'open-tabs.json',
      mime: 'application/json',
      content: JSON.stringify(
        {
          active_file: activeFile?.path ?? null,
          open_files: openFiles.map((item) => ({
            path: item.path,
            dirty: item.content !== item.savedContent,
          })),
        },
        null,
        2,
      ),
    });
    if (activeFile) {
      pushAttachmentWithinBudget(attachments, budgetState, {
        name: 'active-file-content.txt',
        mime: 'text/plain',
        content: `path: ${activeFile.path}\n\n${activeFile.content}`,
      });
      if (activeFile.content !== activeFile.savedContent) {
        pushAttachmentWithinBudget(attachments, budgetState, {
          name: 'active-file-unsaved.diff',
          mime: 'text/x-diff',
          content: [
            `--- saved/${activeFile.path}`,
            `+++ unsaved/${activeFile.path}`,
            '@@ saved @@',
            activeFile.savedContent,
            '@@ unsaved @@',
            activeFile.content,
          ].join('\n'),
        });
      }
    }
  }
  if (includeSelection && selectionText.trim()) {
    pushAttachmentWithinBudget(attachments, budgetState, {
      name: 'selection.txt',
      mime: 'text/plain',
      content: selectionText,
    });
  }
  if (includeGitDiff && gitDiff.trim()) {
    pushAttachmentWithinBudget(attachments, budgetState, {
      name: 'git-diff.patch',
      mime: 'text/x-diff',
      content: gitDiff,
    });
  }
  if (includeTerminal && lastTerminalOutput.trim()) {
    pushAttachmentWithinBudget(attachments, budgetState, {
      name: 'command-runner-last.txt',
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
      label: lastTerminalOutput.trim() ? 'Last command runner output' : 'Command runner output',
      enabled: includeTerminal,
      onToggle: onToggleTerminal,
      hidden: !lastTerminalOutput.trim(),
    },
  ].filter((item) => !item.hidden);
};
