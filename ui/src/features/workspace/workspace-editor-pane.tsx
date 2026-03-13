import Editor, { type OnMount } from '@monaco-editor/react';
import {
  AlertTriangle,
  Check,
  FileText,
  Play,
  Save,
  Search,
  Sparkles,
  Terminal,
  Wrench,
  X,
} from 'lucide-react';
import type { RefObject } from 'react';

import type {
  WorkspaceEditorAction,
  WorkspaceEditorPreviewState,
  WorkspaceEditorReadOnlyResult,
} from '../../app/types';
import { compactPath, monacoLanguageFromPath } from './workspace-helpers';

export type WorkspaceOpenFileTab = {
  id: string;
  path: string;
  name: string;
  content: string;
  savedContent: string;
  version: string | null;
  loading: boolean;
};

type WorkspaceEditorPaneProps = {
  openFiles: WorkspaceOpenFileTab[];
  activeFileId: string | null;
  activeTab: WorkspaceOpenFileTab | null;
  hasUnsavedChanges: boolean;
  editorSaving: boolean;
  terminalBusy: boolean;
  isDecisionBlocking: boolean;
  terminalHeight: number;
  terminalLines: string[];
  terminalInput: string;
  terminalInputDisabled: boolean;
  terminalEndRef: RefObject<HTMLDivElement>;
  editorActionsEnabled: boolean;
  editorActionsDisabledReason: string | null;
  editorActionBusy: boolean;
  editorActiveAction: WorkspaceEditorAction | null;
  editorApplyBusy: boolean;
  editorActionError: string | null;
  editorPreview: WorkspaceEditorPreviewState | null;
  editorReadOnlyResult: WorkspaceEditorReadOnlyResult | null;
  editorPreviewStale: boolean;
  onSelectTab: (tabId: string) => void;
  onCloseTab: (tabId: string) => void;
  onRunActiveFile: () => void;
  onSaveActiveFile: () => void;
  onEditorAction: (action: WorkspaceEditorAction) => void;
  onApplyEditorPreview: () => void;
  onCancelEditorPreview: () => void;
  onClearEditorReadOnly: () => void;
  onEditorMount: OnMount;
  onEditorChange: (value: string) => void;
  onTerminalResizeStart: () => void;
  onTerminalInputChange: (value: string) => void;
  onTerminalSubmit: () => void;
};

const EDITOR_ACTIONS: Array<{
  action: WorkspaceEditorAction;
  label: string;
  Icon: typeof Wrench;
}> = [
  { action: 'fix', label: 'Fix', Icon: Wrench },
  { action: 'improve', label: 'Improve', Icon: Sparkles },
  { action: 'review', label: 'Review', Icon: Search },
  { action: 'explain', label: 'Explain', Icon: FileText },
];

const actionLabel = (action: WorkspaceEditorAction | null): string => {
  if (action === 'fix') return 'Fix';
  if (action === 'improve') return 'Improve';
  if (action === 'review') return 'Review';
  if (action === 'explain') return 'Explain';
  return 'Editor action';
};

export function WorkspaceEditorPane({
  openFiles,
  activeFileId,
  activeTab,
  hasUnsavedChanges,
  editorSaving,
  terminalBusy,
  isDecisionBlocking,
  terminalHeight,
  terminalLines,
  terminalInput,
  terminalInputDisabled,
  terminalEndRef,
  editorActionsEnabled,
  editorActionsDisabledReason,
  editorActionBusy,
  editorActiveAction,
  editorApplyBusy,
  editorActionError,
  editorPreview,
  editorReadOnlyResult,
  editorPreviewStale,
  onSelectTab,
  onCloseTab,
  onRunActiveFile,
  onSaveActiveFile,
  onEditorAction,
  onApplyEditorPreview,
  onCancelEditorPreview,
  onClearEditorReadOnly,
  onEditorMount,
  onEditorChange,
  onTerminalResizeStart,
  onTerminalInputChange,
  onTerminalSubmit,
}: WorkspaceEditorPaneProps) {
  const showPreviewPanel = Boolean(editorPreview);
  const showReadOnlyPanel = !showPreviewPanel && Boolean(editorReadOnlyResult);

  return (
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
                    onClick={() => onSelectTab(tab.id)}
                    className="truncate"
                    title={tab.path}
                  >
                    {tab.name}
                    {dirty ? ' *' : ''}
                  </button>
                  <button
                    onClick={() => onCloseTab(tab.id)}
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
          {activeTab ? (
            <>
              {EDITOR_ACTIONS.map(({ action, label, Icon }) => {
                const disabled =
                  !editorActionsEnabled
                  || editorActionBusy
                  || editorApplyBusy
                  || terminalBusy
                  || isDecisionBlocking;
                return (
                  <button
                    key={action}
                    onClick={() => onEditorAction(action)}
                    disabled={disabled}
                    className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
                    title={`${label} with mercury-edit`}
                  >
                    <Icon className="h-3.5 w-3.5" />
                    {editorActionBusy && editorActiveAction === action ? `${label}...` : label}
                  </button>
                );
              })}
            </>
          ) : null}
          <button
            onClick={onRunActiveFile}
            disabled={!activeTab || terminalBusy || isDecisionBlocking || editorApplyBusy}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            Run
          </button>
          <button
            onClick={onSaveActiveFile}
            disabled={!hasUnsavedChanges || editorSaving || isDecisionBlocking || editorApplyBusy}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5" />
            {editorSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      {editorActionError ? (
        <div className="border-b border-[#312128] bg-[#1b1116] px-3 py-2 text-[12px] text-[#f2b4c0]">
          {editorActionError}
        </div>
      ) : null}

      {!editorActionError && activeTab && !editorActionsEnabled && editorActionsDisabledReason ? (
        <div className="border-b border-[#3d3321] bg-[#1b170f] px-3 py-2 text-[12px] text-[#f1c18c]">
          {editorActionsDisabledReason}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 bg-[#0b0b0f]">
        {activeTab ? (
          <div className={`h-full min-h-0 ${showPreviewPanel || showReadOnlyPanel ? 'grid grid-cols-[minmax(0,1fr)_340px]' : ''}`}>
            <div className="min-h-0">
              <Editor
                theme="vs-dark"
                language={monacoLanguageFromPath(activeTab.path)}
                value={activeTab.content}
                onChange={(value) => onEditorChange(value ?? '')}
                onMount={onEditorMount}
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
            </div>

            {showPreviewPanel && editorPreview ? (
              <aside className="min-h-0 border-l border-[#1f1f24] bg-[#0e0e13] flex flex-col">
                <div className="border-b border-[#1f1f24] px-4 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.14em] text-[#7d7d88]">
                        Patch Preview
                      </div>
                      <div className="mt-1 text-[13px] font-medium text-[#ececf2]">
                        {actionLabel(editorPreview.action)} for {compactPath(editorPreview.targetPath, 44)}
                      </div>
                    </div>
                    <button
                      onClick={onCancelEditorPreview}
                      className="rounded-md border border-[#2a2a31] px-2 py-1 text-[11px] text-[#a8a8b3]"
                    >
                      Dismiss
                    </button>
                  </div>
                  {editorPreview.summary.trim() ? (
                    <p className="mt-2 text-[12px] leading-5 text-[#b2b2bc]">{editorPreview.summary}</p>
                  ) : null}
                  <div className="mt-3 flex items-center gap-2">
                    <button
                      onClick={onApplyEditorPreview}
                      disabled={!editorPreview.applyAvailable || editorApplyBusy || editorPreviewStale}
                      className="inline-flex items-center gap-1 rounded-md border border-[#34543a] bg-[#162219] px-2.5 py-1.5 text-[12px] text-[#cfe8d2] disabled:opacity-50"
                    >
                      <Check className="h-3.5 w-3.5" />
                      {editorApplyBusy ? 'Applying...' : 'Apply'}
                    </button>
                    <button
                      onClick={onCancelEditorPreview}
                      disabled={editorApplyBusy}
                      className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2.5 py-1.5 text-[12px] text-[#bdbdc6] disabled:opacity-50"
                    >
                      <X className="h-3.5 w-3.5" />
                      Cancel
                    </button>
                  </div>
                  {editorPreviewStale ? (
                    <div className="mt-3 flex items-start gap-2 rounded-md border border-[#4b3324] bg-[#21160f] px-3 py-2 text-[12px] text-[#f1c18c]">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      <span>Preview is stale. Regenerate it before apply.</span>
                    </div>
                  ) : null}
                </div>
                <div className="min-h-0 flex-1 overflow-auto px-4 py-3" data-scrollbar="auto">
                  {editorPreview.patchedContent ? (
                    <>
                      <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-[#7d7d88]">
                        Suggested Content
                      </div>
                      <pre className="whitespace-pre-wrap break-words rounded-md border border-[#20202a] bg-[#0a0a0f] p-3 font-mono text-[11px] leading-5 text-[#c9c9d1]">
                        {editorPreview.patchedContent}
                      </pre>
                    </>
                  ) : null}
                  <div className={`${editorPreview.patchedContent ? 'mt-4' : ''} mb-2 text-[11px] uppercase tracking-[0.14em] text-[#7d7d88]`}>
                    Unified Diff
                  </div>
                  <pre className="whitespace-pre-wrap break-words rounded-md border border-[#20202a] bg-[#0a0a0f] p-3 font-mono text-[11px] leading-5 text-[#c9c9d1]">
                    {editorPreview.patch || 'No patch content returned.'}
                  </pre>
                </div>
              </aside>
            ) : null}

            {showReadOnlyPanel && editorReadOnlyResult ? (
              <aside className="min-h-0 border-l border-[#1f1f24] bg-[#0e0e13] flex flex-col">
                <div className="border-b border-[#1f1f24] px-4 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.14em] text-[#7d7d88]">
                        {actionLabel(editorReadOnlyResult.action)} Result
                      </div>
                      <div className="mt-1 text-[13px] font-medium text-[#ececf2]">
                        {compactPath(editorReadOnlyResult.targetPath, 44)}
                      </div>
                    </div>
                    <button
                      onClick={onClearEditorReadOnly}
                      className="rounded-md border border-[#2a2a31] px-2 py-1 text-[11px] text-[#a8a8b3]"
                    >
                      Close
                    </button>
                  </div>
                </div>
                <div className="min-h-0 flex-1 overflow-auto px-4 py-3" data-scrollbar="auto">
                  <pre className="whitespace-pre-wrap break-words rounded-md border border-[#20202a] bg-[#0a0a0f] p-3 font-mono text-[11px] leading-5 text-[#c9c9d1]">
                    {editorReadOnlyResult.message || 'No response content returned.'}
                  </pre>
                </div>
              </aside>
            ) : null}
          </div>
        ) : (
          <div className="h-full w-full p-4 text-[12px] text-[#70707b]">Select a file from Explorer.</div>
        )}
      </div>

      <button
        onMouseDown={onTerminalResizeStart}
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
            onChange={(event) => onTerminalInputChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                onTerminalSubmit();
              }
            }}
            placeholder="Type shell command"
            className="flex-1 bg-transparent border-0 outline-none text-[12px] text-[#d0d0d8]"
            disabled={terminalInputDisabled}
          />
        </div>
      </div>
    </section>
  );
}
