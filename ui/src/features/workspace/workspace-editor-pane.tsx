import Editor, { type OnMount } from '@monaco-editor/react';
import { Play, Save, SquareTerminal, X } from 'lucide-react';
import type { RefObject } from 'react';

import { monacoLanguageFromPath } from './workspace-helpers';

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
  readOnly?: boolean;
  onSelectTab: (tabId: string) => void;
  onCloseTab: (tabId: string) => void;
  onRunActiveFile: () => void;
  onSaveActiveFile: () => void;
  onEditorMount: OnMount;
  onEditorChange: (value: string) => void;
  onTerminalResizeStart: () => void;
  onTerminalInputChange: (value: string) => void;
  onTerminalSubmit: () => void;
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
  readOnly = false,
  onSelectTab,
  onCloseTab,
  onRunActiveFile,
  onSaveActiveFile,
  onEditorMount,
  onEditorChange,
  onTerminalResizeStart,
  onTerminalInputChange,
  onTerminalSubmit,
}: WorkspaceEditorPaneProps) {
  return (
    <section className="min-h-0 flex flex-col overflow-hidden">
      <div className="h-9 border-b border-zinc-800 px-3 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2 overflow-auto" data-scrollbar="auto">
          {openFiles.length === 0 ? (
            <span className="text-[12px] text-zinc-500">No file selected</span>
          ) : (
            openFiles.map((tab) => {
              const isActive = tab.id === activeFileId;
              const dirty = tab.content !== tab.savedContent;
              return (
                <div
                  key={tab.id}
                  className={`group inline-flex max-w-[220px] items-center gap-2 rounded-md border px-2 py-1 text-[12px] ${
                    isActive
                      ? 'border-zinc-700 bg-zinc-800 text-zinc-300'
                      : 'border-zinc-800 bg-zinc-900 text-zinc-400'
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
          <button
            onClick={onRunActiveFile}
            disabled={!activeTab || terminalBusy || isDecisionBlocking}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-300 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            Run
          </button>
          {!readOnly ? (
            <button
              onClick={onSaveActiveFile}
              disabled={!hasUnsavedChanges || editorSaving || isDecisionBlocking}
              className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-300 disabled:opacity-50"
            >
              <Save className="h-3.5 w-3.5" />
              {editorSaving ? 'Saving...' : 'Save'}
            </button>
          ) : null}
        </div>
      </div>

      <div className="flex-1 min-h-0 bg-zinc-950">
        {activeTab ? (
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
              readOnly,
            }}
          />
        ) : (
          <div className="h-full w-full p-4 text-[12px] text-[#70707b]">Select a file from Explorer.</div>
        )}
      </div>

      <button
        onMouseDown={onTerminalResizeStart}
        className="h-1.5 cursor-row-resize bg-zinc-900 hover:bg-zinc-800"
        aria-label="Resize command runner"
        title="Resize command runner"
      />

      <div className="border-t border-zinc-800 bg-zinc-950 flex flex-col" style={{ height: `${terminalHeight}px` }}>
        <div className="h-8 border-b border-zinc-800 px-3 flex items-center gap-2 text-[12px] text-zinc-400">
          <SquareTerminal className="h-3.5 w-3.5" />
          Command Runner
        </div>
        <div className="flex-1 min-h-0 overflow-auto px-3 py-2 font-mono text-[12px] text-zinc-300" data-scrollbar="always">
          {terminalLines.map((line, index) => (
            <div key={`${line}-${index}`} className="whitespace-pre-wrap break-words">
              {line}
            </div>
          ))}
          <div ref={terminalEndRef} />
        </div>
        <div className="h-9 border-t border-zinc-800 px-3 flex items-center gap-2">
          <span className="font-mono text-[12px] text-zinc-500">$</span>
          <input
            value={terminalInput}
            onChange={(event) => onTerminalInputChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                onTerminalSubmit();
              }
            }}
            placeholder="Run one-shot command"
            className="flex-1 bg-transparent border-0 outline-none text-[12px] text-zinc-300"
            disabled={terminalInputDisabled}
          />
        </div>
      </div>
    </section>
  );
}
