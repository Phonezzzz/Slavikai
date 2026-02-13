import Editor, { type OnMount } from '@monaco-editor/react';
import { Play, Save, Terminal, X } from 'lucide-react';

import { monacoLanguageFromPath } from './workspace-helpers';

export type WorkspaceOpenFileTab = {
  id: string;
  path: string;
  name: string;
  content: string;
  savedContent: string;
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
  terminalEndRef: React.RefObject<HTMLDivElement | null>;
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
          <button
            onClick={onRunActiveFile}
            disabled={!activeTab || terminalBusy || isDecisionBlocking}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            Run
          </button>
          <button
            onClick={onSaveActiveFile}
            disabled={!hasUnsavedChanges || editorSaving || isDecisionBlocking}
            className="inline-flex items-center gap-1 rounded-md border border-[#2a2a31] bg-[#121217] px-2 py-1 text-[12px] text-[#bdbdc6] disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5" />
            {editorSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 bg-[#0b0b0f]">
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
            }}
          />
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
