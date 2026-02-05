import {
  ChevronLeft,
  ChevronRight,
  Code2,
  FileText,
  Diff,
  Bug,
  FolderOpen,
  Terminal,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useMemo, useState } from 'react';

interface WorkspaceProps {
  sessionId: string | null;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

type WorkspaceTab = 'code' | 'diff' | 'audit' | 'files' | 'logs';

const mockCodeContent = `// src/app/components/ChatArea.tsx
import { useState } from 'react';

export function ChatArea() {
  const [messages, setMessages] = useState([]);
  
  return (
    <div className="chat-container">
      {messages.map(msg => (
        <div key={msg.id}>{msg.content}</div>
      ))}
    </div>
  );
}`;

const mockDiff = `--- a/src/app/App.tsx
+++ b/src/app/App.tsx
@@ -1,5 +1,5 @@
 import { ChatArea } from './components/ChatArea';
-import { RightPanel } from './components/RightPanel';
+import { Workspace } from './components/Workspace';
 
 export default function App() {
   return (
-    <RightPanel />
+    <Workspace />
   );
 }`;

const mockFiles = [
  { name: 'App.tsx', path: '/src/app/App.tsx', status: 'modified' },
  { name: 'ChatArea.tsx', path: '/src/app/components/ChatArea.tsx', status: 'created' },
  { name: 'Workspace.tsx', path: '/src/app/components/Workspace.tsx', status: 'created' },
  { name: 'Settings.tsx', path: '/src/app/components/Settings.tsx', status: 'created' },
];

type WorkspaceLogLevel = 'info' | 'success' | 'error';

type WorkspaceLog = {
  time: string;
  level: WorkspaceLogLevel;
  message: string;
};

const formatNow = () =>
  new Date().toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

export function Workspace({ sessionId, collapsed, onToggleCollapse }: WorkspaceProps) {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('code');
  const [projectPath, setProjectPath] = useState(() => {
    if (typeof window === 'undefined') {
      return '.';
    }
    return window.localStorage.getItem('slavik.project.path') || '.';
  });
  const [indexing, setIndexing] = useState(false);
  const [logs, setLogs] = useState<WorkspaceLog[]>([]);

  const workspaceTitle = useMemo(() => (indexing ? 'WORKSPACE · indexing' : 'WORKSPACE'), [indexing]);

  const addLog = (level: WorkspaceLogLevel, message: string) => {
    const entry: WorkspaceLog = { time: formatNow(), level, message };
    setLogs((prev) => [entry, ...prev].slice(0, 200));
  };

  const handleIndex = async () => {
    if (!sessionId) {
      addLog('error', 'Нет активной сессии для индексации проекта.');
      return;
    }
    if (indexing) {
      return;
    }
    const trimmedPath = projectPath.trim() || '.';
    setIndexing(true);
    addLog('info', `project index: ${trimmedPath}`);
    try {
      const response = await fetch('/ui/api/tools/project', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Slavik-Session': sessionId,
        },
        body: JSON.stringify({ command: 'index', args: trimmedPath }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        const errorPayload = payload as { error?: { message?: unknown } };
        const message =
          typeof errorPayload.error?.message === 'string'
            ? errorPayload.error?.message
            : 'Не удалось запустить индекс проекта.';
        addLog('error', message);
        return;
      }
      const messages = (payload as { messages?: unknown }).messages;
      if (Array.isArray(messages)) {
        let lastContent = 'Индексирование завершено.';
        for (let index = messages.length - 1; index >= 0; index -= 1) {
          const item = messages[index];
          if (!item || typeof item !== 'object') {
            continue;
          }
          const role = (item as { role?: unknown }).role;
          const content = (item as { content?: unknown }).content;
          if (role === 'assistant' && typeof content === 'string') {
            lastContent = content;
            break;
          }
        }
        addLog('success', lastContent.replace(/\s+/g, ' ').trim());
      } else {
        addLog('success', 'Индексирование завершено.');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Ошибка индексирования проекта.';
      addLog('error', message);
    } finally {
      setIndexing(false);
    }
  };

  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? '0px' : '480px' }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative bg-zinc-950"
    >
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex flex-col h-full w-[480px]"
          >
            {/* Header */}
            <div className="p-4">
              <h2 className="text-sm font-medium text-white/50 mb-3">{workspaceTitle}</h2>
              <div className="flex gap-1 bg-black/40 rounded-lg p-1">
                <button
                  onClick={() => setActiveTab('code')}
                  className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === 'code' ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  <Code2 className="w-3 h-3 inline mr-1" />
                  Code
                </button>
                <button
                  onClick={() => setActiveTab('diff')}
                  className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === 'diff' ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  <Diff className="w-3 h-3 inline mr-1" />
                  Diff
                </button>
                <button
                  onClick={() => setActiveTab('audit')}
                  className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === 'audit' ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  <Bug className="w-3 h-3 inline mr-1" />
                  Audit
                </button>
                <button
                  onClick={() => setActiveTab('files')}
                  className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === 'files' ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  <FolderOpen className="w-3 h-3 inline mr-1" />
                  Files
                </button>
                <button
                  onClick={() => setActiveTab('logs')}
                  className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeTab === 'logs' ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  <Terminal className="w-3 h-3 inline mr-1" />
                  Logs
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {activeTab === 'code' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="h-full"
                >
                  <div className="bg-black/60 border border-white/10 rounded-lg p-4 font-mono text-xs">
                    <pre className="text-white/80 overflow-x-auto whitespace-pre">{mockCodeContent}</pre>
                  </div>
                </motion.div>
              )}

              {activeTab === 'diff' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="h-full"
                >
                  <div className="bg-black/60 border border-white/10 rounded-lg p-4 font-mono text-xs">
                    <pre className="overflow-x-auto whitespace-pre">
                      {mockDiff.split('\n').map((line, i) => (
                        <div
                          key={i}
                          className={`${
                            line.startsWith('+')
                              ? 'text-green-400 bg-green-500/10'
                              : line.startsWith('-')
                              ? 'text-red-400 bg-red-500/10'
                              : line.startsWith('@@')
                              ? 'text-blue-400'
                              : 'text-white/60'
                          }`}
                        >
                          {line}
                        </div>
                      ))}
                    </pre>
                  </div>
                </motion.div>
              )}

              {activeTab === 'audit' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-3"
                >
                  <div className="bg-black/40 border border-white/10 rounded-lg p-4">
                    <div className="flex items-start gap-2 mb-2">
                      <Bug className="w-4 h-4 text-green-400 mt-0.5" />
                      <div className="flex-1">
                        <div className="text-sm font-medium text-white">No issues found</div>
                        <div className="text-xs text-white/50 mt-1">Code quality check passed</div>
                      </div>
                    </div>
                  </div>
                  <div className="bg-black/40 border border-white/10 rounded-lg p-4">
                    <div className="text-xs text-white/50 space-y-1">
                      <div>✓ TypeScript: No errors</div>
                      <div>✓ Linting: Passed</div>
                      <div>✓ Formatting: Consistent</div>
                    </div>
                  </div>
                </motion.div>
              )}

              {activeTab === 'files' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-2"
                >
                  {mockFiles.map((file) => (
                    <div
                      key={file.path}
                      className="bg-black/40 border border-white/10 rounded-lg p-3 hover:bg-black/60 transition-all cursor-pointer"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-white/60" />
                          <div>
                            <div className="text-sm text-white">{file.name}</div>
                            <div className="text-xs text-white/40 font-mono">{file.path}</div>
                          </div>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded-md text-xs font-medium ${
                            file.status === 'modified'
                              ? 'bg-yellow-500/20 text-yellow-400'
                              : 'bg-green-500/20 text-green-400'
                          }`}
                        >
                          {file.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}

              {activeTab === 'logs' && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-1"
                >
                  <div className="mb-3 rounded-lg border border-white/10 bg-black/40 p-3 text-xs text-white/60">
                    <div className="mb-2 font-medium text-white/70">Project index</div>
                    <div className="flex items-center gap-2">
                      <input
                        value={projectPath}
                        onChange={(event) => {
                          const next = event.target.value;
                          setProjectPath(next);
                          if (typeof window !== 'undefined') {
                            window.localStorage.setItem('slavik.project.path', next);
                          }
                        }}
                        className="flex-1 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-xs text-white/80"
                        placeholder="sandbox/project path (.)"
                      />
                      <button
                        type="button"
                        onClick={() => void handleIndex()}
                        disabled={indexing}
                        className="rounded-md border border-white/10 bg-white/10 px-2 py-1 text-xs text-white/80 hover:bg-white/20 disabled:opacity-40"
                      >
                        {indexing ? 'Indexing...' : 'Index'}
                      </button>
                    </div>
                    <div className="mt-2 text-[11px] text-white/40">
                      Индексируется sandbox/project. После индексации агент может искать по проекту.
                    </div>
                  </div>

                  {logs.length === 0 ? (
                    <div className="rounded-lg border border-white/10 bg-black/40 p-2 text-xs text-white/50">
                      Нет событий. Запусти индексирование проекта.
                    </div>
                  ) : (
                    logs.map((log, i) => (
                      <div
                        key={`${log.time}-${i}`}
                        className="bg-black/40 border border-white/10 rounded-lg p-2 font-mono text-xs"
                      >
                        <span className="text-white/40">{log.time}</span>
                        <span
                          className={`ml-2 ${
                            log.level === 'success'
                              ? 'text-green-400'
                              : log.level === 'error'
                              ? 'text-red-400'
                              : 'text-white/60'
                          }`}
                        >
                          [{log.level}]
                        </span>
                        <span className="ml-2 text-white/70">{log.message}</span>
                      </div>
                    ))
                  )}
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button */}
      <button
        onClick={onToggleCollapse}
        className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-xl border border-white/10 flex items-center justify-center transition-all duration-200 z-10"
      >
        {collapsed ? (
          <ChevronLeft className="w-3.5 h-3.5 text-white/60" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-white/60" />
        )}
      </button>
    </motion.div>
  );
}
