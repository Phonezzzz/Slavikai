import { Github, Loader2, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import type { UiDecision } from '../types';

export type WorkspaceGithubImportResult = {
  status: 'done' | 'pending';
  message?: string | null;
};

type WorkspaceSettingsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  pendingDecision?: UiDecision | null;
  onRunGithubImport: (
    repositoryUrl: string,
    branch?: string,
  ) => Promise<WorkspaceGithubImportResult>;
};

type WorkspaceSettingsTab = 'github';
type SubmitState = 'idle' | 'submitting' | 'pending' | 'done' | 'error';

const isHttpGithubUrl = (value: string): boolean => {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
      return false;
    }
    return parsed.hostname === 'github.com' || parsed.hostname.endsWith('.github.com');
  } catch {
    return false;
  }
};

const isSshGithubUrl = (value: string): boolean => /^git@github\.com:[^ ]+$/i.test(value);

const isValidGithubRepository = (value: string): boolean =>
  isHttpGithubUrl(value) || isSshGithubUrl(value);

const extractDecisionSourceEndpoint = (decision: UiDecision | null | undefined): string | null => {
  if (!decision || typeof decision.context !== 'object' || !decision.context) {
    return null;
  }
  const source = decision.context.source_endpoint;
  return typeof source === 'string' ? source : null;
};

export function WorkspaceSettingsModal({
  isOpen,
  onClose,
  pendingDecision = null,
  onRunGithubImport,
}: WorkspaceSettingsModalProps) {
  const [activeTab, setActiveTab] = useState<WorkspaceSettingsTab>('github');
  const [repositoryUrl, setRepositoryUrl] = useState('');
  const [branch, setBranch] = useState('');
  const [submitState, setSubmitState] = useState<SubmitState>('idle');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const pendingProjectDecision = useMemo(() => {
    return (
      pendingDecision?.status === 'pending'
      && extractDecisionSourceEndpoint(pendingDecision) === 'project.command'
    );
  }, [pendingDecision]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setActiveTab('github');
    setSubmitState('idle');
    setStatusMessage(null);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  const handleImport = async () => {
    const repo = repositoryUrl.trim();
    const branchName = branch.trim();
    if (!repo) {
      setSubmitState('error');
      setStatusMessage('Repository URL обязателен.');
      return;
    }
    if (!isValidGithubRepository(repo)) {
      setSubmitState('error');
      setStatusMessage('Введите корректный GitHub URL (https://github.com/... или git@github.com:...).');
      return;
    }
    if (branchName.includes(' ')) {
      setSubmitState('error');
      setStatusMessage('Branch не должен содержать пробелы.');
      return;
    }

    setSubmitState('submitting');
    setStatusMessage(null);
    try {
      const result = await onRunGithubImport(repo, branchName || undefined);
      if (result.status === 'pending') {
        setSubmitState('pending');
        setStatusMessage(
          result.message
            ?? 'Действие ожидает подтверждения в DecisionPanel (AI Assistant).',
        );
        return;
      }
      setSubmitState('done');
      setStatusMessage(result.message ?? 'Импорт запущен успешно.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Не удалось запустить импорт.';
      setSubmitState('error');
      setStatusMessage(message);
    }
  };

  const statusClass =
    submitState === 'error'
      ? 'text-red-300 border-red-900/60 bg-red-950/40'
      : submitState === 'pending'
        ? 'text-amber-200 border-amber-800/60 bg-amber-950/30'
        : submitState === 'done'
          ? 'text-emerald-200 border-emerald-900/60 bg-emerald-950/30'
          : 'text-zinc-300 border-zinc-800 bg-zinc-900/70';

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 px-4 py-12 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">Workspace Settings</h2>
            <p className="text-xs text-zinc-400">
              Настройки рабочей среды проекта.
            </p>
          </div>
          <button
            onClick={onClose}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
            aria-label="Close workspace settings"
            title="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="grid grid-cols-[180px_1fr] min-h-[360px]">
          <aside className="border-r border-zinc-800 bg-zinc-950 p-3">
            <button
              onClick={() => setActiveTab('github')}
              className={`flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-sm ${
                activeTab === 'github'
                  ? 'border-indigo-700 bg-indigo-950/50 text-indigo-100'
                  : 'border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'
              }`}
            >
              <Github className="h-4 w-4" />
              GitHub
            </button>
          </aside>

          <section className="p-5">
            {activeTab === 'github' ? (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-zinc-100">Import Repository</h3>
                  <p className="mt-1 text-xs text-zinc-400">
                    Импорт выполняется через backend command lane и approval lifecycle.
                  </p>
                </div>

                <label className="block space-y-1">
                  <span className="text-xs text-zinc-400">Repository URL</span>
                  <input
                    value={repositoryUrl}
                    onChange={(event) => setRepositoryUrl(event.target.value)}
                    placeholder="https://github.com/org/repo.git"
                    className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-indigo-500"
                  />
                </label>

                <label className="block space-y-1">
                  <span className="text-xs text-zinc-400">Branch (optional)</span>
                  <input
                    value={branch}
                    onChange={(event) => setBranch(event.target.value)}
                    placeholder="main"
                    className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-indigo-500"
                  />
                </label>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      void handleImport();
                    }}
                    disabled={submitState === 'submitting'}
                    className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {submitState === 'submitting' ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Github className="h-4 w-4" />
                    )}
                    Import
                  </button>
                  <span className="text-xs text-zinc-500">
                    {pendingProjectDecision
                      ? 'Pending approval in DecisionPanel.'
                      : 'Approval, если потребуется, появится в DecisionPanel.'}
                  </span>
                </div>

                {statusMessage ? (
                  <div className={`rounded-md border px-3 py-2 text-xs ${statusClass}`}>
                    {statusMessage}
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        </div>
      </div>
    </div>
  );
}
