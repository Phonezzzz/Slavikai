import { ChevronRight, LoaderCircle, Play, Server } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { ProviderModels } from '../types';

type SessionModelPickerProps = {
  sessionId: string | null;
  modelLabel: string;
  providerModels: ProviderModels[];
  selectedModelValue: string | null;
  modelsLoading?: boolean;
  savingModel?: boolean;
  onLoadProviderModels: (provider: string) => Promise<ProviderModels | null>;
  onStartLocalOllama: () => Promise<ProviderModels | null>;
  onSelectModel: (provider: string, model: string) => void;
};

const PROVIDER_ORDER = ['local', 'xai', 'inception', 'deepseek', 'openrouter'];
const PROVIDER_LABELS: Record<string, string> = {
  local: 'Local',
  xai: 'xAI',
  inception: 'Inception',
  deepseek: 'DeepSeek',
  openrouter: 'OpenRouter',
};

const sortProviders = (items: ProviderModels[]): ProviderModels[] => {
  return [...items].sort((left, right) => {
    const leftIndex = PROVIDER_ORDER.indexOf(left.provider);
    const rightIndex = PROVIDER_ORDER.indexOf(right.provider);
    const normalizedLeft = leftIndex === -1 ? PROVIDER_ORDER.length : leftIndex;
    const normalizedRight = rightIndex === -1 ? PROVIDER_ORDER.length : rightIndex;
    if (normalizedLeft !== normalizedRight) {
      return normalizedLeft - normalizedRight;
    }
    return left.provider.localeCompare(right.provider);
  });
};

const compactStatusText = (value: string): string => {
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (normalized.length <= 96) {
    return normalized;
  }
  return `${normalized.slice(0, 93)}...`;
};

export function SessionModelPicker({ sessionId, modelLabel, providerModels, selectedModelValue,
  modelsLoading = false, savingModel = false, onLoadProviderModels, onStartLocalOllama, onSelectModel,
}: SessionModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [activeProvider, setActiveProvider] = useState<string | null>(null);
  const [manualModelId, setManualModelId] = useState('');
  const [loadedProviders, setLoadedProviders] = useState<Set<string>>(() => new Set());
  const [startingOllama, setStartingOllama] = useState(false);

  const sortedProviders = sortProviders(providerModels);
  const currentProvider =
    sortedProviders.find((item) => item.provider === activeProvider) ?? sortedProviders[0] ?? null;
  const currentProviderName = currentProvider?.provider ?? null;
  const currentProviderLoaded =
    currentProviderName !== null && loadedProviders.has(currentProviderName);
  const currentProviderError = currentProvider?.error ?? null;
  const currentProviderModels = (currentProvider?.models ?? []).filter((model) => model.toLowerCase().includes(search.trim().toLowerCase()));
  const manualModelAllowed = !modelsLoading && currentProviderName?.startsWith('custom-')
    && (currentProvider?.status === 'models_unavailable' || currentProvider?.status === 'provider_unavailable');

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setOpen(false);
    };
    const close = () => setOpen(false);
    document.addEventListener('pointerdown', dismiss);
    window.addEventListener('resize', close);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      window.removeEventListener('resize', close);
    };
  }, [open]);
  const handleOpenProvider = async (provider: string) => {
    setActiveProvider(provider);
    setManualModelId('');
    setSearch('');
    setStatus(null);
    if (provider === 'local') {
      return;
    }
    try {
      const loaded = await onLoadProviderModels(provider);
      if (loaded) {
        setLoadedProviders((current) => {
          const next = new Set(current);
          next.add(provider);
          return next;
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to load ${provider} models.`;
      setStatus(message);
    }
  };

  const handleStartLocalOllama = async () => {
    setStartingOllama(true);
    setStatus(null);
    setActiveProvider('local');
    try {
      const loaded = await onStartLocalOllama();
      if (loaded) {
        setLoadedProviders((current) => {
          const next = new Set(current);
          next.add('local');
          return next;
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start Local Ollama.';
      setStatus(message);
    } finally {
      setStartingOllama(false);
    }
  };

  const anchor = triggerRef.current?.getBoundingClientRect();
  return <div ref={rootRef} className="relative mb-2" onKeyDown={(event) => {
    if (event.key === 'Escape') { setOpen(false); triggerRef.current?.focus(); }
  }}>
    <button ref={triggerRef} type="button" disabled={!sessionId} aria-expanded={open}
      aria-label="Выбрать модель" onClick={() => setOpen((value) => !value)}
      className="max-w-full truncate rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 disabled:opacity-50">
      {modelLabel}
    </button>
    {open ? <div role="region" aria-label="Выбор модели"
      style={{
        left: Math.max(16, Math.min(anchor?.left ?? 16, window.innerWidth - 496)),
        bottom: window.innerHeight - (anchor?.top ?? window.innerHeight) + 8,
        maxHeight: Math.max(80, Math.min(window.innerHeight * 0.6, (anchor?.top ?? 96) - 16)),
      }}
      className="fixed z-50 w-[min(480px,calc(100vw-32px))] overflow-auto rounded-xl border border-zinc-700 bg-zinc-950 p-3 shadow-xl">
      <div className="space-y-2">
        <div className="text-xs font-medium text-zinc-300">Model</div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-2">
          <div className="mb-2 text-[11px] text-zinc-500">{modelLabel}</div>
          <input autoFocus type="search" aria-label="Поиск модели" placeholder="Поиск модели"
            value={search} onChange={(event) => setSearch(event.target.value)}
            className="mb-2 w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100" />
          <div className="grid grid-cols-[120px_1fr] gap-2">
            <div className="space-y-1">
              {sortedProviders.map((provider) => {
                const isActive = provider.provider === currentProviderName;
                const label = provider.displayName ?? PROVIDER_LABELS[provider.provider] ?? provider.provider;
                return (
                  <button
                    key={provider.provider}
                    type="button"
                    onClick={() => {
                      void handleOpenProvider(provider.provider);
                    }}
                    disabled={modelsLoading || savingModel || !sessionId}
                    className={`flex w-full items-center justify-between rounded-md border px-2 py-2 text-left text-xs transition-colors ${
                      isActive
                        ? 'border-zinc-700 bg-zinc-800 text-zinc-100'
                        : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:bg-zinc-800'
                    } disabled:opacity-50`}
                  >
                    <span className="inline-flex min-w-0 items-center gap-1.5">
                      <Server className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">{label}</span>
                    </span>
                    <ChevronRight className="h-3 w-3 shrink-0 text-zinc-500" />
                  </button>
                );
              })}
            </div>

            <div className="min-h-[150px] rounded-md border border-zinc-800 bg-zinc-950 p-2">
              {currentProviderName === 'local' && (currentProvider?.models.length ?? 0) === 0 ? (
                <div className="flex h-full min-h-[132px] flex-col justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      void handleStartLocalOllama();
                    }}
                    disabled={startingOllama || modelsLoading || savingModel || !sessionId}
                    className="inline-flex items-center justify-center gap-2 rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-xs font-medium text-zinc-100 hover:bg-zinc-700 disabled:opacity-50"
                  >
                    {startingOllama ? (
                      <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Play className="h-3.5 w-3.5" />
                    )}
                    Local Ollama
                  </button>
                  <div className="text-[11px] leading-4 text-zinc-500">
                    Local models become selectable after Ollama responds with a model list.
                  </div>
                </div>
              ) : null}

              {currentProviderName !== 'local'
              && !currentProviderLoaded
              && (currentProvider?.models.length ?? 0) === 0
              && !currentProviderError ? (
                <div className="flex h-full min-h-[132px] items-center justify-center text-xs text-zinc-500">
                  Select provider to load its models.
                </div>
              ) : null}

              {currentProviderError ? (
                <div className="text-xs leading-5 text-amber-200">
                  {compactStatusText(currentProviderError)}
                </div>
              ) : null}

              {search.trim() && currentProviderModels.length === 0 && (currentProvider?.models.length ?? 0) > 0
                ? <p className="text-xs text-zinc-400">Модели не найдены.</p> : null}
              {currentProviderModels.length > 0 ? (
                <div className="max-h-56 space-y-1 overflow-y-auto pr-1" data-scrollbar="auto">
                  {currentProviderModels.map((model) => {
                    const value = `${currentProviderName}::${model}`;
                    const selected = selectedModelValue === value;
                    return (
                      <button
                        key={value}
                        type="button"
                        onClick={() => {
                          if (currentProviderName) {
                            onSelectModel(currentProviderName, model);
                          }
                        }}
                        disabled={savingModel || !sessionId}
                        className={`w-full rounded-md border px-2 py-1.5 text-left text-xs transition-colors ${
                          selected
                            ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-100'
                            : 'border-transparent bg-transparent text-zinc-300 hover:border-zinc-800 hover:bg-zinc-800'
                        } disabled:opacity-50`}
                      >
                        <span className="block truncate">{model}</span>
                      </button>
                    );
                  })}
                </div>
              ) : null}
              {manualModelAllowed ? (
                <div className="mt-2 flex gap-2">
                  <input
                    aria-label="Manual model ID"
                    placeholder="Enter model ID"
                    value={manualModelId}
                    onChange={(event) => setManualModelId(event.target.value)}
                    className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-100"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (currentProviderName && manualModelId.trim()) onSelectModel(currentProviderName, manualModelId.trim());
                    }}
                    disabled={savingModel || !sessionId || !manualModelId.trim()}
                    className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1.5 text-xs text-zinc-100 disabled:opacity-50"
                  >
                    Set
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>
      {status ? <p role="alert" className="mt-2 text-xs text-amber-200">{status}</p> : null}
    </div> : null}
  </div>;
}
