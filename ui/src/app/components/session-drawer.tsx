import { AnimatePresence, motion } from 'motion/react';
import { ChevronRight, LoaderCircle, Play, Server, Shield, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { ToggleSwitch } from './ui/toggle-switch';
import { useFocusTrap } from '../use-focus-trap';
import {
  SESSION_MODE_VALUES,
  SESSION_MODE_LABELS,
  type ModeTransitionsContract,
  type ProviderModels,
  type SessionMode,
} from '../types';

type ToolKey = 'fs' | 'shell' | 'web' | 'project' | 'img' | 'tts' | 'stt';
type PolicyProfile = 'sandbox' | 'index' | 'yolo';

type SessionDrawerProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modeTransitions?: ModeTransitionsContract | null;
  modeBusy?: boolean;
  onChangeMode: (mode: SessionMode) => Promise<void>;
  modelLabel: string;
  providerModels: ProviderModels[];
  selectedModelValue: string | null;
  modelsLoading?: boolean;
  savingModel?: boolean;
  onLoadProviderModels: (provider: string) => Promise<ProviderModels | null>;
  onStartLocalOllama: () => Promise<ProviderModels | null>;
  onSelectModel: (provider: string, model: string) => void;
};

type SessionSecurityState = {
  toolsState: Record<ToolKey, boolean>;
  policyProfile: PolicyProfile;
};

const DEFAULT_TOOLS_STATE: Record<ToolKey, boolean> = {
  fs: true,
  shell: false,
  web: false,
  project: false,
  img: false,
  tts: false,
  stt: false,
};

const SAFE_MODE_BLOCKED_TOOLS = new Set<ToolKey>(['web', 'shell', 'project', 'tts', 'stt']);

const TOOL_LABELS: Record<ToolKey, string> = {
  fs: 'Filesystem access',
  shell: 'Shell access',
  web: 'Web access',
  project: 'Project tool',
  img: 'Images',
  tts: 'Text to speech',
  stt: 'Speech to text',
};

const POLICY_OPTIONS: Array<{
  value: PolicyProfile;
  title: string;
  description: string;
}> = [
  {
    value: 'sandbox',
    title: 'Restricted',
    description: 'Keeps the current session tightly limited and blocks risky access by default.',
  },
  {
    value: 'index',
    title: 'Project access',
    description: 'Allows project-aware work while keeping shell and web access constrained.',
  },
  {
    value: 'yolo',
    title: 'Unrestricted (dangerous)',
    description: 'Removes the normal safety posture for this session and requires explicit confirmation.',
  },
];

const DANGER_CONFIRMATION_PHRASE = 'YOLO';
const PROVIDER_ORDER = ['local', 'xai', 'inception', 'deepseek', 'openrouter'];
const PROVIDER_LABELS: Record<string, string> = {
  local: 'Local',
  xai: 'xAI',
  inception: 'Inception',
  deepseek: 'DeepSeek',
  openrouter: 'OpenRouter',
};

const extractErrorMessage = (payload: unknown, fallback: string): string => {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const body = payload as { error?: { message?: unknown } };
  if (body.error && typeof body.error.message === 'string' && body.error.message.trim()) {
    return body.error.message;
  }
  return fallback;
};

const isToolKey = (value: unknown): value is ToolKey =>
  value === 'fs'
  || value === 'shell'
  || value === 'web'
  || value === 'project'
  || value === 'img'
  || value === 'tts'
  || value === 'stt';

const isPolicyProfile = (value: unknown): value is PolicyProfile =>
  value === 'sandbox' || value === 'index' || value === 'yolo';

const buildSafetyPreset = (): Record<ToolKey, boolean> => ({
  fs: true,
  shell: false,
  web: false,
  project: false,
  img: false,
  tts: false,
  stt: false,
});

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

const parseSecurityPayload = (payload: unknown): SessionSecurityState => {
  const defaults: SessionSecurityState = {
    toolsState: { ...DEFAULT_TOOLS_STATE },
    policyProfile: 'sandbox',
  };
  if (!payload || typeof payload !== 'object') {
    return defaults;
  }
  const toolsStateRaw = (payload as { tools_state?: unknown }).tools_state;
  const policyRaw = (payload as { policy?: unknown }).policy;
  const nextToolsState = { ...DEFAULT_TOOLS_STATE };
  if (toolsStateRaw && typeof toolsStateRaw === 'object') {
    for (const [key, value] of Object.entries(toolsStateRaw as Record<string, unknown>)) {
      if (isToolKey(key) && typeof value === 'boolean') {
        nextToolsState[key] = value;
      }
    }
  }

  let policyProfile: PolicyProfile = defaults.policyProfile;
  if (policyRaw && typeof policyRaw === 'object') {
    const profileRaw = (policyRaw as { profile?: unknown }).profile;
    if (isPolicyProfile(profileRaw)) {
      policyProfile = profileRaw;
    }
  }

  return {
    toolsState: nextToolsState,
    policyProfile,
  };
};

export function SessionDrawer({
  isOpen,
  onClose,
  onSaved,
  sessionId,
  sessionHeader,
  mode,
  modeTransitions = null,
  modeBusy = false,
  onChangeMode,
  modelLabel,
  providerModels,
  selectedModelValue,
  modelsLoading = false,
  savingModel = false,
  onLoadProviderModels,
  onStartLocalOllama,
  onSelectModel,
}: SessionDrawerProps) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [toolsState, setToolsState] = useState<Record<ToolKey, boolean>>({ ...DEFAULT_TOOLS_STATE });
  const [policyProfile, setPolicyProfile] = useState<PolicyProfile>('sandbox');
  const [dangerConfirmText, setDangerConfirmText] = useState('');
  const [dangerConfirmed, setDangerConfirmed] = useState(false);
  const [activeProvider, setActiveProvider] = useState<string | null>(null);
  const [loadedProviders, setLoadedProviders] = useState<Set<string>>(() => new Set());
  const [startingOllama, setStartingOllama] = useState(false);

  const panelRef = useRef<HTMLElement | null>(null);
  useFocusTrap(isOpen, panelRef);

  const requestHeaders = sessionId ? { [sessionHeader]: sessionId } : {};
  const safeModeEnabled = policyProfile !== 'yolo';
  const sortedProviders = sortProviders(providerModels);
  const currentProvider =
    sortedProviders.find((item) => item.provider === activeProvider) ?? sortedProviders[0] ?? null;
  const currentProviderName = currentProvider?.provider ?? null;
  const currentProviderLoaded =
    currentProviderName !== null && loadedProviders.has(currentProviderName);
  const currentProviderError = currentProvider?.error ?? null;
  const currentProviderModels = currentProvider?.models ?? [];

  const loadControls = async (): Promise<void> => {
    if (!sessionId) {
      setStatus('Select an active session to edit session controls.');
      setToolsState({ ...DEFAULT_TOOLS_STATE });
      setPolicyProfile('sandbox');
      return;
    }
    setLoading(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/session/security', { headers: requestHeaders });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to load session controls.'));
      }
      const parsed = parseSecurityPayload(payload);
      setToolsState(parsed.toolsState);
      setPolicyProfile(parsed.policyProfile);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load session controls.';
      setStatus(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setDangerConfirmText('');
    setDangerConfirmed(false);
    setActiveProvider((current) => current ?? sortedProviders[0]?.provider ?? null);
    void loadControls();
  }, [isOpen, sessionId]);

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

  const handleToolToggle = (tool: ToolKey) => {
    setToolsState((current) => ({ ...current, [tool]: !current[tool] }));
  };

  const handleResetPreset = () => {
    setPolicyProfile('sandbox');
    setToolsState(buildSafetyPreset());
    setDangerConfirmText('');
    setDangerConfirmed(false);
    setStatus('Safety preset restored for the current session.');
  };

  const handleSave = async () => {
    if (!sessionId || saving) {
      return;
    }
    const wantsDangerousMode = policyProfile === 'yolo';
    if (wantsDangerousMode && dangerConfirmText.trim().toUpperCase() !== DANGER_CONFIRMATION_PHRASE) {
      setStatus(`Type "${DANGER_CONFIRMATION_PHRASE}" to confirm unrestricted access.`);
      return;
    }
    if (wantsDangerousMode && !dangerConfirmed) {
      setStatus('Confirm that you understand this session will run with reduced safeguards.');
      return;
    }
    setSaving(true);
    setStatus(null);
    try {
      const response = await fetch('/ui/api/session/security', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestHeaders,
        },
        body: JSON.stringify({
          policy: {
            profile: policyProfile,
            yolo_confirm: wantsDangerousMode,
            yolo_confirm_text: wantsDangerousMode ? DANGER_CONFIRMATION_PHRASE : '',
          },
          tools: {
            state: toolsState,
          },
        }),
      });
      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, 'Failed to save session controls.'));
      }
      const parsed = parseSecurityPayload(payload);
      setToolsState(parsed.toolsState);
      setPolicyProfile(parsed.policyProfile);
      setStatus('Session controls updated.');
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save session controls.';
      setStatus(message);
    } finally {
      setSaving(false);
    }
  };

  const handleOpenProvider = async (provider: string) => {
    setActiveProvider(provider);
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

  return (
    <AnimatePresence>
      {isOpen ? (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/55"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            ref={panelRef}
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[520px] flex-col border-l border-zinc-800 bg-zinc-950 shadow-2xl"
            role="dialog"
            aria-modal="true"
            aria-label="Session controls"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 240 }}
          >
            <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
              <div>
                <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-zinc-500">
                  <Shield className="h-3.5 w-3.5" />
                  Current session
                </div>
                <h2 className="mt-1 text-sm font-semibold text-zinc-100">Session Controls</h2>
                <p className="mt-1 text-xs text-zinc-400">
                  Model, mode, safety level, and tool access for the active session.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
                aria-label="Close session controls"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex-1 space-y-6 overflow-y-auto px-5 py-5" data-scrollbar="auto">
              <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="space-y-1">
                  <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Session</div>
                  <div className="text-sm font-medium text-zinc-100">{modelLabel}</div>
                </div>

                <div className="space-y-2">
                  <div className="text-xs font-medium text-zinc-300">Mode</div>
                  <div className="grid grid-cols-5 gap-2">
                    {SESSION_MODE_VALUES.map((item) => {
                        const transition = modeTransitions?.targets[item] ?? null;
                        const blockedReason =
                          transition && !transition.allowed
                            ? transition.message ?? transition.reasonCode ?? 'blocked'
                            : null;
                        const buttonTitle =
                          blockedReason
                            ?? (transition?.requiresConfirm ? 'Для перехода понадобится confirm.' : null)
                            ?? undefined;
                        return (
                          <button
                            key={item}
                            type="button"
                            title={buttonTitle}
                            onClick={() => {
                              void onChangeMode(item);
                            }}
                            disabled={
                              modeBusy || !sessionId || !transition || !transition.allowed
                            }
                            className={`rounded-md border px-2 py-2 text-[11px] uppercase tracking-wide ${
                              mode === item
                                ? 'border-zinc-700 bg-zinc-800 text-zinc-100'
                                : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:bg-zinc-800'
                            } disabled:opacity-50`}
                          >
                            {SESSION_MODE_LABELS[item]}
                          </button>
                        );
                    })}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="text-xs font-medium text-zinc-300">Model</div>
                  <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-2">
                    <div className="mb-2 text-[11px] text-zinc-500">{modelLabel}</div>
                    <div className="grid grid-cols-[132px,1fr] gap-2">
                      <div className="space-y-1">
                        {sortedProviders.map((provider) => {
                          const isActive = provider.provider === currentProviderName;
                          const label = PROVIDER_LABELS[provider.provider] ?? provider.provider;
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
                        {currentProviderName === 'local' && currentProviderModels.length === 0 ? (
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
                        && currentProviderModels.length === 0
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
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="space-y-1">
                  <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Safety</div>
                  <div className="text-sm font-medium text-zinc-100">Session safety level</div>
                  <div className="text-xs text-zinc-400">
                    Applies only to the current session. Safe mode is derived from the selected profile.
                  </div>
                </div>

                <div>
                  {POLICY_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setPolicyProfile(option.value)}
                      className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                        policyProfile === option.value
                          ? 'bg-zinc-800/70'
                          : 'hover:bg-zinc-900'
                      }`}
                    >
                      <span
                        className={`h-3.5 w-3.5 shrink-0 rounded-full border ${
                          policyProfile === option.value
                            ? 'border-zinc-200 bg-zinc-200'
                            : 'border-zinc-600'
                        }`}
                      />
                      <span className="min-w-0">
                        <span
                          className={`block text-[13px] ${
                            policyProfile === option.value ? 'text-zinc-100' : 'text-zinc-300'
                          }`}
                        >
                          {option.title}
                        </span>
                        <span className="block text-xs text-zinc-500">{option.description}</span>
                      </span>
                    </button>
                  ))}
                </div>

                {policyProfile === 'yolo' ? (
                  <div className="space-y-3 rounded-lg border border-amber-700/50 bg-amber-950/20 p-3">
                    <div className="text-sm font-medium text-amber-200">Danger zone</div>
                    <div className="text-xs text-amber-100/80">
                      This session will run with reduced safeguards. Type <span className="font-mono">YOLO</span> and confirm to continue.
                    </div>
                    <input
                      value={dangerConfirmText}
                      onChange={(event) => setDangerConfirmText(event.target.value)}
                      placeholder="YOLO"
                      className="w-full rounded-md border border-amber-700/50 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none"
                    />
                    <label className="flex items-start gap-2 text-xs text-amber-100/80">
                      <input
                        type="checkbox"
                        checked={dangerConfirmed}
                        onChange={(event) => setDangerConfirmed(event.target.checked)}
                        className="mt-0.5"
                      />
                      <span>I understand that this session may execute with fewer safety safeguards.</span>
                    </label>
                  </div>
                ) : null}
              </section>

              <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Tool access</div>
                    <div className="mt-1 text-sm font-medium text-zinc-100">Current session only</div>
                    <div className="mt-1 text-xs text-zinc-400">
                      Safe mode is derived from the selected session profile. When it is on, risky tools stay disabled even if their toggle is on.
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={handleResetPreset}
                    className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-900"
                  >
                    Reset to safety preset
                  </button>
                </div>

                <div className="text-xs text-zinc-400">
                  Safe mode:{' '}
                  <span className={safeModeEnabled ? 'text-amber-300' : 'text-emerald-300'}>
                    {safeModeEnabled ? 'ON' : 'OFF'}
                  </span>
                  <span className="text-zinc-500">
                    {' '}
                    — {safeModeEnabled ? 'risky tools remain blocked' : 'YOLO profile'}
                  </span>
                </div>

                <div className="space-y-2">
                  {(['fs', 'project', 'shell', 'web', 'img', 'tts', 'stt'] as ToolKey[]).map((tool) => {
                    const blockedBySafeMode = safeModeEnabled && SAFE_MODE_BLOCKED_TOOLS.has(tool);
                    const checked = Boolean(toolsState[tool]);
                    return (
                      <div
                        key={tool}
                        className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-3"
                      >
                        <div>
                          <div className="text-sm font-medium text-zinc-100">{TOOL_LABELS[tool]}</div>
                          <div className="text-xs text-zinc-400">
                            {blockedBySafeMode
                              ? 'Disabled by safe mode.'
                              : checked
                                ? 'Enabled for the current session.'
                                : 'Disabled for the current session.'}
                          </div>
                        </div>
                        <ToggleSwitch
                          checked={blockedBySafeMode ? false : checked}
                          disabled={saving || loading || !sessionId || blockedBySafeMode}
                          label={TOOL_LABELS[tool]}
                          onToggle={() => handleToolToggle(tool)}
                        />
                      </div>
                    );
                  })}
                </div>
              </section>
            </div>

            <div className="border-t border-zinc-800 px-5 py-4">
              {status ? (
                <div className="mb-3 rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-300">
                  {status}
                </div>
              ) : null}
              <div className="flex items-center justify-between">
                <div className="text-xs text-zinc-500">
                  {loading
                    ? 'Loading session controls...'
                    : policyProfile === 'yolo'
                      ? 'Unrestricted session profile selected.'
                      : 'Session controls ready.'}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={onClose}
                    className="rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-900"
                  >
                    Close
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void handleSave();
                    }}
                    disabled={saving || loading || !sessionId}
                    className="rounded-md border border-zinc-700 bg-zinc-100 px-3 py-2 text-sm font-medium text-zinc-950 hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {saving ? 'Saving...' : 'Save session controls'}
                  </button>
                </div>
              </div>
            </div>
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  );
}
