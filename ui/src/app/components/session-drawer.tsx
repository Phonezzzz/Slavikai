import { AnimatePresence, motion } from 'motion/react';
import { ChevronDown, Shield, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { SESSION_MODE_VALUES, type SessionMode } from '../types';

type ToolKey = 'fs' | 'shell' | 'web' | 'project' | 'img' | 'tts' | 'stt' | 'safe_mode';
type PolicyProfile = 'sandbox' | 'index' | 'yolo';

type SessionModelOption = {
  value: string;
  label: string;
  provider: string;
  model: string;
  disabled?: boolean;
};

type SessionDrawerProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modeBusy?: boolean;
  onChangeMode: (mode: SessionMode) => Promise<void>;
  modelLabel: string;
  modelOptions: SessionModelOption[];
  selectedModelValue: string | null;
  modelsLoading?: boolean;
  savingModel?: boolean;
  onSelectModel: (provider: string, model: string) => void;
};

type SessionSecurityState = {
  toolsState: Record<ToolKey, boolean>;
  policyProfile: PolicyProfile;
  yoloArmed: boolean;
  yoloArmedAt: string | null;
};

const DEFAULT_TOOLS_STATE: Record<ToolKey, boolean> = {
  fs: true,
  shell: false,
  web: false,
  project: false,
  img: false,
  tts: false,
  stt: false,
  safe_mode: true,
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
  safe_mode: 'Safe mode',
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
  || value === 'stt'
  || value === 'safe_mode';

const isPolicyProfile = (value: unknown): value is PolicyProfile =>
  value === 'sandbox' || value === 'index' || value === 'yolo';

const normalizeToolsState = (
  source: Record<ToolKey, boolean>,
  safeMode: boolean,
): Record<ToolKey, boolean> => {
  const next = { ...source };
  if (safeMode) {
    for (const key of SAFE_MODE_BLOCKED_TOOLS) {
      next[key] = false;
    }
  }
  next.safe_mode = safeMode;
  return next;
};

const buildSafetyPreset = (): Record<ToolKey, boolean> => ({
  fs: true,
  shell: false,
  web: false,
  project: false,
  img: false,
  tts: false,
  stt: false,
  safe_mode: true,
});

const parseSecurityPayload = (payload: unknown): SessionSecurityState => {
  const defaults: SessionSecurityState = {
    toolsState: { ...DEFAULT_TOOLS_STATE },
    policyProfile: 'sandbox',
    yoloArmed: false,
    yoloArmedAt: null,
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
  let yoloArmed = defaults.yoloArmed;
  let yoloArmedAt = defaults.yoloArmedAt;
  if (policyRaw && typeof policyRaw === 'object') {
    const profileRaw = (policyRaw as { profile?: unknown }).profile;
    const yoloArmedRaw = (policyRaw as { yolo_armed?: unknown }).yolo_armed;
    const yoloArmedAtRaw = (policyRaw as { yolo_armed_at?: unknown }).yolo_armed_at;
    if (isPolicyProfile(profileRaw)) {
      policyProfile = profileRaw;
    }
    if (typeof yoloArmedRaw === 'boolean') {
      yoloArmed = yoloArmedRaw;
    }
    if (typeof yoloArmedAtRaw === 'string' && yoloArmedAtRaw.trim()) {
      yoloArmedAt = yoloArmedAtRaw.trim();
    }
  }

  return {
    toolsState: normalizeToolsState(nextToolsState, nextToolsState.safe_mode),
    policyProfile,
    yoloArmed,
    yoloArmedAt,
  };
};

type ToggleSwitchProps = {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onToggle: () => void;
};

function ToggleSwitch({ checked, disabled = false, label, onToggle }: ToggleSwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onToggle}
      className={`relative inline-flex h-6 w-10 shrink-0 items-center rounded-full p-[2px] transition-colors ${
        checked ? 'bg-emerald-500/75' : 'bg-zinc-700'
      } disabled:cursor-not-allowed disabled:opacity-40`}
    >
      <span
        className={`block h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
          checked ? 'translate-x-4' : 'translate-x-0'
        }`}
      />
    </button>
  );
}

export function SessionDrawer({
  isOpen,
  onClose,
  onSaved,
  sessionId,
  sessionHeader,
  mode,
  modeBusy = false,
  onChangeMode,
  modelLabel,
  modelOptions,
  selectedModelValue,
  modelsLoading = false,
  savingModel = false,
  onSelectModel,
}: SessionDrawerProps) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [toolsState, setToolsState] = useState<Record<ToolKey, boolean>>({ ...DEFAULT_TOOLS_STATE });
  const [policyProfile, setPolicyProfile] = useState<PolicyProfile>('sandbox');
  const [yoloArmed, setYoloArmed] = useState(false);
  const [yoloArmedAt, setYoloArmedAt] = useState<string | null>(null);
  const [dangerConfirmText, setDangerConfirmText] = useState('');
  const [dangerConfirmed, setDangerConfirmed] = useState(false);

  const requestHeaders = sessionId ? { [sessionHeader]: sessionId } : {};
  const safeModeEnabled = toolsState.safe_mode;
  const normalizedToolsState = useMemo(
    () => normalizeToolsState(toolsState, safeModeEnabled),
    [safeModeEnabled, toolsState],
  );

  const loadControls = async (): Promise<void> => {
    if (!sessionId) {
      setStatus('Select an active session to edit session controls.');
      setToolsState({ ...DEFAULT_TOOLS_STATE });
      setPolicyProfile('sandbox');
      setYoloArmed(false);
      setYoloArmedAt(null);
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
      setYoloArmed(parsed.yoloArmed);
      setYoloArmedAt(parsed.yoloArmedAt);
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
    setToolsState((current) => {
      const next = { ...current, [tool]: !current[tool] };
      if (tool === 'safe_mode') {
        return normalizeToolsState(next, !current.safe_mode);
      }
      return next;
    });
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
            yolo_armed: wantsDangerousMode,
            yolo_confirm: wantsDangerousMode,
            yolo_confirm_text: wantsDangerousMode ? DANGER_CONFIRMATION_PHRASE : '',
          },
          tools: {
            state: normalizedToolsState,
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
      setYoloArmed(parsed.yoloArmed);
      setYoloArmedAt(parsed.yoloArmedAt);
      setStatus('Session controls updated.');
      onSaved?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save session controls.';
      setStatus(message);
    } finally {
      setSaving(false);
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
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[520px] flex-col border-l border-zinc-800 bg-zinc-950 shadow-2xl"
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
                  <div className="grid grid-cols-4 gap-2">
                    {SESSION_MODE_VALUES.map((item) => (
                      <button
                        key={item}
                        type="button"
                        onClick={() => {
                          void onChangeMode(item);
                        }}
                        disabled={modeBusy || mode === item || !sessionId}
                        className={`rounded-md border px-2 py-2 text-[11px] uppercase tracking-wide ${
                          mode === item
                            ? 'border-[#3a3a46] bg-[#1b1b22] text-[#e0e0e8]'
                            : 'border-[#2a2a31] bg-[#121217] text-[#a4a4ad] hover:bg-[#181820]'
                        } disabled:opacity-50`}
                      >
                        {item}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="text-xs font-medium text-zinc-300">Model</div>
                  <div className="relative">
                    <select
                      value={selectedModelValue ?? ''}
                      onChange={(event) => {
                        const next = modelOptions.find((option) => option.value === event.target.value);
                        if (next && !next.disabled && next.model.trim()) {
                          onSelectModel(next.provider, next.model);
                        }
                      }}
                      disabled={modelsLoading || savingModel || !sessionId || modelOptions.length === 0}
                      className="w-full appearance-none rounded-md border border-[#252530] bg-[#111116] px-3 py-2 pr-8 text-sm text-[#d4d4db] outline-none"
                    >
                      <option value="" disabled>
                        Select model
                      </option>
                      {modelOptions.map((option) => (
                        <option
                          key={option.value}
                          value={option.value}
                          disabled={option.disabled}
                          className="bg-[#0b0b0d] text-[#ddd]"
                        >
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[#666]" />
                  </div>
                </div>
              </section>

              <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="space-y-1">
                  <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Safety</div>
                  <div className="text-sm font-medium text-zinc-100">Session safety level</div>
                  <div className="text-xs text-zinc-400">
                    Applies only to the current session. Safe mode can still override risky tool access.
                  </div>
                </div>

                <div className="grid gap-2">
                  {POLICY_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setPolicyProfile(option.value)}
                      className={`rounded-lg border px-3 py-3 text-left ${
                        policyProfile === option.value
                          ? 'border-zinc-500 bg-zinc-800 text-zinc-100'
                          : 'border-zinc-800 bg-zinc-950 text-zinc-300 hover:border-zinc-700'
                      }`}
                    >
                      <div className="text-sm font-medium">{option.title}</div>
                      <div className="mt-1 text-xs text-zinc-400">{option.description}</div>
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
                    {yoloArmedAt ? (
                      <div className="text-[11px] text-amber-200/80">Last armed at: {yoloArmedAt}</div>
                    ) : null}
                  </div>
                ) : null}
              </section>

              <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Tool access</div>
                    <div className="mt-1 text-sm font-medium text-zinc-100">Current session only</div>
                    <div className="mt-1 text-xs text-zinc-400">
                      Safe mode is the master switch. When it is on, risky tools stay disabled even if their toggle is on.
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

                <div className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-3">
                  <div>
                    <div className="text-sm font-medium text-zinc-100">Safe mode</div>
                    <div className="text-xs text-zinc-400">
                      Blocks web, shell, project, TTS, and STT for this session.
                    </div>
                  </div>
                  <ToggleSwitch
                    checked={safeModeEnabled}
                    disabled={saving || loading || !sessionId}
                    label="Safe mode"
                    onToggle={() => handleToolToggle('safe_mode')}
                  />
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
                  {loading ? 'Loading session controls...' : yoloArmed ? 'Unrestricted mode armed for current session.' : 'Session controls ready.'}
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
