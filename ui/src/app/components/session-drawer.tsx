import { AnimatePresence, motion } from 'motion/react';
import { Shield, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { ToggleSwitch } from './ui/toggle-switch';
import { ModelQualification } from './model-qualification';
import { parseSessionSecurityState, type ToolKey, type PolicyProfile } from '../session-security';
import { useFocusTrap } from '../use-focus-trap';
import {
  SESSION_MODE_LABELS,
  type SessionMode,
} from '../types';

type SessionDrawerProps = {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: () => void;
  sessionId: string | null;
  sessionHeader: string;
  mode: SessionMode;
  modelLabel: string;
  selectedModelValue: string | null;
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

const buildSafetyPreset = (): Record<ToolKey, boolean> => ({
  fs: true,
  shell: false,
  web: false,
  project: false,
  img: false,
  tts: false,
  stt: false,
});

export function SessionDrawer({
  isOpen,
  onClose,
  onSaved,
  sessionId,
  sessionHeader,
  mode,
  modelLabel,
  selectedModelValue,
}: SessionDrawerProps) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [toolsState, setToolsState] = useState<Record<ToolKey, boolean>>(buildSafetyPreset());
  const [policyProfile, setPolicyProfile] = useState<PolicyProfile>('sandbox');
  const [dangerConfirmText, setDangerConfirmText] = useState('');
  const [dangerConfirmed, setDangerConfirmed] = useState(false);
  const panelRef = useRef<HTMLElement | null>(null);
  useFocusTrap(isOpen, panelRef);
  const epoch = useRef(0);
  const pendingWrites = useRef(new Map<string, Promise<void>>());
  const [loadedSession, setLoadedSession] = useState<string | null>(null);
  const ready = !!sessionId && loadedSession === `${sessionHeader}:${sessionId}`;
  const requestHeaders = sessionId ? { [sessionHeader]: sessionId } : {};
  const safeModeEnabled = policyProfile !== 'yolo';

  useEffect(() => {
    const generation = ++epoch.current;
    const abort = new AbortController();
    setLoadedSession(null);
    setSaving(false);
    setDangerConfirmText('');
    setDangerConfirmed(false);
    if (!isOpen || !sessionId) {
      setLoading(false);
      setStatus(sessionId ? null : 'Select an active session to edit session controls.');
      return () => { ++epoch.current; abort.abort(); };
    }
    const current = () => epoch.current === generation && !abort.signal.aborted;
    setLoading(true);
    setStatus(null);
    void (async () => {
      try {
        await pendingWrites.current.get(sessionId);
        if (!current()) return;
        const response = await fetch('/ui/api/session/security', { headers: { [sessionHeader]: sessionId }, signal: abort.signal });
        const payload: unknown = await response.json();
        if (!current()) return;
        if (!response.ok) throw new Error(extractErrorMessage(payload, 'Failed to load session controls.'));
        const parsed = parseSessionSecurityState(payload, sessionId);
        if (!parsed) throw new Error('Session controls are unavailable or belong to another session.');
        setToolsState(parsed.toolsState);
        setPolicyProfile(parsed.policyProfile);
        setLoadedSession(`${sessionHeader}:${sessionId}`);
      } catch (error) {
        if (current()) setStatus(error instanceof Error ? error.message : 'Failed to load session controls.');
      } finally {
        if (current()) setLoading(false);
      }
    })();
    return () => { ++epoch.current; abort.abort(); };
  }, [isOpen, sessionId, sessionHeader]);

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
    if (!sessionId || !ready || saving || pendingWrites.current.has(sessionId)) {
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
    const generation = epoch.current;
    const current = () => epoch.current === generation;
    setSaving(true);
    setStatus(null);
    const operation = (async () => {
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
        if (!current()) return;
        const parsed = parseSessionSecurityState(payload, sessionId);
        if (!parsed) {
          setLoadedSession(null);
          throw new Error('The update response is unavailable. Reopen controls to read the current state before another change.');
        }
        setToolsState(parsed.toolsState);
        setPolicyProfile(parsed.policyProfile);
        setStatus('Session controls updated.');
        onSaved?.();
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to save session controls.';
        if (current()) { setLoadedSession(null); setStatus(message); }
      } finally {
        if (current()) setSaving(false);
      }
    })();
    pendingWrites.current.set(sessionId, operation);
    try { await operation; } finally { pendingWrites.current.delete(sessionId); }
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
                  Safety level and tool access for the active session. Model and mode are selected by the message input.
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
                {sessionId && selectedModelValue?.startsWith('custom-') && (
                  <ModelQualification key={`${sessionId}:${selectedModelValue}`} sessionId={sessionId} sessionHeader={sessionHeader} />
                )}

                <p className="text-xs text-zinc-400">{SESSION_MODE_LABELS[mode]} · Режим выбирается у поля сообщения.</p>
              </section>

              {ready ? <fieldset disabled={saving} className="contents">
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
              </fieldset> : <p className="text-sm text-zinc-400">{loading ? 'Loading session access settings...' : 'Session access settings are unavailable. Close and reopen this panel to retry.'}</p>}
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
                    : !ready ? 'Session controls unavailable.' : policyProfile === 'yolo'
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
                    disabled={saving || loading || !ready}
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
