import { useMemo, useState } from 'react';

import { SESSION_MODE_LABELS, SESSION_MODE_VALUES } from '../types';
import { DesktopApprovalsPanel } from '../../features/desktop/desktop-approvals-panel';
import type {
  AutoState,
  ModeTransitionsContract,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
} from '../types';

type PlanPanelProps = {
  mode: SessionMode;
  plan: PlanEnvelope | null;
  task: TaskExecutionState | null;
  autoState?: AutoState | null;
  modeTransitions?: ModeTransitionsContract | null;
  busy: boolean;
  error: string | null;
  showModeControls?: boolean;
  onChangeMode: (mode: SessionMode) => Promise<void> | void;
  onDraft: (goal: string) => Promise<void> | void;
  onApprove: () => Promise<void> | void;
  onExecute: () => Promise<void> | void;
  onCancel: () => Promise<void> | void;
};

export function PlanPanel({
  mode,
  plan,
  task,
  autoState = null,
  modeTransitions = null,
  busy,
  error,
  showModeControls = true,
  onChangeMode,
  onDraft,
  onApprove,
  onExecute,
  onCancel,
}: PlanPanelProps) {
  const [goal, setGoal] = useState('');
  const statusText = useMemo(() => {
    if (plan) {
      return `plan: ${plan.status}`;
    }
    return 'plan: none';
  }, [plan]);

  const handleDraft = () => {
    const nextGoal = goal.trim();
    if (!nextGoal) {
      return;
    }
    void onDraft(nextGoal);
  };

  return (
    <div className="border-b border-zinc-800 bg-zinc-900 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          {showModeControls
            ? SESSION_MODE_VALUES.map((item) => {
                  const transition = modeTransitions?.targets[item] ?? null;
                  const blockedReason =
                    transition && !transition.allowed
                      ? transition.message ?? transition.reasonCode ?? 'blocked'
                      : null;
                  const title =
                    blockedReason
                      ?? (transition?.requiresConfirm ? 'Для перехода понадобится confirm.' : null)
                      ?? undefined;
                  return (
                    <button
                      key={item}
                      type="button"
                      title={title}
                      onClick={() => {
                        void onChangeMode(item);
                      }}
                      disabled={busy || !transition || !transition.allowed}
                      className={`rounded-md border px-2 py-1 text-[11px] uppercase tracking-wide ${
                        mode === item
                          ? 'border-zinc-700 bg-zinc-800 text-zinc-200'
                          : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:bg-zinc-900'
                      } disabled:opacity-50`}
                    >
                      {SESSION_MODE_LABELS[item]}
                    </button>
                  );
                })
            : null}
        </div>
        <span className="text-[11px] text-zinc-400">{statusText}</span>
      </div>

      {mode === 'plan' ? (
        <div className="mt-2 space-y-2">
          <div className="flex items-center gap-2">
            <input
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="Цель плана (например: исправить streaming в Computer)"
              className="h-8 flex-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 text-[12px] text-zinc-300 outline-none"
              disabled={busy}
            />
            <button
              type="button"
              onClick={handleDraft}
              disabled={busy || !goal.trim()}
              className="h-8 rounded-md border border-zinc-800 bg-zinc-900 px-3 text-[12px] text-zinc-300 disabled:opacity-50"
            >
              Draft
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void onApprove();
              }}
              disabled={busy || !plan || plan.status !== 'draft'}
              className="h-8 rounded-md border border-zinc-800 bg-zinc-900 px-3 text-[12px] text-zinc-300 disabled:opacity-50"
            >
              Approve
            </button>
            <button
              type="button"
              onClick={() => {
                void onExecute();
              }}
              disabled={busy || !plan || plan.status !== 'approved'}
              className="h-8 rounded-md border border-zinc-800 bg-zinc-900 px-3 text-[12px] text-zinc-300 disabled:opacity-50"
            >
              Execute
            </button>
          </div>
        </div>
      ) : null}

      {mode === 'act' ? (
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[11px] text-zinc-400">
            task: {task ? task.status : 'none'}
          </span>
          <button
            type="button"
            onClick={() => {
              void onCancel();
            }}
            disabled={busy || !task || task.status !== 'running'}
            className="h-8 rounded-md border border-zinc-800 bg-zinc-900 px-3 text-[12px] text-zinc-300 disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      ) : null}

      {mode === 'auto' ? (
        <div className="mt-2 rounded-md border border-zinc-800 bg-zinc-900 p-2 text-[11px] text-zinc-400">
          <div>run: {autoState?.run_id ?? 'none'}</div>
          <div>status: {autoState?.status ?? 'idle'}</div>
          <div>pool: {autoState?.pool_size ?? '-'}</div>
          <div>
            skill: {autoState?.skill?.skill_id && autoState.skill.version
              ? `${autoState.skill.skill_id}@${autoState.skill.version}`
              : 'none'} · {autoState?.skill?.status ?? 'skipped'}
          </div>
        </div>
      ) : null}

      {mode === 'desktop' ? (
        <div className="mt-2">
          <div className="rounded-md border border-amber-900/60 bg-amber-950/20 p-2 text-[11px] text-amber-200">
            Host execution is active. Destructive and sensitive actions require scoped approval.
          </div>
          <DesktopApprovalsPanel />
        </div>
      ) : null}

      {plan?.steps?.length ? (
        <div className="mt-2 max-h-28 overflow-auto rounded-md border border-zinc-800 bg-zinc-900 p-2 text-[11px] text-zinc-400" data-scrollbar="always">
          {plan.steps.map((step) => (
            <div key={step.step_id} className="flex items-center justify-between gap-2 py-0.5">
              <span className="truncate">{step.title}</span>
              <span className="shrink-0 text-zinc-500">{step.status}</span>
            </div>
          ))}
        </div>
      ) : null}

      {error ? <div className="mt-2 text-[11px] text-rose-300">{error}</div> : null}
    </div>
  );
}
