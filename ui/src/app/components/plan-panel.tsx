import { useMemo, useState } from 'react';

import type {
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
} from '../types';

type PlanPanelProps = {
  mode: SessionMode;
  plan: PlanEnvelope | null;
  task: TaskExecutionState | null;
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
    <div className="border-b border-[#1f1f24] bg-[#0f0f14] px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          {showModeControls
            ? (['ask', 'plan', 'act'] as SessionMode[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => {
                    void onChangeMode(item);
                  }}
                  disabled={busy || mode === item}
                  className={`rounded-md border px-2 py-1 text-[11px] uppercase tracking-wide ${
                    mode === item
                      ? 'border-[#3a3a46] bg-[#1b1b22] text-[#e0e0e8]'
                      : 'border-[#2a2a31] bg-[#121217] text-[#a4a4ad] hover:bg-[#181820]'
                  } disabled:opacity-50`}
                >
                  {item}
                </button>
              ))
            : null}
        </div>
        <span className="text-[11px] text-[#8a8a94]">{statusText}</span>
      </div>

      {mode === 'plan' ? (
        <div className="mt-2 space-y-2">
          <div className="flex items-center gap-2">
            <input
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="Цель плана (например: исправить streaming в workspace)"
              className="h-8 flex-1 rounded-md border border-[#2a2a31] bg-[#111117] px-2 text-[12px] text-[#d0d0d8] outline-none"
              disabled={busy}
            />
            <button
              type="button"
              onClick={handleDraft}
              disabled={busy || !goal.trim()}
              className="h-8 rounded-md border border-[#2a2a31] bg-[#141418] px-3 text-[12px] text-[#c4c4cd] disabled:opacity-50"
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
              className="h-8 rounded-md border border-[#2a2a31] bg-[#141418] px-3 text-[12px] text-[#c4c4cd] disabled:opacity-50"
            >
              Approve
            </button>
            <button
              type="button"
              onClick={() => {
                void onExecute();
              }}
              disabled={busy || !plan || plan.status !== 'approved'}
              className="h-8 rounded-md border border-[#2a2a31] bg-[#141418] px-3 text-[12px] text-[#c4c4cd] disabled:opacity-50"
            >
              Execute
            </button>
          </div>
        </div>
      ) : null}

      {mode === 'act' ? (
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[11px] text-[#9a9aa4]">
            task: {task ? task.status : 'none'}
          </span>
          <button
            type="button"
            onClick={() => {
              void onCancel();
            }}
            disabled={busy || !task || task.status !== 'running'}
            className="h-8 rounded-md border border-[#2a2a31] bg-[#141418] px-3 text-[12px] text-[#c4c4cd] disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      ) : null}

      {plan?.steps?.length ? (
        <div className="mt-2 max-h-28 overflow-auto rounded-md border border-[#1f1f24] bg-[#0d0d11] p-2 text-[11px] text-[#a5a5ae]" data-scrollbar="always">
          {plan.steps.map((step) => (
            <div key={step.step_id} className="flex items-center justify-between gap-2 py-0.5">
              <span className="truncate">{step.title}</span>
              <span className="shrink-0 text-[#7e7e88]">{step.status}</span>
            </div>
          ))}
        </div>
      ) : null}

      {error ? <div className="mt-2 text-[11px] text-rose-300">{error}</div> : null}
    </div>
  );
}
