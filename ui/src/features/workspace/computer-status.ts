import type {
  AutoState,
  PlanEnvelope,
  SessionMode,
  TaskExecutionState,
  UiDecision,
} from '../../app/types';
import type { StatusTone } from '../../app/components/ui/status-badge';

export type ComputerRunStatus =
  | 'idle'
  | 'planning'
  | 'running'
  | 'waiting_approval'
  | 'verifying'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type ComputerStatus = {
  status: ComputerRunStatus;
  label: string;
  tone: StatusTone;
  isActive: boolean;
  goal: string | null;
  stepLabel: string | null;
  detail: string | null;
};

const STATUS_META: Record<
  ComputerRunStatus,
  { label: string; tone: StatusTone; active: boolean }
> = {
  idle: { label: 'Idle', tone: 'neutral', active: false },
  planning: { label: 'Planning', tone: 'active', active: true },
  running: { label: 'Running', tone: 'active', active: true },
  waiting_approval: { label: 'Waiting for approval', tone: 'waiting', active: true },
  verifying: { label: 'Verifying', tone: 'active', active: true },
  completed: { label: 'Completed', tone: 'success', active: false },
  failed: { label: 'Failed', tone: 'error', active: false },
  cancelled: { label: 'Cancelled', tone: 'cancelled', active: false },
};

const AUTO_STATUS_MAP: Record<string, ComputerRunStatus> = {
  planning: 'planning',
  coding: 'running',
  merging: 'running',
  verifying: 'verifying',
  waiting_approval: 'waiting_approval',
  completed: 'completed',
  failed_conflict: 'failed',
  failed_verifier: 'failed',
  failed_worker: 'failed',
  failed_internal: 'failed',
  cancelled: 'cancelled',
};

const TASK_STATUS_MAP: Record<string, ComputerRunStatus> = {
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  cancelled: 'cancelled',
};

const PLAN_STATUS_MAP: Record<string, ComputerRunStatus> = {
  draft: 'planning',
  approved: 'planning',
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  cancelled: 'cancelled',
};

function firstNonEmpty(...values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function autoStepLabel(autoState: AutoState): string | null {
  switch (autoState.status) {
    case 'planning':
      return 'Planning approach';
    case 'coding': {
      const active = autoState.coders.filter(
        (coder) => coder && typeof coder === 'object' && coder.status !== 'completed',
      ).length;
      return active > 0 ? `${active} worker${active === 1 ? '' : 's'} active` : 'Working on task';
    }
    case 'merging':
      return 'Merging changes';
    case 'verifying':
      return 'Verifying result';
    case 'waiting_approval':
      return 'Approval requested';
    case 'completed':
      return 'Run complete';
    case 'cancelled':
      return 'Cancelled';
    default:
      return null;
  }
}

function planStepLabel(activePlan: PlanEnvelope): string | null {
  const activeStep = activePlan.steps.find(
    (step) => step.status === 'doing' || step.status === 'waiting_approval',
  );
  if (activeStep) {
    return `Step: ${activeStep.title || activeStep.step_id}`;
  }
  return activePlan.steps.length > 0 ? `${activePlan.steps.length} step(s) planned` : null;
}

function buildStatus(
  status: ComputerRunStatus,
  goal: string | null,
  stepLabel: string | null,
  detail: string | null,
): ComputerStatus {
  const meta = STATUS_META[status];
  return {
    status,
    label: meta.label,
    tone: meta.tone,
    isActive: meta.active,
    goal,
    stepLabel,
    detail,
  };
}

export function deriveComputerStatus(input: {
  mode: SessionMode;
  activePlan: PlanEnvelope | null;
  activeTask: TaskExecutionState | null;
  autoState: AutoState | null;
  decision: UiDecision | null | undefined;
}): ComputerStatus {
  const { activePlan, activeTask, autoState, decision } = input;

  if (decision && decision.status === 'pending' && decision.blocking) {
    return buildStatus(
      'waiting_approval',
      firstNonEmpty(autoState?.goal, activePlan?.goal),
      firstNonEmpty(
        activePlan ? planStepLabel(activePlan) : null,
        autoState ? autoStepLabel(autoState) : null,
      ),
      decision.summary?.trim() || 'A decision requires your input.',
    );
  }

  if (autoState && autoState.status !== 'idle') {
    return buildStatus(
      AUTO_STATUS_MAP[autoState.status] ?? 'running',
      autoState.goal?.trim() || null,
      autoStepLabel(autoState),
      autoState.error?.trim() || null,
    );
  }

  if (activeTask && TASK_STATUS_MAP[activeTask.status]) {
    const step = activeTask.current_step_id
      ? `Step: ${activeTask.current_step_id}`
      : activePlan
        ? planStepLabel(activePlan)
        : null;
    return buildStatus(
      TASK_STATUS_MAP[activeTask.status],
      activePlan?.goal?.trim() || null,
      step,
      null,
    );
  }

  if (activePlan && PLAN_STATUS_MAP[activePlan.status]) {
    return buildStatus(
      PLAN_STATUS_MAP[activePlan.status],
      activePlan.goal?.trim() || null,
      planStepLabel(activePlan),
      null,
    );
  }

  return buildStatus('idle', null, null, null);
}
