import { describe, expect, it } from 'vitest';

import { deriveComputerStatus } from '../../features/workspace/computer-status';
import type {
  AutoState,
  PlanEnvelope,
  TaskExecutionState,
  UiDecision,
} from '../types';

const baseDecision: UiDecision = {
  id: 'd1',
  kind: 'decision',
  decision_type: 'tool_approval',
  status: 'pending',
  blocking: true,
  reason: 'approval_required',
  summary: 'Approve file write',
  proposed_action: {},
  options: [],
  default_option_id: null,
  context: {},
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  resolved_at: null,
};

const autoState = (status: AutoState['status']): AutoState => ({
  run_id: 'run-1',
  status,
  goal: 'Refactor auth',
  pool_size: 3,
  started_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:01Z',
  planner: {},
  plan: null,
  coders: [{ coder_id: 'c1', shard_id: 's1', status: 'running', changed_paths: [] }],
  merge: {},
  verifier: null,
  skill: null,
  approval: null,
  error: null,
});

const plan = (status: PlanEnvelope['status']): PlanEnvelope => ({
  plan_id: 'p1',
  plan_hash: 'h',
  plan_revision: 1,
  status,
  goal: 'Plan goal',
  scope_in: [],
  scope_out: [],
  assumptions: [],
  inputs_needed: [],
  audit_log: [],
  steps: [{ step_id: 's1', title: 'Write tests', description: '', allowed_tool_kinds: [], acceptance_checks: [], status: 'doing', details: null }],
  exit_criteria: [],
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  approved_at: null,
  approved_by: null,
});

describe('deriveComputerStatus', () => {
  it('reports idle with no active state', () => {
    const result = deriveComputerStatus({
      mode: 'ask',
      activePlan: null,
      activeTask: null,
      autoState: null,
      decision: null,
    });
    expect(result.status).toBe('idle');
    expect(result.label).toBe('Idle');
    expect(result.isActive).toBe(false);
  });

  it('maps auto coding to running with a goal', () => {
    const result = deriveComputerStatus({
      mode: 'auto',
      activePlan: null,
      activeTask: null,
      autoState: autoState('coding'),
      decision: null,
    });
    expect(result.status).toBe('running');
    expect(result.tone).toBe('active');
    expect(result.goal).toBe('Refactor auth');
  });

  it('maps auto verifying to verifying', () => {
    const result = deriveComputerStatus({
      mode: 'auto',
      activePlan: null,
      activeTask: null,
      autoState: autoState('verifying'),
      decision: null,
    });
    expect(result.status).toBe('verifying');
  });

  it('maps failed verifier to failed with tone error', () => {
    const result = deriveComputerStatus({
      mode: 'auto',
      activePlan: null,
      activeTask: null,
      autoState: autoState('failed_verifier'),
      decision: null,
    });
    expect(result.status).toBe('failed');
    expect(result.tone).toBe('error');
  });

  it('prefers a blocking decision over an active run', () => {
    const result = deriveComputerStatus({
      mode: 'auto',
      activePlan: null,
      activeTask: null,
      autoState: autoState('coding'),
      decision: baseDecision,
    });
    expect(result.status).toBe('waiting_approval');
    expect(result.tone).toBe('waiting');
    expect(result.detail).toBe('Approve file write');
  });

  it('maps a running plan with an active step label', () => {
    const result = deriveComputerStatus({
      mode: 'plan',
      activePlan: plan('running'),
      activeTask: null,
      autoState: null,
      decision: null,
    });
    expect(result.status).toBe('running');
    expect(result.stepLabel).toBe('Step: Write tests');
  });

  it('maps a running task with the current step', () => {
    const task: TaskExecutionState = {
      task_id: 't1',
      plan_id: 'p1',
      plan_hash: 'h',
      current_step_id: 's1',
      status: 'running',
      started_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    };
    const result = deriveComputerStatus({
      mode: 'act',
      activePlan: plan('running'),
      activeTask: task,
      autoState: null,
      decision: null,
    });
    expect(result.status).toBe('running');
    expect(result.stepLabel).toBe('Step: s1');
  });
});
