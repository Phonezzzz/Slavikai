import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { PlanPanel } from '../components/plan-panel';
import { parseModeTransitions, parsePlanEnvelope, parseTaskExecution } from '../session-payload';
import { SESSION_MODE_VALUES } from '../types';

afterEach(cleanup);
const plan = parsePlanEnvelope({ plan_id: 'P', plan_hash: 'H', plan_revision: 1, status: 'draft', goal: 'Проверить проект',
  scope_in: ['Файлы проекта'], scope_out: ['Deploy'], assumptions: ['Есть доступ к проекту'], inputs_needed: [],
  steps: [{ step_id: 'S', title: 'Прочитать файлы', description: 'Только выбранный проект', allowed_tool_kinds: ['read'], acceptance_checks: ['Найден README'], status: 'todo' }],
  exit_criteria: ['Отчёт готов'], created_at: '', updated_at: '' })!;
const transitions = parseModeTransitions({ current_mode: 'plan', targets: Object.fromEntries(SESSION_MODE_VALUES.map(mode => [mode, {
  allowed: mode === 'act', requires_confirm: mode === 'act', reason_code: null, message: null,
}])) });
const props = () => ({ mode: 'plan' as const, plan, task: null, modeTransitions: transitions, busy: false, cancelling: false,
  blocked: false, error: null, onDraft: vi.fn(), onApprove: vi.fn(), onExecute: vi.fn(), onCancel: vi.fn() });

it('shows the actual plan and approves only the displayed envelope without executing', () => {
  const p = props(); render(<PlanPanel {...p} />);
  expect(screen.getByText('Проверить проект')).toBeTruthy();
  expect(screen.getByText('Только выбранный проект')).toBeTruthy(); expect(screen.getByText('Deploy')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Согласовать план' }));
  expect(p.onApprove).toHaveBeenCalledWith(plan); expect(p.onExecute).not.toHaveBeenCalled();
});

it('executes only the displayed approved plan through the existing confirmation path', () => {
  const p = props(); const approved = { ...plan, status: 'approved' as const, plan_revision: 2 };
  render(<PlanPanel {...p} plan={approved} />);
  fireEvent.click(screen.getByRole('button', { name: 'Выполнить план' }));
  expect(p.onExecute).toHaveBeenCalledWith(approved); expect(p.onApprove).not.toHaveBeenCalled();
});

it('drafts a goal explicitly without chat dispatch', () => {
  const p = props(); render(<PlanPanel {...p} plan={null} />);
  fireEvent.change(screen.getByRole('textbox', { name: 'Цель плана' }), { target: { value: '  Новая цель  ' } });
  fireEvent.click(screen.getByRole('button', { name: 'Подготовить план' }));
  expect(p.onDraft).toHaveBeenCalledWith('Новая цель'); expect(p.onExecute).not.toHaveBeenCalled();
});

it('keeps cancellation available while execution or a confirmation request is busy', () => {
  const p = props(); const task = parseTaskExecution({ task_id: 'T', plan_id: 'P', plan_hash: 'H', status: 'running', current_step_id: 'S', started_at: '', updated_at: '' });
  const { rerender } = render(<PlanPanel {...p} mode="act" busy blocked plan={{ ...plan, status: 'running' }} task={task} />);
  fireEvent.click(screen.getByRole('button', { name: 'Остановить план' })); expect(p.onCancel).toHaveBeenCalledTimes(1);
  rerender(<PlanPanel {...p} mode="act" busy cancelling plan={{ ...plan, status: 'running' }} task={task} />);
  expect((screen.getByRole('button', { name: 'Останавливаем…' }) as HTMLButtonElement).disabled).toBe(true);
});

it('blocks approval and execution during a pending decision', () => {
  const p = props(); render(<PlanPanel {...p} blocked plan={{ ...plan, status: 'approved' }} />);
  fireEvent.click(screen.getByRole('button', { name: 'Выполнить план' })); expect(p.onExecute).not.toHaveBeenCalled();
});
