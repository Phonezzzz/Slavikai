import { useState } from 'react';
import type { ModeTransitionsContract, PlanEnvelope, SessionMode, TaskExecutionState } from '../types';

type PlanPanelProps = {
  mode: SessionMode;
  plan: PlanEnvelope | null;
  task: TaskExecutionState | null;
  modeTransitions: ModeTransitionsContract | null;
  busy: boolean;
  cancelling: boolean;
  blocked: boolean;
  error: string | null;
  onDraft: (goal: string) => Promise<void> | void;
  onApprove: (plan: PlanEnvelope) => Promise<void> | void;
  onExecute: (plan: PlanEnvelope) => Promise<void> | void;
  onCancel: () => Promise<void> | void;
};
const statusLabels = {
  draft: 'Черновик', approved: 'Согласован', running: 'Выполняется', completed: 'Завершён', failed: 'Ошибка', cancelled: 'Отменён',
};
const stepLabels = { todo: 'Ожидает', doing: 'Выполняется', waiting_approval: 'Ждёт подтверждения', blocked: 'Заблокирован', done: 'Готово', failed: 'Ошибка' };
const buttonClass = 'rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-xs text-zinc-200 disabled:opacity-50';

export function PlanPanel({ mode, plan, task, modeTransitions, busy, cancelling, blocked, error, onDraft, onApprove, onExecute, onCancel }: PlanPanelProps) {
  const [goal, setGoal] = useState('');
  const running = plan?.status === 'running' || task?.status === 'running';
  const disabled = busy || blocked || running;
  const canExecute = plan?.status === 'approved' && (mode === 'act' || modeTransitions?.targets.act.allowed === true);
  const lists = plan ? [
    ['Входит в задачу', plan.scope_in], ['Вне задачи', plan.scope_out], ['Допущения', plan.assumptions],
    ['Нужны данные', plan.inputs_needed], ['Условия завершения', plan.exit_criteria],
  ] as const : [];
  return <section aria-label="Карточка плана" className="my-4 rounded-xl border border-zinc-700 bg-zinc-900 p-4 text-sm text-zinc-200">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 className="font-semibold">План задачи</h2>
      {plan ? <span role="status" className="text-xs text-zinc-400">{statusLabels[plan.status]} · Версия {plan.plan_revision}</span> : null}
    </div>
    <p className="mt-2 text-xs text-zinc-400">Подготовка плана не запускает выполнение. Согласуйте показанный план, затем отдельно подтвердите запуск.</p>
    {plan ? <div className="mt-3 space-y-3 break-words">
      <p className="whitespace-pre-wrap">{plan.goal}</p>
      {lists.map(([title, items]) => items.length ? <div key={title}>
        <h3 className="text-xs font-medium text-zinc-400">{title}</h3>
        <ul className="ml-5 list-disc text-xs">{items.map((item, index) => <li key={index}>{item}</li>)}</ul>
      </div> : null)}
      <ol className="space-y-2" aria-label="Шаги плана">{plan.steps.map((step, index) => <li key={step.step_id} className="rounded-lg border border-zinc-800 p-3">
        <div className="flex flex-wrap justify-between gap-2"><span>{index + 1}. {step.title}</span><span className="text-xs text-zinc-400">{stepLabels[step.status]}</span></div>
        <p className="mt-1 whitespace-pre-wrap text-xs text-zinc-400">{step.description}</p>
        {step.allowed_tool_kinds.length ? <p className="mt-1 text-xs text-zinc-400">Действия: {step.allowed_tool_kinds.join(', ')}</p> : null}
        {step.acceptance_checks.length ? <ul className="mt-1 ml-5 list-disc text-xs">{step.acceptance_checks.map((check, i) => <li key={i}>{check}</li>)}</ul> : null}
        {step.details ? <p className="mt-1 whitespace-pre-wrap text-xs text-zinc-400">{step.details}</p> : null}
      </li>)}</ol>
      {plan.audit_log.length ? <details className="text-xs text-zinc-400"><summary>Результаты проверки</summary><pre className="mt-2 whitespace-pre-wrap break-all">{JSON.stringify(plan.audit_log, null, 2)}</pre></details> : null}
    </div> : null}
    {mode === 'plan' && !running ? <form className="mt-4 flex flex-wrap gap-2" onSubmit={(event) => { event.preventDefault(); if (goal.trim() && !disabled) void onDraft(goal.trim()); }}>
      <input aria-label="Цель плана" value={goal} onChange={(event) => setGoal(event.target.value)} disabled={disabled}
        placeholder={plan ? 'Цель нового плана' : 'Что нужно сделать?'} className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs" />
      <button type="submit" className={buttonClass} disabled={disabled || !goal.trim()}>Подготовить план</button>
    </form> : null}
    <div className="mt-3 flex flex-wrap gap-2">
      {plan?.status === 'draft' && mode === 'plan' ? <button type="button" className={buttonClass} disabled={disabled} onClick={() => { void onApprove(plan); }}>Согласовать план</button> : null}
      {plan?.status === 'approved' ? <button type="button" className={buttonClass} title={modeTransitions?.targets.act.message ?? undefined}
        disabled={disabled || !canExecute} onClick={() => { void onExecute(plan); }}>Выполнить план</button> : null}
      {running ? <button type="button" className={buttonClass} disabled={cancelling} onClick={() => { void onCancel(); }}>{cancelling ? 'Останавливаем…' : 'Остановить план'}</button> : null}
    </div>
    {error ? <p role="alert" className="mt-3 text-xs text-rose-300">{error}</p> : null}
  </section>;
}
