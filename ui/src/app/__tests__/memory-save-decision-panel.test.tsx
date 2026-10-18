import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';

import { DecisionPanel } from '../components/decision-panel';
import type { UiDecision } from '../types';

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const memoryDecision: UiDecision = {
  id: 'memory-decision-1',
  kind: 'decision',
  decision_type: 'memory_save',
  status: 'pending',
  blocking: true,
  reason: 'memory_save_confirmation',
  summary: 'Сохранить предложенные изменения Memory (1)?',
  proposed_action: {
    text: 'я предпочитаю короткие ответы',
    claims: [{
      claim_type: 'preference',
      stable_key: 'preference:response_length',
      value_json: { value: 'короткие ответы' },
      confidence: 0.92,
      summary_text: 'Предпочтение: короткие ответы',
    }],
  },
  options: [
    { id: 'confirm', title: 'Сохранить', action: 'confirm', payload: {}, risk: 'medium' },
    {
      id: 'edit_and_confirm',
      title: 'Изменить и сохранить',
      action: 'edit_and_confirm',
      payload: {},
      risk: 'medium',
    },
    { id: 'reject', title: 'Не сохранять', action: 'reject', payload: {}, risk: 'low' },
  ],
  default_option_id: 'reject',
  context: { source_endpoint: 'memory.save' },
  created_at: '2026-08-23T00:00:00Z',
  updated_at: '2026-08-23T00:00:00Z',
  resolved_at: null,
};

describe('DecisionPanel memory save contract', () => {
  it('shows the preview and edits source text instead of editable claims JSON', async () => {
    const onRespond = vi.fn();
    const view = render(
      <DecisionPanel
        decision={memoryDecision}
        busy={false}
        error={null}
        onRespond={onRespond}
      />,
    );

    expect(view.container.textContent).toContain('я предпочитаю короткие ответы');
    expect(view.container.textContent).toContain('Предпочтение: короткие ответы');

    fireEvent.click(view.getByRole('button', { name: 'Изменить и сохранить' }));
    const editor = view.getByRole('textbox') as HTMLTextAreaElement;
    expect(editor.value).toBe('я предпочитаю короткие ответы');
    fireEvent.change(editor, { target: { value: ' \n    отвечай только по-русски\n ' } });
    fireEvent.click(view.getByRole('button', { name: 'Изменить и сохранить' }));

    await waitFor(() => {
      expect(onRespond).toHaveBeenCalledWith(
        'edit_and_confirm',
        { text: ' \n    отвечай только по-русски\n ' },
      );
    });
  });
});

it('requires a separate Memory confirmation after the end-session question', () => {
  const onRespond = vi.fn();
  const first: UiDecision = {
    ...memoryDecision,
    id: 'session-end-1',
    decision_type: 'end_session',
    reason: 'need_user_input',
    summary: 'Сохранить важную информацию перед завершением?',
    proposed_action: {},
    context: { source_endpoint: 'session.end' },
    options: [
      { id: 'yes', title: 'Да', action: 'confirm', payload: {}, risk: 'low' },
      { id: 'no', title: 'Нет', action: 'reject', payload: {}, risk: 'low' },
      { id: 'cancel', title: 'Отмена', action: 'abort', payload: {}, risk: 'low' },
    ],
  };
  const view = render(<DecisionPanel decision={first} busy={false} error={null} onRespond={onRespond} />);
  expect(view.queryByRole('textbox')).toBeNull();
  fireEvent.click(view.getByRole('button', { name: 'Да' }));
  expect(onRespond).toHaveBeenCalledTimes(1);
  expect(onRespond).toHaveBeenLastCalledWith('confirm', null);
  view.rerender(<DecisionPanel decision={{ ...memoryDecision, context: { source_endpoint: 'session.end' } }} busy={false} error={null} onRespond={onRespond} />);
  expect(view.container.textContent).toContain('я предпочитаю короткие ответы');
  expect(onRespond).toHaveBeenCalledTimes(1);
  fireEvent.click(view.getByRole('button', { name: 'Сохранить' }));
  expect(onRespond).toHaveBeenCalledTimes(2);
});
