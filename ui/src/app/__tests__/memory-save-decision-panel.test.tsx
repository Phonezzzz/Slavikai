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
    fireEvent.change(editor, { target: { value: 'отвечай только по-русски' } });
    fireEvent.click(view.getByRole('button', { name: 'Изменить и сохранить' }));

    await waitFor(() => {
      expect(onRespond).toHaveBeenCalledWith(
        'edit_and_confirm',
        { text: 'отвечай только по-русски' },
      );
    });
  });
});
