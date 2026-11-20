import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render } from '@testing-library/react';

import { DecisionPanel } from '../components/decision-panel';
import { getDecisionDisplayState } from '../decision-display';
import { parseUiDecision } from '../session-payload';
import type { UiDecision } from '../types';

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const userAuthDecision: UiDecision = {
  id: 'user-auth-decision-1',
  kind: 'decision',
  decision_type: 'user_auth',
  status: 'pending',
  blocking: true,
  reason: 'NEEDS_USER_AUTH',
  summary: 'Войди в окне Chromium на ПК → Готово',
  proposed_action: {
    reason: 'NEEDS_USER_AUTH',
    run_phase: 'WAITING_USER_INPUT',
    attempt_expires_at: '2026-11-19T17:00:00Z',
    cancel_at: '2026-11-19T17:05:00Z',
    url: 'https://login.example.test/signin',
  },
  options: [
    { id: 'confirm', title: 'Готово', action: 'confirm', payload: {}, risk: 'low' },
    { id: 'retry', title: 'Повторить вход', action: 'retry', payload: {}, risk: 'low' },
    { id: 'reject', title: 'Отмена', action: 'reject', payload: {}, risk: 'low' },
  ],
  default_option_id: 'confirm',
  context: { source_endpoint: 'chat.tool_continue', run_phase: 'WAITING_USER_INPUT' },
  created_at: '2026-11-19T16:55:00Z',
  updated_at: '2026-11-19T16:55:00Z',
  resolved_at: null,
};

describe('user_auth decision (E02-T04)', () => {
  it('parses an expired user_auth decision instead of dropping it', () => {
    const parsed = parseUiDecision({ ...userAuthDecision, status: 'expired' });
    expect(parsed).not.toBeNull();
    expect(parsed?.status).toBe('expired');
    expect(parsed?.decision_type).toBe('user_auth');
  });

  it('renders the pending handoff with the Done button', () => {
    const state = getDecisionDisplayState(userAuthDecision);
    expect(state.shouldRender).toBe(true);
    expect(state.isPending).toBe(true);
    const view = render(
      <DecisionPanel decision={userAuthDecision} busy={false} error={null} onRespond={vi.fn()} />,
    );
    expect(view.getByRole('button', { name: 'Готово' })).toBeTruthy();
    expect(view.container.textContent).toContain('Нужен ручной вход');
  });

  it('shows run cancellation for an expired user_auth decision without buttons', () => {
    const expired: UiDecision = { ...userAuthDecision, status: 'expired' };
    const state = getDecisionDisplayState(expired);
    expect(state.shouldRender).toBe(true);
    expect(state.isPending).toBe(false);
    expect(state.isBlocking).toBe(false);
    const view = render(
      <DecisionPanel decision={expired} busy={false} error={null} onRespond={vi.fn()} />,
    );
    expect(view.container.textContent).toContain('Вход не выполнен');
    expect(view.container.textContent).toContain('запуск отменён');
    expect(view.queryByRole('button', { name: 'Готово' })).toBeNull();
    expect(view.queryByRole('button', { name: 'Повторить вход' })).toBeNull();
  });
});
