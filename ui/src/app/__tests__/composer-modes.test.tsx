import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ComposerModes } from '../components/composer-modes';
import { parseModeTransitions, parseUiDecision } from '../session-payload';
import { SESSION_MODE_VALUES, type ModeTransitionsContract, type SessionMode } from '../types';

afterEach(cleanup);
const transitions = (mode: SessionMode): ModeTransitionsContract => parseModeTransitions({ current_mode: mode,
  targets: Object.fromEntries(SESSION_MODE_VALUES.map((value) => [value, {
    allowed: value !== mode && value !== 'act', reason_code: null, requires_confirm: false, message: null,
  }])),
})!;

it('maps public chat modes to the existing enums without an Act toggle', () => {
  const change = vi.fn(async () => {});
  render(<ComposerModes sessionId="C" mode="ask" transitions={transitions('ask')} busy={false} onChangeMode={change} />);
  fireEvent.click(screen.getByRole('button', { name: 'Агент' }));
  fireEvent.click(screen.getByRole('button', { name: 'План' }));
  expect(change.mock.calls).toEqual([['auto'], ['plan']]);
  expect(screen.queryByRole('button', { name: 'Act' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Обсудить' }).getAttribute('aria-pressed')).toBe('true');
});

it('requires an explicit computer environment and prevents unsupported chat combinations', () => {
  const change = vi.fn(async () => {});
  const { rerender } = render(<ComposerModes sessionId="C" mode="ask" transitions={transitions('ask')} busy={false} onChangeMode={change} />);
  fireEvent.change(screen.getByRole('combobox', { name: 'Рабочая среда' }), { target: { value: 'computer' } });
  expect(change.mock.calls).toEqual([['desktop']]);
  rerender(<ComposerModes sessionId="C" mode="desktop" transitions={transitions('desktop')} busy={false} onChangeMode={change} />);
  expect((screen.getByRole('button', { name: 'План' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.change(screen.getByRole('combobox', { name: 'Рабочая среда' }), { target: { value: 'chat' } });
  expect(change.mock.calls).toEqual([['desktop'], ['ask']]);
});

it.each(['busy', 'unknown', 'no-session'] as const)('disables changes for %s without inventing capability', (state) => {
  const change = vi.fn(async () => {});
  render(<ComposerModes sessionId={state === 'no-session' ? null : 'C'} mode="ask"
    transitions={state === 'unknown' ? null : transitions('ask')} busy={state === 'busy'} onChangeMode={change} />);
  fireEvent.click(screen.getByRole('button', { name: 'Агент' }));
  expect(change).not.toHaveBeenCalled();
  expect((screen.getByRole('combobox') as HTMLSelectElement).disabled).toBe(true);
});

it('keeps server-denied transitions unavailable and explains the reason', () => {
  const rules = transitions('ask');
  rules.targets.plan = { allowed: false, message: 'Модель не поддерживает этот режим.', reasonCode: 'mode_transition_not_allowed', requiresConfirm: false };
  render(<ComposerModes sessionId="C" mode="ask" transitions={rules} busy={false} onChangeMode={vi.fn()} />);
  const button = screen.getByRole('button', { name: 'План' }) as HTMLButtonElement;
  expect(button.disabled).toBe(true);
  expect(button.title).toBe('Модель не поддерживает этот режим.');
});

it('fails closed when the server transition contract is incomplete', () => {
  const raw = { current_mode: 'ask', targets: { ask: { allowed: false, requires_confirm: false } } };
  expect(parseModeTransitions(raw)).toBeNull();
});

it.each(['pending', 'executing', 'resolved'] as const)('honours a blocking %s decision without freezing a resolved action', (status) => {
  const decision = parseUiDecision({ id: 'D', kind: 'approval', status, blocking: true,
    reason: 'scope', summary: 'confirm', created_at: '', updated_at: '' });
  render(<ComposerModes sessionId="C" mode="ask" transitions={transitions('ask')} busy={false}
    decision={decision} onChangeMode={vi.fn()} />);
  expect((screen.getByRole('button', { name: 'Агент' }) as HTMLButtonElement).disabled).toBe(status !== 'resolved');
});
