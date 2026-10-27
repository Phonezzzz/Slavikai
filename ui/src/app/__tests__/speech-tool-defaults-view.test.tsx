import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
const snapshot = { basis: 'speech_tool_class_defaults_at_refresh', values: { tts_timeout_seconds: 7, tts_input_code_points: 12345, tts_request_code_points: 678, stt_timeout_seconds: 9 } };
const setup = (owner = true, value: unknown = snapshot) => {
  const mock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify({ settings: { can_manage_installation: owner, preferences_subject: owner ? 'owner' : 'member', preferences_etag: 'tag', speech_tool_defaults_snapshot: value } })));
  vi.stubGlobal('fetch', mock);
  return mock;
};
const developer = async () => {
  await screen.findByRole('textbox', { name: 'Custom instructions' });
  fireEvent.click(screen.getByRole('button', { name: 'Developer view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Developer' }));
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('shows speech tool defaults only in owner Developer without edits or writes', async () => {
  const mock = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  expect(screen.queryByRole('region', { name: 'Speech tool defaults' })).toBeNull();
  await developer();
  const region = await screen.findByRole('region', { name: 'Speech tool defaults' });
  for (const value of ['7', '12345', '678', '9']) expect(within(region).getByText(value)).toBeTruthy();
  expect(within(region).getByText(/Browser dictation/)).toBeTruthy();
  expect(region.querySelector('input,textarea,select')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('No changes to save.');
  expect(mock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
});
it('withholds diagnostics from a member even if supplied by a bad response', async () => {
  setup(false);
  render(<Settings isOpen onClose={() => undefined} />);
  await developer();
  expect(screen.queryByRole('region', { name: 'Speech tool defaults' })).toBeNull();
  expect(screen.queryByText('12345')).toBeNull();
});
it.each([null, {}, { ...snapshot, basis: 'model_token_limit' }, { ...snapshot, values: { ...snapshot.values, tts_timeout_seconds: 'SECRET_VALUE' } }, { ...snapshot, values: { ...snapshot.values, tts_input_code_points: -1 } }, { ...snapshot, values: { ...snapshot.values, tts_request_code_points: 1.5 } }, { ...snapshot, values: { ...snapshot.values, stt_timeout_seconds: null } }])('does not invent or echo malformed diagnostic values', async value => {
  setup(true, value);
  render(<Settings isOpen onClose={() => undefined} />);
  await developer();
  await screen.findByText('Speech tool defaults are unavailable.');
  expect(screen.queryByRole('region', { name: 'Speech tool defaults' })).toBeNull();
  expect(document.body.textContent).not.toContain('SECRET_');
});
it('discards a late owner response after reopening for a member', async () => {
  const mock = setup(false);
  let resolve!: (response: Response) => void;
  mock.mockImplementationOnce(() => new Promise<Response>(done => { resolve = done; }));
  const h = render(<Settings isOpen onClose={() => undefined} />);
  h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
  h.rerender(<Settings isOpen onClose={() => undefined} />);
  await developer();
  await act(async () => resolve(new Response(JSON.stringify({ settings: { can_manage_installation: true, preferences_subject: 'old-owner', speech_tool_defaults_snapshot: snapshot } }))));
  expect(screen.queryByRole('region', { name: 'Speech tool defaults' })).toBeNull();
  expect(screen.queryByText('12345')).toBeNull();
});
