import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
const snapshot = { basis: 'runtime_context_constants_at_refresh', values: { short_term_max_messages: 7, summary_max_messages: 13, summary_message_code_points: 109 } };
const setup = (owner = true, value: unknown = snapshot) => {
  const mock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify({ settings: { can_manage_installation: owner, preferences_subject: owner ? 'owner' : 'member', preferences_etag: 'tag', runtime_context_limits_snapshot: value } })));
  vi.stubGlobal('fetch', mock);
  return mock;
};
const advanced = async () => {
  await screen.findByRole('textbox', { name: 'Custom instructions' });
  fireEvent.click(screen.getByRole('button', { name: 'Memory and data' }));
  fireEvent.click(screen.getByRole('button', { name: 'Advanced memory settings' }));
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('shows runtime context limits only in owner Advanced without edits or writes', async () => {
  const mock = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  expect(screen.queryByRole('region', { name: 'Runtime context limits' })).toBeNull();
  await advanced();
  const region = await screen.findByRole('region', { name: 'Runtime context limits' });
  for (const value of ['7', '13', '109']) expect(within(region).getByText(value)).toBeTruthy();
  expect(within(region).getByText(/Summary content is measured in Unicode code points/)).toBeTruthy();
  expect(region.querySelector('input,textarea,select')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('No changes to save.');
  expect(mock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
});
it('withholds diagnostics from a member even if supplied by a bad response', async () => {
  setup(false);
  render(<Settings isOpen onClose={() => undefined} />);
  await screen.findByRole('button', { name: 'Memory and data' });
  fireEvent.click(screen.getByRole('button', { name: 'Memory and data' }));
  expect(screen.queryByRole('region', { name: 'Runtime context limits' })).toBeNull();
  expect(screen.queryByText('109')).toBeNull();
});
it.each([null, {}, { ...snapshot, basis: 'model_token_limit' }, { ...snapshot, values: { ...snapshot.values, short_term_max_messages: 'SECRET_VALUE' } }, { ...snapshot, values: { ...snapshot.values, summary_max_messages: -1 } }, { ...snapshot, values: { ...snapshot.values, summary_message_code_points: 1.5 } }])('does not invent or echo malformed diagnostic values', async value => {
  setup(true, value);
  render(<Settings isOpen onClose={() => undefined} />);
  await advanced();
  await screen.findByText('Runtime context limits are unavailable.');
  expect(screen.queryByRole('region', { name: 'Runtime context limits' })).toBeNull();
  expect(document.body.textContent).not.toContain('SECRET_');
});
it('discards a late owner response after reopening for a member', async () => {
  const mock = setup(false);
  let resolve!: (response: Response) => void;
  mock.mockImplementationOnce(() => new Promise<Response>(done => { resolve = done; }));
  const h = render(<Settings isOpen onClose={() => undefined} />);
  h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
  h.rerender(<Settings isOpen onClose={() => undefined} />);
  await screen.findByRole('button', { name: 'Memory and data' });
  fireEvent.click(screen.getByRole('button', { name: 'Memory and data' }));
  await act(async () => resolve(new Response(JSON.stringify({ settings: { can_manage_installation: true, preferences_subject: 'old-owner', runtime_context_limits_snapshot: snapshot } }))));
  expect(screen.queryByRole('region', { name: 'Runtime context limits' })).toBeNull();
  expect(screen.queryByText('109')).toBeNull();
});
