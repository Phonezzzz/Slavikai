import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { SessionDrawer } from '../components/session-drawer';
const props = { isOpen: true, onClose: () => undefined, sessionId: 'A', sessionHeader: 'X-Test', mode: 'ask' as const, modelLabel: 'Selected', selectedModelValue: null };
const values = { temperature: 0.23, top_p: null, max_tokens: 2345, mode: 'default', thinking_enabled: false, reasoning_effort: null, reasoning_summary: null, reasoning_summary_wait: null, diffusing: null, web_search_enabled: true };
const paused = { session_id: 'A', decision_id: 'D', basis: 'paused_approval_model_request_at_refresh', state: 'available', identity: { provider: 'local', model: 'Captured' }, values, unavailable_fields: [] };
const json = (data: unknown) => new Response(JSON.stringify(data));
const setup = (value: unknown = paused) => {
  const fetch = vi.fn(async (url: string, _init?: RequestInit) => url.endsWith('/paused-model-request') ? json(value) : json({ session_id: 'A', security_revision: 'a'.repeat(32), tools_state: { fs: true, shell: false, web: false, project: false, img: false, tts: false, stt: false, safe_mode: true }, policy: { profile: 'sandbox' } }));
  vi.stubGlobal('fetch', fetch); return fetch;
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('reads paused options only on explicit request and sends no mutation', async () => {
  const fetch = setup(); render(<SessionDrawer {...props} />);
  await screen.findByText('Session controls ready.');
  expect(fetch.mock.calls.some(([url]) => url.endsWith('/paused-model-request'))).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: 'Show paused model request' }));
  await screen.findByRole('region', { name: 'Paused model request' });
  expect(screen.getByText('local / Captured')).toBeTruthy();
  expect(screen.getByText('2345')).toBeTruthy();
  expect(screen.getByText(/not a provider acceptance/)).toBeTruthy();
  expect(fetch.mock.calls.every(([, init]) => !init?.method || init.method === 'GET')).toBe(true);
});
it.each([null, { ...paused, session_id: 'B' }, { ...paused, basis: 'loaded_default_model_at_refresh' }, { ...paused, state: 'unavailable' }, { ...paused, decision_id: null }, { ...paused, values: null }])('keeps missing or foreign snapshots unavailable', async value => {
  setup(value); render(<SessionDrawer {...props} />);
  fireEvent.click(screen.getByRole('button', { name: 'Show paused model request' }));
  await screen.findByText('Paused model request is unavailable.');
  expect(screen.queryByText('local / Captured')).toBeNull();
});
it.each(['close', 'session', 'header'])('discards a read after panel scope changes: %s', async transition => {
  const fetch = setup(); let resolve!: (response: Response) => void;
  fetch.mockImplementation((url: string) => url.endsWith('/paused-model-request') ? new Promise<Response>(done => { resolve = done; }) : Promise.resolve(json({})));
  const h = render(<SessionDrawer {...props} />);
  fireEvent.click(screen.getByRole('button', { name: 'Show paused model request' }));
  if (transition === 'close') {
    h.rerender(<SessionDrawer {...props} isOpen={false} />);
    h.rerender(<SessionDrawer {...props} />);
  } else if (transition === 'session') h.rerender(<SessionDrawer {...props} sessionId="B" />);
  else h.rerender(<SessionDrawer {...props} sessionHeader="X-Other" />);
  await act(async () => resolve(json(paused)));
  expect(screen.queryByText('local / Captured')).toBeNull();
});

it('does not echo malformed option values', async () => {
  setup({ ...paused, values: { ...values, temperature: 'SECRET_VALUE', reasoning_effort: 'SECRET_VALUE' } });
  render(<SessionDrawer {...props} />);
  fireEvent.click(screen.getByRole('button', { name: 'Show paused model request' }));
  await screen.findByRole('region', { name: 'Paused model request' });
  expect(document.body.textContent).not.toContain('SECRET_VALUE');
  expect(screen.getAllByText('Unavailable')).toHaveLength(2);
});
