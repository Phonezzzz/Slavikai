import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
const warning = /Some installation changes may already be saved/;
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
const snapshot = (etag: string | null = 'tag', ttl = 30) => ({ settings: { can_manage_installation: true, preferences_subject: 'fixture', preferences_etag: etag, personalization: { tone: 'balanced', system_prompt: '' }, providers: [], memory: { inbox_ttl_days: ttl } } });
const setup = (post: (init: RequestInit) => Promise<Response>, etag: string | null = 'tag') => {
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    if (url === '/ui/api/settings') return init?.method === 'POST' ? post(init) : json(snapshot(etag));
    if (url.startsWith('/ui/api/models')) return json({ providers: [] });
    if (url === '/ui/api/embeddings/status') return json({ state: 'package_missing' });
    if (url.startsWith('/ui/api/memory/items')) return json({ items: [], next_cursor: null, subject: 'memory-fixture', preferences_subject: 'fixture' });
    throw new Error(`Unexpected URL ${url}`);
  });
  vi.stubGlobal('fetch', fetch); return fetch;
};
const editMemory = async () => {
  await screen.findByRole('textbox', { name: 'Custom instructions' });
  fireEvent.click(screen.getByRole('button', { name: 'Memory and data' }));
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '31' } });
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it.each(['http', 'transport', 'json', 'ack'])('discloses possible partial installation save after %s failure without retry or clearing draft', async kind => {
  const onSaved = vi.fn();
  const fetch = setup(async () => {
    if (kind === 'transport') throw new TypeError('Network unavailable');
    if (kind === 'json') return new Response('<html>SECRET_BODY</html>');
    if (kind === 'ack') return json({ unexpected: 'SECRET_BODY' });
    return json({ error: { message: 'Model invalid.' } }, 400);
  });
  render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
  await editMemory();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText(warning);
  expect((screen.getByRole('spinbutton') as HTMLInputElement).value).toBe('31');
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
  expect(onSaved).not.toHaveBeenCalled();
  expect(document.body.textContent).not.toContain('SECRET_BODY');
  expect(screen.queryByText('Settings saved.')).toBeNull();
});
it('retains a provider key draft after an unconfirmed installation save', async () => {
  const fetch = setup(async () => json({ error: { message: 'Rejected.' } }, 400));
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
  const key = await screen.findByLabelText('DeepSeek API key');
  fireEvent.change(key, { target: { value: 'fixture-key-draft' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText(warning);
  expect((key as HTMLInputElement).value).toBe('fixture-key-draft');
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
});
it('does not describe a personal-only conflict as an installation save', async () => {
  setup(async () => json({ error: { message: 'Personal conflict.' } }, 409));
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.change(await screen.findByRole('textbox', { name: 'Custom instructions' }), { target: { value: 'private draft' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('Personal conflict.');
  expect(screen.queryByText(warning)).toBeNull();
});
it('does not claim dispatch when local validation blocks the save', async () => {
  const fetch = setup(async () => { throw new Error('Must not POST'); }, null);
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.change(await screen.findByRole('textbox', { name: 'Custom instructions' }), { target: { value: 'private draft' } });
  await editMemory();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('Refresh Settings before saving preferences.');
  expect(screen.queryByText(warning)).toBeNull();
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(0);
});
it('discards the late failure of a closed panel without retrying it', async () => {
  let resolve!: (response: Response) => void;
  const fetch = setup(() => new Promise<Response>(done => { resolve = done; }));
  const h = render(<Settings isOpen onClose={() => undefined} />);
  await editMemory(); fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
  h.rerender(<Settings isOpen onClose={() => undefined} />);
  await act(async () => resolve(json({ error: { message: 'Old failure.' } }, 400)));
  await screen.findByRole('spinbutton');
  expect(screen.queryByText(warning)).toBeNull();
  expect(screen.queryByText('Old failure.')).toBeNull();
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
});
