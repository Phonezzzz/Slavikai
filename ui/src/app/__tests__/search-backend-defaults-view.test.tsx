import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
const snapshot = { basis: 'backend_class_defaults_at_refresh', values: { top_k: 3, timeout_seconds: 7, response_content_bytes: 12345 } };
const setup = (owner = true, value: unknown = snapshot) => {
  const mock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify({ settings: { can_manage_installation: owner, preferences_subject: owner ? 'owner' : 'member', preferences_etag: 'tag', search_backend_defaults_snapshot: value } })));
  vi.stubGlobal('fetch', mock);
  return mock;
};
const developer = async () => {
  await screen.findByRole('textbox', { name: 'Custom instructions' });
  fireEvent.click(screen.getByRole('button', { name: 'Developer view' }));
  fireEvent.click(screen.getByRole('button', { name: 'Developer' }));
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('shows search backend defaults only in owner Developer without edits or writes', async () => {
  const mock = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  expect(screen.queryByRole('region', { name: 'Search backend defaults' })).toBeNull();
  await developer();
  const region = await screen.findByRole('region', { name: 'Search backend defaults' });
  for (const value of ['3', '7', '12345']) expect(within(region).getByText(value)).toBeTruthy();
  expect(within(region).getByText(/Truncation markers/)).toBeTruthy();
  expect(region.querySelector('input,textarea,select')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('No changes to save.');
  expect(mock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
});
it('withholds diagnostics from a member even if supplied by a bad response', async () => {
  setup(false);
  render(<Settings isOpen onClose={() => undefined} />);
  await developer();
  expect(screen.queryByRole('region', { name: 'Search backend defaults' })).toBeNull();
  expect(screen.queryByText('12345')).toBeNull();
});
it.each([null, {}, { ...snapshot, basis: 'model_token_limit' }, { ...snapshot, values: { ...snapshot.values, top_k: 'SECRET_VALUE' } }, { ...snapshot, values: { ...snapshot.values, timeout_seconds: -1 } }, { ...snapshot, values: { ...snapshot.values, response_content_bytes: 1.5 } }])('does not invent or echo malformed diagnostic values', async value => {
  setup(true, value);
  render(<Settings isOpen onClose={() => undefined} />);
  await developer();
  await screen.findByText('Search backend defaults are unavailable.');
  expect(screen.queryByRole('region', { name: 'Search backend defaults' })).toBeNull();
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
  await act(async () => resolve(new Response(JSON.stringify({ settings: { can_manage_installation: true, preferences_subject: 'old-owner', search_backend_defaults_snapshot: snapshot } }))));
  expect(screen.queryByRole('region', { name: 'Search backend defaults' })).toBeNull();
  expect(screen.queryByText('12345')).toBeNull();
});
