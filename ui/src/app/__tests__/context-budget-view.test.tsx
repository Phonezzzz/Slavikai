import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
const fields = ['total_chars', 'prefs_max_items', 'prefs_chars', 'legacy_notes_chars', 'feedback_max_items', 'feedback_chars', 'canonical_memory_chars', 'vector_code_chars', 'vector_docs_chars', 'workspace_file_chars', 'pinned_atoms_chars', 'session_summary_chars'];
const values = Object.fromEntries(fields.map((key, index) => [key, 7000 + index]));
const setup = (owner = true, diagnostic: unknown = { values, basis: 'server_configuration_at_refresh', source: 'Memory configuration with server defaults' }) => {
  const snapshot = { settings: { can_manage_installation: owner, preferences_subject: 'subject', preferences_etag: 'tag', context_budget_snapshot: diagnostic } };
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url === '/ui/api/settings') return new Response(JSON.stringify(snapshot));
    throw new Error(url);
  });
  vi.stubGlobal('fetch', fetchMock);
  return { fetchMock, snapshot };
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('shows all configured values read-only in owner Advanced without dirtying or saving them', async () => {
  const { fetchMock } = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  expect(screen.queryByRole('region', { name: 'Configured context budget' })).toBeNull();
  fireEvent.click(await screen.findByRole('button', { name: 'Advanced memory settings' }));
  const region = await screen.findByRole('region', { name: 'Configured context budget' });
  for (const value of Object.values(values)) expect(within(region).getByText(String(value))).toBeTruthy();
  expect(within(region).getByText(/Already running sessions may use an earlier snapshot/)).toBeTruthy();
  expect(region.querySelector('input,textarea,select')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('No changes to save.');
  expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: 'Advanced memory settings' }));
  fireEvent.click(screen.getByRole('button', { name: 'Advanced memory settings' }));
  expect(screen.getByText('7000')).toBeTruthy();
});
it('does not render owner diagnostics for a member even if a malformed server snapshot includes them', async () => {
  setup(false);
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  expect(screen.queryByRole('button', { name: 'Advanced memory settings' })).toBeNull();
  expect(screen.queryByText('7000')).toBeNull();
});
it.each([null, { values: { total_chars: 7000 }, basis: 'server_configuration_at_refresh' }, { values, basis: 'effective_running_agent' }])('does not invent defaults for missing or untrusted metadata', async diagnostic => {
  setup(true, diagnostic);
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Advanced memory settings' }));
  await screen.findByText('Configured context budget is unavailable.');
  expect(screen.queryByText('12000')).toBeNull();
});
it('omits diagnostics when saving an unrelated Memory setting', async () => {
  const { fetchMock } = setup();
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Advanced memory settings' }));
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Inbox size' }), { target: { value: '321' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await screen.findByText('Settings saved.');
  const writes = fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST');
  expect(writes).toHaveLength(1);
  const sent = JSON.parse(writes[0][1]?.body as string);
  expect(sent.memory.inbox_max_items).toBe(321);
  expect(sent.context_budget_snapshot).toBeUndefined();
  expect(sent.memory.context_budget).toBeUndefined();
});
it('discards a stale owner snapshot after closing and reopening for a member', async () => {
  const { fetchMock } = setup(false);
  let resolve!: (response: Response) => void;
  fetchMock.mockImplementationOnce(() => new Promise<Response>(done => { resolve = done; }));
  const h = render(<Settings isOpen onClose={() => undefined} />);
  h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
  h.rerender(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  await act(async () => resolve(new Response(JSON.stringify({ settings: { can_manage_installation: true, preferences_subject: 'old-owner', context_budget_snapshot: { values, source: 'old-owner', basis: 'server_configuration_at_refresh' } } }))));
  expect(screen.queryByRole('button', { name: 'Advanced memory settings' })).toBeNull();
  expect(screen.queryByText('7000')).toBeNull();
});
