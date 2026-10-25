import { StrictMode } from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';

const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status });
const item = (id = 'one', revision = 'r1') => ({ atom_id: id, revision, stable_key: `fact:${id}`, claim_type: 'fact', value_json: { text: `${id} full value` }, summary_text: `${id} summary`, status: 'active', pinned: true, last_seen_at: '2026-10-25' });
const page = (items = [item()], cursor: string | null = null, owner = 'member') => ({ items, next_cursor: cursor, subject: `memory-${owner}`, preferences_subject: owner });
const preview = (value = item(), owner = 'member') => ({ item: value, subject: `memory-${owner}`, preferences_subject: owner, operation_id: 'opaque-op', expires_at: Date.now() / 1000 + 300, scope: 'canonical_atom_and_atom_vector', retained: 'Server retained warning' });
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  return { promise: new Promise<T>(done => { resolve = done; }), resolve: (value: T) => resolve(value) };
};
const settings = { can_manage_installation: false, preferences_subject: 'member', preferences_etag: 'member-0' };
const setup = (api: (url: string, init?: RequestInit) => Promise<Response>) => {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    if (url === '/ui/api/settings') return json({ settings: { ...settings } });
    if (url === '/ui/api/embeddings/status') return json({ state: 'package_missing' });
    return api(url, init);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
};
const open = async () => {
  fireEvent.click(await screen.findByRole('button', { name: 'Memory and data' }));
  return screen.findByRole('region', { name: 'Your Memory' });
};
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); settings.preferences_subject = 'member'; });

it('reads own Memory only when its tab is open, pages, previews and explicitly confirms the exact item', async () => {
  let deleted = false;
  const fetchMock = setup(async (url) => {
    if (url === '/ui/api/memory/items?limit=50') return json(page(deleted ? [] : [item()], deleted ? null : 'next'));
    if (url.includes('cursor=next')) return json(page([item('two')]));
    if (url.endsWith('/preview')) return json(preview());
    if (url.endsWith('/confirm')) { deleted = true; return json({ status: 'deleted', subject: 'memory-member', preferences_subject: 'member' }); }
    throw new Error(url);
  });
  render(<Settings isOpen onClose={() => undefined} />);
  await screen.findByRole('button', { name: 'Memory and data' });
  expect(fetchMock.mock.calls.some(([url]) => url.includes('/memory/'))).toBe(false);
  const region = await open();
  await within(region).findByText(/one full value/);
  fireEvent.click(within(region).getByRole('button', { name: 'Load more Memory' }));
  await within(region).findByText(/two full value/);
  fireEvent.click(within(region).getAllByRole('button', { name: 'Review deletion' })[0]);
  await within(region).findByRole('button', { name: 'Confirm delete' });
  expect(fetchMock.mock.calls.filter(([url]) => url.endsWith('/confirm'))).toHaveLength(0);
  fireEvent.click(within(region).getByRole('button', { name: 'Cancel deletion' }));
  expect(fetchMock.mock.calls.filter(([url]) => url.endsWith('/confirm'))).toHaveLength(0);
  fireEvent.click(within(region).getAllByRole('button', { name: 'Review deletion' })[0]);
  const confirm = await within(region).findByRole('button', { name: 'Confirm delete' });
  act(() => { confirm.click(); confirm.click(); });
  await within(region).findByText('Memory item deleted.');
  const confirms = fetchMock.mock.calls.filter(([url]) => url.endsWith('/confirm'));
  expect(confirms).toHaveLength(1);
  expect(JSON.parse(confirms[0][1]?.body as string)).toEqual({ operation_id: 'opaque-op', confirm: true });
  expect(screen.queryByText(/one full value/)).toBeNull();
});

it('loads Memory under the real StrictMode lifecycle', async () => {
  setup(async () => json(page()));
  render(<StrictMode><Settings isOpen onClose={() => undefined} /></StrictMode>);
  const region = await open();
  await within(region).findByText(/one full value/);
  expect(within(region).queryByText('Loading Memory…')).toBeNull();
});

it('discards late private pages after closing and reopening', async () => {
  const old = deferred<Response>();
  let lists = 0;
  setup(async () => ++lists === 1 ? old.promise : json(page([item('new')])));
  const h = render(<Settings isOpen onClose={() => undefined} />);
  await open();
  await screen.findByText('Loading Memory…');
  h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
  h.rerender(<Settings isOpen onClose={() => undefined} />);
  await open();
  await screen.findByText(/new full value/);
  await act(async () => old.resolve(json(page([item('old-private')]))));
  expect(screen.queryByText(/old-private/)).toBeNull();
});

it('rejects an initial list belonging to a different Settings account', async () => {
  setup(async () => json(page([item('foreign')], null, 'other')));
  render(<Settings isOpen onClose={() => undefined} />);
  await open();
  await screen.findByText('Account changed. Refresh Settings before viewing Memory.');
  expect(screen.queryByText(/foreign full value/)).toBeNull();
  expect(screen.queryByRole('button', { name: 'Review deletion' })).toBeNull();
});

it('rejects foreign pages and previews without making a destructive request', async () => {
  let foreignPage = true;
  const mock = setup(async url => {
    if (url.endsWith('/preview')) return json(preview(item(), 'other'));
    if (url.includes('cursor=')) return json(page([item('foreign')], null, 'other'));
    return json(page([item()], foreignPage ? 'cursor' : null));
  });
  render(<Settings isOpen onClose={() => undefined} />);
  await open();
  fireEvent.click(await screen.findByRole('button', { name: 'Load more Memory' }));
  await screen.findByText('Account changed. Refresh Settings before viewing Memory.');
  expect(screen.queryByText(/one full value/)).toBeNull();
  expect(screen.queryByText(/foreign full value/)).toBeNull();
  foreignPage = false;
  fireEvent.click(screen.getByRole('button', { name: 'Reload Memory' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Review deletion' }));
  await screen.findByText('Account changed. Refresh Settings before viewing Memory.');
  expect(screen.queryByRole('button', { name: 'Confirm delete' })).toBeNull();
  expect(mock.mock.calls.some(([url]) => url.endsWith('/confirm'))).toBe(false);
});

it.each(['item', 'revision', 'scope', 'expired'])('rejects an invalid %s preview', async kind => {
  const value = preview();
  if (kind === 'item') value.item.atom_id = 'wrong';
  if (kind === 'revision') value.item.revision = 'new';
  if (kind === 'scope') value.scope = 'bulk';
  if (kind === 'expired') value.expires_at = 1;
  setup(async url => json(url.endsWith('/preview') ? value : page()));
  render(<Settings isOpen onClose={() => undefined} />);
  await open();
  fireEvent.click(await screen.findByRole('button', { name: 'Review deletion' }));
  await screen.findByText('Deletion preview is unavailable or expired. Reload and review again.');
  expect(screen.queryByRole('button', { name: 'Confirm delete' })).toBeNull();
});

it('waits for a pending confirmation before reading Memory in a reopened panel', async () => {
  const done = deferred<Response>();
  let deleted = false;
  let lists = 0;
  setup(async url => {
    if (url.endsWith('/confirm')) { const response = await done.promise; deleted = true; return response; }
    if (url.endsWith('/preview')) return json(preview());
    lists += 1;
    return json(page(deleted ? [] : [item()]));
  });
  const h = render(<Settings isOpen onClose={() => undefined} />);
  await open();
  fireEvent.click(await screen.findByRole('button', { name: 'Review deletion' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Confirm delete' }));
  h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
  h.rerender(<Settings isOpen onClose={() => undefined} />);
  await open();
  expect(lists).toBe(1);
  await act(async () => done.resolve(json({ status: 'deleted', subject: 'memory-member', preferences_subject: 'member' })));
  await screen.findByText('No saved Memory items.');
  expect(lists).toBe(2);
  expect(screen.queryByText('Memory item deleted.')).toBeNull();
});

it('clears old Memory and a pending preview when Settings identity changes', async () => {
  const old = deferred<Response>();
  setup(async url => url.endsWith('/preview') ? old.promise : json(page([item(settings.preferences_subject)], null, settings.preferences_subject)));
  render(<Settings isOpen onClose={() => undefined} />);
  await open();
  fireEvent.click(await screen.findByRole('button', { name: 'Review deletion' }));
  settings.preferences_subject = 'new-account';
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await screen.findByText(/new-account full value/);
  await act(async () => old.resolve(json(preview(item('member')))));
  expect(screen.queryByText(/member full value/)).toBeNull();
  expect(screen.queryByRole('button', { name: 'Confirm delete' })).toBeNull();
});

it.each([409, 410, 503])('keeps honest failure feedback after confirm HTTP %s with no automatic retry', async code => {
  const mock = setup(async url => url.endsWith('/preview') ? json(preview()) : url.endsWith('/confirm') ? json({}, code) : json(page()));
  render(<Settings isOpen onClose={() => undefined} />);
  await open();
  fireEvent.click(await screen.findByRole('button', { name: 'Review deletion' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Confirm delete' }));
  await screen.findByText(/A failed connection does not prove that deletion was cancelled/);
  expect(screen.queryByText('Memory item deleted.')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Confirm delete' })).toBeNull();
  expect(mock.mock.calls.filter(([url]) => url.endsWith('/confirm'))).toHaveLength(1);
  expect(screen.getByText(/one full value/)).toBeTruthy();
});
