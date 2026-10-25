import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { DesktopApprovalsPanel } from '../../features/desktop/desktop-approvals-panel';
import { Settings } from '../components/Settings';
const owner = 'a'.repeat(64), other = 'b'.repeat(64);
const rule = { reference: 'c'.repeat(64), effect: 'allow', source: 'persistent', tool: 'desktop_file_delete', action: 'delete', target_path: '/home/user/Downloads/one.iso', target_kind: 'path', command_restriction_present: true, risk_restriction_present: true };
const view = (subject = owner, rules: unknown[] = [rule]) => ({ basis: 'saved_desktop_rules_at_refresh', preferences_subject: subject, rules });
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('shows the saved safe scope and effect without exposing mutation buttons', async () => {
  const fetch = vi.fn(async (_url: string, _init?: RequestInit) => json(view())); vi.stubGlobal('fetch', fetch);
  render(<DesktopApprovalsPanel subject={owner} />);
  await screen.findByText('/home/user/Downloads/one.iso');
  expect(screen.getByText(/desktop_file_delete.*delete/)).toBeTruthy();
  expect(screen.getByText(/Command restriction present/)).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Remove' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'allow' })).toBeNull();
  expect(fetch.mock.calls.every(([, init]) => !init?.method || init.method === 'GET')).toBe(true);
});
it('does not claim an empty rule list while a read is pending', () => {
  vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})));
  render(<DesktopApprovalsPanel subject={owner} />);
  expect(screen.getByText('Loading saved Desktop rules...')).toBeTruthy();
  expect(screen.queryByText('No saved Desktop rules.')).toBeNull();
});
it('rejects foreign or malformed snapshots without echoing them', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => json(view(other, [{ ...rule, tool: 'SECRET_TOOL' }]))));
  const h = render(<DesktopApprovalsPanel subject={owner} />);
  await screen.findByText('Saved Desktop rules are unavailable.');
  expect(h.container.textContent).not.toContain('SECRET_');
  expect(screen.queryByText('No saved Desktop rules.')).toBeNull();
});
it('drops a late response after changing the captured Settings subject', async () => {
  const old = deferred<Response>(); let count = 0;
  const fetch = vi.fn(async (_url: string, _init?: RequestInit) => ++count === 1 ? old.promise : json(view(other, [{ ...rule, target_path: '/B-only' }]))); vi.stubGlobal('fetch', fetch);
  const h = render(<DesktopApprovalsPanel subject={owner} />);
  h.rerender(<DesktopApprovalsPanel subject={other} />);
  await screen.findByText('/B-only');
  expect(fetch.mock.calls[0][1]?.signal?.aborted).toBe(true);
  await act(async () => { old.resolve(json(view())); });
  expect(screen.queryByText('/home/user/Downloads/one.iso')).toBeNull();
});
it('recovers through explicit read-only Refresh after a failed read', async () => {
  let count = 0; vi.stubGlobal('fetch', vi.fn(async () => ++count === 1 ? json({ error: 'SECRET_ERROR' }, 409) : json(view(owner, []))));
  render(<DesktopApprovalsPanel subject={owner} />);
  await screen.findByText('Saved Desktop rules are unavailable.');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh saved Desktop rules' }));
  await screen.findByText('No saved Desktop rules.');
  expect(screen.queryByText('Saved Desktop rules are unavailable.')).toBeNull();
});
it.each([true, false])('integrates the read view in owner Access only (owner=%s)', async canManage => {
  const fetch = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url === '/ui/api/settings') return json({ settings: { can_manage_installation: canManage, preferences_subject: owner, preferences_etag: 'tag' } });
    if (url === '/ui/api/desktop/approvals?view=settings') return json(view());
    return json({});
  }); vi.stubGlobal('fetch', fetch);
  render(<Settings isOpen onClose={() => undefined} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Access and environment' }));
  if (canManage) {
    fireEvent.click(screen.getByRole('button', { name: 'Show saved Desktop rules' }));
    const section = await screen.findByRole('region', { name: 'Saved Desktop rules' });
    expect(within(section).queryByRole('textbox')).toBeNull();
    await within(section).findByText('/home/user/Downloads/one.iso');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('No changes to save.');
  } else {
    expect(screen.queryByRole('button', { name: 'Show saved Desktop rules' })).toBeNull();
    expect(fetch.mock.calls.some(([url]) => url.includes('/desktop/approvals'))).toBe(false);
  }
  expect(fetch.mock.calls.every(([, init]) => !init?.method || init.method === 'GET')).toBe(true);
});
