import { loadLastSessionId, saveLastSessionId } from '../session-storage';
import { useState } from 'react';
import { useBrowserSubject } from '../browser-subject';
import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import AuthGate from '../AuthGate';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('AuthGate', () => {
  it('renders the application after cookie auth is confirmed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ authenticated: true, auth_required: true }), {
          status: 200,
        }),
      ),
    );

    const view = render(
      <AuthGate>
        <div>application-ready</div>
      </AuthGate>,
    );

    await vi.waitFor(() => expect(view.container.textContent).toContain('application-ready'));
  });

  it('logs in with the token and never stores it in frontend state after success', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ authenticated: false, auth_required: true }), {
          status: 200,
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ authenticated: true }), { status: 200 }),
      )
      .mockResolvedValueOnce(new Response(JSON.stringify({ authenticated: true, auth_required: true, browser_subject: 'a'.repeat(64) }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const view = render(
      <AuthGate>
        <div>application-ready</div>
      </AuthGate>,
    );
    const input = await vi.waitFor(() => view.getByLabelText('API token'));
    fireEvent.change(input, { target: { value: 'secret-token' } });
    const form = input.closest('form');
    if (!form) throw new Error('Auth form not found');
    fireEvent.submit(form);

    await vi.waitFor(() => expect(view.container.textContent).toContain('application-ready'));
    expect(fetchMock).toHaveBeenNthCalledWith(2,
      '/ui/api/auth/login',
      expect.objectContaining({
        credentials: 'same-origin',
        body: JSON.stringify({ token: 'secret-token' }),
      }),
    );
    expect(fetchMock).toHaveBeenLastCalledWith('/ui/api/auth/status', expect.objectContaining({ cache: 'no-store' }));
  });
});

const owner = 'a'.repeat(64), member = 'b'.repeat(64);
function IdentityProbe() {
  const subject = useBrowserSubject();
  const [draft, setDraft] = useState('empty');
  return <><p>{subject ?? 'no-persistence'}</p><input aria-label="Local draft" value={draft} onChange={e => setDraft(e.target.value)} /></>;
}
it('provides a verified subject before child bootstrap and remounts only on a changed subject', async () => {
  let subject = owner;
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ authenticated: true, auth_required: true, browser_subject: subject }))));
  const h = render(<AuthGate><IdentityProbe /></AuthGate>);
  await h.findByText(owner);
  fireEvent.change(h.getByLabelText('Local draft'), { target: { value: 'owner-private' } });
  await act(async () => window.dispatchEvent(new Event('focus')));
  expect((h.getByLabelText('Local draft') as HTMLInputElement).value).toBe('owner-private');
  subject = member;
  await act(async () => window.dispatchEvent(new Event('focus')));
  await h.findByText(member);
  expect((h.getByLabelText('Local draft') as HTMLInputElement).value).toBe('empty');
});

it('ignores a late old identity response after a newer focus check', async () => {
  let resolve!: (response: Response) => void;
  const old = new Promise<Response>(done => { resolve = done; });
  vi.stubGlobal('fetch', vi.fn().mockReturnValueOnce(old).mockResolvedValue(new Response(JSON.stringify({ authenticated: true, browser_subject: member }))));
  const h = render(<AuthGate><IdentityProbe /></AuthGate>);
  fireEvent(window, new Event('focus'));
  await h.findByText(member);
  await act(async () => resolve(new Response(JSON.stringify({ authenticated: true, browser_subject: owner }))));
  expect(h.queryByText(owner)).toBeNull();
  expect(h.getByText(member)).toBeTruthy();
});
it.each([undefined, 'email:private@example.com'])('renders without persistent hints if authenticated metadata has no valid subject', async subject => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ authenticated: true, browser_subject: subject }))));
  const h = render(<AuthGate><IdentityProbe /></AuthGate>);
  await h.findByText('no-persistence');
});
it('does not retarget a captured old-account continuation into a new account namespace', async () => {
  localStorage.clear();
  let subject = owner;
  let finish!: () => void;
  function WriteProbe() {
    const current = useBrowserSubject();
    return <button onClick={() => { finish = () => saveLastSessionId(current, 'late-owner-session'); }}>Capture hint</button>;
  }
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ authenticated: true, browser_subject: subject }))));
  const h = render(<AuthGate><WriteProbe /><IdentityProbe /></AuthGate>);
  await h.findByText(owner);
  fireEvent.click(h.getByText('Capture hint'));
  subject = member;
  await act(async () => window.dispatchEvent(new Event('focus')));
  await h.findByText(member);
  finish();
  expect(loadLastSessionId(member)).toBeNull();
  expect(loadLastSessionId(owner)).toBe('late-owner-session');
  localStorage.clear();
});
it('keeps the verified application and unsaved draft after a transient focus-check failure', async () => {
  const mock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ authenticated: true, browser_subject: owner }))).mockRejectedValueOnce(new Error('Offline'));
  vi.stubGlobal('fetch', mock);
  const h = render(<AuthGate><IdentityProbe /></AuthGate>);
  await h.findByText(owner);
  fireEvent.change(h.getByLabelText('Local draft'), { target: { value: 'unsaved-private' } });
  await act(async () => window.dispatchEvent(new Event('focus')));
  await h.findByText('Не удалось перепроверить доступ.');
  expect((h.getByLabelText('Local draft') as HTMLInputElement).value).toBe('unsaved-private');
  expect(h.getByText(owner)).toBeTruthy();
});
it('removes the private application when a new identity check denies access', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ authenticated: true, browser_subject: owner }))).mockResolvedValueOnce(new Response('{}', { status: 401 })));
  const h = render(<AuthGate><IdentityProbe /></AuthGate>);
  await h.findByText(owner);
  await act(async () => window.dispatchEvent(new Event('focus')));
  expect(h.queryByLabelText('Local draft')).toBeNull();
  expect(h.queryByText(owner)).toBeNull();
  expect(h.queryByLabelText('API token')).toBeNull();
});
it('shows an error if the first identity read is superseded by a failed focus check', async () => {
  let resolve!: (response: Response) => void;
  vi.stubGlobal('fetch', vi.fn().mockImplementationOnce(() => new Promise<Response>(done => { resolve = done; })).mockRejectedValueOnce(new Error('Offline')));
  const h = render(<AuthGate><IdentityProbe /></AuthGate>);
  await act(async () => window.dispatchEvent(new Event('focus')));
  await h.findByText(/Сервер недоступен: Offline/);
  await act(async () => resolve(new Response(JSON.stringify({ authenticated: true, browser_subject: owner }))));
  expect(h.queryByLabelText('Local draft')).toBeNull();
});
