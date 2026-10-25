import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { WorkspaceToolbar } from '../../features/workspace/workspace-toolbar';
import { SessionControlShell as SessionDrawer } from '../components/session-control-shell';
import { loadSessionSecuritySummary, parseSessionSecuritySummary } from '../session-security';
const payload = (id = 'A', profile = 'sandbox') => ({ session_id: id, policy: { profile }, tools_state: { fs: true, shell: false, web: false, project: false, img: false, tts: false, stt: false, safe_mode: profile !== 'yolo' } });
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; };
const props = { isOpen: true, onClose: () => {}, sessionId: 'A', sessionHeader: 'X-Test', mode: 'ask' as const, modelLabel: 'Fixture', selectedModelValue: null };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it('reports unknown security instead of defaults for malformed policy', () => {
  for (const value of [null, {}, { policy: { profile: 'unexpected' } }, { policy: { profile: 'yolo' }, tools_state: { safe_mode: true } }]) {
    const result = parseSessionSecuritySummary(value);
    expect(result.safeMode).toBeNull();
    expect(result.yoloActive).toBe(false);
    expect(result.policyLabel).toBe('Unavailable');
  }
});
it('rejects a foreign session snapshot even when HTTP succeeded', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => json(payload('B'))));
  await expect(loadSessionSecuritySummary('A', 'X-Test')).rejects.toThrow();
});
it.each([json({}, 200), json({ error: { message: 'Unavailable fixture' } }, 503)])('does not enable controls or save after a failed read', async response => {
  const fetch = vi.fn(async (_url: string, _init?: RequestInit) => response.clone()); vi.stubGlobal('fetch', fetch);
  render(<SessionDrawer {...props} />);
  await waitFor(() => expect(screen.queryByText('Loading session controls...')).toBeNull());
  expect((screen.getByRole('button', { name: 'Save session controls' }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.queryByText('Session controls ready.')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  expect(fetch.mock.calls.every(args => args[1]?.method !== 'POST')).toBe(true);
});
it('drops a late read from another session and only saves the current snapshot', async () => {
  const a = deferred<Response>(); const writes: RequestInit[] = [];
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') { writes.push(init); return json(payload('B', 'index')); }
    return new Headers(init?.headers).get('X-Test') === 'A' ? a.promise : json(payload('B', 'index'));
  }));
  const h = render(<StrictMode><SessionDrawer {...props} /></StrictMode>);
  h.rerender(<StrictMode><SessionDrawer {...props} sessionId="B" /></StrictMode>);
  await screen.findByText('Session controls ready.');
  await act(async () => { a.resolve(json(payload('A', 'yolo'))); });
  expect(screen.queryByText('Danger zone')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  await screen.findByText('Session controls updated.');
  expect(writes).toHaveLength(1);
  expect(new Headers(writes[0].headers).get('X-Test')).toBe('B');
  expect(JSON.parse(writes[0].body as string).policy.profile).toBe('index');
});

it('waits for a pending explicit save before reading a reopened panel without retrying it', async () => {
  const saved = deferred<Response>(); const onSaved = vi.fn(); let gets = 0; let posts = 0;
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') { posts++; return saved.promise; }
    gets++; return json(payload('A'));
  }));
  const h = render(<SessionDrawer {...props} onSaveSettled={onSaved} />);
  await screen.findByText('Session controls ready.');
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  h.rerender(<SessionDrawer {...props} isOpen={false} onSaveSettled={onSaved} />);
  h.rerender(<SessionDrawer {...props} onSaveSettled={onSaved} />);
  expect((screen.getByRole('button', { name: 'Save session controls' }) as HTMLButtonElement).disabled).toBe(true);
  expect(gets).toBe(1);
  await act(async () => { saved.resolve(json(payload('A'))); });
  await screen.findByText('Session controls ready.');
  expect(gets).toBe(2); expect(posts).toBe(1); expect(onSaved).toHaveBeenCalledWith('A');
});
it('keeps the existing two explicit confirmations for an unrestricted profile', async () => {
  const posts: RequestInit[] = [];
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') { posts.push(init); return json(payload('A', 'yolo')); }
    return json(payload('A'));
  }));
  render(<SessionDrawer {...props} />);
  await screen.findByText('Session controls ready.');
  fireEvent.click(screen.getByRole('button', { name: /Unrestricted \(dangerous\)/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  expect(posts).toHaveLength(0);
  fireEvent.change(screen.getByPlaceholderText('YOLO'), { target: { value: 'YOLO' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  expect(posts).toHaveLength(0);
  fireEvent.click(screen.getByRole('checkbox'));
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  await screen.findByText('Session controls updated.');
  expect(posts).toHaveLength(1);
  expect(JSON.parse(posts[0].body as string).policy).toEqual({ profile: 'yolo', yolo_confirm: true, yolo_confirm_text: 'YOLO' });
});

it('renders unknown safe mode explicitly in Computer instead of ON or OFF', () => {
  render(<WorkspaceToolbar modelLabel="Fixture" indexing={false} workspaceRoot="/fixture"
    sessionPolicyLabel="Unavailable" sessionYoloActive={false} sessionSafeMode={null} statusMessage={null}
    onBackToChat={() => {}} onOpenSessionDrawer={() => {}} onOpenProjectPicker={() => {}}
    onReindex={() => {}} onSwitchToFiles={() => {}} onOpenRepositoryPanel={() => {}} />);
  expect(screen.getByText('Session safe mode: Unavailable')).toBeTruthy();
  expect(screen.queryByText('Session safe mode: ON')).toBeNull();
  expect(screen.queryByText('Session safe mode: OFF')).toBeNull();
});
it('does not apply a late save response or callback to the next session', async () => {
  const old = deferred<Response>(); const onSaved = vi.fn(); const writes: string[] = [];
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
    const id = new Headers(init?.headers).get('X-Test') ?? '';
    if (init?.method === 'POST') { writes.push(id); return old.promise; }
    return json(payload(id, id === 'A' ? 'sandbox' : 'index'));
  }));
  const h = render(<SessionDrawer {...props} onSaveSettled={onSaved} />);
  await screen.findByText('Session controls ready.');
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  h.rerender(<SessionDrawer {...props} sessionId="B" onSaveSettled={onSaved} />);
  await screen.findByText('Session controls ready.');
  await act(async () => { old.resolve(json(payload('A', 'yolo'))); });
  expect(screen.queryByText('Danger zone')).toBeNull(); expect(onSaved).toHaveBeenCalledWith('A');
  expect(writes).toEqual(['A']);
});
