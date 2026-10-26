import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { WorkspaceToolbar } from '../../features/workspace/workspace-toolbar';
import { SessionControlShell as SessionDrawer } from '../components/session-control-shell';
import { loadSessionSecuritySummary, parseSessionSecuritySummary } from '../session-security';
const payload = (id = 'A', profile = 'sandbox') => ({ session_id: id, security_revision: (id === 'A' ? 'a' : 'b').repeat(32), policy: { profile }, tools_state: { fs: true, shell: false, web: false, project: false, img: false, tts: false, stt: false, safe_mode: profile !== 'yolo' } });
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
  expect(new Headers(writes[0].headers).get('If-Match')).toBe('\"' + 'b'.repeat(32) + '\"');
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

const sources = { basis: 'configuration_at_refresh', fields: { fs: 'session_override', shell: 'installation_configuration', web: 'installation_configuration', project: 'installation_configuration', img: 'installation_configuration', tts: 'installation_configuration', stt: 'installation_configuration' } };
it('labels saved tool configuration sources and clears them for unsaved drafts and source-less save responses', async () => {
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => json(init?.method === 'POST' ? payload() : { ...payload(), tool_sources: sources })));
  render(<SessionDrawer {...props} />);
  await screen.findByText('Saved source: chat override.');
  expect(screen.getAllByText('Saved source: installation configuration.')).toHaveLength(6);
  fireEvent.click(screen.getByRole('switch', { name: 'Filesystem access' }));
  expect(screen.queryByText('Saved source: chat override.')).toBeNull();
  expect(screen.getAllByText('Saved source unavailable; reopen controls to refresh.')).toHaveLength(7);
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  await screen.findByText('Session controls updated.');
  expect(screen.queryByText('Saved source: chat override.')).toBeNull();
});
it.each([undefined, { basis: 'configuration_at_refresh', fields: { fs: 'SECRET_FAKE' } }])('does not invent or echo missing/invalid source metadata', async tool_sources => {
  vi.stubGlobal('fetch', vi.fn(async () => json({ ...payload(), tool_sources })));
  const h = render(<SessionDrawer {...props} />);
  await screen.findByText('Session controls ready.');
  expect(screen.getAllByText('Saved source unavailable; reopen controls to refresh.')).toHaveLength(7);
  expect(h.container.textContent).not.toContain('SECRET_');
});

it('clears source labels for a safety preset and rejects late source metadata from another chat', async () => {
  const old = deferred<Response>();
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => new Headers(init?.headers).get('X-Test') === 'A' ? old.promise : json({ ...payload('B'), tool_sources: { ...sources, fields: { ...sources.fields, fs: 'installation_configuration' } } })));
  const h = render(<SessionDrawer {...props} />);
  h.rerender(<SessionDrawer {...props} sessionId="B" />);
  await screen.findByText('Session controls ready.');
  await act(async () => { old.resolve(json({ ...payload(), tool_sources: sources })); });
  expect(screen.queryByText('Saved source: chat override.')).toBeNull();
  expect(screen.getAllByText('Saved source: installation configuration.')).toHaveLength(7);
  fireEvent.click(screen.getByRole('button', { name: 'Reset to safety preset' }));
  expect(screen.getAllByText('Saved source unavailable; reopen controls to refresh.')).toHaveLength(7);
});


it.each([undefined, null, '', 'invalid', 'A'.repeat(32), 17])('cannot edit a snapshot without valid saved revision (%s)', async security_revision => {
  const fetch = vi.fn(async () => json({ ...payload(), security_revision }));
  vi.stubGlobal('fetch', fetch);
  render(<SessionDrawer {...props} />);
  await waitFor(() => expect(screen.queryByText('Loading session controls...')).toBeNull());
  expect((screen.getByRole('button', { name: 'Save session controls' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  expect(fetch).toHaveBeenCalledTimes(1);
});

it('requires a fresh read and another explicit save after a revision conflict', async () => {
  const writes: RequestInit[] = []; let reads = 0;
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') {
      writes.push(init);
      return writes.length === 1 ? json({ error: { code: 'session_security_revision_conflict', message: 'Changed elsewhere' } }, 409) : json({ ...payload(), security_revision: 'c'.repeat(32) });
    }
    reads++;
    return json({ ...payload(), security_revision: (reads === 1 ? 'a' : 'b').repeat(32) });
  }));
  const h = render(<SessionDrawer {...props} />);
  await screen.findByText('Session controls ready.');
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  await waitFor(() => expect((screen.getByRole('button', { name: 'Save session controls' }) as HTMLButtonElement).disabled).toBe(true));
  expect(writes).toHaveLength(1); expect(reads).toBe(1);
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  expect(writes).toHaveLength(1);
  h.rerender(<SessionDrawer {...props} isOpen={false} />);
  h.rerender(<SessionDrawer {...props} />);
  await screen.findByText('Session controls ready.');
  expect(writes).toHaveLength(1); expect(reads).toBe(2);
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  await screen.findByText('Session controls updated.');
  expect(writes).toHaveLength(2);
  expect(new Headers(writes[0].headers).get('If-Match')).toBe('"' + 'a'.repeat(32) + '"');
  expect(new Headers(writes[1].headers).get('If-Match')).toBe('"' + 'b'.repeat(32) + '"');
});

it('preserves the root guard explanation for a non-revision 409', async () => {
  vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => init?.method === 'POST'
    ? json({ error: { code: 'policy_root_incompatible', message: 'Fixture root is incompatible with this policy.' } }, 409)
    : json(payload())));
  render(<SessionDrawer {...props} />);
  await screen.findByText('Session controls ready.');
  fireEvent.click(screen.getByRole('button', { name: 'Save session controls' }));
  await screen.findByText('Fixture root is incompatible with this policy.');
});
