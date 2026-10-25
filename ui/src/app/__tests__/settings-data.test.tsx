import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';

const settingsUrl = '/ui/api/settings';
const importUrl = '/ui/api/settings/chats/import';
const exportUrl = '/ui/api/settings/chats/export';
const snapshot = (owner = false, memory: unknown = {
  inbox_ttl_days: 41, embeddings: { provider: 'openai', openai_model: 'fixture' },
}) => ({ settings: { can_manage_installation: owner, preferences_subject: 'member', preferences_etag: '"member-0"', memory } });
const json = (value: unknown) => new Response(JSON.stringify(value));
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  return { promise: new Promise<T>(done => { resolve = done; }), resolve: (value: T) => resolve(value) };
};
const file = (name: string, text: Promise<string> = Promise.resolve('{"sessions":[]}')) => {
  const result = new File(['fixture'], name, { type: 'application/json' });
  Object.defineProperty(result, 'text', { value: () => text });
  return result;
};
const input = (container: HTMLElement) => {
  const result = container.querySelector<HTMLInputElement>('input[type="file"]');
  if (!result) throw new Error('Import file input missing');
  return result;
};
const openData = async () => {
  fireEvent.click(await screen.findByRole('button', { name: /^(Data|Memory and data)$/ }));
  await screen.findByRole('button', { name: 'Choose file' });
};
const setup = (extra?: (url: string, init?: RequestInit) => Promise<Response>, value = snapshot()) => {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url === settingsUrl) return json(value);
    if (url === '/ui/api/embeddings/status') return json({ state: 'package_missing' });
    if (extra) return extra(url, init);
    throw new Error(`Unexpected URL: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
};
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('Memory and data settings', () => {
  it('shows actual remote processing and inbox retention to a member without installation controls', async () => {
    const fetchMock = setup();
    render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    expect(screen.getByText('Remote embeddings: OpenAI')).toBeTruthy();
    expect(screen.getByText(/41 days.*inbox entries/)).toBeTruthy();
    expect(screen.getByText(/Sends text for indexing and search queries to OpenAI/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Local' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Export chats' })).toBeTruthy();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
  });

  it('does not claim local processing when the server omitted memory settings', async () => {
    setup(undefined, snapshot(false, null));
    render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    expect(screen.getByText('Memory processing settings are unavailable.')).toBeTruthy();
    expect(screen.queryByText('Local embeddings')).toBeNull();
  });

  it('requires fresh replacement confirmation after file and mode changes and sends one import', async () => {
    const reply = deferred<Response>();
    const fetchMock = setup(async url => { if (url === importUrl) return reply.promise; throw new Error(url); });
    const h = render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    fireEvent.change(input(h.container), { target: { files: [file('first.json')] } });
    await screen.findByText('first.json');
    fireEvent.change(screen.getByRole('combobox', { name: 'Mode' }), { target: { value: 'replace' } });
    expect((screen.getByRole('button', { name: 'Import chats' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('checkbox', { name: /I understand that Replace removes my existing chats/ }));
    fireEvent.change(input(h.container), { target: { files: [file('second.json')] } });
    await screen.findByText('second.json');
    expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(false);
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.change(screen.getByRole('combobox', { name: 'Mode' }), { target: { value: 'merge' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Mode' }), { target: { value: 'replace' } });
    expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(false);
    fireEvent.click(screen.getByRole('checkbox'));
    const button = screen.getByRole('button', { name: 'Import chats' });
    act(() => { button.click(); button.click(); });
    expect(fetchMock.mock.calls.filter(([url]) => url === importUrl)).toHaveLength(1);
    await act(async () => { reply.resolve(json({ imported: 0, mode: 'replace' })); });
    await screen.findByText('Imported 0 sessions.');
  });

  it('ignores an old file read after close and reopen', async () => {
    setup();
    const old = deferred<string>();
    const h = render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    fireEvent.change(input(h.container), { target: { files: [file('private-old.json', old.promise)] } });
    h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
    h.rerender(<Settings isOpen onClose={() => undefined} />);
    await openData();
    await act(async () => { old.resolve('{"sessions":[]}'); });
    expect(screen.queryByText('private-old.json')).toBeNull();
    expect((screen.getByRole('button', { name: 'Import chats' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('keeps the newer file when an earlier read resolves later', async () => {
    setup();
    const old = deferred<string>();
    const h = render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    fireEvent.change(input(h.container), { target: { files: [file('old.json', old.promise)] } });
    fireEvent.change(input(h.container), { target: { files: [file('new.json')] } });
    await screen.findByText('new.json');
    await act(async () => { old.resolve('{"sessions":[]}'); });
    expect(screen.getByText('new.json')).toBeTruthy();
    expect(screen.queryByText('old.json')).toBeNull();
  });

  it('does not download a stale export into a reopened dialog', async () => {
    const reply = deferred<Response>();
    setup(async url => { if (url === exportUrl) return reply.promise; throw new Error(url); });
    const download = vi.fn(() => 'blob:fixture');
    const NativeURL = URL;
    vi.stubGlobal('URL', Object.assign(class extends NativeURL {}, { createObjectURL: download, revokeObjectURL: vi.fn() }));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    const h = render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    fireEvent.click(screen.getByRole('button', { name: 'Export chats' }));
    h.rerender(<Settings isOpen={false} onClose={() => undefined} />);
    h.rerender(<Settings isOpen onClose={() => undefined} />);
    await openData();
    await act(async () => { reply.resolve(json({ sessions: [] })); });
    expect(download).not.toHaveBeenCalled();
    expect(screen.queryByText('Chats export downloaded.')).toBeNull();
  });

  it.each([true, false])('ignores stale import completion after reopening (ok=%s)', async (ok) => {
    const reply = deferred<Response>();
    setup(async url => { if (url === importUrl) return reply.promise; throw new Error(url); });
    const saved = vi.fn();
    const h = render(<Settings isOpen onClose={() => undefined} onSaved={saved} />);
    await openData();
    fireEvent.change(input(h.container), { target: { files: [file('old.json')] } });
    await screen.findByText('old.json');
    fireEvent.click(screen.getByRole('button', { name: 'Import chats' }));
    h.rerender(<Settings isOpen={false} onClose={() => undefined} onSaved={saved} />);
    h.rerender(<Settings isOpen onClose={() => undefined} onSaved={saved} />);
    await openData();
    await act(async () => { reply.resolve(ok ? json({ imported: 0 }) : new Response(JSON.stringify({ error: { message: 'Old import error' } }), { status: 503 })); });
    expect(saved).not.toHaveBeenCalled();
    expect(screen.queryByText('Imported 0 sessions.')).toBeNull();
    expect(screen.queryByText('Old import error')).toBeNull();
    expect(screen.queryByText('old.json')).toBeNull();
    expect((screen.getByRole('button', { name: 'Import chats' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('drops a private file read after a settings identity change without closing', async () => {
    const value = snapshot();
    setup(undefined, value);
    const old = deferred<string>();
    const h = render(<Settings isOpen onClose={() => undefined} />);
    await openData();
    fireEvent.change(input(h.container), { target: { files: [file('old-subject.json', old.promise)] } });
    value.settings.preferences_subject = 'new-subject';
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Refresh' })); });
    await act(async () => { old.resolve('{"sessions":[]}'); });
    expect(screen.queryByText('old-subject.json')).toBeNull();
    expect((screen.getByRole('button', { name: 'Import chats' }) as HTMLButtonElement).disabled).toBe(true);
  });

});
