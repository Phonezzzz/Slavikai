import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { EmbeddingNotice } from '../components/embedding-notice';

const statusUrl = '/ui/api/embeddings/status';
const response = (payload: unknown, status = 200) => new Response(JSON.stringify(payload), { status });
const missing = { provider: 'local', model: 'model-a', state: 'missing', can_download: true };

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('Embedding notice', () => {
  it.each([
    { ...missing, state: 'ready', runtime_ready: true },
    { ...missing, provider: 'openai' },
  ])('does not show a missing-component notice for $provider/$state', async (payload) => {
    const fetcher = vi.fn(async () => response(payload));
    vi.stubGlobal('fetch', fetcher);
    render(<EmbeddingNotice />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.queryByRole('complementary')).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('downloads only after the owner explicitly requests it, using the saved model and consent', async () => {
    let downloaded = false;
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === statusUrl) return response(downloaded ? { ...missing, state: 'ready', runtime_ready: true } : missing);
      expect(init?.method).toBe('POST');
      expect(JSON.parse(String(init?.body))).toEqual({ model: 'model-a', confirm: true });
      downloaded = true;
      return response({ model: 'model-a', state: 'ready' });
    });
    vi.stubGlobal('fetch', fetcher);
    render(<EmbeddingNotice />);
    const download = await screen.findByRole('button', { name: 'Download model to server' });
    expect(fetcher).toHaveBeenCalledTimes(1);
    fireEvent.click(download);
    await vi.waitFor(() => expect(screen.queryByRole('complementary')).toBeNull());
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it.each([
    { ...missing, can_download: false },
    { ...missing, can_download: undefined },
    { ...missing, state: 'package_missing', can_download: false },
    { ...missing, state: 'error', can_download: false },
    { ...missing, state: 'unrecognized' },
  ])('fails closed for unavailable or unknown repair capability: $state/$can_download', async (payload) => {
    vi.stubGlobal('fetch', vi.fn(async () => response(payload)));
    render(<EmbeddingNotice />);
    await screen.findByRole('button', { name: 'Recheck semantic search' });
    expect(screen.queryByRole('button', { name: 'Download model to server' })).toBeNull();
    expect(screen.queryByRole('button', { name: /agent|install/i })).toBeNull();
    if (payload.state === 'package_missing') {
      expect(screen.getByText(/make install-beta/)).toBeTruthy();
      expect(screen.getByText(/Chat remains available/)).toBeTruthy();
    }
  });

  it('allows retry after a diagnostic failure without blocking the composer', async () => {
    let succeeds = false;
    vi.stubGlobal('fetch', vi.fn(async () => {
      if (!succeeds) throw new Error('offline');
      return response(missing);
    }));
    render(<><EmbeddingNotice /><input aria-label="Chat message" /></>);
    fireEvent.change(screen.getByLabelText('Chat message'), { target: { value: 'Hello' } });
    await screen.findByText(/offline/);
    succeeds = true;
    fireEvent.click(screen.getByRole('button', { name: 'Recheck semantic search' }));
    await screen.findByRole('button', { name: 'Download model to server' });
    expect(screen.getByDisplayValue('Hello')).toBeTruthy();
  });

  it('ignores a late status after identity/configuration invalidates the component', async () => {
    let resolve!: (value: Response) => void;
    const old = new Promise<Response>((done) => { resolve = done; });
    vi.stubGlobal('fetch', vi.fn().mockReturnValueOnce(old).mockResolvedValue(response({ ...missing, state: 'ready', runtime_ready: true })));
    const view = render(<EmbeddingNotice key="old-principal" />);
    view.rerender(<EmbeddingNotice key="new-principal" />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { resolve(response(missing)); });
    expect(screen.queryByRole('complementary')).toBeNull();
  });

  it('does not keep an owner download action after the server refuses it', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => url === statusUrl
      ? response(missing) : response({ error: { message: 'Owner access required.' } }, 403)));
    render(<EmbeddingNotice />);
    fireEvent.click(await screen.findByRole('button', { name: 'Download model to server' }));
    await screen.findByText(/Owner access required/);
    expect(screen.queryByRole('button', { name: 'Download model to server' })).toBeNull();
  });

  it('can be dismissed without changing server configuration or installing anything', async () => {
    const fetcher = vi.fn(async () => response({ ...missing, state: 'package_missing' }));
    vi.stubGlobal('fetch', fetcher);
    render(<EmbeddingNotice />);
    fireEvent.click(await screen.findByRole('button', { name: 'Dismiss notice' }));
    expect(screen.queryByRole('complementary')).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('keeps Chat usable while status is indefinitely pending', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})));
    render(<><EmbeddingNotice /><input aria-label="Chat message" /></>);
    fireEvent.change(screen.getByLabelText('Chat message'), { target: { value: 'Hello' } });
    expect(screen.getByDisplayValue('Hello')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Download model to server' })).toBeNull();
  });

  it('polls an explicit download until ready without dispatching another download', async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn().mockResolvedValueOnce(response({ ...missing, state: 'downloading' }))
      .mockResolvedValueOnce(response({ ...missing, state: 'ready', runtime_ready: true }));
    vi.stubGlobal('fetch', fetcher);
    render(<EmbeddingNotice />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByText(/files are downloading/)).toBeTruthy();
    await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
    expect(screen.queryByRole('complementary')).toBeNull();
    expect(fetcher.mock.calls.every(([url]) => url === statusUrl)).toBe(true);
  });

  it('ignores a late download response after a principal/configuration refresh', async () => {
    let resolve!: (value: Response) => void;
    const pending = new Promise<Response>((done) => { resolve = done; });
    let statusReads = 0;
    const fetcher = vi.fn(async (url: string) => {
      if (url !== statusUrl) return pending;
      statusReads += 1;
      return response(statusReads === 1 ? missing : { ...missing, can_download: false });
    });
    vi.stubGlobal('fetch', fetcher);
    const view = render(<EmbeddingNotice key="owner" />);
    fireEvent.click(await screen.findByRole('button', { name: 'Download model to server' }));
    view.rerender(<EmbeddingNotice key="member" />);
    await screen.findByText(/Ask the owner/);
    await act(async () => { resolve(response({ state: 'ready' })); });
    expect(screen.getByText(/Ask the owner/)).toBeTruthy();
    expect(statusReads).toBe(2);
  });

  it('does not abort a slow in-flight status on each polling tick', async () => {
    vi.useFakeTimers();
    let resolve!: (value: Response) => void;
    const slow = new Promise<Response>((done) => { resolve = done; });
    const fetcher = vi.fn().mockResolvedValueOnce(response({ ...missing, state: 'downloading' }))
      .mockReturnValueOnce(slow);
    vi.stubGlobal('fetch', fetcher);
    render(<EmbeddingNotice />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(4500); });
    expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async () => { resolve(response({ ...missing, state: 'ready', runtime_ready: true })); });
    expect(screen.queryByRole('complementary')).toBeNull();
  });

  it.each([
    { state: 'ready', can_activate: true, can_download: false, label: 'Activate local model' },
    { state: 'error', can_activate: false, can_download: true, label: 'Retry model setup' },
  ])('offers an explicit confirmed $label instead of hiding inactive or failed setup', async (setup) => {
    let activated = false;
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === statusUrl) return response(activated
        ? { ...missing, state: 'ready', runtime_ready: true }
        : { ...missing, ...setup, runtime_ready: false });
      expect(JSON.parse(String(init?.body))).toEqual({ model: 'model-a', confirm: true });
      activated = true;
      return response({ state: 'ready' });
    });
    vi.stubGlobal('fetch', fetcher);
    render(<EmbeddingNotice />);
    fireEvent.click(await screen.findByRole('button', { name: setup.label }));
    await vi.waitFor(() => expect(screen.queryByRole('complementary')).toBeNull());
    expect(fetcher).toHaveBeenCalledTimes(3);
  });
});
