import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { Settings } from '../components/Settings';

const SETTINGS_URL = '/ui/api/settings';
const MODELS_URL = '/ui/api/models?skip_custom_probe=1';
const EMBEDDINGS_URL = '/ui/api/embeddings/status';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status });
}

function settings(prompt = 'saved instructions') {
  return { settings: { personalization: { tone: 'balanced', system_prompt: prompt }, providers: [] } };
}

async function editInstructions(value: string): Promise<HTMLTextAreaElement> {
  fireEvent.click(await screen.findByRole('button', { name: 'Show advanced editor' }));
  const input = screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' });
  fireEvent.change(input, { target: { value } });
  return input;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Settings loading isolation', () => {
  it('loads and saves preferences while embedding status is pending, without probing providers', async () => {
    const embedding = deferred<Response>();
    const posts: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') posts.push(JSON.parse(String(init.body)));
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return embedding.promise;
      if (url === MODELS_URL) return new Promise<Response>(() => {});
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<Settings isOpen onClose={() => undefined} />);

    await editInstructions('edited while status is pending');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Global settings saved.');
    expect(posts[0].personalization).toEqual({ tone: 'balanced', system_prompt: 'edited while status is pending' });
    expect(fetchMock.mock.calls.some(([url]) => url.startsWith('/ui/api/models'))).toBe(false);
  });

  it('keeps unsaved preferences editable during an explicit slow provider check', async () => {
    const probe = deferred<Response>();
    let settingsGets = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) { settingsGets += 1; return json(settings()); }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return probe.promise;
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    const input = await editInstructions('unsaved draft');
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findAllByText('checking');
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    expect(input.value).toBe('unsaved draft');
    expect(screen.queryByText('Loading settings...')).toBeNull();
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
    await act(async () => { probe.resolve(json({ providers: [{ provider: 'deepseek', models: ['m1', 'm2'], error: null }] })); });
    expect(settingsGets).toBe(1);
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('unsaved draft');
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    await screen.findByText('2 models');
  });

  it('shows a provider check error in diagnostics without replacing the form or its draft', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return json({ error: { message: 'Provider offline' } }, 503);
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('keep this draft');
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findAllByText('Runtime: Provider offline');
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('keep this draft');
    expect(screen.queryByText('Provider offline')).toBeNull();
  });

  it('isolates embedding errors from settings loading and saving', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === EMBEDDINGS_URL) return json({ error: { message: 'Embedding status offline' } }, 503);
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('keep despite embedding failure');
    expect(screen.queryByText('Embedding status offline')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText('Embedding status offline');
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
  });

  it('ignores an old settings JSON body after close and reopen even if the transport ignores abort', async () => {
    const oldBody = deferred<unknown>();
    const signals: AbortSignal[] = [];
    let gets = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.signal) signals.push(init.signal);
        gets += 1;
        if (gets === 1) {
          const response = json({});
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json(settings('new session value'));
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await vi.waitFor(() => expect(gets).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('new unsaved draft');
    expect(signals[0]?.aborted).toBe(true);
    await act(async () => { oldBody.resolve(settings('old delayed value')); });
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('new unsaved draft');
  });

  it('keeps the newest Refresh loading state when an older refresh finishes', async () => {
    const first = deferred<Response>();
    const second = deferred<Response>();
    let gets = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) {
        gets += 1;
        if (gets === 2) return first.promise;
        if (gets === 3) return second.promise;
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await act(async () => { first.resolve(json(settings('stale refresh'))); });
    expect(screen.getByText('Loading settings...')).toBeTruthy();
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(true);
    await act(async () => { second.resolve(json(settings('newest refresh'))); });
    fireEvent.click(await screen.findByRole('button', { name: 'Show advanced editor' }));
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('newest refresh');
  });

  it.each([false, true])('ignores an old diagnostic JSON body after reopening (failure=%s)', async (failed) => {
    const oldBody = deferred<unknown>();
    const currentProbe = deferred<Response>();
    const signals: AbortSignal[] = [];
    let probes = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) {
        probes += 1;
        if (init?.signal) signals.push(init.signal);
        if (probes === 1) {
          const response = json({}, failed ? 503 : 200);
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return currentProbe.promise;
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(probes).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(probes).toBe(2));
    expect(signals[0]?.aborted).toBe(true);
    await act(async () => {
      oldBody.resolve(failed
        ? { error: { message: 'Old check failed' } }
        : { providers: [{ provider: 'deepseek', models: ['old1', 'old2', 'old3'] }] });
    });
    expect(screen.getAllByText('checking').length).toBeGreaterThan(0);
    expect(screen.queryByText('3 models')).toBeNull();
    expect(screen.queryByText(/Old check failed/)).toBeNull();
    await act(async () => {
      currentProbe.resolve(json({ providers: [{ provider: 'deepseek', models: ['new'] }] }));
    });
    await screen.findByText('1 models');
  });

  it('ignores an old embedding JSON body after reopening', async () => {
    const oldBody = deferred<unknown>();
    const signals: AbortSignal[] = [];
    let checks = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === MODELS_URL) return json({ providers: [] });
      if (url === EMBEDDINGS_URL) {
        checks += 1;
        if (init?.signal) signals.push(init.signal);
        if (checks === 1) {
          const response = json({}, 503);
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json({ model: 'fresh-model', state: 'ready' });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText('fresh-model is available locally.');
    expect(signals[0]?.aborted).toBe(true);
    await act(async () => { oldBody.resolve({ error: { message: 'Old embedding check failed' } }); });
    expect(screen.getByText('fresh-model is available locally.')).toBeTruthy();
    expect(screen.queryByText(/Old embedding check failed/)).toBeNull();
  });

  it('does not hide a settings failure behind an indefinitely pending embedding status', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json({ error: { message: 'Cannot load preferences' } }, 500);
      if (url === EMBEDDINGS_URL || url === MODELS_URL) return new Promise<Response>(() => {});
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByText('Cannot load preferences');
    expect(screen.queryByText('Loading settings...')).toBeNull();
  });
});
