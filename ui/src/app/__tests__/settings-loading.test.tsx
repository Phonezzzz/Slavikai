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
    expect(settingsGets).toBe(2);
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
    await vi.waitFor(() => expect(gets).toBe(2));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await vi.waitFor(() => expect(gets).toBe(3));
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

  it('restarts a hanging embedding status on explicit Recheck and ignores its late body', async () => {
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
          const response = json({});
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json({ model: 'rechecked-model', state: 'ready' });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('keep draft during retry');
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText('rechecked-model is available locally.');
    expect(signals[0]?.aborted).toBe(true);
    await act(async () => { oldBody.resolve({ model: 'stale-model', state: 'missing' }); });
    expect(screen.getByText('rechecked-model is available locally.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('keep draft during retry');
  });

  it('refreshes credential diagnostics without applying remote preferences or waiting for model probes', async () => {
    let gets = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) {
        gets += 1;
        return json({ settings: {
          ...settings('remote instructions').settings,
          audio: { tts: { api_key_set: gets > 1, backend_ready: gets > 1 } },
        } });
      }
      if (url === EMBEDDINGS_URL || url === MODELS_URL) return new Promise<Response>(() => {});
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('local unsaved instructions');
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    expect(screen.getByText('not ready')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findByText('ready');
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('local unsaved instructions');
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
  });

  it.each([200, 503])('ignores a stale credential check after reopening (status=%s)', async (status) => {
    const oldBody = deferred<unknown>();
    let gets = 0;
    let oldSignal: AbortSignal | null | undefined;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        gets += 1;
        if (gets === 2) {
          oldSignal = init?.signal;
          const response = json({}, status);
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json({ settings: {
          ...settings().settings,
          audio: { tts: { api_key_set: gets > 2, backend_ready: gets > 2 } },
        } });
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(gets).toBe(2));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByText('ready');
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => { oldBody.resolve(status === 200 ? settings() : { error: { message: 'Stale credential failure' } }); });
    expect(screen.getByText('ready')).toBeTruthy();
    expect(screen.queryByText(/Stale credential failure/)).toBeNull();
  });

  it('keeps credential refresh errors local to diagnostics', async () => {
    let gets = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) {
        gets += 1;
        return gets === 1 ? json(settings()) : json({ error: { message: 'Snapshot offline' } }, 503);
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('preserve after credential failure');
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findByText('Credential check: Snapshot offline');
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    expect(screen.queryByText(/Snapshot offline/)).toBeNull();
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('preserve after credential failure');
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
  });

  it.each([
    ['Save changes', 200], ['Save changes', 503], ['Load models', 200], ['Load models', 503],
  ] as const)('invalidates old provider checks after key save via %s (status=%s)', async (action, status) => {
    const oldBody = deferred<unknown>();
    const signals: AbortSignal[] = [];
    const posts: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') posts.push(JSON.parse(String(init.body)));
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === '/ui/api/models?provider=deepseek&strict=1') {
        return json({ providers: [{ provider: 'deepseek', models: ['new-model'] }] });
      }
      if (url === MODELS_URL) {
        if (init?.signal) signals.push(init.signal);
        const response = json({}, status);
        vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
        return response;
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(signals).toHaveLength(1));
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement-test-key' } });
    fireEvent.click(screen.getByRole('button', { name: action }));
    await screen.findByText(action === 'Save changes' ? 'Global settings saved.' : 'Loaded 1 deepseek models.');
    expect(posts[0]?.providers).toEqual({ deepseek: { api_key: 'replacement-test-key' } });
    expect(signals[0]?.aborted).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    await act(async () => {
      oldBody.resolve(status === 200
        ? { providers: [{ provider: 'deepseek', models: ['old1', 'old2', 'old3'] }] }
        : { error: { message: 'Old credential failed' } });
    });
    expect(screen.queryByText('3 models')).toBeNull();
    expect(screen.queryByText(/Old credential failed/)).toBeNull();
    expect(screen.queryAllByText('checking')).toHaveLength(0);
  });

  it.each([
    ['Save changes', 200], ['Save changes', 503], ['Load models', 200], ['Load models', 503],
  ] as const)('keeps a new Recheck across delayed %s JSON (status=%s)', async (action, status) => {
    const saveBody = deferred<unknown>();
    const probeSignals: AbortSignal[] = [];
    let posts = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          posts += 1;
          const response = json({}, status);
          vi.spyOn(response, 'json').mockReturnValue(saveBody.promise);
          return response;
        }
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === '/ui/api/models?provider=deepseek&strict=1') {
        return json({ providers: [{ provider: 'deepseek', models: ['new-model'] }] });
      }
      if (url === MODELS_URL) {
        if (init?.signal) probeSignals.push(init.signal);
        return json({ providers: [{ provider: 'deepseek', models: ['new1', 'new2'] }] });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement-test-key' } });
    fireEvent.click(screen.getByRole('button', { name: action }));
    await vi.waitFor(() => expect(posts).toBe(1));
    fireEvent.click(screen.getByRole('button', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await act(async () => { await Promise.resolve(); });
    expect(probeSignals).toHaveLength(0);
    expect(screen.getAllByText('checking').length).toBeGreaterThan(0);
    await act(async () => { saveBody.resolve(status === 200 ? settings() : { error: { message: 'Key save failed' } }); });
    await screen.findByText('2 models');
    expect(probeSignals).toHaveLength(1);
    expect(probeSignals[0]?.aborted).toBe(false);
  });

  it('rechecks the committed embedding model and ignores the previous model status', async () => {
    const oldBody = deferred<unknown>();
    const signals: AbortSignal[] = [];
    let model = 'old-model';
    let checks = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          model = JSON.parse(String(init.body)).memory.embeddings.local_model;
        }
        return json({ settings: { ...settings().settings, memory: { embeddings: { provider: 'local', local_model: model } } } });
      }
      if (url === EMBEDDINGS_URL) {
        checks += 1;
        if (init?.signal) signals.push(init.signal);
        if (checks === 1) {
          const response = json({});
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json({ model, state: 'ready' });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    fireEvent.change(screen.getByDisplayValue('old-model'), { target: { value: 'new-model' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('new-model is available locally.');
    expect(signals[0]?.aborted).toBe(true);
    await act(async () => { oldBody.resolve({ model: 'old-model', state: 'ready' }); });
    expect(screen.getByText('new-model is available locally.')).toBeTruthy();
    expect(screen.queryByText('old-model is available locally.')).toBeNull();
  });

  it.each(['Save changes', 'Load models'])('reopens the form during %s without applying the old write response', async (action) => {
    const oldBody = deferred<unknown>();
    let posts = 0;
    let strictCalls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          posts += 1;
          const response = json({});
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json(settings('reopened server value'));
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === '/ui/api/models?provider=deepseek&strict=1') {
        strictCalls += 1;
        return json({ providers: [{ provider: 'deepseek', models: ['obsolete-model'] }] });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Show advanced editor' });
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'test-key' } });
    fireEvent.click(screen.getByRole('button', { name: action }));
    await vi.waitFor(() => expect(posts).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    await editInstructions('reopened unsaved draft');
    await act(async () => { oldBody.resolve(settings('stale saved snapshot')); });
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('reopened unsaved draft');
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
    expect(strictCalls).toBe(0);
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
