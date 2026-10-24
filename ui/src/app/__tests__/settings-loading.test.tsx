import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { Settings } from '../components/Settings';

const SETTINGS_URL = '/ui/api/settings';
const MODELS_URL = '/ui/api/models?skip_custom_probe=1&discovery_only=1';
const EMBEDDINGS_URL = '/ui/api/embeddings/status';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function json(payload: unknown, status = 200): Response {
  if (payload && typeof payload === 'object' && 'settings' in payload) {
    const value = payload as { settings: Record<string, unknown> };
    payload = { ...value, settings: { preferences_etag: '"preferences-fixture-0"', can_manage_installation: true, preferences_subject: 'fixture', ...value.settings } };
  }
  return new Response(JSON.stringify(payload), { status });
}

function settings(prompt = 'saved instructions') {
  return { settings: { preferences_etag: '"preferences-fixture-0"', preferences_subject: 'fixture', can_manage_installation: true, personalization: { tone: 'balanced', system_prompt: prompt }, providers: [] } };
}

async function editInstructions(value: string): Promise<HTMLTextAreaElement> {
  await screen.findByRole('textbox', { name: 'Custom instructions' });
  const input = screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' });
  fireEvent.change(input, { target: { value } });
  return input;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Settings loading isolation', () => {
  it('offers manual server setup instead of an unusable model download when the package is missing', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'package_missing' });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText(/Local embeddings library is missing/);
    expect(screen.queryByRole('button', { name: 'Download model' })).toBeNull();
  });

  it('replaces ready badge with the runtime error after cached activation fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === '/ui/api/embeddings/download') return json({ error: {
        message: 'Activation failed', code: 'embeddings_model_load_failed',
        details: { model: 'local', state: 'error', error: 'cached files invalid' },
      } }, 503);
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText('local is available locally.');
    fireEvent.click(screen.getByRole('button', { name: 'Check again' }));
    await screen.findByText('cached files invalid');
    expect(screen.queryByText('local is available locally.')).toBeNull();
    expect(screen.queryByText('ready', { exact: true })).toBeNull();
    expect(screen.getByText('error', { exact: true })).toBeTruthy();
  });

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
    await screen.findByText('Settings saved.');
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
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findAllByText('checking');
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
    expect(input.value).toBe('unsaved draft');
    expect(screen.queryByText('Loading settings...')).toBeNull();
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
    await act(async () => { probe.resolve(json({ providers: [{ provider: 'deepseek', models: ['m1', 'm2'], error: null }] })); });
    expect(settingsGets).toBe(2);
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('unsaved draft');
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
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
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findAllByText('Runtime: Provider offline');
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await vi.waitFor(() => expect(gets).toBe(2));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await vi.waitFor(() => expect(gets).toBe(3));
    await act(async () => { first.resolve(json(settings('stale refresh'))); });
    expect(screen.getByText('Loading settings...')).toBeTruthy();
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(true);
    await act(async () => { second.resolve(json(settings('newest refresh'))); });
    await screen.findByRole('textbox', { name: 'Custom instructions' });
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Memory' }));
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
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText('rechecked-model is available locally.');
    expect(signals[0]?.aborted).toBe(true);
    await act(async () => { oldBody.resolve({ model: 'stale-model', state: 'missing' }); });
    expect(screen.getByText('rechecked-model is available locally.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('keep draft during retry');
  });

  it('refreshes credential diagnostics without applying remote preferences or waiting for model probes', async () => {
    let gets = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) {
        gets += 1;
        return json({ settings: {
          ...settings('remote instructions').settings,
          audio: { tts: { api_key_set: gets > 1, api_key_source: gets > 1 ? 'file' : 'missing' } },
        } });
      }
      if (url === EMBEDDINGS_URL || url === MODELS_URL) return new Promise<Response>(() => {});
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('local unsaved instructions');
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    expect(screen.getByText('Not configured')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findByText('Configured');
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
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
          audio: { tts: { api_key_set: gets > 2, api_key_source: gets > 2 ? 'file' : 'missing' } },
        } });
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(gets).toBe(2));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByText('Configured');
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => { oldBody.resolve(status === 200 ? settings() : { error: { message: 'Stale credential failure' } }); });
    expect(screen.getByText('Configured')).toBeTruthy();
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
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await screen.findByText('Credential check: Snapshot offline');
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(signals).toHaveLength(1));
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement-test-key' } });
    fireEvent.click(screen.getByRole('button', { name: action }));
    await screen.findByText(action === 'Save changes' ? 'Settings saved.' : 'Loaded 1 deepseek models.');
    expect(posts[0]?.providers).toEqual({ deepseek: { api_key: 'replacement-test-key' } });
    expect(signals[0]?.aborted).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement-test-key' } });
    fireEvent.click(screen.getByRole('button', { name: action }));
    await vi.waitFor(() => expect(posts).toBe(1));
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
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
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'test-key' } });
    fireEvent.click(screen.getByRole('button', { name: action }));
    await vi.waitFor(() => expect(posts).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
    await editInstructions('reopened unsaved draft');
    await act(async () => { oldBody.resolve(settings('stale saved snapshot')); });
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('reopened unsaved draft');
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
    expect(strictCalls).toBe(0);
  });

  it.each(['none', 'text', 'button', 'switch'])('reconciles a pre-write reopened snapshot while preserving edits (%s)', async (edit) => {
    const pendingPost = deferred<Response>();
    let committed = false;
    let posts = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') { posts += 1; return pendingPost.promise; }
        return json(settings(committed ? 'committed instructions' : 'before write'));
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('committed instructions');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await vi.waitFor(() => expect(posts).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    const input = await screen.findByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' });
    expect(input.value).toBe('before write');
    if (edit === 'text') fireEvent.change(input, { target: { value: 'new unsaved draft' } });
    if (edit === 'button') fireEvent.click(screen.getByRole('button', { name: /Professional/ }));
    if (edit === 'switch') {
      fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
      fireEvent.click(screen.getByRole('switch', { name: 'Convert long paste to file' }));
    }
    await act(async () => {
      committed = true;
      pendingPost.resolve(json(settings('committed instructions')));
    });
    if (edit === 'none') {
      await screen.findByDisplayValue('committed instructions');
      expect(screen.queryByText(/Your draft was kept/)).toBeNull();
    } else {
      await screen.findByText(/Your draft was kept/);
      if (edit === 'switch') {
        expect(screen.getByRole('switch', { name: 'Convert long paste to file' }).getAttribute('aria-checked')).toBe('false');
        fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
      }
      expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value)
        .toBe(edit === 'text' ? 'new unsaved draft' : 'committed instructions');
    }
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(false);
  });

  it.each([false, true])('preserves provider diagnostics across a preference-only save (already completed=%s)', async (completed) => {
    const body = deferred<unknown>();
    const signals: AbortSignal[] = [];
    const posts: Array<Record<string, unknown>> = [];
    let probes = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') posts.push(JSON.parse(String(init.body)));
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      if (url === MODELS_URL) {
        probes += 1;
        if (init?.signal) signals.push(init.signal);
        const response = json({});
        vi.spyOn(response, 'json').mockReturnValue(body.promise);
        return response;
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('preference-only change');
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await vi.waitFor(() => expect(probes).toBe(1));
    const result = { providers: [{ provider: 'deepseek', models: ['one', 'two'] }] };
    if (completed) {
      await act(async () => { body.resolve(result); });
      await screen.findByText('2 models');
    }
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Settings saved.');
    expect(posts[0]).not.toHaveProperty('providers');
    expect(signals[0]?.aborted).toBe(false);
    if (!completed) await act(async () => { body.resolve(result); });
    await screen.findByText('2 models');
    expect(probes).toBe(1);
  });

  it('invalidates an old reconciliation GET when a newer write starts', async () => {
    const firstPost = deferred<Response>();
    const oldBody = deferred<unknown>();
    const readSignals: AbortSignal[] = [];
    let gets = 0;
    let posts = 0;
    const snapshot = (ready: boolean) => ({ settings: {
      ...settings().settings,
      audio: { tts: { api_key_set: ready, api_key_source: ready ? 'file' : 'missing' } },
    } });
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          posts += 1;
          return posts === 1 ? firstPost.promise : json(snapshot(true));
        }
        gets += 1;
        if (init?.signal) readSignals.push(init.signal);
        if (gets === 3) {
          const response = json({});
          vi.spyOn(response, 'json').mockReturnValue(oldBody.promise);
          return response;
        }
        return json(snapshot(false));
      }
      if (url === EMBEDDINGS_URL) return json({ model: 'local', state: 'ready' });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'first-key' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load models' }));
    await vi.waitFor(() => expect(posts).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByLabelText('OpenAI API key');
    await act(async () => { firstPost.resolve(json(snapshot(false))); });
    await vi.waitFor(() => expect(gets).toBe(3));
    fireEvent.change(screen.getByLabelText('OpenAI API key'), { target: { value: 'new-key' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Settings saved.');
    expect(readSignals[2]?.aborted).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    await screen.findByText('Configured');
    await act(async () => { oldBody.resolve(snapshot(false)); });
    expect(screen.getByText('Configured')).toBeTruthy();
    expect(screen.queryByText(/Your draft was kept/)).toBeNull();
  });

  it('waits for a pending download before Recheck reads embedding status', async () => {
    const download = deferred<Response>();
    let downloading = false;
    let reads = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json(settings());
      if (url === EMBEDDINGS_URL) { reads += 1; return json({ model: 'local', state: downloading ? 'downloading' : 'missing' }); }
      if (url === '/ui/api/embeddings/download') return download.promise;
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    fireEvent.click(screen.getByRole('button', { name: 'Download model' }));
    const readsBeforeRecheck = reads;
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await act(async () => { await Promise.resolve(); });
    expect(reads).toBe(readsBeforeRecheck);
    await act(async () => {
      downloading = true;
      download.resolve(json({ model: 'local', state: 'downloading' }));
    });
    await vi.waitFor(() => expect(reads).toBe(readsBeforeRecheck + 1));
    fireEvent.click(screen.getByRole('button', { name: 'Memory' }));
    await screen.findByText('downloading');
  });

  it.each(['provider', 'import'])('does not treat %s controls as persisted preference edits', async (control) => {
    const post = deferred<Response>();
    let committed = false;
    let posts = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') { posts += 1; return post.promise; }
        return json(settings(committed ? 'committed' : 'before'));
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('committed');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await vi.waitFor(() => expect(posts).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByDisplayValue('before');
    if (control === 'provider') {
      fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
      fireEvent.change(screen.getByLabelText('Provider display name'), { target: { value: 'draft provider' } });
    } else {
      fireEvent.click(screen.getByRole('button', { name: 'Data' }));
      fireEvent.change(screen.getByRole('combobox', { name: 'Mode' }), { target: { value: 'replace' } });
    }
    await act(async () => { committed = true; post.resolve(json(settings('committed'))); });
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
    await screen.findByDisplayValue('committed');
    expect(screen.queryByText(/Your draft was kept/)).toBeNull();
  });

  it.each([
    { editDraft: false, status: 200 },
    { editDraft: true, status: 200 },
    { editDraft: true, status: 503 },
  ])('resumes reconciliation after a partial key save (draft=$editDraft, status=$status)', async ({ editDraft, status }) => {
    const firstPost = deferred<Response>();
    let committed = false;
    let posts = 0;
    let gets = 0;
    const payloads: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          posts += 1;
          payloads.push(JSON.parse(String(init.body)));
          return posts === 1 ? firstPost.promise : json(settings('before'), posts === 2 ? status : 200);
        }
        gets += 1;
        return json({ settings: { ...settings(committed ? 'committed' : 'before').settings, appearance: { theme: committed ? 'oled' : 'default' } } });
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      if (url === '/ui/api/models?provider=deepseek&strict=1') return json({ providers: [{ provider: 'deepseek', models: ['test-model'] }] });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('committed');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await vi.waitFor(() => expect(posts).toBe(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    const input = await screen.findByDisplayValue('before');
    if (editDraft) fireEvent.change(input, { target: { value: 'unsaved draft' } });
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'new-key' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load models' }));
    await vi.waitFor(() => expect(posts).toBe(2));
    await act(async () => { committed = true; firstPost.resolve(json(settings('committed'))); });
    await vi.waitFor(() => expect(gets).toBeGreaterThanOrEqual(3));
    if (status === 200) await screen.findByRole('option', { name: 'test-model' });
    else await vi.waitFor(() => expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Load models' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
    const reconciled = await screen.findByDisplayValue(editDraft ? 'unsaved draft' : 'committed');
    fireEvent.change(reconciled, { target: { value: 'next local preference' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await vi.waitFor(() => expect(payloads[payloads.length - 1]?.personalization).toEqual({ tone: 'balanced', system_prompt: 'next local preference' }));
    expect(payloads[payloads.length - 1]).not.toHaveProperty('appearance');
    expect(payloads[payloads.length - 1]).not.toHaveProperty('providers');
  });

  it.each([
    ['endpoint', 'API response'], ['documented', 'Documented list'],
    ['unverified', 'Unverified'], ['not_checked', 'Not checked'],
  ])('shows %s catalog evidence without claiming speech or inference readiness', async (source, label) => {
    const snapshot = { settings: {
      ...settings().settings,
      model: { provider: 'deepseek', model: 'chosen-model' },
      audio: { tts: { api_key_set: true, api_key_source: 'env' } },
    } };
    const posts: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') posts.push(JSON.parse(String(init.body)));
        return json(snapshot);
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [{ provider: 'deepseek', models: ['different-model'], catalog_source: source }] });
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('keep these instructions');
    expect(fetchMock.mock.calls.some(([url]) => url.startsWith('/ui/api/models'))).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Models and connections' }));
    await screen.findByText('Configured');
    expect(screen.getByText('Key source: Environment')).toBeTruthy();
    expect(screen.getByText('Speech inference has not been tested.')).toBeTruthy();
    expect(screen.queryByText('ready')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    if (source === 'not_checked') await screen.findByText('not checked');
    else await screen.findByText(`Catalog source: ${label}`);
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Settings saved.');
    expect(posts[0]).not.toHaveProperty('model');
    expect(posts[0].personalization).toEqual({ tone: 'balanced', system_prompt: 'keep these instructions' });
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

describe('principal preferences', () => {
  it('saves only member preferences with a subject-bound precondition', async () => {
    const writes: Array<{ body: Record<string, unknown>; etag: string | null }> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          writes.push({ body: JSON.parse(String(init.body)), etag: new Headers(init.headers).get('If-Match') });
        }
        return json({ settings: { ...settings().settings, can_manage_installation: false, preferences_etag: '"member-0"' } });
      }
      if (url === EMBEDDINGS_URL) return new Promise<Response>(() => {});
      throw new Error(`Unexpected URL ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('member private draft');
    expect(screen.queryByRole('button', { name: 'Models and connections' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Memory' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Settings saved.');
    expect(writes).toEqual([{ body: { personalization: { tone: 'balanced', system_prompt: 'member private draft' } }, etag: '"member-0"' }]);
  });

  it('keeps a conflicting draft and never retries against a new revision', async () => {
    let writes = 0;
    const onSaved = vi.fn();
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') { writes += 1; return json({ error: { message: 'Preferences changed. Refresh before saving.' } }, 409); }
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      throw new Error(`Unexpected URL ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    const input = await editInstructions('unsaved conflicting draft');
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Preferences changed. Refresh before saving.');
    expect(writes).toBe(1);
    expect(input.value).toBe('unsaved conflicting draft');
    expect(onSaved).not.toHaveBeenCalled();
  });
});

describe('settings scope transactions', () => {
  it('reports installation commit separately when personal CAS conflicts', async () => {
    const onSaved = vi.fn();
    const writes: Array<{ body: Record<string, unknown>; etag: string | null }> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') {
          writes.push({ body: JSON.parse(String(init.body)), etag: new Headers(init.headers).get('If-Match') });
          if (writes.length === 2) return json({ error: { message: 'Preferences changed.' } }, 409);
          return json({ settings: { ...settings().settings, preferences_etag: '"new-revision"' } });
        }
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    await editInstructions('private pending draft');
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Installation settings saved. Preferences were not saved: Preferences changed.');
    expect(writes).toHaveLength(2);
    expect(onSaved).toHaveBeenCalledTimes(1);
    expect(writes[0].body).not.toHaveProperty('personalization');
    expect(writes[1].body).not.toHaveProperty('providers');
    expect(writes[1].etag).toBe('"preferences-fixture-0"');
    fireEvent.click(screen.getByRole('button', { name: 'Conversation and interface' }));
    expect(screen.getByRole<HTMLTextAreaElement>('textbox', { name: 'Custom instructions' }).value).toBe('private pending draft');
  });

  it('does not dispatch an old personal draft after closing and changing identity', async () => {
    const pending = deferred<Response>();
    const writes: Array<Record<string, unknown>> = [];
    let member = false;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') { writes.push(JSON.parse(String(init.body))); return pending.promise; }
        return json({ settings: { ...settings(member ? 'member saved' : 'owner saved').settings, preferences_etag: member ? '"member-0"' : '"owner-0"', preferences_subject: member ? 'member' : 'owner', can_manage_installation: !member } });
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('owner unsaved private');
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await vi.waitFor(() => expect(writes).toHaveLength(1));
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    member = true;
    rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByDisplayValue('member saved');
    await act(async () => pending.resolve(json(settings('owner saved'))));
    await screen.findByDisplayValue('member saved');
    expect(writes).toHaveLength(1);
    expect(writes[0]).not.toHaveProperty('personalization');
  });
});

describe('review regressions', () => {
  it('hides previous principal fields when reopening fails to load settings', async () => {
    let fail = false;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return fail ? json({ error: { message: 'Identity settings unavailable' } }, 503) : json(settings('old principal private'));
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      throw new Error(`Unexpected URL ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    await editInstructions('old private draft');
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    fail = true;
    rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByText('Identity settings unavailable');
    expect(screen.queryByDisplayValue('old private draft')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Models and connections' })).toBeNull();
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save changes' }).disabled).toBe(true);
  });

  it('notifies consumers once after both scope commits, never between them', async () => {
    const personal = deferred<Response>();
    const onSaved = vi.fn();
    let writes = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') { writes += 1; return writes === 1 ? json(settings()) : personal.promise; }
        return json(settings());
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [] });
      throw new Error(`Unexpected URL ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    await editInstructions('new private draft');
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    fireEvent.change(screen.getByLabelText('DeepSeek API key'), { target: { value: 'replacement' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await vi.waitFor(() => expect(writes).toBe(2));
    expect(onSaved).not.toHaveBeenCalled();
    await act(async () => personal.resolve(json(settings('new private draft'))));
    await screen.findByText('Settings saved.');
    expect(onSaved).toHaveBeenCalledTimes(1);
  });
});


describe('provider connection cards', () => {
  it('keeps credential editing and explicit catalog evidence in one owner card', async () => {
    const snapshot = settings();
    const fetchMock = vi.fn(async (url: string) => {
      if (url === SETTINGS_URL) return json(snapshot);
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      if (url === MODELS_URL) return json({ providers: [{ provider: 'deepseek', models: ['m1', 'm2'], catalog_source: 'endpoint' }] });
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Models and connections' }));
    const card = within(screen.getByRole('group', { name: 'DeepSeek connection' }));
    expect(card.getByLabelText('DeepSeek API key')).toBeTruthy();
    expect(card.getByText('unchecked')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Diagnostics' })).toBeNull();
    expect(fetchMock.mock.calls.some(([url]) => url === MODELS_URL)).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Recheck connections' }));
    await card.findByText('2 models');
    expect(card.getByText('Catalog source: API response')).toBeTruthy();
    expect(screen.queryByText('Provider status')).toBeNull();
  });
});


describe('personal settings grouping', () => {
  it.each([false, true])('keeps personal fields together and saves only their groups (owner=%s)', async (owner) => {
    const writes: Array<{ body: unknown; etag: string | null }> = [];
    const snapshot = { settings: { ...settings().settings, can_manage_installation: owner,
      preferences_etag: '"personal-0"', composer: { long_paste_to_file_enabled: true, long_paste_threshold_chars: 19000 } } };
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === SETTINGS_URL) {
        if (init?.method === 'POST') writes.push({ body: JSON.parse(String(init.body)), etag: new Headers(init.headers).get('If-Match') });
        return json(snapshot);
      }
      if (url === EMBEDDINGS_URL) return json({ state: 'ready' });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Conversation and interface' });
    const instructions = await screen.findByRole('textbox', { name: 'Custom instructions' });
    expect(screen.getByRole('button', { name: 'OLED' })).toBeTruthy();
    expect(screen.getByRole('switch', { name: 'Convert long paste to file' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Appearance' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Composer' })).toBeNull();
    const advanced = screen.getByRole('button', { name: 'Advanced input settings' });
    expect(advanced.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('spinbutton', { name: 'Long paste threshold (characters)' })).toBeNull();
    fireEvent.click(advanced);
    expect((screen.getByRole('spinbutton', { name: 'Long paste threshold (characters)' }) as HTMLInputElement).value).toBe('19000');
    fireEvent.click(advanced);
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('No changes to save.');
    expect(writes).toEqual([]);
    fireEvent.change(instructions, { target: { value: 'personal instructions' } });
    fireEvent.click(screen.getByRole('button', { name: /Professional/ }));
    fireEvent.click(screen.getByRole('button', { name: 'OLED' }));
    fireEvent.click(screen.getByRole('switch', { name: 'Convert long paste to file' }));
    fireEvent.click(advanced);
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Long paste threshold (characters)' }), { target: { value: '23000' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Settings saved.');
    expect(writes).toEqual([{ etag: '"personal-0"', body: {
      personalization: { tone: 'professional', system_prompt: 'personal instructions' },
      appearance: { theme: 'oled' },
      composer: { long_paste_to_file_enabled: false, long_paste_threshold_chars: 23000 },
    } }]);
  });
});
