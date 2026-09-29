import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { Settings } from '../components/Settings';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('generic provider onboarding', () => {
  it('checks models, saves an instance and keeps the current model unchanged', async () => {
    const calls: Array<{ path: string; body?: Record<string, string> }> = [];
    let saved = false;
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, string> : undefined;
      calls.push({ path: url, body });
      if (url === '/ui/api/settings') {
        return new Response(JSON.stringify({
          settings: {
            model: { provider: 'deepseek', model: 'deepseek-chat' },
            providers: saved ? [{
              provider: 'custom-0123456789abcdef0123456789abcdef',
              display_name: 'Example',
              base_url: 'https://example.test/v1',
              api_key_stored: true,
            }] : [],
          },
        }), { status: 200 });
      }
      if (url === '/ui/api/models?skip_custom_probe=1') {
        return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      }
      if (url === '/ui/api/embeddings/status') {
        return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      }
      if (url === '/ui/api/provider-instances/probe') {
        return new Response(JSON.stringify({
          base_url: 'https://example.test/v1',
          models: ['opaque/model'],
          status: 'ready',
          message: null,
        }), { status: 200 });
      }
      if (url === '/ui/api/provider-instances') {
        saved = true;
        return new Response(JSON.stringify({
          provider: 'custom-0123456789abcdef0123456789abcdef',
          display_name: 'Example', base_url: 'https://example.test/v1',
        }), { status: 200 });
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    const onSaved = vi.fn();
    render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);

    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    await screen.findByRole('button', { name: 'Test connection' });
    fireEvent.change(screen.getByRole('textbox', { name: 'Provider display name' }), { target: { value: 'Example' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test/' } });
    fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'ui-test-secret' } });
    expect((screen.getByRole('button', { name: 'Save provider' }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

    await screen.findByText('Connected. Found 1 models.');
    expect((screen.getByRole('button', { name: 'Save provider' }) as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
    await vi.waitFor(() => expect(onSaved).toHaveBeenCalledOnce());

    const probe = calls.find((call) => call.path === '/ui/api/provider-instances/probe');
    const create = calls.find((call) => call.path === '/ui/api/provider-instances');
    expect(probe?.body?.api_key).toBe('ui-test-secret');
    expect(create?.body?.model).toBeUndefined();
    expect(calls.some((call) => call.path === '/ui/api/settings' && call.body?.model !== undefined)).toBe(false);
    expect(screen.queryByDisplayValue('ui-test-secret')).toBeNull();
    expect(screen.getAllByText('Example').length).toBeGreaterThan(0);
  });

  it('keeps unsaved instructions when a provider is saved', async () => {
    let saved = false;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/ui/api/settings') return new Response(JSON.stringify({ settings: {
        personalization: { tone: 'balanced', system_prompt: 'server value' },
        providers: saved ? [{ provider: 'custom-0123456789abcdef0123456789abcdef', display_name: 'Example', base_url: 'https://example.test/v1' }] : [],
      } }), { status: 200 });
      if (url === '/ui/api/models?skip_custom_probe=1') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      if (url === '/ui/api/embeddings/status') return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      if (url === '/ui/api/provider-instances/probe') return new Response(JSON.stringify({ base_url: 'https://example.test/v1', models: ['opaque/model'], status: 'ready' }), { status: 200 });
      if (url === '/ui/api/provider-instances') { saved = true; return new Response(JSON.stringify({ provider: 'custom-0123456789abcdef0123456789abcdef', display_name: 'Example', base_url: 'https://example.test/v1' }), { status: 200 }); }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Show advanced editor' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Custom instructions' }), { target: { value: 'unsaved value' } });
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Provider display name' }), { target: { value: 'Example' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test' } });
    fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    await screen.findByText('Connected. Found 1 models.');
    fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
    await screen.findByText('Provider saved. Select it in the chat model picker when ready.');
    fireEvent.click(screen.getByRole('button', { name: 'Assistant' }));
    expect((screen.getByRole('textbox', { name: 'Custom instructions' }) as HTMLTextAreaElement).value).toBe('unsaved value');
  });

  it('clears unsaved onboarding fields and key when Settings closes', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/ui/api/settings') return new Response(JSON.stringify({ settings: { providers: [] } }), { status: 200 });
      if (url === '/ui/api/models?skip_custom_probe=1') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      if (url === '/ui/api/embeddings/status') return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      if (url === '/ui/api/provider-instances/probe') return new Response(JSON.stringify({ base_url: 'https://example.test/v1', models: [], status: 'models_unavailable', message: 'Catalog unavailable.' }), { status: 200 });
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Provider display name' }), { target: { value: 'Unsaved' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test' } });
    fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'unsaved-secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    await screen.findByText('Catalog unavailable.');
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await vi.waitFor(() => expect((screen.getByLabelText('New provider API key') as HTMLInputElement).value).toBe(''));
    expect((screen.getByRole('textbox', { name: 'Provider display name' }) as HTMLInputElement).value).toBe('');
    expect((screen.getByRole('textbox', { name: 'Provider Base URL' }) as HTMLInputElement).value).toBe('');
    expect(screen.queryByText('Catalog unavailable.')).toBeNull();
  });

  it('releases onboarding busy state when Settings closes during a slow probe', async () => {
    const probeResolvers: Array<(value: Response) => void> = [];
    const resolveProbe = (index: number, payload: unknown): void => {
      const resolve = probeResolvers[index];
      if (!resolve) {
        throw new Error(`probe ${index} was not started`);
      }
      resolve(new Response(JSON.stringify(payload), { status: 200 }));
    };
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/ui/api/settings') return new Response(JSON.stringify({ settings: { providers: [] } }), { status: 200 });
      if (url === '/ui/api/models?skip_custom_probe=1') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      if (url === '/ui/api/embeddings/status') return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      if (url === '/ui/api/provider-instances/probe') {
        return new Promise<Response>((resolve) => { probeResolvers.push(resolve); });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Provider display name' }), { target: { value: 'Example' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test' } });
    fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    await vi.waitFor(() => expect(probeResolvers.length).toBe(1));
    await screen.findByRole('button', { name: 'Checking...' });

    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    // Reopening triggers an async settings load; wait for it to finish.
    // findByRole keeps the assertion strict: a stuck-busy UI would render
    // 'Checking...' here instead and this would time out.
    await screen.findByRole('button', { name: 'Test connection' });

    // Late completion of the stale probe must not disturb the reopened form.
    resolveProbe(0, { base_url: 'https://example.test/v1', models: [], status: 'ready' });
    await new Promise((resolve) => setTimeout(resolve, 50));
    screen.getByRole('button', { name: 'Test connection' });
    expect(screen.queryByText(/Connected\. Found/)).toBeNull();
  });

  it('does not let a stale probe clear busy state of a newer probe', async () => {
    const probeResolvers: Array<(value: Response) => void> = [];
    const resolveProbe = (index: number, payload: unknown): void => {
      const resolve = probeResolvers[index];
      if (!resolve) {
        throw new Error(`probe ${index} was not started`);
      }
      resolve(new Response(JSON.stringify(payload), { status: 200 }));
    };
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/ui/api/settings') return new Response(JSON.stringify({ settings: { providers: [] } }), { status: 200 });
      if (url === '/ui/api/models?skip_custom_probe=1') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      if (url === '/ui/api/embeddings/status') return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      if (url === '/ui/api/provider-instances/probe') {
        return new Promise<Response>((resolve) => { probeResolvers.push(resolve); });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const fillOnboardingFields = async (): Promise<void> => {
      fireEvent.change(await screen.findByRole('textbox', { name: 'Provider display name' }), { target: { value: 'Example' } });
      fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test' } });
      fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'secret' } });
    };
    const { rerender } = render(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    await fillOnboardingFields();
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    await vi.waitFor(() => expect(probeResolvers.length).toBe(1));

    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} />);
    await fillOnboardingFields();
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    await vi.waitFor(() => expect(probeResolvers.length).toBe(2));
    await screen.findByRole('button', { name: 'Checking...' });

    // Stale probe A completes: newer probe B's busy state must survive.
    resolveProbe(0, { base_url: 'https://example.test/v1', models: ['a'], status: 'ready' });
    await new Promise((resolve) => setTimeout(resolve, 50));
    screen.getByRole('button', { name: 'Checking...' });

    // Newer probe B completes: releases its own busy state.
    resolveProbe(1, { base_url: 'https://example.test/v1', models: ['a'], status: 'ready' });
    await screen.findByText('Connected. Found 1 models.');
    screen.getByRole('button', { name: 'Test connection' });
  });

  it('reconciles a stale save success without touching the new onboarding form', async () => {
    const saveResolvers: Array<(value: Response) => void> = [];
    const probeResolvers: Array<(value: Response) => void> = [];
    const requestedUrls: string[] = [];
    const resolveDeferred = (
      resolvers: Array<(value: Response) => void>,
      index: number,
      payload: unknown,
    ): void => {
      const resolve = resolvers[index];
      if (!resolve) {
        throw new Error('deferred request was not started');
      }
      resolve(new Response(JSON.stringify(payload), { status: 200 }));
    };
    const onSaved = vi.fn();
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      requestedUrls.push(url);
      if (url === '/ui/api/settings') return new Response(JSON.stringify({ settings: { providers: [] } }), { status: 200 });
      if (url === '/ui/api/models?skip_custom_probe=1') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      if (url === '/ui/api/embeddings/status') return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      if (url === '/ui/api/provider-instances/probe') {
        return new Promise<Response>((resolve) => { probeResolvers.push(resolve); });
      }
      if (url === '/ui/api/provider-instances') {
        return new Promise<Response>((resolve) => { saveResolvers.push(resolve); });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    const fillOnboardingFields = async (name: string): Promise<void> => {
      fireEvent.change(await screen.findByRole('textbox', { name: 'Provider display name' }), { target: { value: name } });
      fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test' } });
      fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'secret' } });
    };
    const { rerender } = render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    await fillOnboardingFields('Old');
    fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
    await vi.waitFor(() => expect(saveResolvers.length).toBe(1));

    // Close during the slow save, reopen, let the new load finish, enter new data.
    rerender(<Settings isOpen={false} onClose={() => undefined} />);
    rerender(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    await fillOnboardingFields('New');

    // Start a fresh probe so the new epoch owns the busy state.
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    await vi.waitFor(() => expect(probeResolvers.length).toBe(1));
    await screen.findByRole('button', { name: 'Checking...' });

    // The stale save succeeds on the server: provider A must be reconciled
    // into the list exactly once, but the new form must stay untouched.
    resolveDeferred(saveResolvers, 0, {
      provider: 'custom-0123456789abcdef0123456789abcdef',
      display_name: 'Old',
      base_url: 'https://example.test/v1',
    });
    await vi.waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));

    // New form fields are unchanged.
    expect((screen.getByRole('textbox', { name: 'Provider display name' }) as HTMLInputElement).value).toBe('New');
    expect((screen.getByRole('textbox', { name: 'Provider Base URL' }) as HTMLInputElement).value).toBe('https://example.test');
    expect((screen.getByLabelText('New provider API key') as HTMLInputElement).value).toBe('secret');
    // Stale status/probe state did not overwrite the new form.
    expect(screen.queryByText('Provider saved. Select it in the chat model picker when ready.')).toBeNull();
    expect(screen.queryByText(/Unexpected URL/)).toBeNull();
    // Provider A reconciled exactly once into the saved provider list.
    expect(screen.getAllByText('Old')).toHaveLength(1);
    // The stale save's finally did not clear the new probe's busy state.
    screen.getByRole('button', { name: 'Checking...' });

    // The new probe completes normally afterwards.
    resolveDeferred(probeResolvers, 0, { base_url: 'https://example.test/v1', models: ['m'], status: 'ready' });
    await screen.findByText('Connected. Found 1 models.');
    screen.getByRole('button', { name: 'Test connection' });
  });

  it('keeps a confirmed stale save when a reopened settings load finishes later', async () => {
    let settingsRequests = 0;
    let resolveReopenedSettings: ((response: Response) => void) | undefined;
    let resolveSave: ((response: Response) => void) | undefined;
    const onSaved = vi.fn();
    const emptySettings = () => new Response(JSON.stringify({ settings: { providers: [] } }), { status: 200 });
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/ui/api/settings') {
        settingsRequests += 1;
        if (settingsRequests === 1) return emptySettings();
        return new Promise<Response>((resolve) => { resolveReopenedSettings = resolve; });
      }
      if (url === '/ui/api/models?skip_custom_probe=1') {
        return new Response(JSON.stringify({ providers: [] }), { status: 200 });
      }
      if (url === '/ui/api/embeddings/status') {
        return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      }
      if (url === '/ui/api/provider-instances') {
        return new Promise<Response>((resolve) => { resolveSave = resolve; });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));

    const { rerender } = render(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Provider display name' }), { target: { value: 'Old' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Provider Base URL' }), { target: { value: 'https://example.test' } });
    fireEvent.change(screen.getByLabelText('New provider API key'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save provider' }));
    await vi.waitFor(() => expect(resolveSave).toBeDefined());

    rerender(<Settings isOpen={false} onClose={() => undefined} onSaved={onSaved} />);
    rerender(<Settings isOpen onClose={() => undefined} onSaved={onSaved} />);
    await vi.waitFor(() => expect(resolveReopenedSettings).toBeDefined());

    // The POST commits after the reopened GET has captured its old snapshot.
    resolveSave?.(new Response(JSON.stringify({
      provider: 'custom-0123456789abcdef0123456789abcdef',
      display_name: 'Old',
      base_url: 'https://example.test/v1',
    }), { status: 200 }));
    await vi.waitFor(() => expect(onSaved).toHaveBeenCalledOnce());

    // The older GET arrives last and must not erase the confirmed provider.
    resolveReopenedSettings?.(emptySettings());
    await screen.findByRole('textbox', { name: 'Provider display name' });
    expect(screen.getAllByText('Old')).toHaveLength(1);
  });

  it('loads initial diagnostics from /ui/api/models?skip_custom_probe=1', async () => {
    const requestedUrls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      requestedUrls.push(url);
      if (url === '/ui/api/settings') {
        return new Response(JSON.stringify({ settings: { providers: [] } }), { status: 200 });
      }
      if (url === '/ui/api/models?skip_custom_probe=1') {
        return new Response(JSON.stringify({ providers: [
          { provider: 'deepseek', display_name: 'deepseek', models: ['m1', 'm2'], error: null },
        ] }), { status: 200 });
      }
      if (url === '/ui/api/embeddings/status') {
        return new Response(JSON.stringify({ model: 'local', state: 'ready' }), { status: 200 });
      }
      throw new Error(`Unexpected URL: ${url}`);
    }));
    render(<Settings isOpen onClose={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'API Keys' }));
    // Wait for the initial load to finish (onboarding form is gated on it).
    await screen.findByRole('textbox', { name: 'Provider display name' });
    // Initial Settings load requested exactly the skip_custom_probe diagnostics URL...
    expect(requestedUrls).toContain('/ui/api/models?skip_custom_probe=1');
    expect(requestedUrls).not.toContain('/ui/api/models');
    // ...and consumed its response instead of falling into the loadSettings
    // catch path (any unexpected fetch would surface as an "Unexpected URL" status).
    expect(screen.queryByText(/Unexpected URL/)).toBeNull();
  });
});
