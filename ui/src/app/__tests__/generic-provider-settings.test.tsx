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
      if (url === '/ui/api/models') {
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
    expect(screen.queryByLabelText('Manual model ID')).toBeNull();
    expect(screen.queryByLabelText('Discovered model')).toBeNull();
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
      if (url === '/ui/api/models') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
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
      if (url === '/ui/api/models') return new Response(JSON.stringify({ providers: [] }), { status: 200 });
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
});
