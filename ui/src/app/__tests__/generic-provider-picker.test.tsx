import { useState } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SessionDrawer } from '../components/session-drawer';
import type { ProviderModels } from '../types';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const provider = 'custom-0123456789abcdef0123456789abcdef';

function renderPicker(result: ProviderModels, onSelectModel = vi.fn()) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url === '/ui/api/session/security') return new Response(JSON.stringify({}), { status: 200 });
    throw new Error(`Unexpected URL: ${url}`);
  }));
  const onLoadProviderModels = vi.fn(async (_selectedProvider: string) => result);
  function Wrapper() {
    const [providers, setProviders] = useState<ProviderModels[]>([
      { provider, displayName: 'Example', models: [], error: null },
    ]);
    return <SessionDrawer
    isOpen
    onClose={() => undefined}
    sessionId="session-test"
    sessionHeader="X-Slavik-Session"
    mode="ask"
    onChangeMode={async () => undefined}
    modelLabel="Model not selected"
    providerModels={providers}
    selectedModelValue={null}
    onLoadProviderModels={async (selectedProvider) => {
      const loaded = await onLoadProviderModels(selectedProvider);
      setProviders([loaded]);
      return loaded;
    }}
    onStartLocalOllama={async () => null}
    onSelectModel={onSelectModel}
  />;
  }
  render(<Wrapper />);
  return { onLoadProviderModels, onSelectModel };
}

describe('chat session model picker for generic providers', () => {
  it('loads a successful catalog and offers only its IDs', async () => {
    const { onLoadProviderModels, onSelectModel } = renderPicker({
      provider, displayName: 'Example', models: ['opaque/model'], error: null, status: 'ready',
    });
    fireEvent.click(screen.getByRole('button', { name: 'Example' }));
    await vi.waitFor(() => expect(onLoadProviderModels).toHaveBeenCalledWith(provider));
    expect(screen.queryByLabelText('Manual model ID')).toBeNull();
    fireEvent.click(await screen.findByRole('button', { name: 'opaque/model' }));
    expect(onSelectModel).toHaveBeenCalledWith(provider, 'opaque/model');
  });

  it('offers manual ID in the picker when the catalog is unavailable', async () => {
    const { onSelectModel } = renderPicker({
      provider, displayName: 'Example', models: [], error: 'Catalog unavailable.', status: 'models_unavailable',
    });
    fireEvent.click(screen.getByRole('button', { name: 'Example' }));
    const manual = await screen.findByRole('textbox', { name: 'Manual model ID' });
    fireEvent.change(manual, { target: { value: 'manual/opaque' } });
    fireEvent.click(screen.getByRole('button', { name: 'Set' }));
    expect(onSelectModel).toHaveBeenCalledWith(provider, 'manual/opaque');
  });

  it('does not offer manual ID when the saved key is rejected', async () => {
    renderPicker({
      provider, displayName: 'Example', models: [], error: 'API key rejected.', status: 'invalid_api_key',
    });
    fireEvent.click(screen.getByRole('button', { name: 'Example' }));
    await screen.findByText('API key rejected.');
    expect(screen.queryByLabelText('Manual model ID')).toBeNull();
  });
});
