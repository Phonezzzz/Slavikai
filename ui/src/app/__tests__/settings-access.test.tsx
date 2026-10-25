import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Settings } from '../components/Settings';
import { GlobalSettingsShell } from '../components/global-settings-shell';

const setup = (owner: boolean) => {
  const value = { settings: { can_manage_installation: owner, preferences_subject: owner ? 'owner' : 'member', preferences_etag: '"fixture-0"',
    personalization: { tone: 'balanced', system_prompt: 'private fixture' },
    memory: { inbox_max_items: 317, inbox_ttl_days: 43, inbox_writes_per_minute: 9,
      embeddings: { provider: 'local', local_model: 'fixture-saved-custom', openai_model: 'fixture-remote-custom' } },
    model: { provider: 'local', model: 'fixture-model' }, tools: { state: { fs: true, shell: false } }, policy: { profile: 'sandbox' },
  } };
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url === '/ui/api/settings') return new Response(JSON.stringify(value));
    if (url === '/ui/api/embeddings/status') return new Response(JSON.stringify({ state: 'missing', model: 'fixture-saved-custom' }));
    throw new Error(`Unexpected URL: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return { value, fetchMock };
};
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('Settings access and developer presentation', () => {
  it.each([false, true])('changes only presentation when Developer view is toggled (owner=%s)', async owner => {
    const h = setup(owner);
    const original = JSON.stringify(h.value);
    const view = render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    const reads = h.fetchMock.mock.calls.length;
    expect(screen.queryByRole('button', { name: 'Developer' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Developer view' }));
    fireEvent.click(screen.getByRole('button', { name: 'Developer' }));
    expect(!!screen.queryByRole('spinbutton', { name: 'Write rate limit' })).toBe(owner);
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('No changes to save.');
    expect(h.fetchMock.mock.calls.length).toBe(reads);
    expect(JSON.stringify(h.value)).toBe(original);
    fireEvent.click(screen.getByRole('button', { name: 'Developer view' }));
    expect(screen.queryByRole('button', { name: 'Developer' })).toBeNull();
    expect(screen.getByRole('textbox', { name: 'Custom instructions' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Access and environment' }));
    expect((screen.getByRole('button', { name: 'Open chat controls' }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Developer view' }));
    view.rerender(<Settings isOpen={false} onClose={() => undefined} />);
    view.rerender(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('button', { name: 'Open chat controls' });
    expect(screen.queryByRole('button', { name: 'Developer' })).toBeNull();
  });

  it('closes Settings and delegates to the existing chat controls through the real shell', async () => {
    const h = setup(false);
    const events: string[] = [];
    render(<GlobalSettingsShell isOpen onClose={() => events.push('close')} onSaved={() => undefined}
      onOpenSessionControls={() => events.push('open')} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Access and environment' }));
    fireEvent.click(screen.getByRole('button', { name: 'Open chat controls' }));
    expect(events).toEqual(['close', 'open']);
    expect(h.fetchMock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false);
  });

  it('preserves hidden advanced values while saving owner operational settings', async () => {
    const h = setup(true);
    render(<Settings isOpen onClose={() => undefined} />);
    await screen.findByRole('textbox', { name: 'Custom instructions' });
    fireEvent.click(screen.getByRole('button', { name: 'Memory and data' }));
    expect(screen.getByRole('button', { name: 'Download model' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Local' })).toBeTruthy();
    expect(screen.queryByRole('spinbutton', { name: 'Inbox size' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Advanced memory settings' }));
    expect((screen.getByRole('textbox', { name: 'Local embedding model' }) as HTMLInputElement).value).toBe('fixture-saved-custom');
    fireEvent.change(screen.getByRole('textbox', { name: 'Local embedding model' }), { target: { value: 'edited-local' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Inbox size' }), { target: { value: '999' } });
    fireEvent.click(screen.getByRole('button', { name: 'Advanced memory settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Developer view' }));
    fireEvent.click(screen.getByRole('button', { name: 'Developer' }));
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Write rate limit' }), { target: { value: '11' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await screen.findByText('Settings saved.');
    const posts = h.fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST');
    expect(posts).toHaveLength(1);
    expect(JSON.parse(String(posts[0][1]?.body))).toEqual({ memory: {
      inbox_max_items: 999, inbox_ttl_days: 43, inbox_writes_per_minute: 11,
      embeddings: { provider: 'local', local_model: 'edited-local', openai_model: 'fixture-remote-custom' },
    } });
  });
});
