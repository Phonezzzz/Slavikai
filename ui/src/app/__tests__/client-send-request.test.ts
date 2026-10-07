import { beforeEach, expect, it, vi } from 'vitest';
import { acknowledgeClientSend, markClientSendRegenerated, prepareClientSend } from '../client-send-request';

beforeEach(() => {
  localStorage.clear();
  let next = 0;
  vi.stubGlobal('crypto', {
    randomUUID: () => `request-${++next}`,
    subtle: { digest: async (_algorithm: string, input: Uint8Array) => input.buffer },
  });
});

it('retains one send identity through lost acknowledgement and reload', async () => {
  const first = await prepareClientSend('session', 'sensitive request');
  markClientSendRegenerated('session', first.key);
  vi.resetModules();
  const reloaded = await import('../client-send-request');
  const retry = await reloaded.prepareClientSend('session', 'sensitive request');
  expect(retry.key).toBe(first.key);
  expect(retry.regenerated).toBe(true);
  expect(localStorage.getItem('slavikai:pending-send:session')).not.toContain('sensitive request');
  reloaded.acknowledgeClientSend('session', retry.key);
  expect((await reloaded.prepareClientSend('session', 'sensitive request')).key).not.toBe(first.key);
});

it('isolates sessions and intentional replacement, and fences late acknowledgement', async () => {
  const first = await prepareClientSend('a', 'first');
  const other = await prepareClientSend('b', 'first');
  expect(other.key).not.toBe(first.key);
  const replacement = await prepareClientSend('a', 'changed');
  acknowledgeClientSend('a', first.key);
  expect((await prepareClientSend('a', 'changed')).key).toBe(replacement.key);
});

it('sends the stable identity through the actual transport on retry', async () => {
  const { act, renderHook } = await import('@testing-library/react');
  const { useSessionTransport } = await import('../use-session-transport');
  vi.stubGlobal('EventSource', class { close() {} });
  const headers: HeadersInit[] = [];
  let lost = true;
  vi.stubGlobal('fetch', vi.fn(async (_url: string, options: RequestInit) => {
    headers.push(options.headers ?? {});
    if (lost) { lost = false; throw new Error('response lost'); }
    return { ok: true, json: async () => ({ messages: [] }), headers: new Headers() };
  }));
  const { result, unmount } = renderHook(() => useSessionTransport({
    sessionHeader: 'X-Slavik-Session', selectedConversation: 'session', forceCanvasNext: false,
    consumeForceCanvasNext: vi.fn(), onSessionIdChange: vi.fn(), onStatusMessage: vi.fn(),
    onStreamWarning: vi.fn(), onRuntimePayload: vi.fn(), onOpenStreamedArtifact: vi.fn(),
    setArtifactViewerArtifactId: vi.fn(), loadSessions: async () => [],
  }));
  await act(async () => { expect(await result.current.handleSendChat({ content: 'hello' })).toBe(false); });
  await act(async () => { expect(await result.current.handleSendChat({ content: 'hello' })).toBe(true); });
  const first = new Headers(headers[0]).get('Idempotency-Key');
  expect(first).toBeTruthy();
  expect(new Headers(headers[1]).get('Idempotency-Key')).toBe(first);
  unmount();
});
