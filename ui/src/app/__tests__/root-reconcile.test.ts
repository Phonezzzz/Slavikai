import { act, renderHook, waitFor } from '@testing-library/react';
import type { MutableRefObject } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { DEFAULT_COMPOSER_SETTINGS } from '../session-payload';
import type { SessionTransportBridge } from '../session-bridges';
import type { SessionSummary } from '../types';
import { useSessionRuntimeController } from '../use-session-runtime-controller';

vi.mock('../session-security', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../session-security')>();
  return {
    ...actual,
    loadSessionSecuritySummary: vi.fn(async () => actual.DEFAULT_SESSION_SECURITY_SUMMARY),
  };
});

vi.mock('../session-storage', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../session-storage')>();
  return {
    ...actual,
    loadLastSessionId: vi.fn(() => 'C'),
    loadLastModel: vi.fn(() => null),
    saveLastSessionId: vi.fn(),
    saveLastModel: vi.fn(),
  };
});

type Deferred<T> = {
  resolve: (value: T) => void;
  promise: Promise<T>;
};

const deferred = <T,>(): Deferred<T> => {
  let resolve: Deferred<T>['resolve'] = () => {};
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { resolve, promise };
};

const jsonResponse = (payload: unknown): Response =>
  ({
    ok: true,
    status: 200,
    headers: new Headers(),
    json: async () => payload,
  }) as Response;

const sessionSummary = (sessionId: string): SessionSummary => ({
  session_id: sessionId,
  title: sessionId,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  message_count: 0,
  chat_message_count: 0,
  workspace_message_count: 0,
  last_message_lane: null,
});

const sessionPayload = (sessionId: string, workspaceRoot: string) => ({
  session: {
    session_id: sessionId,
    messages: [],
    artifacts: [],
    computer_events: [],
    selected_model: null,
    workspace_root: workspaceRoot,
    mode: 'ask',
    decision: null,
    active_plan: null,
    active_task: null,
    auto_state: null,
    mode_transitions: null,
  },
});

const createHarness = () => {
  const sessionRequests = new Map<string, Deferred<Response>>();
  const rootRequests = new Map<string, Deferred<Response>>();
  const roots = new Map([
    ['C', '/root-C'],
    ['A', '/root-A'],
    ['B', '/root-B'],
  ]);

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const headers = new Headers(init?.headers);
    const sessionId = headers.get('X-Test') ?? '';

    if (url === '/ui/api/workspace/root') {
      const pending = rootRequests.get(sessionId);
      if (pending) {
        return pending.promise;
      }
      return jsonResponse({ root_path: roots.get(sessionId) ?? '' });
    }

    const sessionMatch = url.match(/^\/ui\/api\/sessions\/([^/]+)$/);
    if (sessionMatch) {
      const requestedSession = decodeURIComponent(sessionMatch[1]);
      const pending = sessionRequests.get(requestedSession);
      if (pending) {
        return pending.promise;
      }
      return jsonResponse(
        sessionPayload(requestedSession, roots.get(requestedSession) ?? ''),
      );
    }
    if (url.endsWith('/output')) {
      return jsonResponse({ output: null });
    }
    if (url.endsWith('/files')) {
      return jsonResponse({ files: [] });
    }
    throw new Error(`Unexpected fetch: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);

  const transport: SessionTransportBridge = {
    applyLoadedConversation: vi.fn(),
    applySessionPayload: vi.fn(() => ({ lane: 'chat' as const })),
    clearConversationState: vi.fn(),
  };
  const transportRef: MutableRefObject<SessionTransportBridge | null> = {
    current: transport,
  };
  const sessions = ['C', 'A', 'B'].map(sessionSummary);
  const hook = renderHook(() =>
    useSessionRuntimeController({
      sessionHeader: 'X-Test',
      providerModels: [],
      loadSessions: vi.fn(async () => sessions),
      loadModels: vi.fn(async () => []),
      loadComposerSettings: vi.fn(async () => DEFAULT_COMPOSER_SETTINGS),
      onStatusMessage: vi.fn(),
      transportRef,
    }),
  );

  return { ...hook, sessionRequests, rootRequests };
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('useSessionRuntimeController root reconciliation', () => {
  it('drops A reconcile as soon as switch to B begins, while B load is pending', async () => {
    const harness = createHarness();
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));

    const rootA = deferred<Response>();
    harness.rootRequests.set('A', rootA);
    await act(async () => {
      await harness.result.current.handleSelectConversation('A');
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-A');

    const sessionB = deferred<Response>();
    harness.sessionRequests.set('B', sessionB);
    let switchToB: Promise<void> = Promise.resolve();
    act(() => {
      switchToB = harness.result.current.handleSelectConversation('B');
    });
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('B'));

    await act(async () => {
      rootA.resolve(jsonResponse({ root_path: '/root-A-stale' }));
      await rootA.promise;
      await Promise.resolve();
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-A');

    await act(async () => {
      sessionB.resolve(jsonResponse(sessionPayload('B', '/root-B')));
      await switchToB;
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-B');
  });

  it('A -> B -> A reentrant switch never applies the stale B load', async () => {
    const harness = createHarness();
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));
    await act(async () => {
      await harness.result.current.handleSelectConversation('A');
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-A');

    const sessionB = deferred<Response>();
    harness.sessionRequests.set('B', sessionB);
    let switchToB: Promise<void> = Promise.resolve();
    act(() => {
      switchToB = harness.result.current.handleSelectConversation('B');
    });
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('B'));

    const sessionA2 = deferred<Response>();
    const rootA2 = deferred<Response>();
    harness.sessionRequests.set('A', sessionA2);
    harness.rootRequests.set('A', rootA2);
    let switchBackToA: Promise<void> = Promise.resolve();
    act(() => {
      switchBackToA = harness.result.current.handleSelectConversation('A');
    });
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('A'));

    await act(async () => {
      sessionB.resolve(jsonResponse(sessionPayload('B', '/root-B-stale')));
      await switchToB;
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-A');

    await act(async () => {
      sessionA2.resolve(jsonResponse(sessionPayload('A', '/root-A-current')));
      await switchBackToA;
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-A-current');

    await act(async () => {
      rootA2.resolve(jsonResponse({ root_path: '/root-A-current' }));
      await rootA2.promise;
    });
    expect(harness.result.current.workspaceRoot).toBe('/root-A-current');
  });
});
