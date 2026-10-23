import { act, renderHook, waitFor } from '@testing-library/react';
import type { MutableRefObject } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { loadLastModel, saveLastModel } from '../session-storage';
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

const createHarness = (restore: 'bootstrap' | 'create' | null = null, initialState?: Deferred<Response>) => {
  const restoredModel = { provider: 'local', model: 'restored' };
  vi.mocked(loadLastModel).mockReturnValue(restore ? restoredModel : null);
  const availableModels = [{ provider: 'local', models: ['restored'], error: null }];
  const sessionRequests = new Map<string, Deferred<Response>>();
  const rootRequests = new Map<string, Deferred<Response>>();
  const modelRequests = new Map<string, Deferred<Response>>();
  const workflowRequests = new Map<string, Deferred<Response>>();
  if (initialState) workflowRequests.set('C:/ui/api/state', initialState);
  const createRequests: { current: Deferred<Response> | null } = { current: null };
  const roots = new Map([
    ['C', '/root-C'],
    ['A', '/root-A'],
    ['B', '/root-B'],
  ]);

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const headers = new Headers(init?.headers);
    const sessionId = headers.get('X-Test') ?? '';

    if (url === '/ui/api/sessions' && init?.method === 'POST') {
      if (createRequests.current) return createRequests.current.promise;
      return jsonResponse({ session_id: 'N', selected_model: null });
    }

    const workflowPending = workflowRequests.get(`${sessionId}:${url}`);
    if (workflowPending) return workflowPending.promise;

    if (url === '/ui/api/session-model') {
      const pending = modelRequests.get(sessionId);
      if (pending) return pending.promise;
    }

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
  const onStatusMessage = vi.fn();
  const hook = renderHook(() =>
    useSessionRuntimeController({
      sessionHeader: 'X-Test',
      providerModels: restore ? availableModels : [],
      loadSessions: vi.fn(async () => sessions),
      loadModels: vi.fn(async () => restore === 'bootstrap' ? availableModels : []),
      loadComposerSettings: vi.fn(async () => DEFAULT_COMPOSER_SETTINGS),
      onStatusMessage,
      transportRef,
    }),
  );

  return { ...hook, sessionRequests, rootRequests, modelRequests, workflowRequests, onStatusMessage, fetchMock, createRequests };
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('useSessionRuntimeController root reconciliation', () => {
  const workflowActions = [
    ['handleChangeMode', '/ui/api/mode'],
    ['handlePlanDraft', '/ui/api/plan/draft'],
    ['handlePlanApprove', '/ui/api/plan/approve'],
    ['handlePlanExecute', '/ui/api/plan/execute'],
    ['handlePlanCancel', '/ui/api/plan/cancel'],
  ] as const;
  type WorkflowAction = typeof workflowActions[number][0];
  const invoke = (runtime: ReturnType<typeof useSessionRuntimeController>, action: WorkflowAction) => {
    if (action === 'handleChangeMode') return runtime.handleChangeMode('plan');
    if (action === 'handlePlanDraft') return runtime.handlePlanDraft('goal');
    return runtime[action]();
  };
  const workflowResponse = () => jsonResponse({ mode: 'auto', active_plan: null, active_task: null });

  it.each(workflowActions)('fences late %s success and finally from the next chat request', async (action, endpoint) => {
    const h = createHarness();
    await waitFor(() => expect(h.result.current.selectedConversation).toBe('C'));
    const old = deferred<Response>(); h.workflowRequests.set(`C:${endpoint}`, old);
    let first: Promise<void>;
    act(() => { first = invoke(h.result.current, action); });
    await act(async () => { await h.result.current.handleSelectConversation('B'); });
    expect(h.result.current.modeBusy).toBe(false);
    const next = deferred<Response>(); h.workflowRequests.set(`B:${endpoint}`, next);
    let second: Promise<void>;
    act(() => { second = invoke(h.result.current, action); });
    await act(async () => { old.resolve(workflowResponse()); await first!; });
    expect(h.result.current.sessionMode).toBe('ask');
    expect(h.result.current.modeBusy).toBe(true);
    await act(async () => { next.resolve(workflowResponse()); await second!; });
    expect(h.result.current.sessionMode).toBe('auto');
    expect(h.result.current.modeBusy).toBe(false);
  });

  it.each(workflowActions)('fences late %s error after returning to the same chat', async (action, endpoint) => {
    const h = createHarness();
    await waitFor(() => expect(h.result.current.selectedConversation).toBe('C'));
    const old = deferred<Response>(); h.workflowRequests.set(`C:${endpoint}`, old);
    let first: Promise<void>;
    act(() => { first = invoke(h.result.current, action); });
    await act(async () => { await h.result.current.handleSelectConversation('B'); await h.result.current.handleSelectConversation('C'); });
    h.onStatusMessage.mockClear();
    await act(async () => {
      old.resolve({ ok: false, status: 409, json: async () => ({ error: { message: 'stale workflow' } }) } as Response);
      await first!;
    });
    expect(h.result.current.modeError).toBeNull();
    expect(h.onStatusMessage).not.toHaveBeenCalled();
  });

  it.each(workflowActions)('submits %s only once before React renders', async (action, endpoint) => {
    const h = createHarness();
    await waitFor(() => expect(h.result.current.selectedConversation).toBe('C'));
    const pending = deferred<Response>(); h.workflowRequests.set(`C:${endpoint}`, pending);
    let first: Promise<void>; let duplicate: Promise<void>;
    act(() => { first = invoke(h.result.current, action); duplicate = invoke(h.result.current, action); });
    await act(async () => { pending.resolve(workflowResponse()); await Promise.all([first!, duplicate!]); });
    expect(h.fetchMock.mock.calls.filter(([url]) => url === endpoint)).toHaveLength(1);
  });

  it('does not overwrite a completed mode change with the initial state response', async () => {
    const initial = deferred<Response>();
    const h = createHarness(null, initial);
    await waitFor(() => expect(h.fetchMock.mock.calls.some(([url]) => url === '/ui/api/state')).toBe(true));
    const change = deferred<Response>(); h.workflowRequests.set('C:/ui/api/mode', change);
    let request: Promise<void>;
    act(() => { request = h.result.current.handleChangeMode('auto'); });
    await act(async () => { change.resolve(workflowResponse()); await request!; });
    expect(h.result.current.sessionMode).toBe('auto');
    await act(async () => { initial.resolve(jsonResponse({ mode: 'ask' })); await initial.promise; });
    expect(h.result.current.sessionMode).toBe('auto');
  });

  it('keeps an in-flight workflow request owned when New Chat fails', async () => {
    const h = createHarness();
    await waitFor(() => expect(h.result.current.selectedConversation).toBe('C'));
    const pending = deferred<Response>(); h.workflowRequests.set('C:/ui/api/mode', pending);
    let request: Promise<void>;
    act(() => { request = h.result.current.handleChangeMode('plan'); });
    h.createRequests.current = deferred<Response>();
    let creation: Promise<void>;
    act(() => { creation = h.result.current.handleCreateConversation(); });
    await act(async () => {
      h.createRequests.current!.resolve({ ok: false, status: 500, json: async () => ({}) } as Response);
      await creation!;
      pending.resolve(workflowResponse()); await request!;
    });
    expect(h.result.current.selectedConversation).toBe('C');
    expect(h.result.current.sessionMode).toBe('auto');
    expect(h.result.current.modeBusy).toBe(false);
  });

  it('does not publish a workflow error after unmount', async () => {
    const h = createHarness();
    await waitFor(() => expect(h.result.current.selectedConversation).toBe('C'));
    const pending = deferred<Response>(); h.workflowRequests.set('C:/ui/api/mode', pending);
    let request: Promise<void>;
    act(() => { request = h.result.current.handleChangeMode('plan'); });
    h.onStatusMessage.mockClear();
    h.unmount();
    await act(async () => {
      pending.resolve({ ok: false, status: 409, json: async () => ({ error: { message: 'unmounted' } }) } as Response);
      await request!;
    });
    expect(h.onStatusMessage).not.toHaveBeenCalled();
  });

  it('does not publish an old plan execute confirmation in the next chat', async () => {
    const h = createHarness();
    await waitFor(() => expect(h.result.current.selectedConversation).toBe('C'));
    const pending = deferred<Response>(); h.workflowRequests.set('C:/ui/api/plan/execute', pending);
    let request: Promise<void>;
    act(() => { request = h.result.current.handlePlanExecute(); });
    await act(async () => { await h.result.current.handleSelectConversation('B'); });
    await act(async () => {
      pending.resolve({ ok: false, status: 409, json: async () => ({
        mode: 'plan', error: { code: 'switch_to_act_required' },
        decision: { id: 'old-plan', kind: 'approval', status: 'pending', reason: 'plan', summary: 'old', created_at: '', updated_at: '' },
      }) } as Response);
      await request!;
    });
    expect(h.result.current.sessionMode).toBe('ask');
    expect(h.result.current.pendingDecision).toBeNull();
  });

  it('ignores late model success and keeps the next chat request busy', async () => {
    const harness = createHarness();
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));
    const oldModel = deferred<Response>();
    harness.modelRequests.set('C', oldModel);
    let oldRequest: Promise<boolean>;
    act(() => { oldRequest = harness.result.current.handleSetModel('local', 'old'); });
    await act(async () => { await harness.result.current.handleSelectConversation('B'); });
    expect(harness.result.current.savingModel).toBe(false);
    const newModel = deferred<Response>();
    harness.modelRequests.set('B', newModel);
    let newRequest: Promise<boolean>;
    act(() => { newRequest = harness.result.current.handleSetModel('local', 'new'); });
    await act(async () => {
      oldModel.resolve(jsonResponse({ selected_model: { provider: 'local', model: 'old' } }));
      expect(await oldRequest!).toBe(false);
    });
    expect(harness.result.current.selectedModel).toBeNull();
    expect(harness.result.current.savingModel).toBe(true);
    await act(async () => {
      newModel.resolve(jsonResponse({ selected_model: { provider: 'local', model: 'new' } }));
      expect(await newRequest!).toBe(true);
    });
    expect(harness.result.current.selectedModel?.model).toBe('new');
    expect(harness.result.current.savingModel).toBe(false);
  });
  it('ignores a late error after returning to the original chat', async () => {
    const harness = createHarness();
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));
    const pending = deferred<Response>();
    harness.modelRequests.set('C', pending);
    let request: Promise<boolean>;
    act(() => { request = harness.result.current.handleSetModel('local', 'old'); });
    await act(async () => { await harness.result.current.handleSelectConversation('B'); });
    await act(async () => { await harness.result.current.handleSelectConversation('C'); });
    harness.onStatusMessage.mockClear();
    await act(async () => {
      pending.resolve({ ok: false, status: 409, json: async () => ({ error: { message: 'stale error' } }) } as Response);
      expect(await request!).toBe(false);
    });
    expect(harness.onStatusMessage).not.toHaveBeenCalled();
    expect(harness.result.current.savingModel).toBe(false);
    expect(harness.result.current.selectedModel).toBeNull();
  });

  it('does not submit the same model action twice before React renders', async () => {
    const harness = createHarness();
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));
    const pending = deferred<Response>();
    harness.modelRequests.set('C', pending);
    let request: Promise<boolean>;
    await act(async () => {
      request = harness.result.current.handleSetModel('local', 'new');
      expect(await harness.result.current.handleSetModel('local', 'duplicate')).toBe(false);
    });
    await act(async () => {
      pending.resolve(jsonResponse({ selected_model: { provider: 'local', model: 'new' } }));
      expect(await request!).toBe(true);
    });
    expect(harness.result.current.selectedModel?.model).toBe('new');
  });

  it.each([['bootstrap', true], ['bootstrap', false], ['create', true], ['create', false]] as const)('fences late %s model restoration success=%s after chat switch', async (source, success) => {
    const harness = createHarness(source);
    const pending = deferred<Response>();
    harness.modelRequests.set(source === 'bootstrap' ? 'C' : 'N', pending);
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));
    let creation: Promise<void> | undefined;
    if (source === 'create') {
      act(() => { creation = harness.result.current.handleCreateConversation(); });
    }
    await waitFor(() => expect(harness.fetchMock).toHaveBeenCalledWith('/ui/api/session-model', expect.anything()));
    await act(async () => { await harness.result.current.handleSelectConversation('B'); });
    vi.mocked(saveLastModel).mockClear();
    harness.onStatusMessage.mockClear();
    await act(async () => {
      pending.resolve(success
        ? jsonResponse({ selected_model: { provider: 'local', model: 'restored' } })
        : { ok: false, status: 409, json: async () => ({ error: { message: 'stale restore' } }) } as Response);
      await creation;
      await pending.promise;
    });
    expect(harness.result.current.selectedConversation).toBe('B');
    expect(harness.result.current.selectedModel).toBeNull();
    expect(saveLastModel).not.toHaveBeenCalled();
    expect(harness.onStatusMessage).not.toHaveBeenCalled();
  });

  it('keeps a current model selection when New Chat fails', async () => {
    const harness = createHarness();
    await waitFor(() => expect(harness.result.current.selectedConversation).toBe('C'));
    const model = deferred<Response>();
    const creation = deferred<Response>();
    harness.modelRequests.set('C', model);
    harness.createRequests.current = creation;
    let selection: Promise<boolean>;
    let newChat: Promise<void>;
    act(() => { selection = harness.result.current.handleSetModel('local', 'new'); });
    act(() => { newChat = harness.result.current.handleCreateConversation(); });
    await act(async () => {
      creation.resolve({ ok: false, status: 500, json: async () => ({ error: { message: 'failed create' } }) } as Response);
      await newChat!;
      model.resolve(jsonResponse({ selected_model: { provider: 'local', model: 'new' } }));
      expect(await selection!).toBe(true);
    });
    expect(harness.result.current.selectedConversation).toBe('C');
    expect(harness.result.current.selectedModel?.model).toBe('new');
  });

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
