import { describe, expect, it, vi } from 'vitest';

import { RootReconcileGuard } from '../root-reconcile';

type Deferred = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  promise: Promise<unknown>;
};

const deferred = (): Deferred => {
  let resolve: Deferred['resolve'] = () => {};
  let reject: Deferred['reject'] = () => {};
  const promise = new Promise<unknown>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { resolve, reject, promise };
};

/**
 * Mirrors the controller flow with the REAL production guard:
 * - switch starts  -> guard.invalidate() synchronously (before any await)
 * - reconcile      -> guard.begin() then await fetch; isCurrent() before setWorkspaceRoot
 * - setWorkspaceRoot only when isCurrent() passes
 */
const makeReconcileController = () => {
  const guard = new RootReconcileGuard();
  const appliedRoots: string[] = [];
  let currentSession: string | null = null;

  const switchSession = (sessionId: string | null) => {
    if (sessionId === currentSession) {
      return;
    }
    guard.invalidate();
    currentSession = sessionId;
  };

  const reconcile = async (
    _sessionId: string,
    fetchPromise: Promise<unknown>,
  ): Promise<void> => {
    const generation = guard.begin();
    let payload: unknown = null;
    try {
      payload = await fetchPromise;
    } catch {
      return;
    }
    if (!guard.isCurrent(generation)) {
      return;
    }
    const rootPath =
      payload && typeof payload === 'object'
        ? ((payload as { root_path?: unknown }).root_path as string | undefined)
        : undefined;
    if (rootPath) {
      appliedRoots.push(rootPath);
    }
  };

  return { appliedRoots, switchSession, reconcile };
};

describe('RootReconcileGuard (production class, controller flow)', () => {
  it('applies root when no switch happened', async () => {
    const controller = makeReconcileController();
    const fetchA = deferred();
    controller.switchSession('A');
    const pending = controller.reconcile('A', fetchA.promise);
    fetchA.resolve({ root_path: '/root-A' });
    await pending;
    expect(controller.appliedRoots).toEqual(['/root-A']);
  });

  it('A reconcile resolves while B load pending -> A root MUST NOT apply', async () => {
    const controller = makeReconcileController();
    const fetchA = deferred();
    const fetchB = deferred();

    controller.switchSession('A');
    const reconcileA = controller.reconcile('A', fetchA.promise);

    // user switches to B while A is still in-flight; B's load is pending
    controller.switchSession('B');
    const reconcileB = controller.reconcile('B', fetchB.promise);

    // A resolves after B switch began -> stale, must be dropped
    fetchA.resolve({ root_path: '/root-A-stale' });
    await reconcileA;
    expect(controller.appliedRoots).toEqual([]);

    // B resolves -> current, must apply
    fetchB.resolve({ root_path: '/root-B' });
    await reconcileB;
    expect(controller.appliedRoots).toEqual(['/root-B']);
  });

  it('A -> B -> A reentrant: only the last A reconcile applies', async () => {
    const controller = makeReconcileController();
    const fetchA1 = deferred();
    const fetchB = deferred();
    const fetchA2 = deferred();

    controller.switchSession('A');
    const reconcileA1 = controller.reconcile('A', fetchA1.promise);

    controller.switchSession('B');
    const reconcileB = controller.reconcile('B', fetchB.promise);

    controller.switchSession('A');
    const reconcileA2 = controller.reconcile('A', fetchA2.promise);

    // stale A1 resolves after switch to B -> dropped
    fetchA1.resolve({ root_path: '/root-A1-stale' });
    await reconcileA1;
    // stale B resolves after switch back to A -> dropped
    fetchB.resolve({ root_path: '/root-B-stale' });
    await reconcileB;

    expect(controller.appliedRoots).toEqual([]);

    fetchA2.resolve({ root_path: '/root-A2' });
    await reconcileA2;
    expect(controller.appliedRoots).toEqual(['/root-A2']);
  });

  it('begin() increments generation; isCurrent() tracks latest', () => {
    const guard = new RootReconcileGuard();
    const g1 = guard.begin();
    const g2 = guard.begin();
    expect(guard.isCurrent(g1)).toBe(false);
    expect(guard.isCurrent(g2)).toBe(true);
  });

  it('invalidate() drops any in-flight generation', () => {
    const guard = new RootReconcileGuard();
    const g1 = guard.begin();
    guard.invalidate();
    expect(guard.isCurrent(g1)).toBe(false);
  });
});

describe('controller helper (deterministic deferral)', () => {
  it('vi.fn integration sanity', () => {
    const spy = vi.fn();
    spy('/root');
    expect(spy).toHaveBeenCalledWith('/root');
  });
});
