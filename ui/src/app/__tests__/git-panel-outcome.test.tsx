import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, renderHook } from '@testing-library/react';
import React from 'react';

import { GitPanel } from '../../features/workspace/git-panel';
import type { GitStatusResult } from '../../features/workspace/workspace-api';
import * as workspaceApi from '../../features/workspace/workspace-api';
import type { SessionTransportBridge } from '../session-bridges';
import type { UiDecision } from '../types';
import { useRepositoryActions } from '../use-repository-actions';

vi.mock('../../features/workspace/workspace-api', () => ({
  fetchWorkspaceGitStatus: vi.fn(),
  postWorkspaceGitStage: vi.fn(),
  postWorkspaceGitUnstage: vi.fn(),
  postWorkspaceGitCommit: vi.fn(),
}));

const cleanStatus: GitStatusResult = {
  ok: true,
  error: null,
  branch: 'main',
  upstream: null,
  ahead: 0,
  behind: 0,
  staged: [],
  unstaged: [],
  untracked: [],
  conflicted: [],
};

const renderPanel = (props?: Partial<React.ComponentProps<typeof GitPanel>>) => {
  return render(
    <GitPanel
      sessionId="s1"
      sessionHeader="X-Test"
      requestHeaders={{}}
      workspaceRoot="/root"
      refreshToken={0}
      {...props}
    />,
  );
};

const gitDecision: UiDecision = {
  id: 'git-decision-1',
  kind: 'approval',
  decision_type: 'tool_approval',
  status: 'pending',
  blocking: true,
  reason: 'approval_required',
  summary: 'Git commit',
  proposed_action: {},
  options: [],
  default_option_id: null,
  context: { source_endpoint: 'workspace.git' },
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  resolved_at: null,
};

const renderRepositoryActions = () => {
  const transport: SessionTransportBridge = {
    applyLoadedConversation: vi.fn(),
    applySessionPayload: vi.fn(() => ({ lane: 'chat' as const })),
    clearConversationState: vi.fn(),
  };
  return renderHook(() =>
    useRepositoryActions({
      sessionId: 's1',
      sessionHeader: 'X-Test',
      pendingDecision: gitDecision,
      transportRef: { current: transport },
      applyRuntimePayload: vi.fn(),
      loadSessions: vi.fn(async () => []),
      onSessionIdChange: vi.fn(),
      onStatusMessage: vi.fn(),
    }),
  );
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('GitPanel decision outcome', () => {
  it('success outcome: reloads status and clears pending', async () => {
    const fetchStatus = vi
      .mocked(workspaceApi.fetchWorkspaceGitStatus)
      .mockResolvedValue(cleanStatus);
    const { rerender } = renderPanel();
    await vi.waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));

    rerender(
      <GitPanel
        sessionId="s1"
        sessionHeader="X-Test"
        requestHeaders={{}}
        workspaceRoot="/root"
        refreshToken={0}
        decisionOutcome={{ kind: 'success', message: 'Git operation completed.' }}
      />,
    );
    await vi.waitFor(() => expect(fetchStatus.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it('rejected outcome: shows cancelled message without extra reload', async () => {
    const fetchStatus = vi
      .mocked(workspaceApi.fetchWorkspaceGitStatus)
      .mockResolvedValue(cleanStatus);
    const { container, rerender } = renderPanel();
    await vi.waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));

    rerender(
      <GitPanel
        sessionId="s1"
        sessionHeader="X-Test"
        requestHeaders={{}}
        workspaceRoot="/root"
        refreshToken={0}
        decisionOutcome={{ kind: 'rejected', message: 'Git operation cancelled.' }}
      />,
    );
    await vi.waitFor(() => {
      expect(container.textContent).toContain('Git operation cancelled.');
    });
    expect(fetchStatus.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it('failed outcome: shows error to the user', async () => {
    const fetchStatus = vi
      .mocked(workspaceApi.fetchWorkspaceGitStatus)
      .mockResolvedValue(cleanStatus);
    const { container, rerender } = renderPanel();
    await vi.waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));

    rerender(
      <GitPanel
        sessionId="s1"
        sessionHeader="X-Test"
        requestHeaders={{}}
        workspaceRoot="/root"
        refreshToken={0}
        decisionOutcome={{ kind: 'failed', message: 'git commit failed' }}
      />,
    );
    await vi.waitFor(() => {
      expect(container.textContent).toContain('git commit failed');
    });
    expect(fetchStatus.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it('unrelated decision (no outcome) does not mask git state', async () => {
    const fetchStatus = vi
      .mocked(workspaceApi.fetchWorkspaceGitStatus)
      .mockResolvedValue(cleanStatus);
    const { container } = renderPanel();
    await vi.waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));
    expect(container.textContent).not.toContain('Git operation');
  });
});

describe('useRepositoryActions git outcome derivation', () => {
  it('preserves the real backend failure message', () => {
    const hook = renderRepositoryActions();
    act(() => {
      hook.result.current.handleDecisionResume({
        ok: false,
        source_endpoint: 'workspace.git',
        data: { operation: 'commit', message: 'nothing to commit' },
      });
    });
    expect(hook.result.current.gitDecisionOutcome).toEqual({
      kind: 'failed',
      message: 'nothing to commit',
    });
  });

  it('derives success and reject only for workspace.git decisions', () => {
    const hook = renderRepositoryActions();
    act(() => {
      hook.result.current.handleDecisionResume({
        ok: true,
        source_endpoint: 'workspace.git',
        data: { operation: 'stage', message: 'ok' },
      });
    });
    expect(hook.result.current.gitDecisionOutcome).toEqual({
      kind: 'success',
      message: 'Git stage completed.',
    });

    act(() => {
      hook.result.current.handleDecisionRejected();
    });
    expect(hook.result.current.gitDecisionOutcome).toEqual({
      kind: 'rejected',
      message: 'Git operation cancelled.',
    });
  });
});
