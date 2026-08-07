import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import React from 'react';

import { GitPanel } from '../../features/workspace/git-panel';
import type { GitStatusResult } from '../../features/workspace/workspace-api';
import * as workspaceApi from '../../features/workspace/workspace-api';

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
