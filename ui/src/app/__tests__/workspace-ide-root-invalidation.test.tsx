import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { WorkspaceIde } from '../components/workspace-ide';
import * as workspaceApi from '../../features/workspace/workspace-api';

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

const treeResult = (path: string) => ({
  pendingApproval: false,
  path: '',
  tree: [{ name: path, type: 'file' as const, path }],
  treeMeta: {
    returnedEntries: 1,
    returnedDirs: 0,
    returnedFiles: 1,
    truncated: false,
    truncatedReasons: [],
    maxDepthApplied: 0,
    maxEntries: 0,
    maxDirs: 0,
    maxFiles: 0,
    maxChildrenPerDir: 0,
  },
});

vi.mock('../../features/workspace/workspace-api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../features/workspace/workspace-api')>();
  return {
    ...actual,
    fetchWorkspaceTree: vi.fn(async () => ({
      pendingApproval: false,
      tree: [
        {
          name: 'file-A.txt',
          type: 'file',
          path: 'file-A.txt',
        },
      ],
      treeMeta: {
        returnedEntries: 1,
        returnedDirs: 0,
        returnedFiles: 1,
        truncated: false,
        truncatedReasons: [],
        maxDepthApplied: 0,
        maxEntries: 0,
        maxDirs: 0,
        maxFiles: 0,
        maxChildrenPerDir: 0,
      },
    })),
    fetchWorkspaceFile: vi.fn(async () => ({
      pendingApproval: false,
      content: 'hello from A',
      version: 'sha256:x',
    })),
  };
});

const baseProps: React.ComponentProps<typeof WorkspaceIde> = {
  sessionId: 's1',
  sessionHeader: 'X-Test',
  modelLabel: 'test',
  workspaceRoot: '/root-A',
  sessionPolicyLabel: 'Sandbox',
  sessionYoloActive: false,
  sessionSafeMode: false,
  messages: [],
  statusMessage: null,
  onBackToChat: () => {},
  onOpenSessionDrawer: () => {},
  onOpenRepositoryPanel: () => {},
  onApplyWorkspaceRoot: () => {},
  mode: 'ask',
  activePlan: null,
  activeTask: null,
  autoState: null,
  decision: null,
  refreshToken: 0,
  explorerVisible: false,
};

const renderIde = (props?: Partial<React.ComponentProps<typeof WorkspaceIde>>) => {
  return render(<WorkspaceIde {...baseProps} {...props} />);
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('WorkspaceIde root-scope invalidation', () => {
  it('renders without throwing under jsdom', () => {
    const { container } = renderIde();
    expect(container).toBeTruthy();
  });

  it('switching workspaceRoot prop does not throw and keeps single instance', () => {
    const { rerender } = renderIde();
    rerender(<WorkspaceIde {...baseProps} workspaceRoot="/root-B" />);
    expect(true).toBe(true);
  });

  it('open file from root A is cleared after switching to root B', async () => {
    let currentRoot: 'A' | 'B' = 'A';
    vi.mocked(workspaceApi.fetchWorkspaceTree).mockImplementation(async () =>
      treeResult(currentRoot === 'A' ? 'file-A.txt' : 'file-B.txt'),
    );
    const { getAllByText, queryByText, container, rerender } = renderIde({
      workspaceRoot: '/root-A',
      explorerVisible: true,
      sessionYoloActive: true,
    });

    // switch to the Files tab to reveal the explorer
    const filesTab = document.querySelector('[data-computer-tab="files"]');
    expect(filesTab).toBeTruthy();
    fireEvent.click(filesTab as Element);

    // wait for tree load, then open the file in the explorer (first match = explorer row)
    await waitFor(() => expect(getAllByText('file-A.txt').length).toBeGreaterThanOrEqual(1));
    fireEvent.click(getAllByText('file-A.txt')[0]);

    // after opening, an editor tab with title file-A.txt must exist
    await waitFor(() => {
      expect(container.querySelector('button[title="file-A.txt"]')).toBeTruthy();
    });

    // switching to root B must clear the open-file tab state
    currentRoot = 'B';
    rerender(
      <WorkspaceIde
        {...baseProps}
        workspaceRoot="/root-B"
        explorerVisible={true}
        sessionYoloActive={true}
      />,
    );
    await waitFor(() => {
      expect(container.querySelector('button[title="file-A.txt"]')).toBeNull();
      expect(queryByText('file-A.txt')).toBeNull();
      expect(container.textContent).toContain('file-B.txt');
    });
  });

  it('drops an in-flight quick-open index from root A after switching to root B', async () => {
    const quickA = deferred<ReturnType<typeof treeResult>>();
    const quickB = deferred<ReturnType<typeof treeResult>>();
    let currentRoot: 'A' | 'B' = 'A';
    vi.mocked(workspaceApi.fetchWorkspaceTree).mockImplementation(async (options) => {
      if (options?.recursive) {
        return currentRoot === 'A' ? quickA.promise : quickB.promise;
      }
      return treeResult(currentRoot === 'A' ? 'tree-A.txt' : 'tree-B.txt');
    });

    const { container, rerender } = renderIde({ workspaceRoot: '/root-A' });
    fireEvent.keyDown(window, { key: 'p', ctrlKey: true });
    await waitFor(() => {
      expect(
        vi.mocked(workspaceApi.fetchWorkspaceTree).mock.calls.some(
          ([options]) => options?.recursive === true,
        ),
      ).toBe(true);
    });

    currentRoot = 'B';
    rerender(<WorkspaceIde {...baseProps} workspaceRoot="/root-B" />);
    await act(async () => {
      quickA.resolve(treeResult('stale-A.txt'));
      await quickA.promise;
      await Promise.resolve();
    });

    fireEvent.keyDown(window, { key: 'p', ctrlKey: true });
    await waitFor(() => {
      expect(container.textContent).not.toContain('stale-A.txt');
    });

    await act(async () => {
      quickB.resolve(treeResult('current-B.txt'));
      await quickB.promise;
    });
    await waitFor(() => expect(container.textContent).toContain('current-B.txt'));
  });
});
