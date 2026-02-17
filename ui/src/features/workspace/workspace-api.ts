import { getNumber, getRecord, getString, isRecord } from '../../codecs/guards';
import {
  explainWorkspaceFailure,
  extractApiError,
  parseWorkspaceTree,
  type WorkspaceNode,
} from './workspace-helpers';

type JsonResponse = {
  response: Response;
  payload: unknown;
};

const fetchJson = async (
  url: string,
  init?: RequestInit,
): Promise<JsonResponse> => {
  const response = await fetch(url, init);
  const payload: unknown = await response.json();
  return { response, payload };
};

const throwWorkspaceError = (payload: unknown, fallback: string): never => {
  const errorInfo = extractApiError(payload, fallback);
  throw new Error(explainWorkspaceFailure(errorInfo));
};

export const fetchWorkspaceRoot = async (
  headers: Record<string, string>,
): Promise<{ rootPath: string; policy: Record<string, unknown> | null }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/root', { headers });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load workspace root.');
  }
  return {
    rootPath: getString(payload, 'root_path') ?? '',
    policy: getRecord(payload, 'policy'),
  };
};

export const fetchWorkspaceTree = async (
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; tree: WorkspaceNode[] }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/tree', { headers });
  if (response.status === 202) {
    return { pendingApproval: true, tree: [] };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load workspace tree.');
  }
  return {
    pendingApproval: false,
    tree: parseWorkspaceTree(isRecord(payload) ? payload.tree : null),
  };
};

export const fetchWorkspaceGitDiff = async (
  headers: Record<string, string>,
): Promise<string> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/git-diff', { headers });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load git diff.');
  }
  return getString(payload, 'diff') ?? '';
};

export const fetchWorkspaceFile = async (
  path: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; content: string }> => {
  const { response, payload } = await fetchJson(
    `/ui/api/workspace/file?path=${encodeURIComponent(path)}`,
    { headers },
  );
  if (response.status === 202) {
    return { pendingApproval: true, content: '' };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load file.');
  }
  return {
    pendingApproval: false,
    content: getString(payload, 'content') ?? '',
  };
};

export const putWorkspaceFile = async (
  path: string,
  content: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/file', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ path, content }),
  });
  if (response.status === 202) {
    return { pendingApproval: true };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to save file.');
  }
  return { pendingApproval: false };
};

export const postWorkspaceRun = async (
  path: string,
  headers: Record<string, string>,
): Promise<{
  pendingApproval: boolean;
  stdout: string;
  stderr: string;
  exitCode: number;
}> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/run', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ path }),
  });
  if (response.status === 202) {
    return {
      pendingApproval: true,
      stdout: '',
      stderr: '',
      exitCode: 0,
    };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to run file.');
  }
  return {
    pendingApproval: false,
    stdout: getString(payload, 'stdout')?.trim() ?? '',
    stderr: getString(payload, 'stderr')?.trim() ?? '',
    exitCode: getNumber(payload, 'exit_code') ?? 0,
  };
};

export const postWorkspaceRootSelect = async (
  rootPath: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/root/select', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ root_path: rootPath }),
  });
  if (response.status === 202) {
    return { pendingApproval: true };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to change workspace root.');
  }
  return { pendingApproval: false };
};

export const postWorkspaceIndex = async (
  headers: Record<string, string>,
): Promise<{
  total: number;
  processed: number;
  indexedCode: number;
  indexedDocs: number;
  skipped: number;
}> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/index', {
    method: 'POST',
    headers,
  });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to index workspace.');
  }
  return {
    total: getNumber(payload, 'total') ?? 0,
    processed: getNumber(payload, 'processed') ?? 0,
    indexedCode: getNumber(payload, 'indexed_code') ?? 0,
    indexedDocs: getNumber(payload, 'indexed_docs') ?? 0,
    skipped: getNumber(payload, 'skipped') ?? 0,
  };
};
