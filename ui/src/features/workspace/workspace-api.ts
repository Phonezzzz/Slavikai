import { getNumber, getString, isRecord } from '../../codecs/guards';
import {
  explainWorkspaceFailure,
  extractApiError,
  parseWorkspaceTreeMeta,
  parseWorkspaceTree,
  type WorkspaceTreeMeta,
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
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  return { response, payload };
};

const throwWorkspaceError = (payload: unknown, fallback: string): never => {
  const errorInfo = extractApiError(payload, fallback);
  throw new Error(explainWorkspaceFailure(errorInfo));
};

export const fetchWorkspaceRoot = async (
  headers: Record<string, string>,
): Promise<{ rootPath: string }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/root', { headers });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load workspace root.');
  }
  return {
    rootPath: getString(payload, 'root_path') ?? '',
  };
};

export const fetchWorkspaceTree = async (
  options: { path?: string; recursive?: boolean; maxDepth?: number } | undefined,
  headers: Record<string, string>,
  signal?: AbortSignal,
): Promise<{ pendingApproval: boolean; tree: WorkspaceNode[]; path: string; treeMeta: WorkspaceTreeMeta }> => {
  const params = new URLSearchParams();
  if (options?.path && options.path.trim()) {
    params.set('path', options.path.trim());
  }
  if (options?.recursive) {
    params.set('recursive', '1');
  }
  if (typeof options?.maxDepth === 'number') {
    params.set('max_depth', String(options.maxDepth));
  }
  const query = params.toString();
  const treeUrl = query ? `/ui/api/workspace/tree?${query}` : '/ui/api/workspace/tree';
  const { response, payload } = await fetchJson(treeUrl, { headers, signal });
  if (response.status === 202) {
    return {
      pendingApproval: true,
      tree: [],
      path: '',
      treeMeta: parseWorkspaceTreeMeta(null),
    };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load workspace tree.');
  }
  return {
    pendingApproval: false,
    tree: parseWorkspaceTree(isRecord(payload) ? payload.tree : null),
    path: getString(payload, 'path') ?? '',
    treeMeta: parseWorkspaceTreeMeta(isRecord(payload) ? payload.tree_meta : null),
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
): Promise<{ pendingApproval: boolean; content: string; version: string | null }> => {
  const { response, payload } = await fetchJson(
    `/ui/api/workspace/file?path=${encodeURIComponent(path)}`,
    { headers },
  );
  if (response.status === 202) {
    return { pendingApproval: true, content: '', version: null };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to load file.');
  }
  return {
    pendingApproval: false,
    content: getString(payload, 'content') ?? '',
    version: getString(payload, 'version'),
  };
};

export const putWorkspaceFile = async (
  path: string,
  content: string,
  version: string | null,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; version: string | null }> => {
  const body: Record<string, unknown> = { path, content };
  if (version) {
    body.version = version;
  }
  const { response, payload } = await fetchJson('/ui/api/workspace/file', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify(body),
  });
  if (response.status === 202) {
    return { pendingApproval: true, version: null };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to save file.');
  }
  return { pendingApproval: false, version: getString(payload, 'version') };
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

export const postWorkspaceCommandRunnerRun = async (
  command: string,
  cwdMode: 'session_root' | 'sandbox',
  headers: Record<string, string>,
): Promise<{
  pendingApproval: boolean;
  stdout: string;
  stderr: string;
  exitCode: number;
  cwd: string;
  cwdMode: 'session_root' | 'sandbox';
}> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/terminal/run', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ command, cwd_mode: cwdMode }),
  });
  if (response.status === 202) {
    return {
      pendingApproval: true,
      stdout: '',
      stderr: '',
      exitCode: 0,
      cwd: '',
      cwdMode,
    };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to run command runner command.');
  }
  return {
    pendingApproval: false,
    stdout: getString(payload, 'stdout') ?? '',
    stderr: getString(payload, 'stderr') ?? '',
    exitCode: getNumber(payload, 'exit_code') ?? 0,
    cwd: getString(payload, 'cwd') ?? '',
    cwdMode: getString(payload, 'cwd_mode') === 'sandbox' ? 'sandbox' : 'session_root',
  };
};

export const postWorkspaceFileCreate = async (
  path: string,
  content: string,
  overwrite: boolean,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; version: string | null }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/file/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ path, content, overwrite }),
  });
  if (response.status === 202) {
    return { pendingApproval: true, version: null };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to create file.');
  }
  return { pendingApproval: false, version: getString(payload, 'version') };
};

export const postWorkspaceFileRename = async (
  oldPath: string,
  newPath: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/file/rename', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ old_path: oldPath, new_path: newPath }),
  });
  if (response.status === 202) {
    return { pendingApproval: true };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to rename file.');
  }
  return { pendingApproval: false };
};

export const postWorkspaceFileMove = async (
  fromPath: string,
  toPath: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/file/move', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ from_path: fromPath, to_path: toPath }),
  });
  if (response.status === 202) {
    return { pendingApproval: true };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to move file.');
  }
  return { pendingApproval: false };
};

export const deleteWorkspaceFile = async (
  path: string,
  recursive: boolean,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/file', {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ path, recursive }),
  });
  if (response.status === 202) {
    return { pendingApproval: true };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to delete file.');
  }
  return { pendingApproval: false };
};

export const postWorkspacePatch = async (
  path: string,
  patch: string,
  dryRun: boolean,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; output: string }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/patch', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ path, patch, dry_run: dryRun }),
  });
  if (response.status === 202) {
    return { pendingApproval: true, output: '' };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to apply patch.');
  }
  return { pendingApproval: false, output: getString(payload, 'output') ?? '' };
};

export const postWorkspaceRootSelect = async (
  rootPath: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; rootPath: string }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/root/select', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: JSON.stringify({ root_path: rootPath }),
  });
  if (response.status === 202) {
    return { pendingApproval: true, rootPath: '' };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to change workspace root.');
  }
  return {
    pendingApproval: false,
    rootPath: getString(payload, 'root_path') ?? '',
  };
};

export const postWorkspaceIndex = async (
  headers: Record<string, string>,
): Promise<{ indexedCode: number; indexedDocs: number; skipped: number }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/index', {
    method: 'POST',
    headers,
  });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to index workspace.');
  }
  return {
    indexedCode: getNumber(payload, 'indexed_code') ?? 0,
    indexedDocs: getNumber(payload, 'indexed_docs') ?? 0,
    skipped: getNumber(payload, 'skipped') ?? 0,
  };
};

export type BrowseEntry = {
  name: string;
  path: string;
};

export type BrowseResult = {
  path: string;
  parent: string | null;
  entries: BrowseEntry[];
  truncated: boolean;
  breadcrumbs: BrowseEntry[];
  error?: string;
};

export const fetchWorkspaceBrowse = async (
  path: string,
  headers: Record<string, string>,
): Promise<BrowseResult> => {
  const url = '/ui/api/workspace/browse' + (path ? '?path=' + encodeURIComponent(path) : '');
  const { response, payload } = await fetchJson(url, { headers });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to browse directories.');
  }
  return {
    path: getString(payload, 'path') ?? '',
    parent: getString(payload, 'parent') ?? null,
    entries: Array.isArray((payload as Record<string, unknown>).entries)
      ? ((payload as Record<string, unknown>).entries as Array<Record<string, unknown>>)
        .filter(
          (e): e is BrowseEntry =>
            typeof e.name === 'string' && typeof e.path === 'string',
        )
      : [],
    truncated: (payload as Record<string, unknown>).truncated === true,
    breadcrumbs: Array.isArray(
      (payload as Record<string, unknown>).breadcrumbs,
    )
      ? (
        (payload as Record<string, unknown>).breadcrumbs as Array<
          Record<string, unknown>
        >
      )
        .filter(
          (b): b is BrowseEntry =>
            typeof b.name === 'string' && typeof b.path === 'string',
        )
      : [],
  };
};

export type GitFileEntry = {
  path: string;
  status: string;
};

export type GitStatusResult = {
  ok: boolean;
  error?: string | null;
  branch: string | null;
  upstream: string | null;
  ahead: number;
  behind: number;
  staged: GitFileEntry[];
  unstaged: GitFileEntry[];
  untracked: GitFileEntry[];
  conflicted: GitFileEntry[];
};

export const fetchWorkspaceGitStatus = async (
  headers: Record<string, string>,
  signal?: AbortSignal,
): Promise<GitStatusResult> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/git/status', {
    headers,
    signal,
  });
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to get git status.');
  }
  const data = payload as Record<string, unknown>;
  const parseEntries = (key: string): GitFileEntry[] =>
    Array.isArray(data[key])
      ? (data[key] as Array<Record<string, unknown>>)
        .filter(
          (e): e is GitFileEntry =>
            typeof e.path === 'string' && typeof e.status === 'string',
        )
      : [];
  return {
    ok: data.ok === true,
    error: typeof data.error === 'string' ? data.error : null,
    branch: typeof data.branch === 'string' ? data.branch : null,
    upstream: typeof data.upstream === 'string' ? data.upstream : null,
    ahead: typeof data.ahead === 'number' ? data.ahead : 0,
    behind: typeof data.behind === 'number' ? data.behind : 0,
    staged: parseEntries('staged'),
    unstaged: parseEntries('unstaged'),
    untracked: parseEntries('untracked'),
    conflicted: parseEntries('conflicted'),
  };
};

export const postWorkspaceGitStage = async (
  paths: string[] | null,
  all: boolean,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; ok: boolean; message: string }> => {
  const { response, payload } = await fetchJson('/ui/api/workspace/git/stage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(all ? { all: true } : { paths }),
  });
  if (response.status === 202) {
    return { pendingApproval: true, ok: false, message: 'Approval required.' };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to stage files.');
  }
  const data = payload as Record<string, unknown>;
  return {
    pendingApproval: false,
    ok: data.ok === true,
    message: typeof data.message === 'string' ? data.message : '',
  };
};

export const postWorkspaceGitUnstage = async (
  paths: string[] | null,
  all: boolean,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; ok: boolean; message: string }> => {
  const { response, payload } = await fetchJson(
    '/ui/api/workspace/git/unstage',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(all ? { all: true } : { paths }),
    },
  );
  if (response.status === 202) {
    return { pendingApproval: true, ok: false, message: 'Approval required.' };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to unstage files.');
  }
  const data = payload as Record<string, unknown>;
  return {
    pendingApproval: false,
    ok: data.ok === true,
    message: typeof data.message === 'string' ? data.message : '',
  };
};

export const postWorkspaceGitCommit = async (
  message: string,
  headers: Record<string, string>,
): Promise<{ pendingApproval: boolean; ok: boolean; message: string }> => {
  const { response, payload } = await fetchJson(
    '/ui/api/workspace/git/commit',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify({ message }),
    },
  );
  if (response.status === 202) {
    return { pendingApproval: true, ok: false, message: 'Approval required.' };
  }
  if (!response.ok) {
    throwWorkspaceError(payload, 'Failed to commit.');
  }
  const data = payload as Record<string, unknown>;
  return {
    pendingApproval: false,
    ok: data.ok === true,
    message: typeof data.message === 'string' ? data.message : '',
  };
};
