import { FileCode2, FileText } from 'lucide-react';

export type WorkspaceNode = {
  name: string;
  type: 'dir' | 'file';
  path: string;
  hasChildren?: boolean;
  childrenTruncated?: boolean;
  children?: WorkspaceNode[];
};

export type WorkspaceTreeMeta = {
  returnedEntries: number;
  returnedDirs: number;
  returnedFiles: number;
  truncated: boolean;
  truncatedReasons: string[];
  maxDepthApplied: number;
  maxEntries: number;
  maxDirs: number;
  maxFiles: number;
  maxChildrenPerDir: number;
};

export type FlatWorkspaceRow = {
  key: string;
  depth: number;
  node: WorkspaceNode;
  parentKey: string;
};

export type ApiErrorPayload = {
  message: string;
  code: string | null;
  details: Record<string, unknown> | null;
};

export const extractApiError = (payload: unknown, fallback: string): ApiErrorPayload => {
  if (!payload || typeof payload !== 'object') {
    return { message: fallback, code: null, details: null };
  }
  const maybeError = payload as {
    error?: {
      message?: unknown;
      code?: unknown;
      details?: unknown;
    };
  };
  if (maybeError.error && typeof maybeError.error.message === 'string' && maybeError.error.message.trim()) {
    return {
      message: maybeError.error.message,
      code: typeof maybeError.error.code === 'string' ? maybeError.error.code : null,
      details:
        maybeError.error.details && typeof maybeError.error.details === 'object'
          ? (maybeError.error.details as Record<string, unknown>)
          : null,
    };
  }
  return { message: fallback, code: null, details: null };
};

export const explainWorkspaceFailure = (error: ApiErrorPayload): string => {
  if (error.code === 'approval_required') {
    return 'Ожидает подтверждения действия (approval required).';
  }
  const normalized = error.message.toLowerCase();
  if (normalized.includes('safe mode')) {
    return `Блокировано safe-mode: ${error.message}`;
  }
  if (normalized.includes('инструмент') && normalized.includes('отключ')) {
    return `Инструмент отключён в Settings: ${error.message}`;
  }
  return error.message;
};

export const parseWorkspaceTree = (value: unknown): WorkspaceNode[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const parseNode = (raw: unknown): WorkspaceNode | null => {
    if (!raw || typeof raw !== 'object') {
      return null;
    }
    const candidate = raw as {
      name?: unknown;
      type?: unknown;
      path?: unknown;
      has_children?: unknown;
      children_truncated?: unknown;
      children?: unknown;
    };
    if (typeof candidate.name !== 'string' || !candidate.name.trim()) {
      return null;
    }
    if (candidate.type !== 'dir' && candidate.type !== 'file') {
      return null;
    }
    if (candidate.type === 'file') {
      const path =
        typeof candidate.path === 'string' && candidate.path.trim()
          ? candidate.path.trim()
          : candidate.name.trim();
      return {
        name: candidate.name.trim(),
        type: 'file',
        path,
      };
    }
    const path =
      typeof candidate.path === 'string' && candidate.path.trim()
        ? candidate.path.trim()
        : candidate.name.trim();
    const hasChildren = candidate.has_children === true;
    const childrenTruncated = candidate.children_truncated === true;
    const childrenRaw = Array.isArray(candidate.children) ? candidate.children : [];
    const children = childrenRaw
      .map((item) => parseNode(item))
      .filter((item): item is WorkspaceNode => item !== null);
    return {
      name: candidate.name.trim(),
      type: 'dir',
      path,
      hasChildren,
      childrenTruncated,
      children,
    };
  };

  return value
    .map((item) => parseNode(item))
    .filter((item): item is WorkspaceNode => item !== null);
};

export const parseWorkspaceTreeMeta = (value: unknown): WorkspaceTreeMeta => {
  if (!value || typeof value !== 'object') {
    return {
      returnedEntries: 0,
      returnedDirs: 0,
      returnedFiles: 0,
      truncated: false,
      truncatedReasons: [],
      maxDepthApplied: 0,
      maxEntries: 0,
      maxDirs: 0,
      maxFiles: 0,
      maxChildrenPerDir: 0,
    };
  }
  const candidate = value as {
    returned_entries?: unknown;
    returned_dirs?: unknown;
    returned_files?: unknown;
    truncated?: unknown;
    truncated_reasons?: unknown;
    max_depth_applied?: unknown;
    max_entries?: unknown;
    max_dirs?: unknown;
    max_files?: unknown;
    max_children_per_dir?: unknown;
  };
  const truncatedReasons = Array.isArray(candidate.truncated_reasons)
    ? candidate.truncated_reasons
      .filter((item): item is string => typeof item === 'string')
      .map((item) => item.trim())
      .filter((item) => item.length > 0)
    : [];
  return {
    returnedEntries: typeof candidate.returned_entries === 'number' ? candidate.returned_entries : 0,
    returnedDirs: typeof candidate.returned_dirs === 'number' ? candidate.returned_dirs : 0,
    returnedFiles: typeof candidate.returned_files === 'number' ? candidate.returned_files : 0,
    truncated: candidate.truncated === true,
    truncatedReasons,
    maxDepthApplied: typeof candidate.max_depth_applied === 'number' ? candidate.max_depth_applied : 0,
    maxEntries: typeof candidate.max_entries === 'number' ? candidate.max_entries : 0,
    maxDirs: typeof candidate.max_dirs === 'number' ? candidate.max_dirs : 0,
    maxFiles: typeof candidate.max_files === 'number' ? candidate.max_files : 0,
    maxChildrenPerDir:
      typeof candidate.max_children_per_dir === 'number'
        ? candidate.max_children_per_dir
        : 0,
  };
};

export const findFirstFilePath = (nodes: WorkspaceNode[]): string | null => {
  for (const node of nodes) {
    if (node.type === 'file' && node.path) {
      return node.path;
    }
    if (node.type === 'dir' && node.children) {
      const nested = findFirstFilePath(node.children);
      if (nested) {
        return nested;
      }
    }
  }
  return null;
};

export const nodeKey = (node: WorkspaceNode, parentKey: string): string => {
  if (node.type === 'file') {
    return `file:${node.path}`;
  }
  if (node.path) {
    return `dir:${node.path}`;
  }
  const base = parentKey ? `${parentKey}/${node.name}` : node.name;
  return `dir:${base}`;
};

export const flattenWorkspaceTree = (
  nodes: WorkspaceNode[],
  expandedNodes: Set<string>,
): FlatWorkspaceRow[] => {
  const rows: FlatWorkspaceRow[] = [];
  const walk = (items: WorkspaceNode[], parentKey: string, depth: number): void => {
    for (const node of items) {
      const key = nodeKey(node, parentKey);
      rows.push({
        key,
        depth,
        node,
        parentKey,
      });
      if (node.type === 'dir' && expandedNodes.has(key) && node.children && node.children.length > 0) {
        walk(node.children, key, depth + 1);
      }
    }
  };
  walk(nodes, '', 0);
  return rows;
};

export const fileIcon = (path: string) => {
  const normalized = path.toLowerCase();
  if (
    normalized.endsWith('.py')
    || normalized.endsWith('.ts')
    || normalized.endsWith('.tsx')
    || normalized.endsWith('.js')
    || normalized.endsWith('.jsx')
  ) {
    return <FileCode2 className="h-3.5 w-3.5 text-[#6f9cff]" />;
  }
  return <FileText className="h-3.5 w-3.5 text-[#8f8f97]" />;
};

export const terminalTimestamp = (): string => new Date().toLocaleTimeString();

const extensionFromPath = (path: string): string => {
  const index = path.lastIndexOf('.');
  if (index < 0 || index === path.length - 1) {
    return '';
  }
  return path.slice(index + 1).toLowerCase();
};

export const monacoLanguageFromPath = (path: string): string => {
  const ext = extensionFromPath(path);
  if (ext === 'py') return 'python';
  if (ext === 'ts') return 'typescript';
  if (ext === 'tsx') return 'typescript';
  if (ext === 'js') return 'javascript';
  if (ext === 'jsx') return 'javascript';
  if (ext === 'json') return 'json';
  if (ext === 'md') return 'markdown';
  if (ext === 'toml') return 'ini';
  if (ext === 'yaml' || ext === 'yml') return 'yaml';
  if (ext === 'html') return 'html';
  if (ext === 'css') return 'css';
  if (ext === 'sh') return 'shell';
  if (ext === 'sql') return 'sql';
  return 'plaintext';
};

export const compactPath = (path: string, max = 60): string => {
  if (path.length <= max) {
    return path;
  }
  const keep = Math.max(16, max - 3);
  return `...${path.slice(path.length - keep)}`;
};

export const policyLabel = (value: unknown): string => {
  if (value === 'index') return 'Index';
  if (value === 'yolo') return 'YOLO';
  return 'Sandbox';
};
