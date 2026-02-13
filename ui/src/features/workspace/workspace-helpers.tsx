import { FileCode2, FileText } from 'lucide-react';

export type WorkspaceNode = {
  name: string;
  type: 'dir' | 'file';
  path?: string;
  children?: WorkspaceNode[];
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
    const childrenRaw = Array.isArray(candidate.children) ? candidate.children : [];
    const children = childrenRaw
      .map((item) => parseNode(item))
      .filter((item): item is WorkspaceNode => item !== null);
    return {
      name: candidate.name.trim(),
      type: 'dir',
      children,
    };
  };

  return value
    .map((item) => parseNode(item))
    .filter((item): item is WorkspaceNode => item !== null);
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
  const base = parentKey ? `${parentKey}/${node.name}` : node.name;
  if (node.type === 'file' && node.path) {
    return `file:${node.path}`;
  }
  return `dir:${base}`;
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
