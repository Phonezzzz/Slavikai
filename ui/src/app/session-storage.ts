import type { SelectedModel } from './types';

export type AppView = 'chat' | 'workspace';

const LAST_SESSION_KEY = 'slavik.last.session';
const LAST_MODEL_KEY = 'slavik.last.model';
const WORKSPACE_EXPLORER_VISIBLE_KEY = 'slavik.workspace.explorer.visible';
const WORKSPACE_PATHS = new Set(['/workspace', '/ui/workspace']);

export const loadLastSessionId = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  const raw = window.localStorage.getItem(LAST_SESSION_KEY);
  if (!raw || !raw.trim()) {
    return null;
  }
  return raw.trim();
};

export const saveLastSessionId = (sessionId: string | null): void => {
  if (typeof window === 'undefined') {
    return;
  }
  if (!sessionId) {
    window.localStorage.removeItem(LAST_SESSION_KEY);
    return;
  }
  window.localStorage.setItem(LAST_SESSION_KEY, sessionId);
};

export const loadLastModel = (): SelectedModel | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  const raw = window.localStorage.getItem(LAST_MODEL_KEY);
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as { provider?: unknown; model?: unknown };
    if (typeof parsed.provider === 'string' && typeof parsed.model === 'string') {
      return { provider: parsed.provider, model: parsed.model };
    }
  } catch {
    return null;
  }
  return null;
};

export const saveLastModel = (model: SelectedModel | null): void => {
  if (typeof window === 'undefined' || !model) {
    return;
  }
  window.localStorage.setItem(LAST_MODEL_KEY, JSON.stringify(model));
};

export const loadWorkspaceExplorerVisible = (): boolean => {
  if (typeof window === 'undefined') {
    return true;
  }
  return window.localStorage.getItem(WORKSPACE_EXPLORER_VISIBLE_KEY) !== 'false';
};

export const saveWorkspaceExplorerVisible = (visible: boolean): void => {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(WORKSPACE_EXPLORER_VISIBLE_KEY, visible ? 'true' : 'false');
};

export const viewFromPathname = (pathname: string): AppView =>
  WORKSPACE_PATHS.has(pathname) ? 'workspace' : 'chat';

export const pathForView = (view: AppView): string => (view === 'workspace' ? '/workspace' : '/ui/');
