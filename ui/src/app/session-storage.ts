import type { SelectedModel } from './types';

export type AppView = 'chat' | 'workspace';

const LAST_SESSION_KEY = 'slavik.last.session';
const LAST_MODEL_KEY = 'slavik.last.model';
const WORKSPACE_EXPLORER_VISIBLE_KEY = 'slavik.workspace.explorer.visible';
const WORKSPACE_PATHS = new Set(['/workspace', '/ui/workspace']);

export const readBrowserHint = (subject: string | null, key: string): string | null => {
  if (typeof window === 'undefined' || !subject || !/^[a-f0-9]{64}$/.test(subject)) return null;
  try { return window.localStorage.getItem(`slavik.principal.${subject}.${key}`); }
  catch { return null; }
};
export const writeBrowserHint = (subject: string | null, key: string, value: string | null): void => {
  if (typeof window === 'undefined' || !subject || !/^[a-f0-9]{64}$/.test(subject)) return;
  try {
    const scopedKey = `slavik.principal.${subject}.${key}`;
    if (value === null) window.localStorage.removeItem(scopedKey);
    else window.localStorage.setItem(scopedKey, value);
  } catch { /* Browser hints are optional when storage is unavailable. */ }
};
export const loadLastSessionId = (subject: string | null): string | null => readBrowserHint(subject, LAST_SESSION_KEY)?.trim() || null;
export const saveLastSessionId = (subject: string | null, sessionId: string | null): void => writeBrowserHint(subject, LAST_SESSION_KEY, sessionId);
export const loadLastModel = (subject: string | null): SelectedModel | null => {
  const raw = readBrowserHint(subject, LAST_MODEL_KEY);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && 'provider' in parsed && 'model' in parsed && typeof parsed.provider === 'string' && typeof parsed.model === 'string') return { provider: parsed.provider, model: parsed.model };
  } catch { return null; }
  return null;
};
export const saveLastModel = (subject: string | null, model: SelectedModel | null): void => {
  if (model) writeBrowserHint(subject, LAST_MODEL_KEY, JSON.stringify(model));
};
export const loadWorkspaceExplorerVisible = (subject: string | null): boolean => readBrowserHint(subject, WORKSPACE_EXPLORER_VISIBLE_KEY) !== 'false';
export const saveWorkspaceExplorerVisible = (subject: string | null, visible: boolean): void => writeBrowserHint(subject, WORKSPACE_EXPLORER_VISIBLE_KEY, visible ? 'true' : 'false');

export const viewFromPathname = (pathname: string): AppView =>
  WORKSPACE_PATHS.has(pathname) ? 'workspace' : 'chat';

export const pathForView = (view: AppView): string => (view === 'workspace' ? '/workspace' : '/ui/');
