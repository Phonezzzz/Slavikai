import { policyLabel as formatPolicyLabel } from '../features/workspace/workspace-helpers';

export type SessionSecuritySummary = {
  policyLabel: string;
  yoloActive: boolean;
  safeMode: boolean | null;
};

export const DEFAULT_SESSION_SECURITY_SUMMARY: SessionSecuritySummary = {
  policyLabel: 'Unavailable',
  yoloActive: false,
  safeMode: null,
};

const extractErrorMessage = (payload: unknown, fallback: string): string => {
  if (!payload || typeof payload !== 'object') {
    return fallback;
  }
  const body = payload as { error?: { message?: unknown } };
  if (body.error && typeof body.error.message === 'string' && body.error.message.trim()) {
    return body.error.message;
  }
  return fallback;
};

export const SESSION_TOOL_KEYS = ['fs', 'shell', 'web', 'project', 'img', 'tts', 'stt'] as const;
export type ToolKey = typeof SESSION_TOOL_KEYS[number];
export type PolicyProfile = 'sandbox' | 'index' | 'yolo';
export type SessionSecurityState = { toolsState: Record<ToolKey, boolean>; policyProfile: PolicyProfile };
const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
export const parseSessionSecurityState = (value: unknown, sessionId?: string): SessionSecurityState | null => {
  if (!record(value) || (sessionId !== undefined && value.session_id !== sessionId)
    || !record(value.policy) || !record(value.tools_state)) return null;
  const profile = value.policy.profile;
  if (profile !== 'sandbox' && profile !== 'index' && profile !== 'yolo') return null;
  const tools = value.tools_state;
  if (tools.safe_mode !== (profile !== 'yolo') || !SESSION_TOOL_KEYS.every(key => typeof tools[key] === 'boolean')) return null;
  return {
    policyProfile: profile,
    toolsState: Object.fromEntries(SESSION_TOOL_KEYS.map(key => [key, tools[key]])) as Record<ToolKey, boolean>,
  };
};
export const parseSessionSecuritySummary = (value: unknown): SessionSecuritySummary => {
  const state = parseSessionSecurityState(value);
  if (!state) return DEFAULT_SESSION_SECURITY_SUMMARY;
  return {
    policyLabel: formatPolicyLabel(state.policyProfile),
    yoloActive: state.policyProfile === 'yolo',
    safeMode: state.policyProfile !== 'yolo',
  };
};

export const loadSessionSecuritySummary = async (
  sessionId: string,
  sessionHeader: string,
): Promise<SessionSecuritySummary> => {
  const response = await fetch('/ui/api/session/security', {
    headers: {
      [sessionHeader]: sessionId,
    },
  });
  const payload: unknown = await response.json();
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, 'Failed to load session controls.'));
  }
  if (!parseSessionSecurityState(payload, sessionId)) throw new Error('Session controls are unavailable or belong to another session.');
  return parseSessionSecuritySummary(payload);
};
