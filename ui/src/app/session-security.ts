import { policyLabel as formatPolicyLabel } from '../features/workspace/workspace-helpers';

export type SessionSecuritySummary = {
  policyLabel: string;
  yoloActive: boolean;
  safeMode: boolean;
};

export const DEFAULT_SESSION_SECURITY_SUMMARY: SessionSecuritySummary = {
  policyLabel: 'Sandbox',
  yoloActive: false,
  safeMode: true,
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

export const parseSessionSecuritySummary = (value: unknown): SessionSecuritySummary => {
  if (!value || typeof value !== 'object') {
    return DEFAULT_SESSION_SECURITY_SUMMARY;
  }
  const policyRaw = (value as { policy?: unknown }).policy;

  let nextPolicyLabel = DEFAULT_SESSION_SECURITY_SUMMARY.policyLabel;
  let nextYoloActive = DEFAULT_SESSION_SECURITY_SUMMARY.yoloActive;
  let nextSafeMode = DEFAULT_SESSION_SECURITY_SUMMARY.safeMode;

  if (policyRaw && typeof policyRaw === 'object') {
    const profile = (policyRaw as { profile?: unknown }).profile;
    nextPolicyLabel = formatPolicyLabel(profile);
    nextYoloActive = profile === 'yolo';
    nextSafeMode = profile !== 'yolo';
  }

  return {
    policyLabel: nextPolicyLabel,
    yoloActive: nextYoloActive,
    safeMode: nextSafeMode,
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
  return parseSessionSecuritySummary(payload);
};
