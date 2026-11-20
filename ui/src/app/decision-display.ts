import type { UiDecision } from './types';

export type DecisionDisplayState = {
  decision: UiDecision | null;
  isPending: boolean;
  isBlocking: boolean;
  shouldRender: boolean;
  busy: boolean;
  error: string | null;
};

export const getDecisionDisplayState = (
  decision: UiDecision | null | undefined,
  busy = false,
  error: string | null = null,
): DecisionDisplayState => {
  const nextDecision = decision ?? null;
  const isPending = nextDecision?.status === 'pending';
  // E02-T04: отмена run по второму дедлайну ручного входа видна в UI как
  // expired user_auth decision (без кнопок), а не исчезает молча.
  const isExpiredUserAuth =
    nextDecision?.status === 'expired' && nextDecision.decision_type === 'user_auth';
  return {
    decision: nextDecision,
    isPending,
    isBlocking: isPending && nextDecision.blocking === true,
    shouldRender: isPending || isExpiredUserAuth,
    busy,
    error,
  };
};
