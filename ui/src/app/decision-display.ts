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
  return {
    decision: nextDecision,
    isPending,
    isBlocking: isPending && nextDecision.blocking === true,
    shouldRender: isPending,
    busy,
    error,
  };
};
